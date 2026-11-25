/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#define LOG_TAG "Hentak"

#include <hentak/Hentak.h>

#include "Events.h"
#include "Rust.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include <aidl/android/hardware/vibrator/IVibrator.h>
#include <aidl/vendor/yasvara/hentak/ITagger.h>
#include <android/binder_manager.h>
#include <binder/IBinder.h>
#include <binder/IServiceManager.h>
#include <binder/Parcel.h>
#include <log/log.h>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <utils/String16.h>

namespace android::hentak {
namespace {

constexpr int kSlots = 8;
constexpr int64_t kFree = -1;

struct Slot {
    std::atomic<int64_t> uid{kFree};
    std::atomic<float> sensitivity{0.f};
};
Slot gSlots[kSlots];
constexpr int kNoMmapSlots = 32;
std::atomic<int64_t> gNoMmap[kNoMmapSlots];
std::mutex gWriteLock;
std::atomic<float> gGain{1.f};

// Mirrors com.android.server.yasvara.hentak.HentakService.
constexpr char16_t kService[] = u"yasvara_hentak";
constexpr char16_t kDescriptor[] = u"org.yasvara.hentak.IHentak";
constexpr uint32_t kTransactionPulse = IBinder::FIRST_CALL_TRANSACTION;
constexpr char kFollower[] = "android.hardware.vibrator.IVibrator/hentak";
// the learned tagger in the same HAL, and its session memory (hentak::tagger::shm).
constexpr char kTagger[] = "vendor.yasvara.hentak.ITagger/default";
constexpr size_t kTaggerShmBytes = 4096 + (size_t{1} << 17) * 4;
constexpr int64_t kStepNs = 5'000'000;
constexpr int64_t kLookupNs = 5'000'000'000;
constexpr int64_t kRefreshNs = 50'000'000;  // the HAL treats a level older than 150 ms as 0
// Playout per track: levels arrive in bursts as the mixer consumes buffers (4–20 ms of audio
// each) and leave one per step.
constexpr int kPlays = 8;
constexpr int kFifo = 16;
constexpr int kMaxBacklog = 10;   // 50 ms: beyond it the oldest levels go
constexpr int kHoldSteps = 4;     // a burst running late keeps the last level this long
// ANDROID_PRIORITY_URGENT_AUDIO, as the mixer threads: at nice 0 a game's own threads make the
// 5 ms steps late and the motor's envelope stutters.
constexpr int kSenderNice = -19;
constexpr int kForgetSteps = 40;  // then the track counts as quiet; its slot frees after 200 ms

// Multi-producer (mixer and fast mixer threads), single consumer (sender thread).
constexpr uint32_t kQueue = 256;
struct QueueSlot {
    std::atomic<uint32_t> seq{0};
    Event event;
};
QueueSlot gQueue[kQueue];
std::atomic<uint32_t> gHead{0};
uint32_t gTail = 0;  // sender thread only
std::atomic<int> gEventFd{-1};
std::once_flag gSenderOnce;
std::atomic<uint32_t> gPosted{0}, gSent{0}, gDropped{0}, gLevelsSent{0}, gStarved{0};
std::atomic<bool> gFollowing{false};

int64_t nowNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct Play {
    uintptr_t track = 0;
    float fifo[kFifo];
    int head = 0, size = 0;
    float last = 0;
    int starve = 0;
};

void sendEvent(sp<IBinder>& service, const Event& e) {
    if (service == nullptr || !service->isBinderAlive()) {
        service = defaultServiceManager()->checkService(String16(kService));
        if (service == nullptr) return;
    }
    Parcel data;
    data.writeInterfaceToken(String16(kDescriptor));
    data.writeInt32(e.uid);
    data.writeInt32(e.kind);
    data.writeFloat(e.intensity);
    data.writeFloat(e.sharpness);
    if (service->transact(kTransactionPulse, data, nullptr, IBinder::FLAG_ONEWAY) == OK) {
        gSent.fetch_add(1, std::memory_order_relaxed);
    }
}

void queueLevels(Play (&plays)[kPlays], const Event& e) {
    Play* play = nullptr;
    for (Play& p : plays) {
        if (p.track == e.track) play = &p;
    }
    if (play == nullptr) {
        // A free slot, else the one quiet the longest.
        for (Play& p : plays) {
            if (p.track == 0) {
                play = &p;
                break;
            }
        }
        if (play == nullptr) {
            play = &plays[0];
            for (Play& p : plays) {
                if (p.starve > play->starve) play = &p;
            }
        }
        *play = Play{};
        play->track = e.track;
    }
    for (int i = 0; i < e.count; i++) {
        if (play->size == kFifo) {
            play->head = (play->head + 1) % kFifo;
            play->size--;
        }
        play->fifo[(play->head + play->size) % kFifo] = e.levels[i];
        play->size++;
    }
    while (play->size > kMaxBacklog) {
        play->head = (play->head + 1) % kFifo;
        play->size--;
    }
}

// Next envelope step: the loudest of all tracks. Returns false once every track went quiet.
bool stepLevels(Play (&plays)[kPlays], float* out) {
    bool any = false;
    *out = 0;
    for (Play& p : plays) {
        if (p.track == 0) continue;
        float v;
        if (p.size > 0) {
            v = p.fifo[p.head];
            p.head = (p.head + 1) % kFifo;
            p.size--;
            p.last = v;
            p.starve = 0;
        } else if (++p.starve > kForgetSteps) {
            p = Play{};
            continue;
        } else {
            // A step due with nothing to play: the envelope holds, then gaps.
            if (p.last > 0) gStarved.fetch_add(1, std::memory_order_relaxed);
            v = p.starve <= kHoldSteps ? p.last : 0.f;
        }
        *out = std::max(*out, v);
        any = true;
    }
    return any;
}

bool anyArmed() {
    for (const Slot& s : gSlots) {
        if (s.uid.load(std::memory_order_relaxed) != kFree) return true;
    }
    return false;
}

// Attaches the tagger and hands its session memory to the Rust worker, which owns the mapping
// from then on. Null when the HAL has no tagger.
std::shared_ptr<aidl::vendor::yasvara::hentak::ITagger> attachTagger() {
    using aidl::vendor::yasvara::hentak::ITagger;
    std::shared_ptr<ITagger> tagger = ITagger::fromBinder(ndk::SpAIBinder(AServiceManager_checkService(kTagger)));
    if (tagger == nullptr) return nullptr;
    ndk::ScopedFileDescriptor fd;
    if (!tagger->attach(&fd).isOk() || fd.get() < 0) return nullptr;
    void* base = mmap(nullptr, kTaggerShmBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
    if (base == MAP_FAILED) return nullptr;
    if (!hentak_rs_set_tagger(static_cast<uint8_t*>(base), kTaggerShmBytes)) {
        munmap(base, kTaggerShmBytes);
        return nullptr;
    }
    ALOGI("tagger attached");
    return tagger;
}

void senderLoop() {
    using aidl::android::hardware::vibrator::IVibrator;
    const int fd = gEventFd.load();
    sp<IBinder> service;
    std::shared_ptr<IVibrator> hal;
    std::shared_ptr<aidl::vendor::yasvara::hentak::ITagger> tagger;
    int64_t lookedUpNs = INT64_MIN / 2;
    int64_t taggerLookedUpNs = INT64_MIN / 2;
    Play plays[kPlays];
    bool active = false;
    int64_t nextStepNs = 0;
    float sent = 0;
    int64_t sentNs = 0;
    for (;;) {
        pollfd p = {.fd = fd, .events = POLLIN};
        timespec wait = {};
        if (active) {
            const int64_t ns = std::max<int64_t>(0, nextStepNs - nowNs());
            wait = {.tv_sec = static_cast<time_t>(ns / 1000000000), .tv_nsec = static_cast<long>(ns % 1000000000)};
        }
        if (ppoll(&p, 1, active ? &wait : nullptr, nullptr) > 0) {
            uint64_t n;
            (void)read(fd, &n, sizeof(n));
        }
        const int64_t now = nowNs();
        if (hal == nullptr && now - lookedUpNs >= kLookupNs) {
            lookedUpNs = now;
            hal = IVibrator::fromBinder(ndk::SpAIBinder(AServiceManager_checkService(kFollower)));
            gFollowing.store(hal != nullptr, std::memory_order_relaxed);
        }
        if (tagger != nullptr && !AIBinder_isAlive(tagger->asBinder().get())) {
            hentak_rs_set_tagger(nullptr, 0);  // the HAL restarted: attach again
            tagger = nullptr;
        }
        if (tagger == nullptr && hal != nullptr && now - taggerLookedUpNs >= kLookupNs && anyArmed()) {
            taggerLookedUpNs = now;
            tagger = attachTagger();
        }
        for (;;) {
            QueueSlot& s = gQueue[gTail % kQueue];
            const uint32_t seq = s.seq.load(std::memory_order_acquire);
            if (seq != gTail + 1) {
                // Lapped by producers: skip to the oldest event still there.
                if (seq > gTail + 1) {
                    gDropped.fetch_add(seq - gTail - 1, std::memory_order_relaxed);
                    gTail = seq - 1;
                    continue;
                }
                break;
            }
            const Event e = s.event;
            gTail++;
            if (e.kind == LEVELS) {
                if (hal == nullptr) continue;
                queueLevels(plays, e);
                if (!active) {
                    active = true;
                    nextStepNs = now;
                }
            } else if (hal == nullptr) {
                // No envelope follower: discrete events, played by system_server.
                sendEvent(service, e);
            }
        }
        while (active && now >= nextStepNs) {
            float level;
            active = stepLevels(plays, &level);
            level = std::min(1.f, level * gGain.load(std::memory_order_relaxed));
            if (hal != nullptr && (std::fabs(level - sent) > 0.01f || (level > 0 && now - sentNs > kRefreshNs))) {
                const ndk::ScopedAStatus status = level > 0 ? hal->setAmplitude(level) : hal->off();
                if (status.getStatus() == STATUS_DEAD_OBJECT) {
                    hal = nullptr;
                    gFollowing.store(false, std::memory_order_relaxed);
                } else {
                    gLevelsSent.fetch_add(1, std::memory_order_relaxed);
                }
                sent = level;
                sentNs = now;
            }
            nextStepNs += kStepNs;
            if (now - nextStepNs > 4 * kStepNs) nextStepNs = now + kStepNs;  // fell behind
        }
    }
}

// The Rust worker's events, into the same queue.
void postFromRust(int32_t uid, int32_t kind, float intensity, float sharpness, uintptr_t track,
                  int32_t count, const float* levels) {
    Event e = {.uid = uid, .kind = kind, .intensity = intensity, .sharpness = sharpness,
               .track = track, .count = std::clamp(count, 0, Tap::kLevelBatch)};
    if (levels != nullptr) std::copy_n(levels, e.count, e.levels);
    post(e);
}

void startSender() {
    std::call_once(gSenderOnce, [] {
        hentak_rs_init(postFromRust);
        const int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (fd < 0) {
            ALOGE("eventfd failed: %d", errno);
            return;
        }
        gEventFd.store(fd, std::memory_order_release);
        std::thread([] {
            pthread_setname_np(pthread_self(), "hentak_sender");
            setpriority(PRIO_PROCESS, 0, kSenderNice);
            senderLoop();
        }).detach();
    });
}


}  // namespace

void post(const Event& e) {
    const uint32_t i = gHead.fetch_add(1, std::memory_order_relaxed);
    QueueSlot& s = gQueue[i % kQueue];
    s.event = e;
    s.seq.store(i + 1, std::memory_order_release);
    gPosted.fetch_add(1, std::memory_order_relaxed);
    const int fd = gEventFd.load(std::memory_order_acquire);
    if (fd >= 0) {
        const uint64_t one = 1;
        (void)write(fd, &one, sizeof(one));  // non-blocking eventfd: safe on RT threads
    }
}

float sensitivity(uid_t uid) {
    for (const Slot& s : gSlots) {
        if (s.uid.load(std::memory_order_relaxed) == static_cast<int64_t>(uid)) {
            return s.sensitivity.load(std::memory_order_relaxed);
        }
    }
    return 0.f;
}

bool avoidMmap(uid_t uid) {
    if (sensitivity(uid) > 0) return true;
    for (const auto& u : gNoMmap) {
        const int64_t v = u.load(std::memory_order_relaxed);
        if (v <= 0) break;
        if (v == static_cast<int64_t>(uid)) return true;
    }
    return false;
}

bool handleParameters(const std::string& keyValues) {
    constexpr char kNoMmapKey[] = "hentak_nommap=";
    const size_t noMmapAt = keyValues.find(kNoMmapKey);
    if (noMmapAt != std::string::npos) {
        std::lock_guard<std::mutex> lock(gWriteLock);
        const char* p = keyValues.c_str() + noMmapAt + sizeof(kNoMmapKey) - 1;
        int n = 0;
        while (n < kNoMmapSlots) {
            char* end;
            const long uid = strtol(p, &end, 10);
            if (end == p) break;
            if (uid > 0) gNoMmap[n++].store(uid, std::memory_order_relaxed);
            if (*end != ',') break;
            p = end + 1;
        }
        for (int i = n; i < kNoMmapSlots; i++) gNoMmap[i].store(kFree, std::memory_order_relaxed);
        ALOGI("no MMAP for %d uid(s)", n);
    }
    constexpr char kGainKey[] = "hentak_gain=";
    const size_t gainAt = keyValues.find(kGainKey);
    if (gainAt != std::string::npos) {
        const float gain = strtof(keyValues.c_str() + gainAt + sizeof(kGainKey) - 1, nullptr);
        gGain.store(std::isfinite(gain) ? std::clamp(gain, 0.f, 1.5f) : 1.f);
    }
    constexpr char kCatalogKey[] = "hentak_catalog=";
    const size_t catalogAt = keyValues.find(kCatalogKey);
    if (catalogAt != std::string::npos) {
        std::string v = keyValues.substr(catalogAt + sizeof(kCatalogKey) - 1);
        v = v.substr(0, v.find(';'));
        const size_t colon = v.find(':');
        const long uid = strtol(v.c_str(), nullptr, 10);
        const std::string path = colon == std::string::npos ? "" : v.substr(colon + 1);
        if (uid > 0) {
            const bool ok = hentak_rs_set_catalog(static_cast<int32_t>(uid), path.c_str());
            ALOGI("catalog for %ld: %s", uid, ok ? path.c_str() : "none");
        }
    }
    constexpr char kKey[] = "hentak_uids=";
    const size_t at = keyValues.find(kKey);
    if (at == std::string::npos) {
        return gainAt != std::string::npos || noMmapAt != std::string::npos ||
               catalogAt != std::string::npos;
    }
    std::string list = keyValues.substr(at + sizeof(kKey) - 1);
    list = list.substr(0, list.find(';'));

    if (!list.empty()) startSender();
    std::lock_guard<std::mutex> lock(gWriteLock);
    for (Slot& s : gSlots) s.uid.store(kFree, std::memory_order_relaxed);
    int n = 0;
    const char* p = list.c_str();
    while (*p && n < kSlots) {
        char* end;
        const long uid = strtol(p, &end, 10);
        if (end == p) break;
        float sens = 1.f;
        if (*end == ':') sens = strtof(end + 1, &end);
        if (uid > 0) {
            gSlots[n].sensitivity.store(std::clamp(sens, 0.05f, 2.f), std::memory_order_relaxed);
            gSlots[n].uid.store(uid, std::memory_order_release);
            n++;
        }
        p = *end == ',' ? end + 1 : end;
        if (*p == 0 || (*end != ',' && *end != 0)) break;
    }
    ALOGI("armed %d uid(s): %s", n, list.c_str());
    return true;
}

std::string dump() {
    std::string out = "hentak:";
    for (const Slot& s : gSlots) {
        const int64_t uid = s.uid.load(std::memory_order_relaxed);
        if (uid == kFree) continue;
        char buf[48];
        snprintf(buf, sizeof(buf), " %lld:%.2f", static_cast<long long>(uid),
                 s.sensitivity.load(std::memory_order_relaxed));
        out += buf;
    }
    char buf[160];
    snprintf(buf, sizeof(buf), " gain=%.2f %s posted=%u sent=%u levels=%u dropped=%u starved=%u ",
             gGain.load(), gFollowing.load() ? "follower" : "events", gPosted.load(), gSent.load(),
             gLevelsSent.load(), gDropped.load(), gStarved.load());
    char rs[256];
    hentak_rs_dump(rs, sizeof(rs));
    return out + buf + rs;
}

}  // namespace android::hentak
