/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#define LOG_TAG "Swara"

#include <swara/SwaraState.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include <log/log.h>
#include <system/audio.h>

namespace android::swara {
namespace {

constexpr int kSlots = 128;
constexpr int64_t kFree = -1;

constexpr uint32_t kMuted = 1u << 0;
constexpr uint32_t kMicMuted = 1u << 1;
constexpr uint32_t kShareMic = 1u << 2;

struct Slot {
    std::atomic<int64_t> uid{kFree};
    std::atomic<float> gain{1.f};
    std::atomic<float> micGain{1.f};
    std::atomic<uint32_t> flags{0};
};

Slot gSlots[kSlots];
// Highest used slot + 1, so a lookup scans only what was ever used.
std::atomic<int> gUsed{0};
std::mutex gWriteLock;
std::atomic<void (*)()> gCaptureListener{nullptr};

const Slot* find(uid_t uid) {
    const int used = gUsed.load(std::memory_order_acquire);
    for (int i = 0; i < used; i++) {
        if (gSlots[i].uid.load(std::memory_order_relaxed) == static_cast<int64_t>(uid)) {
            return &gSlots[i];
        }
    }
    return nullptr;
}

// Caller holds gWriteLock.
Slot* findOrAdd(uid_t uid) {
    const int used = gUsed.load(std::memory_order_relaxed);
    int freeSlot = -1;
    for (int i = 0; i < used; i++) {
        const int64_t u = gSlots[i].uid.load(std::memory_order_relaxed);
        if (u == static_cast<int64_t>(uid)) return &gSlots[i];
        if (u == kFree && freeSlot < 0) freeSlot = i;
    }
    if (freeSlot < 0) {
        if (used >= kSlots) return nullptr;
        freeSlot = used;
    }
    Slot& s = gSlots[freeSlot];
    s.gain.store(1.f, std::memory_order_relaxed);
    s.micGain.store(1.f, std::memory_order_relaxed);
    s.flags.store(0, std::memory_order_relaxed);
    s.uid.store(uid, std::memory_order_release);
    if (freeSlot >= used) gUsed.store(freeSlot + 1, std::memory_order_release);
    return &s;
}

void notifyCapture() {
    auto listener = gCaptureListener.load();
    if (listener != nullptr) {
        // AudioPolicyService calls back into AudioFlinger; never from the setParameters thread.
        std::thread(listener).detach();
    }
}

// Finds "key=value" in a ';' separated list.
bool value(const std::string& kv, const char* key, std::string* out) {
    const std::string k = std::string(key) + "=";
    size_t pos = 0;
    while (pos < kv.size()) {
        size_t end = kv.find(';', pos);
        if (end == std::string::npos) end = kv.size();
        if (kv.compare(pos, k.size(), k) == 0) {
            *out = kv.substr(pos + k.size(), end - pos - k.size());
            return true;
        }
        pos = end + 1;
    }
    return false;
}

float clampf(float v, float lo, float hi) {
    return std::isnan(v) ? 1.f : v < lo ? lo : v > hi ? hi : v;
}

// Soft clip above full scale: linear to 0.9, then a tanh knee into 1.
}  // namespace

inline float softClip(float x) {
    const float a = std::fabs(x);
    if (a <= 0.9f) return x;
    const float y = 0.9f + 0.1f * std::tanh((a - 0.9f) * 10.f);
    return x < 0 ? -y : y;
}

float playbackGain(uid_t uid) {
    if (gUsed.load(std::memory_order_relaxed) == 0) return 1.f;
    const Slot* s = find(uid);
    if (s == nullptr) return 1.f;
    if (s->flags.load(std::memory_order_relaxed) & kMuted) return 0.f;
    return s->gain.load(std::memory_order_relaxed);
}

float captureGain(uid_t uid) {
    if (gUsed.load(std::memory_order_relaxed) == 0) return 1.f;
    const Slot* s = find(uid);
    return s != nullptr ? s->micGain.load(std::memory_order_relaxed) : 1.f;
}

bool captureMuted(uid_t uid) {
    if (gUsed.load(std::memory_order_relaxed) == 0) return false;
    const Slot* s = find(uid);
    return s != nullptr && (s->flags.load(std::memory_order_relaxed) & kMicMuted);
}

bool shareMicDuringCall(uid_t uid) {
    if (gUsed.load(std::memory_order_relaxed) == 0) return false;
    const Slot* s = find(uid);
    return s != nullptr && (s->flags.load(std::memory_order_relaxed) & kShareMic);
}

bool handleParameters(const std::string& kv) {
    if (kv.find("swara_") == std::string::npos) return false;
    std::string v;
    bool captureChanged = false;
    {
        std::lock_guard<std::mutex> l(gWriteLock);
        if (value(kv, "swara_reset", &v)) {
            const int used = gUsed.load(std::memory_order_relaxed);
            for (int i = 0; i < used; i++) gSlots[i].uid.store(kFree);
            gUsed.store(0, std::memory_order_release);
            captureChanged = true;
            ALOGI("all rules cleared");
        }
        if (value(kv, "swara_clear", &v)) {
            const uid_t uid = static_cast<uid_t>(strtoul(v.c_str(), nullptr, 10));
            const Slot* s = find(uid);
            if (s != nullptr) {
                const_cast<Slot*>(s)->uid.store(kFree, std::memory_order_release);
                captureChanged = true;
                ALOGI("uid %u cleared", uid);
            }
        }
        if (value(kv, "swara_uid", &v)) {
            const uid_t uid = static_cast<uid_t>(strtoul(v.c_str(), nullptr, 10));
            Slot* s = findOrAdd(uid);
            if (s == nullptr) {
                ALOGW("rule table full, uid %u ignored", uid);
                return true;
            }
            uint32_t flags = s->flags.load(std::memory_order_relaxed);
            const uint32_t oldCapture = flags & (kMicMuted | kShareMic);
            const float oldMicGain = s->micGain.load(std::memory_order_relaxed);
            auto flag = [&](const char* key, uint32_t bit) {
                std::string b;
                if (value(kv, key, &b)) flags = b == "1" ? (flags | bit) : (flags & ~bit);
            };
            flag("swara_muted", kMuted);
            flag("swara_mic_muted", kMicMuted);
            flag("swara_share_mic", kShareMic);
            if (value(kv, "swara_gain", &v)) {
                s->gain.store(clampf(strtof(v.c_str(), nullptr), 0.f, 1.f));
            }
            if (value(kv, "swara_mic_gain", &v)) {
                s->micGain.store(clampf(strtof(v.c_str(), nullptr), 0.f, 2.f));
            }
            s->flags.store(flags, std::memory_order_relaxed);
            captureChanged = (flags & (kMicMuted | kShareMic)) != oldCapture
                    || s->micGain.load() != oldMicGain;
            ALOGI("uid %u gain %.2f%s mic %.2f%s%s", uid, s->gain.load(),
                    flags & kMuted ? " muted" : "", s->micGain.load(),
                    flags & kMicMuted ? " mic-muted" : "", flags & kShareMic ? " share-mic" : "");
        }
    }
    if (captureChanged) notifyCapture();
    return true;
}

void setCaptureChangedListener(void (*listener)()) {
    gCaptureListener.store(listener);
}

std::string dump() {
    std::string out;
    const int used = gUsed.load(std::memory_order_acquire);
    for (int i = 0; i < used; i++) {
        const int64_t uid = gSlots[i].uid.load();
        if (uid == kFree) continue;
        const uint32_t f = gSlots[i].flags.load();
        char line[160];
        snprintf(line, sizeof(line), "uid=%lld gain=%.2f muted=%d mic_gain=%.2f mic_muted=%d"
                " share_mic=%d\n", static_cast<long long>(uid), gSlots[i].gain.load(),
                (f & kMuted) != 0, gSlots[i].micGain.load(), (f & kMicMuted) != 0,
                (f & kShareMic) != 0);
        out += line;
    }
    return out;
}

void applyGain(void* buffer, size_t samples, int format, float gain) {
    if (gain == 1.f || buffer == nullptr) return;
    // Below unity nothing can clip, so the knee is skipped and the scale stays linear.
    const bool boost = gain > 1.f;
    auto softClip = [boost](float x) { return boost ? swara::softClip(x) : x; };
    switch (format) {
        case AUDIO_FORMAT_PCM_16_BIT: {
            auto* p = static_cast<int16_t*>(buffer);
            for (size_t i = 0; i < samples; i++) {
                p[i] = static_cast<int16_t>(softClip(p[i] * gain / 32768.f) * 32767.f);
            }
            break;
        }
        case AUDIO_FORMAT_PCM_FLOAT: {
            auto* p = static_cast<float*>(buffer);
            for (size_t i = 0; i < samples; i++) p[i] = softClip(p[i] * gain);
            break;
        }
        case AUDIO_FORMAT_PCM_32_BIT: {
            auto* p = static_cast<int32_t*>(buffer);
            for (size_t i = 0; i < samples; i++) {
                p[i] = static_cast<int32_t>(softClip(p[i] * gain / 2147483648.f) * 2147483647.f);
            }
            break;
        }
        case AUDIO_FORMAT_PCM_24_BIT_PACKED: {
            auto* p = static_cast<uint8_t*>(buffer);
            for (size_t i = 0; i < samples; i++, p += 3) {
                int32_t s = (p[0] << 8) | (p[1] << 16) | (p[2] << 24);
                s = static_cast<int32_t>(softClip(s * gain / 2147483648.f) * 2147483647.f);
                p[0] = (s >> 8) & 0xff;
                p[1] = (s >> 16) & 0xff;
                p[2] = (s >> 24) & 0xff;
            }
            break;
        }
        default:
            break;
    }
}

}  // namespace android::swara
