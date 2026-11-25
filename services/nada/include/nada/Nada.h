/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include <system/audio.h>

// dsp/src/ffi.rs
extern "C" {
struct NadaEngine;
NadaEngine* nada_insert_new(int32_t io, int32_t kind);
void nada_insert_free(NadaEngine* e);
void nada_insert_process(NadaEngine* e, float* buf, size_t frames, uint32_t channels,
                         uint32_t rate);
uint32_t nada_insert_latency(const NadaEngine* e);
void nada_insert_set_level(NadaEngine* e, float gain);
void nada_set_route(int32_t io, uint32_t route);
void nada_set_parameters(const char* kv);
size_t nada_query(const char* key, char* out, size_t len);
bool nada_avoid_mmap(uint32_t uid);
bool nada_processing();
}

// Nada: CPU effects (Rust, on ARM64 cores) applied to an output's mix in place. A kFast
// insert never holds samples back. A kMixer insert holds them back by latencyFrames() while
// its chain is active, and the thread takes that out of its timestamps. With every stage
// off a call costs one atomic load and leaves the buffer bit-exact.
namespace android::nada {

enum Kind : int32_t { kFast = 0, kMixer = 1 };
enum Route : uint32_t { kNone = 0, kSpeaker, kWired, kUsb, kBluetooth, kOther };

// One per mix point; process() belongs to that mix thread alone.
class Insert {
  public:
    Insert(audio_io_handle_t io, Kind kind) : mEngine(nada_insert_new(io, kind)) {}
    ~Insert() { nada_insert_free(mEngine); }
    Insert(const Insert&) = delete;
    Insert& operator=(const Insert&) = delete;

    // Interleaved float frames; channels beyond the first two pass through.
    void process(float* buf, size_t frames, uint32_t channels, uint32_t rate) {
        nada_insert_process(mEngine, buf, frames, channels, rate);
    }

    uint32_t latencyFrames() const { return nada_insert_latency(mEngine); }

    // Gain the mixer applied to the loudest track (volume included). Level-dependent stages
    // undo it, so they act on the content as mastered and not on the volume setting.
    // Lock-free; any thread.
    void setLevel(float gain) { nada_insert_set_level(mEngine, gain); }

  private:
    NadaEngine* const mEngine;
};

// Profiles are kept per device class. Call paths and capture taps are never processed.
inline Route routeOf(audio_devices_t d) {
    switch (d) {
        case AUDIO_DEVICE_OUT_SPEAKER:
        case AUDIO_DEVICE_OUT_SPEAKER_SAFE:
            return kSpeaker;
        case AUDIO_DEVICE_OUT_WIRED_HEADSET:
        case AUDIO_DEVICE_OUT_WIRED_HEADPHONE:
        case AUDIO_DEVICE_OUT_LINE:
            return kWired;
        case AUDIO_DEVICE_OUT_USB_DEVICE:
        case AUDIO_DEVICE_OUT_USB_HEADSET:
        case AUDIO_DEVICE_OUT_USB_ACCESSORY:
        case AUDIO_DEVICE_OUT_DGTL_DOCK_HEADSET:
            return kUsb;
        case AUDIO_DEVICE_OUT_BLUETOOTH_A2DP:
        case AUDIO_DEVICE_OUT_BLUETOOTH_A2DP_HEADPHONES:
        case AUDIO_DEVICE_OUT_BLUETOOTH_A2DP_SPEAKER:
        case AUDIO_DEVICE_OUT_BLE_HEADSET:
        case AUDIO_DEVICE_OUT_BLE_SPEAKER:
        case AUDIO_DEVICE_OUT_BLE_BROADCAST:
        case AUDIO_DEVICE_OUT_HEARING_AID:
            return kBluetooth;
        case AUDIO_DEVICE_OUT_HDMI:
        case AUDIO_DEVICE_OUT_HDMI_ARC:
        case AUDIO_DEVICE_OUT_HDMI_EARC:
        case AUDIO_DEVICE_OUT_SPDIF:
        case AUDIO_DEVICE_OUT_ANLG_DOCK_HEADSET:
            return kOther;
        default:
            return kNone;
    }
}

inline void setRoute(audio_io_handle_t io, Route route) { nada_set_route(io, route); }

inline std::string query(const char* key) {
    std::string out(nada_query(key, nullptr, 0) + 1, '\0');
    out.resize(std::min(nada_query(key, out.data(), out.size()), out.size() - 1));
    return out;
}

}  // namespace android::nada
