/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <system/audio.h>

// Hentak: game haptics from the game's own audio.
//
// Playback tracks of an armed uid feed a transient detector as the mixer consumes them. Each hit
// opens a haptic envelope that decays as the sound does; sustained low rumble adds a continuous
// one. A game with a catalog of its own sound files keeps voice, music and UI quiet. A sender thread paces the envelope out in 5 ms steps to the vibrator HAL's "hentak"
// instance, which tracks it on the motor. Without that instance, hits go as discrete events to
// system_server's "yasvara_hentak" instead. The mixer side never locks, allocates or blocks: an
// armed uid costs a downmix and a ring push per buffer, any other uid one relaxed load.
namespace android::hentak {

// Sensitivity for uid, 0 when it is not armed.
float sensitivity(uid_t uid);

// True for a game with Hentak on, armed or not: MMAP output bypasses the tap, so such a game gets
// a mixed track instead, from its first stream on.
bool avoidMmap(uid_t uid);

// One per playback track. The consuming (mixer) thread hands the frames over to the Rust worker
// (rust/src/rt.rs), which runs the detectors and posts events to the sender.
class Tap {
  public:
    static constexpr int kLevelBatch = 6;

    Tap();
    ~Tap();
    Tap(const Tap&) = delete;
    Tap& operator=(const Tap&) = delete;

    void feed(uid_t uid, float sensitivity, const void* pcm, size_t frames,
              audio_format_t format, uint32_t channels, uint32_t sampleRate);

  private:
    void* mRs;
};

// Handles a "hentak_*" key/value string from setParameters: "hentak_uids=10123:1.0,10456:0.7"
// (empty to disarm), "hentak_gain=0.8" (the user's haptic intensity),
// "hentak_nommap=10123,10456" (every game with Hentak on) and "hentak_catalog=10123:<path>" (the
// game's sound catalog from hentak_scan; an empty path drops it). Returns false when keyValues
// holds no Hentak key.
bool handleParameters(const std::string& keyValues);

// Armed uids and event counters, for getParameters("hentak_dump").
std::string dump();

}  // namespace android::hentak
