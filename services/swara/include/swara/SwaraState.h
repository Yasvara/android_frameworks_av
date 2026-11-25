/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <sys/types.h>

// Swara: per-uid playback and capture rules.
//
// Readers (mixer and record threads) never lock: rules live in a fixed table of atomics and a
// uid with no rule costs one relaxed load. Writers come from AudioFlinger::setParameters,
// called by system_server only.
namespace android::swara {

// Playback gain for uid: 1 without a rule, 0 when muted.
float playbackGain(uid_t uid);

// Capture gain for uid: 1 without a rule. Mic mute is applied by AudioPolicyService.
float captureGain(uid_t uid);
bool captureMuted(uid_t uid);
// The uid may record while a call or another app holds the mic.
bool shareMicDuringCall(uid_t uid);

// Handles a "swara_*" key/value string from setParameters. Returns false when keyValues
// holds no Swara key, true when it was consumed.
bool handleParameters(const std::string& keyValues);

// Called after a capture rule changes, so AudioPolicyService re-evaluates its clients.
// Runs off the caller's thread.
void setCaptureChangedListener(void (*listener)());

// One line per rule, for getParameters("swara_dump") and dumpsys.
std::string dump();

// Scales interleaved PCM in place (16-bit, 24-bit packed, 32-bit, float), soft-clipping
// above full scale.
void applyGain(void* buffer, size_t samples, int format, float gain);

}  // namespace android::swara
