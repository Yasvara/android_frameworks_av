/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <hentak/Hentak.h>

namespace android::hentak {

enum Kind : int32_t { IMPACT = 0, BLAST = 1, RUMBLE = 2, LEVELS = 3 };

struct Event {
    int32_t uid;
    int32_t kind;
    float intensity;
    float sharpness;
    // LEVELS: the next envelope steps of one track, 5 ms apart.
    uintptr_t track;
    int32_t count;
    float levels[Tap::kLevelBatch];
};
// Hands an event from a detector to the sender thread. Lock-free and non-blocking: safe on the
// mixer's RT threads.
void post(const Event& e);

}  // namespace android::hentak
