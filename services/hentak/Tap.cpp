/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#include <hentak/Hentak.h>

#include "Rust.h"

namespace android::hentak {

Tap::Tap() : mRs(hentak_rs_tap_new()) {}

Tap::~Tap() {
    hentak_rs_tap_free(mRs);
}

void Tap::feed(uid_t uid, float sens, const void* pcm, size_t frames, audio_format_t format,
               uint32_t channels, uint32_t sampleRate) {
    if (format != AUDIO_FORMAT_PCM_16_BIT && format != AUDIO_FORMAT_PCM_FLOAT) return;
    hentak_rs_tap_feed(mRs, static_cast<int32_t>(uid), sens, pcm, frames,
                       format == AUDIO_FORMAT_PCM_FLOAT, channels, sampleRate);
}

}  // namespace android::hentak
