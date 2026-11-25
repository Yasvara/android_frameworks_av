/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

// rust/src/ffi.rs
extern "C" {
typedef void (*hentak_rs_post_fn)(int32_t uid, int32_t kind, float intensity, float sharpness,
                                  uintptr_t track, int32_t count, const float* levels);
void hentak_rs_init(hentak_rs_post_fn post);
void* hentak_rs_tap_new();
void hentak_rs_tap_free(void* tap);
void hentak_rs_tap_feed(void* tap, int32_t uid, float sensitivity, const void* pcm, size_t frames,
                        bool is_float, uint32_t channels, uint32_t rate);
bool hentak_rs_set_catalog(int32_t uid, const char* path);
size_t hentak_rs_dump(char* buf, size_t len);
bool hentak_rs_set_tagger(uint8_t* base, size_t len);
bool hentak_rs_set_sounds(int32_t uid, const uint8_t* base, size_t len);
}
