//! C ABI for audioserver (../include/nada/Nada.h).

use crate::ctl::{self, Msg};
use crate::engine::{Engine, Shared};
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// One per mix point. `kind` 0: FastMixer, 1: a mixer thread without one. Allocates.
#[no_mangle]
pub extern "C" fn nada_insert_new(io: i32, kind: i32) -> *mut Engine {
    let sh = Arc::new(Shared::new(NEXT_ID.fetch_add(1, Relaxed), io, kind));
    ctl::post(Msg::Add(sh.clone()));
    Box::into_raw(Box::new(Engine::new(sh)))
}

/// # Safety
/// `e` comes from `nada_insert_new` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn nada_insert_free(e: *mut Engine) {
    if e.is_null() {
        return;
    }
    // SAFETY: the caller's contract.
    let e = unsafe { Box::from_raw(e) };
    ctl::post(Msg::Remove(e.shared.id));
}

/// The mix thread: `frames` interleaved float frames of `channels` channels, in place.
/// Real-time safe except on the first call and on a sample-rate change.
///
/// # Safety
/// `e` is live and used by this thread only; `buf` holds `frames * channels` floats.
#[no_mangle]
pub unsafe extern "C" fn nada_insert_process(e: *mut Engine, buf: *mut f32, frames: usize, channels: u32, rate: u32) {
    if e.is_null() || buf.is_null() || frames == 0 || channels == 0 || rate == 0 {
        return;
    }
    // SAFETY: the caller's contract.
    let e = unsafe { &mut *e };
    if e.set_rate(rate) {
        ctl::post(Msg::Rate(e.shared.id));
    }
    // SAFETY: the caller's contract.
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, frames * channels as usize) };
    e.process(buf, channels as usize);
}

/// Frames the chain currently holds audio back. Lock-free; any thread.
///
/// # Safety
/// `e` comes from `nada_insert_new` and is live.
#[no_mangle]
pub unsafe extern "C" fn nada_insert_latency(e: *const Engine) -> u32 {
    if e.is_null() {
        return 0;
    }
    // SAFETY: the caller's contract.
    unsafe { &*e }.shared.latency.load(Relaxed)
}

/// Gain the mixer applied to the loudest track. Lock-free; any thread.
///
/// # Safety
/// `e` comes from `nada_insert_new` and is live.
#[no_mangle]
pub unsafe extern "C" fn nada_insert_set_level(e: *const Engine, gain: f32) {
    if e.is_null() {
        return;
    }
    // SAFETY: the caller's contract.
    unsafe { &*e }.shared.set_level(gain);
}

/// The output's device class changed (index into `ctl::ROUTES`).
#[no_mangle]
pub extern "C" fn nada_set_route(io: i32, route: u32) {
    ctl::post(Msg::Route(io, route.min(5) as u8));
}

/// A "nada_*" key/value string from setParameters.
///
/// # Safety
/// `kv` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn nada_set_parameters(kv: *const c_char) {
    if kv.is_null() {
        return;
    }
    // SAFETY: the caller's contract.
    if let Ok(s) = unsafe { CStr::from_ptr(kv) }.to_str() {
        ctl::post(Msg::Params(s.to_string()));
    }
}

/// Answers "nada_dump", "nada_schema" or "nada_active" into `out` (NUL-terminated).
/// Returns the full length of the answer.
///
/// # Safety
/// `key` is a NUL-terminated string; `out` has room for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn nada_query(key: *const c_char, out: *mut c_char, len: usize) -> usize {
    if key.is_null() {
        return 0;
    }
    // SAFETY: the caller's contract.
    let Ok(key) = unsafe { CStr::from_ptr(key) }.to_str() else { return 0 };
    let ans = ctl::query(key);
    if !out.is_null() && len > 0 {
        let n = ans.len().min(len - 1);
        // SAFETY: `out` has `len` bytes and n < len.
        unsafe {
            std::ptr::copy_nonoverlapping(ans.as_ptr(), out as *mut u8, n);
            *out.add(n) = 0;
        }
    }
    ans.len()
}

#[no_mangle]
pub extern "C" fn nada_processing() -> bool {
    ctl::processing()
}

#[no_mangle]
pub extern "C" fn nada_avoid_mmap(uid: u32) -> bool {
    ctl::avoid_mmap(uid)
}
