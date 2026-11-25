//! C ABI for audioserver (frameworks/av/services/hentak).

use crate::catalog::Catalog;
use crate::front::Pcm;
use crate::rt::{self, PostFn, Tap};
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

/// Where events go: the C++ sender's queue.
#[no_mangle]
pub extern "C" fn hentak_rs_init(post: PostFn) {
    rt::init(post);
}

/// One per playback track; not on a real-time thread (it allocates the ring).
#[no_mangle]
pub extern "C" fn hentak_rs_tap_new() -> *mut c_void {
    Arc::into_raw(Tap::new()) as *mut c_void
}

/// # Safety
/// `tap` comes from `hentak_rs_tap_new` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn hentak_rs_tap_free(tap: *mut c_void) {
    if tap.is_null() {
        return;
    }
    // SAFETY: the caller's contract.
    let t = unsafe { Arc::from_raw(tap as *const Tap) };
    t.release();
}

/// Mixer thread: the frames just consumed, interleaved, 16-bit or float.
///
/// # Safety
/// `tap` is live; `pcm` holds `frames * channels` samples of the given format.
#[no_mangle]
pub unsafe extern "C" fn hentak_rs_tap_feed(
    tap: *mut c_void,
    uid: i32,
    sensitivity: f32,
    pcm: *const c_void,
    frames: usize,
    is_float: bool,
    channels: u32,
    rate: u32,
) {
    if tap.is_null() || pcm.is_null() || frames == 0 {
        return;
    }
    // SAFETY: the caller's contract.
    let t = unsafe { &*(tap as *const Tap) };
    let n = frames * channels as usize;
    // SAFETY: the caller's contract.
    let data = unsafe {
        if is_float {
            Pcm::F32(std::slice::from_raw_parts(pcm as *const f32, n))
        } else {
            Pcm::I16(std::slice::from_raw_parts(pcm as *const i16, n))
        }
    };
    t.feed(uid, sensitivity, data, channels as usize, rate);
}

/// Loads `path` (a `hentak_scan` catalog) for `uid`; an empty or unreadable path drops it.
///
/// # Safety
/// `path` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn hentak_rs_set_catalog(uid: i32, path: *const c_char) -> bool {
    // SAFETY: the caller's contract.
    let path = if path.is_null() { "" } else { unsafe { CStr::from_ptr(path) }.to_str().unwrap_or("") };
    let catalog = std::fs::read(path).ok().and_then(|b| Catalog::from_bytes(&b));
    let ok = catalog.is_some();
    rt::set_catalog(uid, catalog);
    ok
}

/// Writes a one-line status, NUL-terminated, truncated to `len`; returns its length.
///
/// # Safety
/// `buf` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn hentak_rs_dump(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }
    let s = rt::dump();
    let n = s.len().min(len - 1);
    // SAFETY: the caller's contract.
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    n
}

/// Vibrator HAL, `ITagger`: opens the tagger's side of a session over memory it created and
/// mapped. `window48` is the network's window in 48 kHz samples. Null if the memory is too small.
///
/// # Safety
/// `base` maps `len` bytes of session memory, page aligned, until `hentak_tagger_close`.
#[no_mangle]
pub unsafe extern "C" fn hentak_tagger_open(base: *mut u8, len: usize, window48: usize) -> *mut c_void {
    // SAFETY: the caller's contract; the HAL created the memory, so it formats it.
    match unsafe { crate::tagger::shm::Session::new(base, len, true) } {
        Some(s) => Box::into_raw(Box::new(crate::tagger::Runner::new(s, window48))) as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `runner` comes from `hentak_tagger_open` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn hentak_tagger_close(runner: *mut c_void) {
    if !runner.is_null() {
        // SAFETY: the caller's contract.
        drop(unsafe { Box::from_raw(runner as *mut crate::tagger::Runner) });
    }
}

/// Waits up to 40 ms for the next window and writes its features: `mel` (128 × frames) and
/// `bands` (16 × frames), row-major, the last `frames` frames. Returns `frames`, or 0 when there
/// is nothing new; `end` gets the window's end sample.
///
/// # Safety
/// `runner` is live; `mel` and `bands` hold 128 × frames and 16 × frames floats; `end` is valid.
#[no_mangle]
pub unsafe extern "C" fn hentak_tagger_next(runner: *mut c_void, mel: *mut f32, bands: *mut f32, frames: usize, end: *mut u64) -> usize {
    use crate::tagger::{BANDS, MELS};
    thread_local! {
        static BUF: std::cell::RefCell<(Vec<f32>, Vec<f32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    // SAFETY: the caller's contract.
    let r = unsafe { &mut *(runner as *mut crate::tagger::Runner) };
    BUF.with_borrow_mut(|(m, b)| {
        let Some((e, t)) = r.next(m, b) else { return 0 };
        if t < frames {
            return 0;
        }
        for (rows, n, out) in [(m, MELS, mel), (b, BANDS, bands)] {
            for k in 0..n {
                // SAFETY: `out` holds n × frames floats; each row copies its last `frames` frames.
                unsafe { std::ptr::copy_nonoverlapping(rows[k * t + t - frames..].as_ptr(), out.add(k * frames), frames) };
            }
        }
        // SAFETY: the caller's contract.
        unsafe { *end = e };
        frames
    })
}

/// Publishes the labels and levels (12 each) of the window ending at `end`.
///
/// # Safety
/// `runner` is live; `labels` and `levels` hold 12 floats each.
#[no_mangle]
pub unsafe extern "C" fn hentak_tagger_publish(runner: *mut c_void, end: u64, labels: *const f32, levels: *const f32) {
    // SAFETY: the caller's contract.
    let r = unsafe { &*(runner as *const crate::tagger::Runner) };
    let read = |p: *const f32| -> [f32; 12] { std::array::from_fn(|i| unsafe { *p.add(i) }) };
    r.publish(&crate::tagger::shm::Verdict { end, labels: read(labels), levels: read(levels) });
}

/// audioserver: hands over the `ITagger` session memory it mapped (null drops the session). The
/// worker unmaps it once nothing uses it. False if the memory isn't a session (the caller still
/// owns it then).
///
/// # Safety
/// `base` is an mmap of `len` bytes nobody else unmaps.
#[no_mangle]
pub unsafe extern "C" fn hentak_rs_set_tagger(base: *mut u8, len: usize) -> bool {
    if base.is_null() {
        rt::set_tagger(None);
        return true;
    }
    // SAFETY: the caller's contract; the HAL formatted the memory.
    match unsafe { crate::tagger::shm::Session::adopt(base, len) } {
        Some(s) => {
            rt::set_tagger(Some(s));
            true
        }
        None => false,
    }
}
