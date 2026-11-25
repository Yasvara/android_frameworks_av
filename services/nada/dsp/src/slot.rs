//! One stage position in the chain, replaced without stopping audio.
//!
//! The control thread publishes a new stage through an atomic pointer. The audio thread
//! picks it up between calls, crossfades from the old stage, and hands the old one back
//! through a second pointer. The audio thread never allocates or frees.

use crate::{Frame, Stage};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering::AcqRel, Ordering::Acquire, Ordering::Release};

/// A stage built for one sample rate; `stage: None` switches the position off.
pub struct StageBox {
    pub rate: u32,
    pub stage: Option<Box<dyn Stage>>,
}

#[derive(Default)]
pub struct SlotShared {
    pending: AtomicPtr<StageBox>,
    retired: AtomicPtr<StageBox>,
}

impl SlotShared {
    /// Control thread only (one at a time).
    pub fn publish(&self, b: StageBox) {
        self.drain();
        let old = self.pending.swap(Box::into_raw(Box::new(b)), AcqRel);
        free(old);
        self.drain();
    }

    pub fn drain(&self) {
        free(self.retired.swap(null_mut(), AcqRel));
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.load(Acquire).is_null()
    }
}

impl Drop for SlotShared {
    fn drop(&mut self) {
        free(*self.pending.get_mut());
        free(*self.retired.get_mut());
    }
}

fn free(p: *mut StageBox) {
    if !p.is_null() {
        // SAFETY: every non-null pointer in a slot came from Box::into_raw and has one owner.
        drop(unsafe { Box::from_raw(p) });
    }
}

/// Audio-thread side.
pub struct SlotRt {
    cur: *mut StageBox,
    prev: *mut StageBox,
    fade: u32,
    fade_len: u32,
}

// SAFETY: the boxes are owned by this slot and only touched by the thread that owns it.
unsafe impl Send for SlotRt {}

impl Default for SlotRt {
    fn default() -> Self {
        SlotRt { cur: null_mut(), prev: null_mut(), fade: 0, fade_len: 1 }
    }
}

fn stage<'a>(p: *mut StageBox, rate: u32) -> Option<&'a mut Box<dyn Stage>> {
    // SAFETY: non-null slot pointers are live boxes owned by the calling thread.
    let b = unsafe { p.as_mut() }?;
    if b.rate != rate {
        return None;
    }
    b.stage.as_mut()
}

impl SlotRt {
    /// Takes a published stage when no fade is running and the hand-back pointer is free.
    #[inline]
    pub fn poll(&mut self, sh: &SlotShared, fade_len: u32) {
        if self.fade != 0 || !sh.has_pending() || !sh.retired.load(Acquire).is_null() {
            return;
        }
        let p = sh.pending.swap(null_mut(), AcqRel);
        if p.is_null() {
            return;
        }
        self.prev = self.cur;
        self.cur = p;
        self.fade_len = fade_len.max(1);
        self.fade = self.fade_len;
    }

    #[inline]
    pub fn live(&self, rate: u32) -> bool {
        self.fade != 0 || stage(self.cur, rate).is_some()
    }

    #[inline]
    pub fn run(&mut self, sh: &SlotShared, rate: u32, x: &mut [Frame], scratch: &mut [Frame]) {
        if self.fade == 0 {
            if let Some(s) = stage(self.cur, rate) {
                s.process(x);
            }
            return;
        }
        let old = &mut scratch[..x.len()];
        old.copy_from_slice(x);
        if let Some(s) = stage(self.prev, rate) {
            s.process(old);
        }
        if let Some(s) = stage(self.cur, rate) {
            s.process(x);
        }
        let len = self.fade_len as f64;
        for (i, (new, old)) in x.iter_mut().zip(old.iter()).enumerate() {
            let left = self.fade.saturating_sub(i as u32) as f64;
            let t = 1.0 - left / len;
            *new = [old[0] + (new[0] - old[0]) * t, old[1] + (new[1] - old[1]) * t];
        }
        self.fade = self.fade.saturating_sub(x.len() as u32);
        if self.fade == 0 && !self.prev.is_null() {
            sh.retired.store(self.prev, Release);
            self.prev = null_mut();
        }
    }

    /// Drops the stage at once (overload). Returns false when the hand-back pointer is busy.
    pub fn kill(&mut self, sh: &SlotShared) -> bool {
        if self.fade != 0 || self.cur.is_null() || !sh.retired.load(Acquire).is_null() {
            return false;
        }
        sh.retired.store(self.cur, Release);
        self.cur = null_mut();
        true
    }

    pub fn reset(&mut self, rate: u32) {
        for p in [self.cur, self.prev] {
            if let Some(s) = stage(p, rate) {
                s.reset();
            }
        }
    }

    /// Frees both stages. Not for the audio path: used on a format change and at teardown.
    pub fn clear(&mut self) {
        free(self.cur);
        free(self.prev);
        *self = SlotRt::default();
    }
}

impl Drop for SlotRt {
    fn drop(&mut self) {
        free(self.cur);
        free(self.prev);
    }
}
