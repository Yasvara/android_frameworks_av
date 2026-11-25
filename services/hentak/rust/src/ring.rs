//! Single-producer single-consumer sample ring. The producer is the mixer thread: it never blocks,
//! allocates or waits, and drops what does not fit.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub struct Ring {
    buf: Box<[UnsafeCell<f32>]>,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    dropped: AtomicU32,
}

// SAFETY: slots between tail and head belong to the consumer, the rest to the producer; the
// release/acquire pairs on head and tail hand them over.
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    /// `capacity` is rounded up to a power of two.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two();
        Self {
            buf: (0..cap).map(|_| UnsafeCell::new(0.0)).collect(),
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            dropped: AtomicU32::new(0),
        }
    }

    /// Producer only. Returns how many samples went in.
    pub fn push(&self, data: &[f32]) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let free = self.buf.len() - head.wrapping_sub(tail);
        let n = data.len().min(free);
        for (i, &x) in data[..n].iter().enumerate() {
            // SAFETY: the slot is free (see the Sync impl).
            unsafe { *self.buf[head.wrapping_add(i) & self.mask].get() = x };
        }
        self.head.store(head.wrapping_add(n), Ordering::Release);
        if n < data.len() {
            self.dropped.fetch_add((data.len() - n) as u32, Ordering::Relaxed);
        }
        n
    }

    /// Consumer only. Returns how many samples came out.
    pub fn pop(&self, out: &mut [f32]) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let n = out.len().min(head.wrapping_sub(tail));
        for (i, o) in out[..n].iter_mut().enumerate() {
            // SAFETY: the slot was published by the producer's release store of head.
            *o = unsafe { *self.buf[tail.wrapping_add(i) & self.mask].get() };
        }
        self.tail.store(tail.wrapping_add(n), Ordering::Release);
        n
    }

    pub fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_drops() {
        let r = Ring::new(8);
        assert_eq!(r.push(&[1.0; 6]), 6);
        let mut out = [0.0; 4];
        assert_eq!(r.pop(&mut out), 4);
        assert_eq!(r.push(&[2.0; 8]), 6);
        assert_eq!(r.dropped(), 2);
        let mut all = [0.0; 16];
        assert_eq!(r.pop(&mut all), 8);
        assert_eq!(&all[..3], &[1.0, 1.0, 2.0]);
    }
}
