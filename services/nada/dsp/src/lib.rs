//! Nada: the CPU audio engine of audioserver.
//!
//! Every sample here is processed on ARM64 cores. Nothing in this crate runs on the ADSP.
//! The engine sits in the mixer threads themselves (FastMixer included) and works in place on
//! the mix buffer. On a fast mixer no stage holds audio back. On a mixer thread without one
//! (the deep buffer) the chain may delay by a fixed budget, reported to the thread so its
//! timestamps stay true.

pub mod biquad;
pub mod colour;
pub mod config;
pub mod conv;
pub mod ctl;
pub mod dyneq;
pub mod engine;
pub mod ffi;
pub mod fft;
pub mod ir;
pub mod lookahead;
pub mod loud;
pub mod mbc;
pub mod slot;
pub mod stages;

/// One stereo frame on the internal bus.
pub type Frame = [f64; 2];

/// A processing stage. `process` runs on an audio thread: no allocation, lock or syscall.
pub trait Stage: Send {
    fn process(&mut self, x: &mut [Frame]);
    /// Clears filter memory; called after a non-finite sample.
    fn reset(&mut self);
}

#[inline]
pub fn db(x: f64) -> f64 {
    10f64.powf(x / 20.0)
}

#[inline]
pub fn to_db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}
