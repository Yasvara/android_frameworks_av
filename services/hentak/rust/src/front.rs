//! Mixer-thread side: mono downmix and decimation to about 16 kHz, into the worker's ring.
//!
//! The factor is the nearest integer to rate / 16 kHz, so 44.1 kHz runs at 14.7 kHz; the mel bands
//! are fixed in Hz, so features stay comparable across rates. Cost per input frame: one add per
//! channel and 16 multiply-adds.

use crate::ring::Ring;

pub const TARGET_RATE: u32 = 16_000;
const TAPS_PER_PHASE: usize = 16;
const MAX_FACTOR: usize = 6;
const MAX_TAPS: usize = TAPS_PER_PHASE * MAX_FACTOR;
const BATCH: usize = 128;

pub enum Pcm<'a> {
    I16(&'a [i16]),
    F32(&'a [f32]),
}

pub struct Front {
    factor: usize,
    len: usize,
    taps: [f32; MAX_TAPS],
    // Each sample is written twice, len apart, so the newest len samples are always contiguous.
    hist: [f32; 2 * MAX_TAPS],
    at: usize,
    phase: usize,
    out: [f32; BATCH],
    fill: usize,
}

impl Front {
    pub fn new(rate: u32) -> Self {
        let factor = ((rate as f32 / TARGET_RATE as f32).round() as usize).clamp(1, MAX_FACTOR);
        let len = TAPS_PER_PHASE * factor;
        let mut taps = [0f32; MAX_TAPS];
        // Windowed sinc at 0.9 of the output Nyquist, unity at DC.
        let fc = 0.45 / factor as f32;
        let mid = (len - 1) as f32 / 2.0;
        let mut sum = 0.0;
        for (i, t) in taps[..len].iter_mut().enumerate() {
            let x = i as f32 - mid;
            let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f32::consts::PI * fc * x).sin() / (std::f32::consts::PI * x) };
            let w = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos()
                + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos();
            *t = sinc * w;
            sum += *t;
        }
        taps[..len].iter_mut().for_each(|t| *t /= sum);
        Self { factor, len, taps, hist: [0.0; 2 * MAX_TAPS], at: 0, phase: 0, out: [0.0; BATCH], fill: 0 }
    }

    pub fn out_rate(rate: u32) -> u32 {
        rate / ((rate as f32 / TARGET_RATE as f32).round() as u32).clamp(1, MAX_FACTOR as u32)
    }

    /// RT-safe: no locks, no allocation, bounded work.
    pub fn feed(&mut self, pcm: Pcm<'_>, channels: usize, ring: &Ring) {
        if channels == 0 {
            return;
        }
        let scale = 1.0 / channels as f32;
        match pcm {
            Pcm::I16(s) => {
                for f in s.chunks_exact(channels) {
                    let x: i32 = f.iter().map(|&v| v as i32).sum();
                    self.sample(x as f32 * (scale / 32768.0), ring);
                }
            }
            Pcm::F32(s) => {
                for f in s.chunks_exact(channels) {
                    self.sample(f.iter().sum::<f32>() * scale, ring);
                }
            }
        }
        if self.fill > 0 {
            ring.push(&self.out[..self.fill]);
            self.fill = 0;
        }
    }

    #[inline]
    fn sample(&mut self, x: f32, ring: &Ring) {
        self.hist[self.at] = x;
        self.hist[self.at + self.len] = x;
        self.at += 1;
        if self.at == self.len {
            self.at = 0;
        }
        self.phase += 1;
        if self.phase < self.factor {
            return;
        }
        self.phase = 0;
        let h = &self.hist[self.at..self.at + self.len];
        let y: f32 = h.iter().zip(&self.taps[..self.len]).map(|(a, b)| a * b).sum();
        self.out[self.fill] = y;
        self.fill += 1;
        if self.fill == BATCH {
            ring.push(&self.out);
            self.fill = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_through(freq: f32) -> f32 {
        let ring = Ring::new(1 << 16);
        let mut f = Front::new(48_000);
        let pcm: Vec<f32> = (0..48_000).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin()).collect();
        f.feed(Pcm::F32(&pcm), 1, &ring);
        let mut out = vec![0.0; 16_000];
        let n = ring.pop(&mut out);
        (out[1000..n].iter().map(|v| v * v).sum::<f32>() / (n - 1000) as f32).sqrt()
    }

    #[test]
    fn passes_band_rejects_alias() {
        assert!((rms_through(1000.0) - 0.707).abs() < 0.02);
        assert!(rms_through(12_000.0) < 0.01);
    }
}
