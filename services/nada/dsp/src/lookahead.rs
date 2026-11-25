//! Look-ahead true-peak limiter, for mix points that may hold audio back (the deep buffer).
//! The fast mixers keep the zero-delay limiter in `stages`.

use crate::ir::bessel_i0;
use crate::stages::{LimP, LimStats};
use crate::{Frame, Stage};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

/// Taps of one inter-sample phase.
const TAPS: usize = 12;
/// Samples the detector waits before it can read between two samples.
pub const WAIT: usize = TAPS / 2;
/// Shortest look-ahead that still leaves a gain ramp.
pub const MIN_LOOKAHEAD: u32 = (WAIT + 8) as u32;

/// Kaiser-windowed sinc reading the signal `frac` of a sample after history tap `WAIT - 1`.
fn phase(frac: f64) -> [f64; TAPS] {
    let (beta, half) = (8.0, WAIT as f64);
    let mut h: [f64; TAPS] = std::array::from_fn(|j| {
        let d = j as f64 - (half - 1.0) - frac;
        let a = std::f64::consts::PI * d;
        let sinc = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
        sinc * bessel_i0(beta * (1.0 - (d / half).powi(2)).max(0.0).sqrt()) / bessel_i0(beta)
    });
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h
}

/// Minimum of the last `win` values, amortised constant time, no allocation after `new`.
struct SlideMin {
    val: Vec<f64>,
    at: Vec<u64>,
    head: usize,
    len: usize,
    win: u64,
    n: u64,
}

impl SlideMin {
    fn new(win: usize) -> Self {
        SlideMin { val: vec![1.0; win + 1], at: vec![0; win + 1], head: 0, len: 0, win: win as u64, n: 0 }
    }

    #[inline]
    fn push(&mut self, v: f64) -> f64 {
        let cap = self.val.len();
        while self.len > 0 && self.val[(self.head + self.len - 1) % cap] >= v {
            self.len -= 1;
        }
        let b = (self.head + self.len) % cap;
        self.val[b] = v;
        self.at[b] = self.n;
        self.len += 1;
        if self.at[self.head] + self.win <= self.n {
            self.head = (self.head + 1) % cap;
            self.len -= 1;
        }
        self.n += 1;
        self.val[self.head]
    }

    fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
        self.n = 0;
    }
}

/// Output gain, then a ceiling on the reconstructed waveform. The detector reads the signal
/// at four points per sample; the gain is held over the look-ahead and smoothed, so it has
/// reached its value when the peak leaves the delay line. Exactly unity (a pure delay) while
/// nothing is above the ceiling. Delay: `lookahead` frames.
pub struct LookLimiter {
    gain: [f64; 2],
    ceil: f64,
    rel: f64,
    ph: [[f64; TAPS]; 3],
    /// Level below which no inter-sample peak can reach the ceiling.
    quiet: f64,
    hist: [Vec<f64>; 2],
    hpos: usize,
    hot: usize,
    r: f64,
    min: SlideMin,
    avg: Vec<f64>,
    apos: usize,
    sum: f64,
    ones: usize,
    delay: Vec<Frame>,
    dpos: usize,
    stats: Arc<LimStats>,
}

impl LookLimiter {
    pub fn new(p: &LimP, fs: f64, stats: Arc<LimStats>) -> Self {
        let look = p.lookahead.max(MIN_LOOKAHEAD) as usize;
        let ramp = look - WAIT;
        let ph = [phase(0.25), phase(0.5), phase(0.75)];
        let l1 = ph.iter().map(|h| h.iter().map(|v| v.abs()).sum::<f64>()).fold(1.0, f64::max);
        LookLimiter {
            gain: p.gain,
            ceil: p.ceiling,
            rel: 1.0 - (-1.0 / (p.release_ms.max(1.0) * 1e-3 * fs)).exp(),
            ph,
            quiet: p.ceiling / l1,
            hist: [vec![0.0; 2 * TAPS], vec![0.0; 2 * TAPS]],
            hpos: 0,
            hot: 0,
            r: 1.0,
            // Two extra samples: a peak between two samples must hold both down.
            min: SlideMin::new(ramp + 2),
            avg: vec![1.0; ramp],
            apos: 0,
            sum: ramp as f64,
            ones: ramp,
            delay: vec![[0.0; 2]; look],
            dpos: 0,
            stats,
        }
    }

    /// Largest magnitude of the waveform around the sample that entered `WAIT` samples ago.
    #[inline]
    fn peak(&self) -> f64 {
        let mut p = 0f64;
        for h in &self.hist {
            let w = &h[self.hpos..self.hpos + TAPS];
            p = p.max(w[WAIT - 1].abs());
            for ph in &self.ph {
                let v: f64 = w.iter().zip(ph.iter()).map(|(x, c)| x * c).sum();
                p = p.max(v.abs());
            }
        }
        p
    }
}

impl Stage for LookLimiter {
    fn process(&mut self, x: &mut [Frame]) {
        let ramp = self.avg.len();
        let look = self.delay.len();
        let mut low = 1.0f64;
        for f in x.iter_mut() {
            let v = [f[0] * self.gain[0], f[1] * self.gain[1]];
            for (h, s) in self.hist.iter_mut().zip(v) {
                h[self.hpos] = s;
                h[self.hpos + TAPS] = s;
            }
            self.hpos = (self.hpos + 1) % TAPS;
            if v[0].abs().max(v[1].abs()) > self.quiet {
                self.hot = TAPS;
            }
            let want = if self.hot > 0 {
                self.hot -= 1;
                let p = self.peak();
                if p > self.ceil { self.ceil / p } else { 1.0 }
            } else {
                1.0
            };
            self.r = if want < self.r { want } else { (self.r + (1.0 - self.r) * self.rel).min(want) };
            if self.r > 0.999_999 {
                self.r = 1.0;
            }
            let m = self.min.push(self.r);
            self.sum += m - self.avg[self.apos];
            self.avg[self.apos] = m;
            self.apos = (self.apos + 1) % ramp;
            self.ones = if m == 1.0 { (self.ones + 1).min(ramp) } else { 0 };
            let g = if self.ones >= ramp {
                // Drop the rounding the running sum has collected.
                self.sum = ramp as f64;
                1.0
            } else {
                (self.sum / ramp as f64).min(1.0)
            };
            low = low.min(g);
            let d = std::mem::replace(&mut self.delay[self.dpos], v);
            self.dpos = (self.dpos + 1) % look;
            *f = [(d[0] * g).clamp(-self.ceil, self.ceil), (d[1] * g).clamp(-self.ceil, self.ceil)];
        }
        if low < 1.0 {
            self.stats.blocks.fetch_add(1, Relaxed);
            let cdb = (-2000.0 * low.log10()) as u32;
            if cdb > self.stats.max_cdb.load(Relaxed) {
                self.stats.max_cdb.store(cdb, Relaxed);
            }
        }
    }

    fn reset(&mut self) {
        self.hist.iter_mut().for_each(|h| h.fill(0.0));
        self.delay.fill([0.0; 2]);
        self.avg.fill(1.0);
        self.sum = self.avg.len() as f64;
        self.ones = self.avg.len();
        self.r = 1.0;
        self.hot = 0;
        self.min.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::Limiter;

    fn lim(ceiling: f64) -> LookLimiter {
        let p = LimP { gain: [1.0; 2], ceiling, release_ms: 80.0, lookahead: 72 };
        LookLimiter::new(&p, 48000.0, Arc::new(LimStats::default()))
    }

    /// Peak of the reconstructed waveform, read at 16 points per sample.
    fn true_peak(x: &[f64]) -> f64 {
        let mut peak = 0f64;
        for n in 64..x.len() - 64 {
            for k in 0..16 {
                let t = n as f64 + k as f64 / 16.0;
                let mut acc = 0.0;
                for (j, v) in x.iter().enumerate().take(n + 64).skip(n - 63) {
                    let d = t - j as f64;
                    let a = std::f64::consts::PI * d;
                    let w = 0.5 + 0.5 * (a / 64.0).cos();
                    acc += v * w * if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                }
                peak = peak.max(acc.abs());
            }
        }
        peak
    }

    #[test]
    fn below_the_ceiling_it_is_a_pure_delay() {
        let mut l = lim(0.9);
        let src: Vec<Frame> = (0..4800).map(|i| [0.3 * (i as f64 * 0.7).sin(), 0.25 * (i as f64 * 1.9).cos()]).collect();
        let mut out = src.clone();
        for part in out.chunks_mut(144) {
            l.process(part);
        }
        assert!(out[..72].iter().all(|f| *f == [0.0; 2]));
        assert_eq!(out[72..], src[..src.len() - 72]);
        assert_eq!(l.stats.blocks.load(Relaxed), 0);
    }

    #[test]
    fn no_sample_passes_the_ceiling_and_unity_returns() {
        let mut l = lim(0.5);
        let mut x: Vec<Frame> = (0..96000)
            .map(|i| if (2000..9000).contains(&i) { [if (i / 37) % 2 == 0 { 2.0 } else { -2.0 }, 1.5 * (i as f64 * 0.3).sin()] } else { [0.2, -0.2] })
            .collect();
        l.process(&mut x);
        assert!(x.iter().all(|f| f[0].abs() <= 0.5 && f[1].abs() <= 0.5));
        // The release reaches exactly one again (1.8 s after the burst).
        assert!(x[86000..].iter().all(|f| *f == [0.2, -0.2]));
    }

    #[test]
    fn holds_the_peak_between_samples() {
        // Samples stay below the ceiling while the waveform between them goes over it.
        for (w, tol) in [(std::f64::consts::FRAC_PI_2, 1.01), (0.11 * std::f64::consts::TAU, 1.035)] {
            let src: Vec<f64> = (0..6000).map(|i| 1.25 * (i as f64 * w + std::f64::consts::FRAC_PI_4).sin()).collect();
            let mut x: Vec<Frame> = src.iter().map(|v| [*v, *v]).collect();
            lim(0.9).process(&mut x);
            let out: Vec<f64> = x[2000..].iter().map(|f| f[0]).collect();
            let tp = true_peak(&out);
            assert!(tp <= 0.9 * tol, "true peak {tp} at w={w}");
            assert!(tp > 0.8, "over-limited: {tp}");
        }
        // The zero-delay limiter looks at samples only and lets the same peak through.
        let w = std::f64::consts::FRAC_PI_2;
        let mut x: Vec<Frame> = (0..6000).map(|i| [1.25 * (i as f64 * w + std::f64::consts::FRAC_PI_4).sin(); 2]).collect();
        let p = LimP { gain: [1.0; 2], ceiling: 0.9, release_ms: 80.0, lookahead: 0 };
        Limiter::new(&p, 48000.0, Arc::new(LimStats::default())).process(&mut x);
        let out: Vec<f64> = x[2000..].iter().map(|f| f[0]).collect();
        assert!(true_peak(&out) > 1.2);
    }

    #[test]
    fn sliding_minimum() {
        let mut m = SlideMin::new(3);
        let got: Vec<f64> = [5.0, 4.0, 6.0, 7.0, 8.0, 1.0, 9.0, 9.0, 9.0].iter().map(|v| m.push(*v)).collect();
        assert_eq!(got, [5.0, 4.0, 4.0, 4.0, 6.0, 1.0, 1.0, 1.0, 9.0]);
    }
}
