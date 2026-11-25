//! Loudness levelling to a target in LUFS (ITU-R BS.1770 meter). Zero delay: the gain moves
//! slowly, behind the measurement.

use crate::biquad::{Biquad, Coef};
use crate::{Frame, Stage};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoudP {
    pub target: f64,
    /// Largest correction either way, dB.
    pub max_db: f64,
    /// Time constant of a gain rise, s; a fall is four times faster.
    pub tau_s: f64,
}

/// Blocks of 100 ms; the meter integrates the last 3 s of them (short-term loudness).
const BLOCKS: usize = 30;
const ABS_GATE: f64 = -70.0;
const REL_GATE: f64 = -10.0;

/// The two K-weighting sections of BS.1770 for any sample rate. At 48 kHz they give the
/// coefficients printed in the standard.
pub fn k_weighting(fs: f64) -> [Coef; 2] {
    let (f0, g, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Coef {
        b0: (vh + vb * k / q + k * k) / a0,
        b1: 2.0 * (k * k - vh) / a0,
        b2: (vh - vb * k / q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
    };
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let hp = Coef { b0: 1.0, b1: -2.0, b2: 1.0, a1: 2.0 * (k * k - 1.0) / a0, a2: (1.0 - k / q + k * k) / a0 };
    [shelf, hp]
}

/// Gated loudness of mean-square block powers, LUFS; `None` when every block is gated out.
pub fn gated(blocks: &[f64]) -> Option<f64> {
    let lufs = |ms: f64| -0.691 + 10.0 * ms.max(1e-30).log10();
    let abs: Vec<f64> = blocks.iter().copied().filter(|&b| lufs(b) > ABS_GATE).collect();
    if abs.is_empty() {
        return None;
    }
    let rel = lufs(abs.iter().sum::<f64>() / abs.len() as f64) + REL_GATE;
    let (sum, n) = abs.iter().filter(|&&b| lufs(b) > rel).fold((0.0, 0usize), |(s, n), b| (s + b, n + 1));
    (n > 0).then(|| lufs(sum / n as f64))
}

pub struct Loud {
    k: [Biquad; 2],
    target: f64,
    max_db: f64,
    up: f64,
    down: f64,
    block: usize,
    acc: f64,
    count: usize,
    ring: [f64; BLOCKS],
    filled: usize,
    pos: usize,
    gain_db: f64,
    g: f64,
    step: f64,
}

impl Loud {
    pub fn new(p: &LoudP, fs: f64) -> Self {
        let block = (fs / 10.0).round() as usize;
        let k = |tau: f64| 1.0 - (-0.1 / tau.max(0.1)).exp();
        let [a, b] = k_weighting(fs);
        Loud {
            k: [Biquad::new(a), Biquad::new(b)],
            target: p.target,
            max_db: p.max_db,
            up: k(p.tau_s),
            down: k(p.tau_s / 4.0),
            block,
            acc: 0.0,
            count: 0,
            ring: [0.0; BLOCKS],
            filled: 0,
            pos: 0,
            gain_db: 0.0,
            g: 1.0,
            step: 0.0,
        }
    }

    fn end_block(&mut self) {
        self.ring[self.pos] = self.acc / self.count as f64;
        self.pos = (self.pos + 1) % BLOCKS;
        self.filled = (self.filled + 1).min(BLOCKS);
        self.acc = 0.0;
        self.count = 0;
        // Silence and pauses hold the gain: noise is never pulled up.
        if let Some(l) = gated(&self.ring[..self.filled]) {
            let want = (self.target - l).clamp(-self.max_db, self.max_db);
            let k = if want > self.gain_db { self.up } else { self.down };
            self.gain_db += (want - self.gain_db) * k;
        }
        self.step = (crate::db(self.gain_db) - self.g) / self.block as f64;
    }
}

impl Stage for Loud {
    fn process(&mut self, x: &mut [Frame]) {
        for f in x.iter_mut() {
            let w = { let t = self.k[0].tick2(*f); self.k[1].tick2(t) };
            self.acc += w[0] * w[0] + w[1] * w[1];
            self.count += 1;
            if self.count == self.block {
                self.end_block();
            }
            self.g += self.step;
            *f = [f[0] * self.g, f[1] * self.g];
        }
    }

    fn reset(&mut self) {
        self.k.iter_mut().for_each(Biquad::reset);
        self.acc = 0.0;
        self.count = 0;
        self.filled = 0;
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lufs(x: &[Frame], fs: f64) -> f64 {
        let [a, b] = k_weighting(fs);
        let (mut a, mut b) = (Biquad::new(a), Biquad::new(b));
        let block = (fs * 0.4) as usize;
        let mut ms = Vec::new();
        for c in x.chunks_exact(block) {
            ms.push(c.iter().map(|f| b.tick2(a.tick2(*f))).map(|w| w[0] * w[0] + w[1] * w[1]).sum::<f64>() / block as f64);
        }
        gated(&ms).unwrap()
    }

    fn sine(amp: f64, secs: f64) -> Vec<Frame> {
        (0..(48000.0 * secs) as usize).map(|i| [amp * (std::f64::consts::TAU * 1000.0 * i as f64 / 48000.0).sin(); 2]).collect()
    }

    #[test]
    fn k_weighting_matches_the_standard_at_48k() {
        let [s, h] = k_weighting(48000.0);
        for (got, want) in [
            (s.b0, 1.53512485958697),
            (s.b1, -2.69169618940638),
            (s.b2, 1.19839281085285),
            (s.a1, -1.69065929318241),
            (s.a2, 0.73248077421585),
            (h.a1, -1.99004745483398),
            (h.a2, 0.99007225036621),
        ] {
            assert!((got - want).abs() < 1e-8, "{got} {want}");
        }
        // A full-scale 1 kHz sine on both channels reads close to 0 LUFS.
        assert!((lufs(&sine(1.0, 2.0), 48000.0) - 0.0).abs() < 0.8);
    }

    #[test]
    fn levels_to_the_target_within_its_range() {
        let p = LoudP { target: -16.0, max_db: 12.0, tau_s: 1.0 };
        let mut l = Loud::new(&p, 48000.0);
        let mut x = sine(0.05, 20.0);
        let before = lufs(&x[..96000], 48000.0);
        for part in x.chunks_mut(480) {
            l.process(part);
        }
        let after = lufs(&x[x.len() - 96000..], 48000.0);
        assert!((after + 16.0).abs() < 0.3, "{before} -> {after}");
        // Far too quiet: stops at the range.
        let mut l = Loud::new(&LoudP { max_db: 6.0, ..p }, 48000.0);
        let mut x = sine(0.005, 20.0);
        let before = lufs(&x[..96000], 48000.0);
        l.process(&mut x);
        let after = lufs(&x[x.len() - 96000..], 48000.0);
        assert!((after - before - 6.0).abs() < 0.2, "{before} -> {after}");
    }

    #[test]
    fn silence_holds_the_gain() {
        let p = LoudP { target: -16.0, max_db: 12.0, tau_s: 1.0 };
        let mut l = Loud::new(&p, 48000.0);
        let mut x = sine(0.05, 10.0);
        l.process(&mut x);
        let held = l.gain_db;
        let mut quiet = vec![[0.0; 2]; 48000 * 5];
        l.process(&mut quiet);
        // Only the block that straddles the pause moves it, slightly.
        assert!((l.gain_db - held).abs() < 0.3 && held > 5.0, "{held} {}", l.gain_db);
        let after = l.gain_db;
        let mut more = vec![[0.0; 2]; 48000 * 5];
        l.process(&mut more);
        assert_eq!(l.gain_db, after);
    }
}
