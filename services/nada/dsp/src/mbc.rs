//! Three-band compressor. Zero delay.

use crate::biquad::{design, Biquad, Kind};
use crate::stages::knee_cut;
use crate::{Frame, Stage};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MbcP {
    pub xlow: f64,
    pub xhigh: f64,
    pub threshold: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup: f64,
}

/// Bass reacts slower and treble faster than the middle, so no band follows the waveform.
const TIME: [f64; 3] = [2.0, 1.0, 0.5];

struct Band {
    att: f64,
    rel: f64,
    env: f64,
}

/// Bands from fourth-order Linkwitz-Riley crossovers: they sum to an all-pass, so with no
/// band compressing the magnitude is flat but the phase turns around the two crossovers. A
/// band is pulled down as deep as its compressor asks, without touching the others.
pub struct Mbc {
    lo: [Biquad; 2],
    rest: [Biquad; 2],
    mid: [Biquad; 2],
    top: [Biquad; 2],
    lo_ap: Biquad,
    bands: [Band; 3],
    threshold: f64,
    slope: f64,
    makeup: f64,
}

#[inline]
fn chain(q: &mut [Biquad], x: Frame) -> Frame {
    q.iter_mut().fold(x, |v, b| b.tick2(v))
}

impl Mbc {
    pub fn new(p: &MbcP, fs: f64) -> Self {
        let xl = p.xlow;
        let xh = p.xhigh.max(xl * 2.0);
        let bq = |k, f| Biquad::new(design(k, fs, f, 0.0, 0.7071));
        let k = |ms: f64| (-1.0 / (ms.max(0.05) * 1e-3 * fs)).exp();
        Mbc {
            lo: [bq(Kind::LowPass, xl); 2],
            rest: [bq(Kind::HighPass, xl); 2],
            mid: [bq(Kind::LowPass, xh); 2],
            top: [bq(Kind::HighPass, xh); 2],
            lo_ap: bq(Kind::AllPass, xh),
            bands: TIME.map(|t| Band { att: k(p.attack_ms * t), rel: k(p.release_ms * t), env: 0.0 }),
            threshold: p.threshold,
            slope: 1.0 / p.ratio - 1.0,
            makeup: crate::db(p.makeup),
        }
    }
}

impl Stage for Mbc {
    fn process(&mut self, x: &mut [Frame]) {
        for f in x.iter_mut() {
            let lo = chain(&mut self.lo, *f);
            let lo = self.lo_ap.tick2(lo);
            let rest = chain(&mut self.rest, *f);
            let mid = chain(&mut self.mid, rest);
            let hi = chain(&mut self.top, rest);
            let mut out = [0.0; 2];
            for (b, v) in self.bands.iter_mut().zip([lo, mid, hi]) {
                let peak = v[0].abs().max(v[1].abs());
                let over = if peak > 1e-6 { 20.0 * peak.log10() - self.threshold } else { -200.0 };
                let cut = knee_cut(over, self.slope);
                let k = if cut > b.env { b.att } else { b.rel };
                b.env = k * b.env + (1.0 - k) * cut;
                let g = if b.env > 1e-9 { 10f64.powf(-b.env / 20.0) } else { 1.0 };
                out[0] += g * v[0];
                out[1] += g * v[1];
            }
            *f = [out[0] * self.makeup, out[1] * self.makeup];
        }
    }

    fn reset(&mut self) {
        for b in self.lo.iter_mut().chain(&mut self.rest).chain(&mut self.mid).chain(&mut self.top) {
            b.reset();
        }
        self.lo_ap.reset();
        self.bands.iter_mut().for_each(|b| b.env = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_db;

    const P: MbcP = MbcP { xlow: 200.0, xhigh: 4000.0, threshold: -20.0, ratio: 4.0, attack_ms: 2.0, release_ms: 100.0, makeup: 0.0 };

    fn level(tones: &[(f64, f64)], at: f64) -> f64 {
        let mut m = Mbc::new(&P, 48000.0);
        let mut x: Vec<Frame> = (0..48000)
            .map(|i| {
                let v: f64 = tones.iter().map(|(f, a)| a * (std::f64::consts::TAU * f * i as f64 / 48000.0).sin()).sum();
                [v, v]
            })
            .collect();
        m.process(&mut x);
        let (mut s, mut c) = (0.0, 0.0);
        for (i, v) in x.iter().enumerate().skip(24000) {
            let a = std::f64::consts::TAU * at * i as f64 / 48000.0;
            s += v[0] * a.sin();
            c += v[0] * a.cos();
        }
        to_db(2.0 * (s * s + c * c).sqrt() / 24000.0)
    }

    #[test]
    fn quiet_input_keeps_its_level_everywhere() {
        for f in [40.0, 200.0, 1000.0, 4000.0, 12000.0] {
            assert!((level(&[(f, 0.02)], f) - to_db(0.02)).abs() < 0.05, "{f}");
        }
    }

    #[test]
    fn a_loud_band_leaves_the_others_alone() {
        // Loud bass is pulled down; a quiet 1 kHz tone next to it keeps its level.
        let bass = level(&[(60.0, 0.9), (1000.0, 0.02)], 60.0);
        assert!(bass < to_db(0.9) - 6.0, "{bass}");
        let mid = level(&[(60.0, 0.9), (1000.0, 0.02)], 1000.0);
        assert!((mid - to_db(0.02)).abs() < 0.5, "{mid}");
    }
}
