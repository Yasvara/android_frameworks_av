//! Dynamic equaliser: bands whose gain follows their own level. Zero delay.

use crate::biquad::{design, Biquad, Kind};
use crate::{db, Frame, Stage};

/// Level span over which a band moves from no change to its full range, in dB.
const SPAN_DB: f64 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynBand {
    pub f: f64,
    pub q: f64,
    /// Band level where the gain starts to move, dBFS.
    pub threshold: f64,
    /// Gain reached `SPAN_DB` above the threshold, dB. Negative tames, positive lifts.
    pub range: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
}

struct Band {
    bp: Biquad,
    thr: f64,
    ln_range: f64,
    env: f64,
    att: f64,
    rel: f64,
}

/// Each band is `x + (g - 1) * bandpass(x)`: a peaking curve of gain `g` whose gain can move
/// every sample without touching filter coefficients. The band-pass output is also the level
/// detector, so the band reacts to what it would change and to nothing else.
pub struct DynEq {
    bands: Vec<Band>,
}

impl DynEq {
    pub fn new(bands: &[DynBand], fs: f64) -> Self {
        let k = |ms: f64| 1.0 - (-1.0 / (ms.max(0.05) * 1e-3 * fs)).exp();
        DynEq {
            bands: bands
                .iter()
                .map(|b| Band {
                    bp: Biquad::new(design(Kind::BandPass, fs, b.f, 0.0, b.q)),
                    thr: db(b.threshold),
                    ln_range: db(b.range).ln(),
                    env: 0.0,
                    att: k(b.attack_ms),
                    rel: k(b.release_ms),
                })
                .collect(),
        }
    }
}

impl Stage for DynEq {
    fn process(&mut self, x: &mut [Frame]) {
        let inv_span = 1.0 / db(SPAN_DB).ln();
        for f in x.iter_mut() {
            let dry = *f;
            for b in self.bands.iter_mut() {
                let band = b.bp.tick2(dry);
                let level = band[0].abs().max(band[1].abs());
                b.env += (level - b.env) * if level > b.env { b.att } else { b.rel };
                if b.env > b.thr {
                    let t = ((b.env / b.thr).ln() * inv_span).min(1.0);
                    let g = (t * b.ln_range).exp() - 1.0;
                    f[0] += g * band[0];
                    f[1] += g * band[1];
                }
            }
        }
    }

    fn reset(&mut self) {
        for b in self.bands.iter_mut() {
            b.bp.reset();
            b.env = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_db;

    fn level(amp: f64, f: f64, band: DynBand) -> f64 {
        let mut d = DynEq::new(&[band], 48000.0);
        let mut x: Vec<Frame> = (0..48000).map(|i| [amp * (std::f64::consts::TAU * f * i as f64 / 48000.0).sin(); 2]).collect();
        d.process(&mut x);
        let rms = (x[24000..].iter().map(|v| v[0] * v[0]).sum::<f64>() / 24000.0).sqrt();
        to_db(rms * std::f64::consts::SQRT_2 / amp)
    }

    #[test]
    fn gain_follows_the_band_level() {
        let cut = DynBand { f: 3000.0, q: 1.2, threshold: -30.0, range: -6.0, attack_ms: 5.0, release_ms: 120.0 };
        // 20 dB under the threshold: untouched.
        assert!(level(db(-50.0), 3000.0, cut).abs() < 0.01);
        // Far over it: the full range.
        assert!((level(db(-6.0), 3000.0, cut) + 6.0).abs() < 0.2);
        // Half way up the span: about half the range.
        let mid = level(db(-24.0), 3000.0, cut);
        assert!(mid < -2.0 && mid > -4.0, "{mid}");
        // A loud tone two octaves away does not move the band and is not changed.
        assert!(level(db(-6.0), 200.0, cut).abs() < 0.3);
        let lift = DynBand { f: 80.0, q: 0.9, threshold: -40.0, range: 4.0, attack_ms: 5.0, release_ms: 120.0 };
        assert!((level(db(-10.0), 80.0, lift) - 4.0).abs() < 0.2);
    }

    #[test]
    fn nothing_is_delayed() {
        let mut d = DynEq::new(&[DynBand { f: 1000.0, q: 1.0, threshold: -60.0, range: 6.0, attack_ms: 5.0, release_ms: 120.0 }], 48000.0);
        let mut x = vec![[0.0; 2]; 64];
        x[0] = [1.0; 2];
        d.process(&mut x);
        assert!(x[0][0] >= 1.0);
    }
}
