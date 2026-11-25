//! Exciter and tube: the stages that make new harmonics. They run at four times the sample
//! rate so the harmonics above half the rate are filtered out instead of folding back into
//! the audible band. The filters hold the signal back, so this stage exists only on mix
//! points that may delay audio (the deep buffer), never on a fast mixer.

use crate::biquad::{design, Biquad, Kind};
use crate::ir::bessel_i0;
use crate::{Frame, Stage};

const OS: usize = 4;
/// Delay of the stage in frames; also the taps of one interpolation phase.
pub const LATENCY: usize = 32;
const LEN: usize = OS * LATENCY + 1;
const BIAS: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColourP {
    /// Exciter amount 0..1; 0 is off.
    pub exc: f64,
    pub exc_fc: f64,
    /// Tube drive 0..1; `None` is off.
    pub tube: Option<f64>,
}

/// Quarter-band low-pass at the oversampled rate, gain 4 (interpolation). Its taps at
/// multiples of four around the centre are zero, so phase 0 is the input itself.
fn prototype() -> Vec<f64> {
    let (beta, c) = (7.857, (LEN / 2) as f64);
    (0..LEN)
        .map(|k| {
            let d = k as f64 - c;
            let a = std::f64::consts::PI * d / OS as f64;
            let sinc = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
            sinc * bessel_i0(beta * (1.0 - (d / c).powi(2)).max(0.0).sqrt()) / bessel_i0(beta)
        })
        .collect()
}

/// The dry signal passes through a plain delay, bit for bit. Only the difference the
/// non-linear curves make is interpolated, shaped, band-limited and added back, so filter
/// ripple can touch the added harmonics but not the music.
pub struct Colour {
    /// Phases 1..3 of the interpolator, ordered oldest input first.
    up: [Vec<f64>; OS - 1],
    /// Decimator, unity gain.
    down: Vec<f64>,
    xh: [Vec<f64>; 2],
    xpos: usize,
    vh: [Vec<f64>; 2],
    vpos: usize,
    dry: Vec<Frame>,
    dpos: usize,
    exc: f64,
    pre: Biquad,
    post: Biquad,
    tube: Option<(f64, f64, f64)>,
    dc: Frame,
    dc_a: f64,
}

impl Colour {
    pub fn new(p: &ColourP, fs: f64) -> Self {
        let h = prototype();
        let up = std::array::from_fn(|p| (1..=LATENCY).map(|i| h[p + 1 + OS * (LATENCY - i)]).collect());
        let hi = fs * OS as f64;
        let fc = p.exc_fc.min(0.45 * fs);
        Colour {
            up,
            down: h.iter().map(|v| v / OS as f64).collect(),
            xh: [vec![0.0; 2 * (LATENCY + 1)], vec![0.0; 2 * (LATENCY + 1)]],
            xpos: 0,
            vh: [vec![0.0; 2 * LEN], vec![0.0; 2 * LEN]],
            vpos: 0,
            dry: vec![[0.0; 2]; LATENCY],
            dpos: 0,
            exc: p.exc,
            pre: Biquad::new(design(Kind::HighPass, hi, fc, 0.0, 0.7071)),
            post: Biquad::new(design(Kind::HighPass, hi, fc, 0.0, 0.7071)),
            tube: p.tube.map(|drive| {
                let k = 0.5 + 3.0 * drive;
                let tb = (k * BIAS).tanh();
                // Unity gain for small signals.
                (k, tb, 1.0 / (k * (1.0 - tb * tb)))
            }),
            dc: [0.0; 2],
            dc_a: 1.0 - (-std::f64::consts::TAU * 10.0 / hi).exp(),
        }
    }

    /// What the curves add to one oversampled frame.
    #[inline]
    fn shape(&mut self, u: Frame) -> Frame {
        let mut d = [0.0; 2];
        if self.exc > 0.0 {
            // Harmonics of the top band only: square for even, cube for odd, then the
            // high-pass again to drop the DC and difference tones the square leaves.
            let hb = self.pre.tick2(u);
            let s = hb.map(|v| {
                let w = (4.0 * v).clamp(-1.5, 1.5);
                0.25 * w * w - w * w * w / 3.0
            });
            let e = self.post.tick2(s);
            d = [self.exc * e[0], self.exc * e[1]];
        }
        if let Some((k, tb, norm)) = self.tube {
            for c in 0..2 {
                // Biased tanh: the two half-waves saturate differently, as a triode does.
                let t = ((k * (u[c] + BIAS)).tanh() - tb) * norm - u[c];
                self.dc[c] += self.dc_a * (t - self.dc[c]);
                d[c] += t - self.dc[c];
            }
        }
        d
    }

    #[inline]
    fn push(&mut self, d: Frame) {
        for (h, v) in self.vh.iter_mut().zip(d) {
            h[self.vpos] = v;
            h[self.vpos + LEN] = v;
        }
        self.vpos = (self.vpos + 1) % LEN;
    }
}

impl Stage for Colour {
    fn process(&mut self, x: &mut [Frame]) {
        const N: usize = LATENCY + 1;
        for f in x.iter_mut() {
            for (h, v) in self.xh.iter_mut().zip(*f) {
                h[self.xpos] = v;
                h[self.xpos + N] = v;
            }
            self.xpos = (self.xpos + 1) % N;
            // Phase 0 is the input half the filter length ago.
            let mid = self.xpos + N - 1 - LATENCY / 2;
            let d0 = self.shape([self.xh[0][mid], self.xh[1][mid]]);
            self.push(d0);
            // Decimate here, on a phase-0 sample, so the total delay is whole frames.
            let mut y = [0.0; 2];
            for (c, h) in self.vh.iter().enumerate() {
                y[c] = h[self.vpos..self.vpos + LEN].iter().zip(self.down.iter()).map(|(v, k)| v * k).sum();
            }
            for p in 0..OS - 1 {
                let mut u = [0.0; 2];
                for (c, h) in self.xh.iter().enumerate() {
                    u[c] = h[self.xpos + 1..self.xpos + N].iter().zip(self.up[p].iter()).map(|(v, k)| v * k).sum();
                }
                let d = self.shape(u);
                self.push(d);
            }
            let dry = std::mem::replace(&mut self.dry[self.dpos], *f);
            self.dpos = (self.dpos + 1) % LATENCY;
            *f = [dry[0] + y[0], dry[1] + y[1]];
        }
    }

    fn reset(&mut self) {
        self.xh.iter_mut().chain(self.vh.iter_mut()).for_each(|h| h.fill(0.0));
        self.dry.fill([0.0; 2]);
        self.pre.reset();
        self.post.reset();
        self.dc = [0.0; 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_db;

    /// Level of the component at `f` in dB re full scale, by quadrature correlation.
    fn tone(x: &[f64], f: f64) -> f64 {
        let (mut s, mut c) = (0.0, 0.0);
        for (i, v) in x.iter().enumerate() {
            let a = std::f64::consts::TAU * f * i as f64 / 48000.0;
            s += v * a.sin();
            c += v * a.cos();
        }
        to_db(2.0 * (s * s + c * c).sqrt() / x.len() as f64)
    }

    fn run(p: ColourP, f: f64, amp: f64) -> Vec<f64> {
        let mut c = Colour::new(&p, 48000.0);
        let mut x: Vec<Frame> = (0..9600 + 4800).map(|i| [amp * (std::f64::consts::TAU * f * i as f64 / 48000.0).sin(); 2]).collect();
        for part in x.chunks_mut(1920) {
            c.process(part);
        }
        x[9600..].iter().map(|v| v[0]).collect()
    }

    #[test]
    fn dry_path_is_an_exact_delay() {
        let mut c = Colour::new(&ColourP { exc: 0.0, exc_fc: 4000.0, tube: Some(0.3) }, 48000.0);
        let mut x = vec![[0.0; 2]; 256];
        x[10] = [1e-9, -1e-9];
        c.process(&mut x);
        let peak = x.iter().enumerate().max_by(|a, b| a.1[0].abs().total_cmp(&b.1[0].abs())).unwrap().0;
        assert_eq!(peak, 10 + LATENCY);
        assert!((x[peak][0] - 1e-9).abs() < 1e-12 && (x[peak][1] + 1e-9).abs() < 1e-12);
    }

    #[test]
    fn tube_harmonics_do_not_fold_back() {
        // 15 kHz, driven hard. Its third harmonic (45 kHz) would land on 3 kHz at 48 kHz.
        let p = ColourP { exc: 0.0, exc_fc: 4000.0, tube: Some(0.6) };
        let out = run(p, 15000.0, 0.5);
        let fund = tone(&out, 15000.0);
        assert!(tone(&out, 3000.0) - fund < -75.0, "alias {} dBc", tone(&out, 3000.0) - fund);
        // The same curve at the base rate does fold back, audibly.
        let k = 0.5 + 3.0 * 0.6;
        let tb = (k * BIAS).tanh();
        let naive: Vec<f64> = (0..4800)
            .map(|i| (((k * (0.5 * (std::f64::consts::TAU * 15000.0 * i as f64 / 48000.0).sin() + BIAS)).tanh()) - tb) / (k * (1.0 - tb * tb)))
            .collect();
        assert!(tone(&naive, 3000.0) - tone(&naive, 15000.0) > -45.0);
    }

    #[test]
    fn tube_adds_even_and_odd_harmonics() {
        let out = run(ColourP { exc: 0.0, exc_fc: 4000.0, tube: Some(0.5) }, 1000.0, 0.5);
        let fund = tone(&out, 1000.0);
        assert!(tone(&out, 2000.0) - fund > -40.0 && tone(&out, 3000.0) - fund > -40.0);
        assert!((fund - to_db(0.5)).abs() < 2.0);
    }

    #[test]
    fn exciter_works_on_the_top_band_only() {
        let p = ColourP { exc: 1.0, exc_fc: 4000.0, tube: None };
        let hi = run(p, 6000.0, 0.1);
        assert!(tone(&hi, 12000.0) - tone(&hi, 6000.0) > -40.0);
        assert!((tone(&hi, 6000.0) - to_db(0.1)).abs() < 0.5);
        // A bass tone is far below the corner: nothing is added.
        let lo = run(p, 100.0, 0.5);
        assert!(tone(&lo, 200.0) - tone(&lo, 100.0) < -90.0);
        assert!((tone(&lo, 100.0) - to_db(0.5)).abs() < 0.01);
    }
}
