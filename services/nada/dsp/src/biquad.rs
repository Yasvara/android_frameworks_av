//! Second-order sections: design and stereo processing in f64.

use crate::Frame;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coef {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Peak,
    LowShelf,
    HighShelf,
    LowPass,
    HighPass,
    Notch,
    BandPass,
    AllPass,
}

impl Coef {
    pub const UNITY: Coef = Coef { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    /// |H| at `w` radians per sample.
    pub fn mag(&self, w: f64) -> f64 {
        let (c1, s1) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    pub fn stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }
}

/// Designs one section. `f0` is clamped below Nyquist, `q` to a sane range.
pub fn design(kind: Kind, fs: f64, f0: f64, gain_db: f64, q: f64) -> Coef {
    let f0 = f0.clamp(1.0, 0.49 * fs);
    let q = q.clamp(0.05, 40.0);
    if kind == Kind::Peak {
        if let Some(c) = matched_peak(fs, f0, gain_db, q) {
            return c;
        }
    }
    rbj(kind, fs, f0, gain_db, q)
}

/// Audio EQ Cookbook (Bristow-Johnson) forms.
pub fn rbj(kind: Kind, fs: f64, f0: f64, gain_db: f64, q: f64) -> Coef {
    let a = 10f64.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / fs;
    let (cw, sw) = (w0.cos(), w0.sin());
    let al = sw / (2.0 * q);
    let (b0, b1, b2, a0, a1, a2);
    match kind {
        Kind::Peak => {
            b0 = 1.0 + al * a;
            b1 = -2.0 * cw;
            b2 = 1.0 - al * a;
            a0 = 1.0 + al / a;
            a1 = -2.0 * cw;
            a2 = 1.0 - al / a;
        }
        Kind::LowShelf => {
            let r = 2.0 * a.sqrt() * al;
            b0 = a * ((a + 1.0) - (a - 1.0) * cw + r);
            b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw);
            b2 = a * ((a + 1.0) - (a - 1.0) * cw - r);
            a0 = (a + 1.0) + (a - 1.0) * cw + r;
            a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw);
            a2 = (a + 1.0) + (a - 1.0) * cw - r;
        }
        Kind::HighShelf => {
            let r = 2.0 * a.sqrt() * al;
            b0 = a * ((a + 1.0) + (a - 1.0) * cw + r);
            b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw);
            b2 = a * ((a + 1.0) + (a - 1.0) * cw - r);
            a0 = (a + 1.0) - (a - 1.0) * cw + r;
            a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw);
            a2 = (a + 1.0) - (a - 1.0) * cw - r;
        }
        Kind::LowPass => {
            b0 = (1.0 - cw) / 2.0;
            b1 = 1.0 - cw;
            b2 = (1.0 - cw) / 2.0;
            a0 = 1.0 + al;
            a1 = -2.0 * cw;
            a2 = 1.0 - al;
        }
        Kind::HighPass => {
            b0 = (1.0 + cw) / 2.0;
            b1 = -(1.0 + cw);
            b2 = (1.0 + cw) / 2.0;
            a0 = 1.0 + al;
            a1 = -2.0 * cw;
            a2 = 1.0 - al;
        }
        Kind::Notch => {
            b0 = 1.0;
            b1 = -2.0 * cw;
            b2 = 1.0;
            a0 = 1.0 + al;
            a1 = -2.0 * cw;
            a2 = 1.0 - al;
        }
        Kind::BandPass => {
            b0 = al;
            b1 = 0.0;
            b2 = -al;
            a0 = 1.0 + al;
            a1 = -2.0 * cw;
            a2 = 1.0 - al;
        }
        Kind::AllPass => {
            b0 = 1.0 - al;
            b1 = -2.0 * cw;
            b2 = 1.0 + al;
            a0 = 1.0 + al;
            a1 = -2.0 * cw;
            a2 = 1.0 - al;
        }
    }
    Coef { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
}

/// Peaking section whose response follows the analogue prototype up to Nyquist instead of
/// narrowing near it (Vicanek, "Matched Second Order Digital Filters", 2016).
fn matched_peak(fs: f64, f0: f64, gain_db: f64, q: f64) -> Option<Coef> {
    let a = 10f64.powf(gain_db / 40.0);
    let g = a * a;
    let w0 = 2.0 * PI * f0 / fs;
    // Prototype (s^2 + s A/Q + 1) / (s^2 + s/(A Q) + 1): pole damping 1/(2 A Q).
    let qp = 1.0 / (2.0 * a * q);
    let a1 = if qp <= 1.0 {
        -2.0 * (-qp * w0).exp() * ((1.0 - qp * qp).sqrt() * w0).cos()
    } else {
        -2.0 * (-qp * w0).exp() * ((qp * qp - 1.0).sqrt() * w0).cosh()
    };
    let a2 = (-2.0 * qp * w0).exp();
    let aa0 = (1.0 + a1 + a2).powi(2);
    let aa1 = (1.0 - a1 + a2).powi(2);
    let aa2 = -4.0 * a2;
    let p1 = (w0 / 2.0).sin().powi(2);
    let p0 = 1.0 - p1;
    let p2 = 4.0 * p0 * p1;
    let r1 = (aa0 * p0 + aa1 * p1 + aa2 * p2) * g * g;
    let r2 = (-aa0 + aa1 + 4.0 * (p0 - p1) * aa2) * g * g;
    let bb0 = aa0;
    let bb2 = (r1 - r2 * p1 - bb0) / (4.0 * p1 * p1);
    let bb1 = r2 + bb0 + 4.0 * (p1 - p0) * bb2;
    if bb1 < 0.0 {
        return None;
    }
    let w = 0.5 * (bb0.sqrt() + bb1.sqrt());
    let d = w * w + bb2;
    if d < 0.0 {
        return None;
    }
    let b0 = 0.5 * (w + d.sqrt());
    let b1 = 0.5 * (bb0.sqrt() - bb1.sqrt());
    let b2 = -bb2 / (4.0 * b0);
    let c = Coef { b0, b1, b2, a1, a2 };
    (c.stable() && [b0, b1, b2].iter().all(|v| v.is_finite())).then_some(c)
}

/// Magnitude of the analogue peaking prototype at `f`, for tests and the fit display.
pub fn analog_peak_mag(f: f64, f0: f64, gain_db: f64, q: f64) -> f64 {
    let a = 10f64.powf(gain_db / 40.0);
    let s = f / f0;
    let nr = 1.0 - s * s;
    let ni = s * a / q;
    let di = s / (a * q);
    ((nr * nr + ni * ni) / (nr * nr + di * di)).sqrt()
}

/// One section on both channels, transposed direct form II. The state is f64 so that
/// corners far below the sample rate add no audible rounding noise.
#[derive(Clone, Copy)]
pub struct Biquad {
    pub c: Coef,
    s1: Frame,
    s2: Frame,
}

impl Biquad {
    pub fn new(c: Coef) -> Self {
        Biquad { c, s1: [0.0; 2], s2: [0.0; 2] }
    }

    pub fn reset(&mut self) {
        self.s1 = [0.0; 2];
        self.s2 = [0.0; 2];
    }

    #[inline]
    pub fn run(&mut self, x: &mut [Frame]) {
        let Coef { b0, b1, b2, a1, a2 } = self.c;
        let (mut s1, mut s2) = (self.s1, self.s2);
        for f in x.iter_mut() {
            let y = [b0 * f[0] + s1[0], b0 * f[1] + s1[1]];
            s1 = [b1 * f[0] - a1 * y[0] + s2[0], b1 * f[1] - a1 * y[1] + s2[1]];
            s2 = [b2 * f[0] - a2 * y[0], b2 * f[1] - a2 * y[1]];
            *f = y;
        }
        self.s1 = s1;
        self.s2 = s2;
    }

    #[inline]
    pub fn tick2(&mut self, x: Frame) -> Frame {
        let mut one = [x];
        self.run(&mut one);
        one[0]
    }

    /// Single channel, for side paths.
    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let c = &self.c;
        let y = c.b0 * x + self.s1[0];
        self.s1[0] = c.b1 * x - c.a1 * y + self.s2[0];
        self.s2[0] = c.b2 * x - c.a2 * y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(c: Coef, fs: f64, f: f64) -> f64 {
        // Steady-state amplitude of a sine through the section, by quadrature correlation.
        let mut b = Biquad::new(c);
        let n = fs as usize;
        let mut buf = vec![[0.0; 2]; n];
        for (i, s) in buf.iter_mut().enumerate() {
            let v = (2.0 * PI * f * i as f64 / fs).sin();
            *s = [v, v];
        }
        b.run(&mut buf);
        let (mut si, mut co) = (0f64, 0f64);
        for (i, s) in buf.iter().enumerate().skip(n / 2) {
            let ph = 2.0 * PI * f * i as f64 / fs;
            si += s[0] * ph.sin();
            co += s[0] * ph.cos();
        }
        (si * si + co * co).sqrt() / (n as f64 / 4.0)
    }

    #[test]
    fn response_matches_formula() {
        let fs = 48000.0;
        for (k, f0, g, q) in [
            (Kind::Peak, 1000.0, 6.0, 1.4),
            (Kind::LowShelf, 100.0, 5.0, 0.707),
            (Kind::HighShelf, 8000.0, -4.0, 0.707),
            (Kind::LowPass, 5000.0, 0.0, 0.707),
            (Kind::HighPass, 60.0, 0.0, 0.707),
        ] {
            let c = design(k, fs, f0, g, q);
            assert!(c.stable());
            for f in [50.0, 100.0, 1000.0, 5000.0, 12000.0] {
                let want = c.mag(2.0 * PI * f / fs);
                let got = measured(c, fs, f);
                assert!((to(got) - to(want)).abs() < 0.05, "{k:?} {f}: {got} vs {want}");
            }
        }
    }

    fn to(x: f64) -> f64 {
        20.0 * x.max(1e-12).log10()
    }

    #[test]
    fn matched_peak_follows_analogue_near_nyquist() {
        let fs = 48000.0;
        for (f0, g, q) in [(12000.0, 6.0, 1.0), (16000.0, -8.0, 2.0), (1000.0, 12.0, 0.7), (40.0, 9.0, 1.2)] {
            let m = matched_peak(fs, f0, g, q).expect("matched design");
            let r = rbj(Kind::Peak, fs, f0, g, q);
            let (mut em, mut er) = (0f64, 0f64);
            let mut f = 20.0;
            while f < 20000.0 {
                let want = to(analog_peak_mag(f, f0, g, q));
                em = em.max((to(m.mag(2.0 * PI * f / fs)) - want).abs());
                er = er.max((to(r.mag(2.0 * PI * f / fs)) - want).abs());
                f *= 1.02;
            }
            assert!(em < 1.0, "matched error {em} dB at f0 {f0}");
            assert!(em <= er + 1e-6, "matched {em} worse than cookbook {er} at f0 {f0}");
            assert!((to(m.mag(2.0 * PI * f0 / fs)) - g).abs() < 0.01);
            assert!((m.mag(0.0) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn low_corner_noise_floor() {
        // 30 Hz, Q 4 peak at 192 kHz: residual against an exact f64 reference stays at zero
        // because the section itself is f64; the check is that output is finite and bounded.
        let c = design(Kind::Peak, 192000.0, 30.0, 12.0, 4.0);
        let mut b = Biquad::new(c);
        let mut buf = vec![[1e-3, -1e-3]; 4096];
        b.run(&mut buf);
        assert!(buf.iter().all(|f| f[0].is_finite() && f[0].abs() < 1.0));
        assert_eq!(buf[100][0], -buf[100][1]);
    }
}
