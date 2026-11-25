//! The stages besides the convolver. None of them delays the signal.

use crate::biquad::{design, Biquad, Coef, Kind};
use crate::{Frame, Stage};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::Arc;

/// Fixed gain: the headroom the later boosts need.
pub struct Trim(pub f64);

impl Stage for Trim {
    fn process(&mut self, x: &mut [Frame]) {
        let g = self.0;
        for f in x.iter_mut() {
            *f = [f[0] * g, f[1] * g];
        }
    }
    fn reset(&mut self) {}
}

/// Parametric equaliser: minimum-phase sections in series.
pub struct Eq(pub Vec<Biquad>);

impl Eq {
    pub fn new(c: &[Coef]) -> Self {
        Eq(c.iter().map(|&c| Biquad::new(c)).collect())
    }
}

impl Stage for Eq {
    fn process(&mut self, x: &mut [Frame]) {
        for b in self.0.iter_mut() {
            b.run(x);
        }
    }
    fn reset(&mut self) {
        self.0.iter_mut().for_each(Biquad::reset);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BassP {
    pub shelf: Coef,
    pub fc: f64,
    pub harm: f64,
    /// Lowest frequency the speaker plays, Hz; 0 keeps the fundamental.
    pub cut: f64,
}

/// Weights of the 2nd, 3rd and 4th harmonic.
const HARM: [f64; 3] = [0.5, 0.35, 0.15];

/// Bass: a low shelf, plus psychoacoustic bass. The band below the speaker's limit is taken
/// from the mono sum, divided by its own envelope and shaped by Chebyshev polynomials, which
/// turn a full-scale sine into exact harmonics; scaled back by the envelope, the overtones
/// follow the bass in level and the ear hears the missing fundamental from them. A high-pass
/// keeps only what lies above the limit. Optionally the fundamental itself is removed from the
/// signal, so the speaker and the limiter spend nothing on what cannot be heard. Harmonics of
/// a band this low stay far below Nyquist: no oversampling, nothing delayed.
pub struct Bass {
    shelf: Biquad,
    cut: Option<[Biquad; 2]>,
    src: [Biquad; 3],
    post: [Biquad; 3],
    env: f64,
    att: f64,
    rel: f64,
    harm: f64,
}

impl Bass {
    pub fn new(p: &BassP, fs: f64) -> Self {
        let bq = |k, f| Biquad::new(design(k, fs, f, 0.0, 0.7071));
        let top = if p.cut > 0.0 { p.cut } else { p.fc };
        let k = |ms: f64| 1.0 - (-1.0 / (ms * 1e-3 * fs)).exp();
        Bass {
            shelf: Biquad::new(p.shelf),
            cut: (p.cut > 0.0).then(|| [bq(Kind::HighPass, p.cut); 2]),
            src: [bq(Kind::HighPass, 30.0), bq(Kind::LowPass, top), bq(Kind::LowPass, top)],
            post: [bq(Kind::HighPass, top), bq(Kind::HighPass, top), bq(Kind::LowPass, top * 5.0)],
            env: 0.0,
            att: k(2.0),
            rel: k(80.0),
            harm: p.harm,
        }
    }
}

impl Stage for Bass {
    fn process(&mut self, x: &mut [Frame]) {
        if self.harm <= 0.0 && self.cut.is_none() {
            self.shelf.run(x);
            return;
        }
        for f in x.iter_mut() {
            let mut d = self.shelf.tick2(*f);
            if let Some(c) = &mut self.cut {
                d = { let t = c[0].tick2(d); c[1].tick2(t) };
            }
            if self.harm > 0.0 {
                let band = self.src.iter_mut().fold(0.5 * (f[0] + f[1]), |v, q| q.tick(v));
                let a = band.abs();
                self.env += (a - self.env) * if a > self.env { self.att } else { self.rel };
                if self.env > 1e-7 {
                    let u = (band / self.env).clamp(-1.0, 1.0);
                    let u2 = u * u;
                    let t = HARM[0] * (2.0 * u2 - 1.0) + HARM[1] * (4.0 * u2 - 3.0) * u + HARM[2] * (8.0 * u2 * (u2 - 1.0) + 1.0);
                    let h = self.post.iter_mut().fold(t * self.env, |v, q| q.tick(v)) * self.harm;
                    d = [d[0] + h, d[1] + h];
                }
            }
            *f = d;
        }
    }
    fn reset(&mut self) {
        self.shelf.reset();
        self.env = 0.0;
        self.src.iter_mut().chain(self.post.iter_mut()).chain(self.cut.iter_mut().flatten()).for_each(Biquad::reset);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialP {
    /// Crossfeed level in dB (0 = off) and its corner.
    pub feed: f64,
    pub fc: f64,
    /// Side gain of the middle band: 1 leaves the image alone.
    pub width: f64,
    /// Side gains below `xlow` and above `xhigh`.
    pub low: f64,
    pub high: f64,
    pub xlow: f64,
    pub xhigh: f64,
}

/// Headphone crossfeed (Bauer's stereophonic-to-binaural network), then mid/side width. One
/// width for every band is a plain mid/side gain. Different widths split the side signal with
/// fourth-order Linkwitz-Riley crossovers, whose bands sum to an all-pass; the mid signal goes
/// through the same all-passes, so mid and side stay in phase and a bass width of 0 is mono.
pub struct Spatial {
    a0_lo: f64,
    b1_lo: f64,
    a0_hi: f64,
    a1_hi: f64,
    b1_hi: f64,
    gain: f64,
    lo: Frame,
    hi: Frame,
    last: Frame,
    feed: bool,
    width: f64,
    low: f64,
    high: f64,
    split: Option<Split>,
}

struct Split {
    lo: [Biquad; 2],
    rest: [Biquad; 2],
    mid: [Biquad; 2],
    top: [Biquad; 2],
    /// Second crossover's all-pass on the bass band.
    lo_ap: Biquad,
    /// Both crossovers' all-passes on the mid signal.
    m_ap: [Biquad; 2],
}

#[inline]
fn chain(q: &mut [Biquad], x: f64) -> f64 {
    q.iter_mut().fold(x, |v, b| b.tick(v))
}

impl Spatial {
    pub fn new(p: &SpatialP, fs: f64) -> Self {
        let feed = p.feed > 0.0;
        let gb_lo = p.feed * -5.0 / 6.0 - 3.0;
        let gb_hi = p.feed / 6.0 - 3.0;
        let g_lo = 10f64.powf(gb_lo / 20.0);
        let g_hi = 1.0 - 10f64.powf(gb_hi / 20.0);
        let fc_hi = p.fc * 2f64.powf((gb_lo - 20.0 * g_hi.log10()) / 12.0);
        let x = (-2.0 * std::f64::consts::PI * p.fc / fs).exp();
        let y = (-2.0 * std::f64::consts::PI * fc_hi / fs).exp();
        Spatial {
            a0_lo: g_lo * (1.0 - x),
            b1_lo: x,
            a0_hi: 1.0 - g_hi * (1.0 - y),
            a1_hi: -y,
            b1_hi: y,
            gain: 1.0 / (1.0 - g_hi + g_lo),
            lo: [0.0; 2],
            hi: [0.0; 2],
            last: [0.0; 2],
            feed,
            width: p.width,
            low: p.low,
            high: p.high,
            split: (p.low != p.width || p.high != p.width).then(|| {
                let bq = |k, f| Biquad::new(design(k, fs, f, 0.0, 0.7071));
                let xh = p.xhigh.max(p.xlow * 2.0);
                Split {
                    lo: [bq(Kind::LowPass, p.xlow); 2],
                    rest: [bq(Kind::HighPass, p.xlow); 2],
                    mid: [bq(Kind::LowPass, xh); 2],
                    top: [bq(Kind::HighPass, xh); 2],
                    lo_ap: bq(Kind::AllPass, xh),
                    m_ap: [bq(Kind::AllPass, p.xlow), bq(Kind::AllPass, xh)],
                }
            }),
        }
    }
}

impl Stage for Spatial {
    fn process(&mut self, x: &mut [Frame]) {
        if self.feed {
            for f in x.iter_mut() {
                for c in 0..2 {
                    self.lo[c] = self.a0_lo * f[c] + self.b1_lo * self.lo[c];
                    self.hi[c] = self.a0_hi * f[c] + self.a1_hi * self.last[c] + self.b1_hi * self.hi[c];
                }
                self.last = *f;
                *f = [(self.hi[0] + self.lo[1]) * self.gain, (self.hi[1] + self.lo[0]) * self.gain];
            }
        }
        if let Some(sp) = &mut self.split {
            let (w, wl, wh) = (self.width, self.low, self.high);
            for f in x.iter_mut() {
                let m = chain(&mut sp.m_ap, 0.5 * (f[0] + f[1]));
                let s = 0.5 * (f[0] - f[1]);
                let lo = chain(&mut sp.lo, s);
                let lo = sp.lo_ap.tick(lo);
                let rest = chain(&mut sp.rest, s);
                let s = wl * lo + w * chain(&mut sp.mid, rest) + wh * chain(&mut sp.top, rest);
                *f = [m + s, m - s];
            }
        } else if self.width != 1.0 {
            let w = self.width;
            for f in x.iter_mut() {
                let m = 0.5 * (f[0] + f[1]);
                let s = 0.5 * (f[0] - f[1]) * w;
                *f = [m + s, m - s];
            }
        }
    }
    fn reset(&mut self) {
        self.lo = [0.0; 2];
        self.hi = [0.0; 2];
        self.last = [0.0; 2];
        if let Some(sp) = &mut self.split {
            for b in sp.lo.iter_mut().chain(&mut sp.rest).chain(&mut sp.mid).chain(&mut sp.top).chain(&mut sp.m_ap) {
                b.reset();
            }
            sp.lo_ap.reset();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompP {
    pub threshold: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup: f64,
}

/// Feed-forward compressor, channels linked, 6 dB soft knee, no look-ahead.
pub struct Comp {
    p: CompP,
    att: f64,
    rel: f64,
    env: f64,
}

const KNEE: f64 = 6.0;

/// Gain reduction in dB for a level `over` dB above the threshold, 6 dB soft knee.
/// `slope` is `1 / ratio - 1`.
#[inline]
pub fn knee_cut(over: f64, slope: f64) -> f64 {
    if 2.0 * over < -KNEE {
        0.0
    } else if 2.0 * over <= KNEE {
        -slope * (over + KNEE / 2.0).powi(2) / (2.0 * KNEE)
    } else {
        -slope * over
    }
}

impl Comp {
    pub fn new(p: &CompP, fs: f64) -> Self {
        let k = |ms: f64| (-1.0 / (ms.max(0.01) * 1e-3 * fs)).exp();
        Comp { p: *p, att: k(p.attack_ms), rel: k(p.release_ms), env: 0.0 }
    }
}

impl Stage for Comp {
    fn process(&mut self, x: &mut [Frame]) {
        let (t, slope) = (self.p.threshold, 1.0 / self.p.ratio - 1.0);
        for f in x.iter_mut() {
            let peak = f[0].abs().max(f[1].abs());
            // Below the knee nothing happens; skip the logarithm.
            let over = if peak > 1e-6 { 20.0 * peak.log10() - t } else { -200.0 };
            let cut = knee_cut(over, slope);
            let k = if cut > self.env { self.att } else { self.rel };
            self.env = k * self.env + (1.0 - k) * cut;
            let g = if self.env > 1e-9 || self.p.makeup != 0.0 {
                10f64.powf((self.p.makeup - self.env) / 20.0)
            } else {
                1.0
            };
            *f = [f[0] * g, f[1] * g];
        }
    }
    fn reset(&mut self) {
        self.env = 0.0;
    }
}

/// What the limiter did, for diagnostics.
#[derive(Default)]
pub struct LimStats {
    /// Calls in which gain was reduced.
    pub blocks: AtomicU32,
    /// Largest reduction seen, in 0.01 dB.
    pub max_cdb: AtomicU32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimP {
    /// Output gain per channel: level and balance.
    pub gain: [f64; 2],
    pub ceiling: f64,
    pub release_ms: f64,
    /// Frames of look-ahead; 0 is this zero-delay limiter, otherwise `lookahead::LookLimiter`.
    pub lookahead: u32,
}

/// Output gain and a sample-peak ceiling. Gain drops within the same sample, so nothing is
/// delayed and no sample leaves above the ceiling; it is a safety net behind the headroom
/// trim, and stays at exactly unity when the signal is below the ceiling.
pub struct Limiter {
    gain: [f64; 2],
    ceil: f64,
    rel: f64,
    g: f64,
    stats: Arc<LimStats>,
}

impl Limiter {
    pub fn new(p: &LimP, fs: f64, stats: Arc<LimStats>) -> Self {
        Limiter {
            gain: p.gain,
            ceil: p.ceiling,
            rel: 1.0 - (-1.0 / (p.release_ms.max(1.0) * 1e-3 * fs)).exp(),
            g: 1.0,
            stats,
        }
    }
}

impl Stage for Limiter {
    fn process(&mut self, x: &mut [Frame]) {
        let mut low = 1.0f64;
        for f in x.iter_mut() {
            let v = [f[0] * self.gain[0], f[1] * self.gain[1]];
            let peak = v[0].abs().max(v[1].abs());
            let want = if peak > self.ceil { self.ceil / peak } else { 1.0 };
            self.g = if want < self.g { want } else { (self.g + (1.0 - self.g) * self.rel).min(want) };
            if self.g > 0.999_999 {
                self.g = 1.0;
            }
            low = low.min(self.g);
            *f = [v[0] * self.g, v[1] * self.g];
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
        self.g = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_holds_ceiling_and_is_transparent_below() {
        let st = Arc::new(LimStats::default());
        let mut l = Limiter::new(&LimP { gain: [1.0; 2], ceiling: 0.5, release_ms: 50.0, lookahead: 0 }, 48000.0, st.clone());
        let mut quiet: Vec<Frame> = (0..480).map(|i| [0.4 * (i as f64 * 0.1).sin(), 0.3]).collect();
        let want = quiet.clone();
        l.process(&mut quiet);
        assert_eq!(quiet, want);
        assert_eq!(st.blocks.load(Relaxed), 0);
        let mut loud: Vec<Frame> = (0..4800).map(|i| [1.7 * (i as f64 * 0.05).sin(), -0.2]).collect();
        l.process(&mut loud);
        assert!(loud.iter().all(|f| f[0].abs() <= 0.5 + 1e-12));
        assert!(st.blocks.load(Relaxed) == 1 && st.max_cdb.load(Relaxed) > 1000);
    }

    #[test]
    fn crossfeed_keeps_mono_level_and_narrows_bass() {
        let mut s = Spatial::new(&SpatialP { feed: 4.5, fc: 700.0, ..FLAT }, 48000.0);
        let mut mono = vec![[0.5, 0.5]; 48000];
        s.process(&mut mono);
        assert!((mono[47999][0] - 0.5).abs() < 1e-6);
        s.reset();
        let mut left = vec![[0.5, 0.0]; 48000];
        s.process(&mut left);
        // At DC the far ear hears the near channel a few dB down, not silence.
        let ratio = left[47999][1] / left[47999][0];
        assert!(ratio > 0.3 && ratio < 0.9, "{ratio}");
    }

    #[test]
    fn width_is_mid_side() {
        let mut s = Spatial::new(&SpatialP { width: 0.0, low: 0.0, high: 0.0, ..FLAT }, 48000.0);
        let mut x = vec![[1.0, 0.0]];
        s.process(&mut x);
        assert_eq!(x[0], [0.5, 0.5]);
    }

    const FLAT: SpatialP = SpatialP { feed: 0.0, fc: 700.0, width: 1.0, low: 1.0, high: 1.0, xlow: 150.0, xhigh: 4000.0 };

    fn tone(f: f64, buf: &[Frame]) -> f64 {
        let (mut s, mut c) = (0f64, 0f64);
        for (i, v) in buf.iter().enumerate().skip(24000) {
            let ph = std::f64::consts::TAU * f * i as f64 / 48000.0;
            s += v[0] * ph.sin();
            c += v[0] * ph.cos();
        }
        2.0 * (s * s + c * c).sqrt() / (buf.len() - 24000) as f64
    }

    fn sines(tones: &[(f64, f64)]) -> Vec<Frame> {
        (0..48000)
            .map(|i| {
                let v: f64 = tones.iter().map(|(f, a)| a * (std::f64::consts::TAU * f * i as f64 / 48000.0).sin()).sum();
                [v, v]
            })
            .collect()
    }

    #[test]
    fn psychoacoustic_bass_moves_the_fundamental_above_the_limit() {
        let p = BassP { shelf: Coef::UNITY, fc: 100.0, harm: 1.0, cut: 200.0 };
        let mut x = sines(&[(70.0, 0.5), (1000.0, 0.1)]);
        Bass::new(&p, 48000.0).process(&mut x);
        // The fundamental is gone, its overtones above the limit carry it.
        assert!(tone(70.0, &x) < 0.5 * crate::db(-20.0), "fundamental {}", tone(70.0, &x));
        let over = tone(280.0, &x).hypot(tone(210.0, &x));
        assert!(over > 0.5 * crate::db(-16.0), "overtones {over}");
        // The midrange is untouched.
        assert!((tone(1000.0, &x) - 0.1).abs() < 0.002);
        // Overtones scale with the bass, not with a fixed curve.
        let mut q = sines(&[(70.0, 0.05)]);
        Bass::new(&p, 48000.0).process(&mut q);
        let ratio = tone(210.0, &q).hypot(tone(280.0, &q)) / over;
        assert!((ratio - 0.1).abs() < 0.01, "{ratio}");
    }

    #[test]
    fn shelf_only_bass_is_the_shelf() {
        let shelf = design(Kind::LowShelf, 48000.0, 100.0, 4.0, 0.7071);
        let mut a = sines(&[(50.0, 0.3), (3000.0, 0.2)]);
        let mut b = a.clone();
        Bass::new(&BassP { shelf, fc: 100.0, harm: 0.0, cut: 0.0 }, 48000.0).process(&mut a);
        Biquad::new(shelf).run(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn imager_bands_and_exact_flat_width() {
        // One width for every band is a plain mid/side gain.
        let mut s = Spatial::new(&SpatialP { width: 1.4, low: 1.4, high: 1.4, ..FLAT }, 48000.0);
        let mut x = vec![[1.0, 0.0]];
        s.process(&mut x);
        assert!((x[0][0] - 1.2).abs() < 1e-12 && (x[0][1] + 0.2).abs() < 1e-12);
        // Mono bass: a left-only 50 Hz reaches both sides equally, a left-only 2 kHz does not.
        let mut s = Spatial::new(&SpatialP { low: 0.0, ..FLAT }, 48000.0);
        let mut x: Vec<Frame> = sines(&[(50.0, 0.5), (2000.0, 0.5)]).into_iter().map(|f| [f[0], 0.0]).collect();
        s.process(&mut x);
        let right: Vec<Frame> = x.iter().map(|f| [f[1], 0.0]).collect();
        assert!((tone(50.0, &right) / tone(50.0, &x) - 1.0).abs() < 0.05);
        assert!(tone(2000.0, &right) < 0.01);
    }

    #[test]
    fn compressor_reduces_above_threshold_only() {
        let p = CompP { threshold: -20.0, ratio: 4.0, attack_ms: 1.0, release_ms: 50.0, makeup: 0.0 };
        let mut c = Comp::new(&p, 48000.0);
        let mut quiet = vec![[0.01, 0.01]; 4800];
        c.process(&mut quiet);
        assert_eq!(quiet[4799], [0.01, 0.01]);
        let mut loud = vec![[1.0, 1.0]; 48000];
        c.process(&mut loud);
        // 20 dB over at 4:1 leaves 5 dB over: -15 dBFS.
        let out = 20.0 * loud[47999][0].log10();
        assert!((out + 15.0).abs() < 0.1, "{out}");
    }
}

#[cfg(test)]
mod peak_probe {
    use super::*;

    /// Peak gain of the bass stage over a few signals, harmonics at 100 %.
    fn worst(cut: f64) -> f64 {
        let p = BassP { shelf: Coef::UNITY, fc: 120.0, harm: 1.0, cut };
        let mut worst = 0f64;
        let mut seed = 1u64;
        let mut noise = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let sets: Vec<Vec<(f64, f64)>> = vec![
            vec![(50.0, 0.9), (1000.0, 0.1)],
            vec![(80.0, 0.6), (120.0, 0.3), (2000.0, 0.1)],
            vec![(40.0, 0.5), (60.0, 0.5)],
            vec![(150.0, 0.7), (440.0, 0.3)],
        ];
        for tones in sets {
            let mut x: Vec<Frame> = (0..96000)
                .map(|i| {
                    let v: f64 = tones.iter().map(|(f, a)| a * (std::f64::consts::TAU * f * i as f64 / 48000.0).sin()).sum();
                    [v, v]
                })
                .collect();
            let pin = x.iter().fold(0f64, |m, f| m.max(f[0].abs()));
            Bass::new(&p, 48000.0).process(&mut x);
            let pout = x[24000..].iter().fold(0f64, |m, f| m.max(f[0].abs()));
            worst = worst.max(pout / pin);
        }
        // Kick-like bursts in noise.
        let mut x: Vec<Frame> = (0..96000)
            .map(|i| {
                let t = (i % 24000) as f64 / 48000.0;
                let v = 0.8 * (std::f64::consts::TAU * 55.0 * t).sin() * (-t * 8.0).exp() + 0.1 * noise();
                [v, v]
            })
            .collect();
        let pin = x.iter().fold(0f64, |m, f| m.max(f[0].abs()));
        Bass::new(&p, 48000.0).process(&mut x);
        let pout = x.iter().fold(0f64, |m, f| m.max(f[0].abs()));
        worst.max(pout / pin)
    }

    #[test]
    fn overtones_need_little_headroom() {
        // config::plan budgets 1 + 0.35 * amount with the fundamental kept, nothing with it cut.
        let (keep, cut) = (worst(0.0), worst(250.0));
        assert!(keep < 1.35 && cut < 1.0, "{keep} {cut}");
    }
}
