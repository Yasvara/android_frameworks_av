//! Log-mel frames: 16 ms Hann window every 5 ms (the haptic step), 32 bands from 60 Hz to 7.6 kHz.
//! A 256-point real FFT runs as a 128-point complex one.

pub const N: usize = 256;
pub const BANDS: usize = 32;
const HALF: usize = N / 2;
const BINS: usize = HALF + 1;
const F_LO: f32 = 60.0;
const F_HI: f32 = 7600.0;
const BODY_HZ: f32 = 2000.0;
pub const FRAME_MS: u32 = 5;

#[derive(Clone, Copy)]
pub struct Frame {
    /// dB re a full-scale sine, per band.
    pub mel: [f32; BANDS],
    /// dB re a full-scale sine, whole frame.
    pub level: f32,
    /// Amplitude below 2 kHz, linear: what the envelope follows.
    pub body: f32,
}

impl Default for Frame {
    fn default() -> Self {
        Self { mel: [-120.0; BANDS], level: -120.0, body: 0.0 }
    }
}

struct Fft {
    cos: [f32; HALF / 2],
    sin: [f32; HALF / 2],
    rev: [u8; HALF],
    // Real split: e^{-2πik/N}.
    wr: [f32; BINS],
    wi: [f32; BINS],
}

impl Fft {
    fn new() -> Self {
        let mut f = Fft { cos: [0.0; HALF / 2], sin: [0.0; HALF / 2], rev: [0; HALF], wr: [0.0; BINS], wi: [0.0; BINS] };
        for k in 0..HALF / 2 {
            let a = -2.0 * std::f64::consts::PI * k as f64 / HALF as f64;
            f.cos[k] = a.cos() as f32;
            f.sin[k] = a.sin() as f32;
        }
        let bits = HALF.trailing_zeros();
        for i in 0..HALF {
            f.rev[i] = ((i as u32).reverse_bits() >> (32 - bits)) as u8;
        }
        for k in 0..BINS {
            let a = -2.0 * std::f64::consts::PI * k as f64 / N as f64;
            f.wr[k] = a.cos() as f32;
            f.wi[k] = a.sin() as f32;
        }
        f
    }

    /// Power spectrum of N real samples into BINS bins.
    fn power(&self, x: &[f32; N], out: &mut [f32; BINS]) {
        let mut re = [0f32; HALF];
        let mut im = [0f32; HALF];
        for i in 0..HALF {
            let j = self.rev[i] as usize;
            re[i] = x[2 * j];
            im[i] = x[2 * j + 1];
        }
        let mut size = 2;
        while size <= HALF {
            let half = size / 2;
            let step = HALF / size;
            for start in (0..HALF).step_by(size) {
                for k in 0..half {
                    let (c, s) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
        for k in 0..BINS {
            let (zr, zi) = (re[k % HALF], im[k % HALF]);
            let (cr, ci) = (re[(HALF - k) % HALF], -im[(HALF - k) % HALF]);
            let (er, ei) = ((zr + cr) * 0.5, (zi + ci) * 0.5);
            // O = (Z - conj Z') / 2i
            let (or, oi) = ((zi - ci) * 0.5, -(zr - cr) * 0.5);
            let (w_r, w_i) = (self.wr[k], self.wi[k]);
            let xr = er + w_r * or - w_i * oi;
            let xi = ei + w_r * oi + w_i * or;
            out[k] = xr * xr + xi * xi;
        }
    }
}

pub struct Mel {
    fft: Fft,
    window: [f32; N],
    frame: [f32; N],
    hop: usize,
    pending: usize,
    norm: f32,
    // Triangular weights, flattened: band b uses weights[from[b]..from[b+1]] starting at bin[b].
    bin: [u8; BANDS],
    from: [u16; BANDS + 1],
    weights: Vec<f32>,
    body_bands: usize,
}

fn mel_of(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn hz_of(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

impl Mel {
    pub fn new(rate: u32) -> Self {
        let rate_f = rate as f32;
        let mut window = [0f32; N];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos();
        }
        let sum_sq: f32 = window.iter().map(|w| w * w).sum();
        let hi = F_HI.min(0.475 * rate_f);
        let (mlo, mhi) = (mel_of(F_LO), mel_of(hi));
        let edge: Vec<f32> = (0..BANDS + 2).map(|i| hz_of(mlo + (mhi - mlo) * i as f32 / (BANDS + 1) as f32)).collect();
        let bin_hz = rate_f / N as f32;
        let mut bin = [0u8; BANDS];
        let mut from = [0u16; BANDS + 1];
        let mut weights = Vec::new();
        let mut body_bands = 0;
        for b in 0..BANDS {
            let (l, c, r) = (edge[b], edge[b + 1], edge[b + 2]);
            let mut ws = Vec::new();
            let mut first = None;
            for k in 0..BINS {
                let f = k as f32 * bin_hz;
                let w = if f > l && f <= c { (f - l) / (c - l) } else if f > c && f < r { (r - f) / (r - c) } else { 0.0 };
                if w > 0.0 {
                    first.get_or_insert(k);
                    ws.push(w);
                } else if first.is_some() {
                    break;
                }
            }
            // A band narrower than a bin takes the nearest bin whole.
            if ws.is_empty() {
                first = Some(((c / bin_hz).round() as usize).min(BINS - 1));
                ws.push(1.0);
            }
            bin[b] = first.unwrap() as u8;
            from[b] = weights.len() as u16;
            weights.extend(ws);
            if c < BODY_HZ {
                body_bands = b + 1;
            }
        }
        from[BANDS] = weights.len() as u16;
        Self {
            fft: Fft::new(),
            window,
            frame: [0.0; N],
            hop: (rate / (1000 / FRAME_MS)) as usize,
            pending: 0,
            norm: N as f32 * sum_sq / 4.0,
            bin,
            from,
            weights,
            body_bands,
        }
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Feeds samples; calls `emit` once per 5 ms with the frame ending at that sample.
    pub fn push(&mut self, mut x: &[f32], mut emit: impl FnMut(&Frame)) {
        while !x.is_empty() {
            let take = (self.hop - self.pending).min(x.len());
            let at = N - self.hop + self.pending;
            self.frame[at..at + take].copy_from_slice(&x[..take]);
            self.pending += take;
            x = &x[take..];
            if self.pending == self.hop {
                let f = self.compute();
                emit(&f);
                self.frame.copy_within(self.hop.., 0);
                self.pending = 0;
            }
        }
    }

    fn compute(&self) -> Frame {
        let mut w = [0f32; N];
        for ((o, x), win) in w.iter_mut().zip(&self.frame).zip(&self.window) {
            *o = x * win;
        }
        let mut p = [0f32; BINS];
        self.fft.power(&w, &mut p);
        let inv = 1.0 / self.norm;
        let mut out = Frame::default();
        let total: f32 = p.iter().sum::<f32>() * inv;
        out.level = 10.0 * (total + 1e-12).log10();
        let mut body = 0.0;
        for b in 0..BANDS {
            let (s, e) = (self.from[b] as usize, self.from[b + 1] as usize);
            let k0 = self.bin[b] as usize;
            let e_b: f32 = self.weights[s..e].iter().enumerate().map(|(i, w)| w * p[k0 + i]).sum::<f32>() * inv;
            if b < self.body_bands {
                body += e_b;
            }
            out.mel[b] = 10.0 * (e_b + 1e-12).log10();
        }
        out.body = body.sqrt();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_lands_in_its_band() {
        let mut m = Mel::new(16_000);
        let x: Vec<f32> = (0..1600).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin()).collect();
        let mut last = Frame::default();
        m.push(&x, |f| last = *f);
        assert!(last.level.abs() < 1.0, "level {}", last.level);
        let peak = (0..BANDS).max_by(|&a, &b| last.mel[a].total_cmp(&last.mel[b])).unwrap();
        let c = |b: usize| hz_of(mel_of(F_LO) + (mel_of(F_HI) - mel_of(F_LO)) * (b + 1) as f32 / (BANDS + 1) as f32);
        assert!((c(peak) - 1000.0).abs() < 150.0, "peak band {peak} at {}", c(peak));
    }
}
