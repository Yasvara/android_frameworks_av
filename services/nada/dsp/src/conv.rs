//! Convolver with no added delay.
//!
//! The impulse response is split three ways. Taps 0..64 run as a direct FIR, so the first
//! output sample depends only on the current input sample. Taps 64..2048 run as 64-sample
//! frequency-domain partitions whose result is due exactly when their input block ends.
//! Taps from 2048 run as 1024-sample partitions; their result is due one full block later,
//! and that slack is used to spread the work over the sixteen 64-sample rounds in between,
//! so the cost per call stays flat for any call size.

use crate::fft::{Fft, R};
use crate::{Frame, Stage};
use std::sync::Arc;

pub const P1: usize = 64;
const N1: usize = 2 * P1;
const B1: usize = P1 + 1;
pub const P2: usize = 1024;
const N2: usize = 2 * P2;
const B2: usize = P2 + 1;
const ROUNDS: usize = P2 / P1;
const L1_MAX: usize = N2 / P1 - 1;
const MAC_SLICES: usize = ROUNDS - 2;
/// Longest impulse response, in taps at the stream rate.
pub const MAX_TAPS: usize = 1 << 17;

pub struct Path {
    pub i: usize,
    pub o: usize,
    head: [R; P1],
    h1: Vec<R>,
    h2: Vec<R>,
}

/// An impulse response prepared for one sample rate. Immutable and shared.
pub struct Kernel {
    pub paths: Vec<Path>,
    pub k1: usize,
    pub k2: usize,
    pub taps: usize,
    /// Largest gain any frequency can get on an output channel.
    pub peak_gain: f64,
    f1: Fft,
    f2: Fft,
}

impl Kernel {
    /// One channel: the same response on left and right. Two: left, right. Four: true stereo
    /// in the order L→L, L→R, R→L, R→R.
    pub fn new(ir: &[Vec<f32>]) -> Option<Kernel> {
        let map: &[(usize, usize, usize)] = match ir.len() {
            1 => &[(0, 0, 0), (1, 1, 0)],
            2 => &[(0, 0, 0), (1, 1, 1)],
            4 => &[(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)],
            _ => return None,
        };
        let taps = ir.iter().map(|c| c.len()).max()?.min(MAX_TAPS);
        if taps == 0 || ir.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let k1 = if taps > P1 { (taps - P1).div_ceil(P1).min(L1_MAX) } else { 0 };
        let k2 = if taps > N2 { (taps - N2).div_ceil(P2) } else { 0 };
        let (f1, f2) = (Fft::new(N1), Fft::new(N2));
        let seg = |h: &[f32], from: usize, len: usize, f: &Fft, out: &mut Vec<R>| {
            let n = f.len();
            let mut re = vec![0.0 as R; n];
            let mut im = vec![0.0 as R; n];
            for (d, s) in re.iter_mut().zip(h.iter().skip(from).take(len)) {
                *d = *s as R;
            }
            f.forward(&mut re, &mut im);
            out.extend_from_slice(&re[..n / 2 + 1]);
            out.extend_from_slice(&im[..n / 2 + 1]);
        };
        let mut paths = Vec::new();
        for &(i, o, c) in map {
            let h = &ir[c][..ir[c].len().min(taps)];
            let mut head = [0.0 as R; P1];
            for (d, s) in head.iter_mut().zip(h.iter()) {
                *d = *s as R;
            }
            let (mut h1, mut h2) = (Vec::new(), Vec::new());
            for k in 0..k1 {
                seg(h, P1 * (k + 1), P1, &f1, &mut h1);
            }
            for m in 0..k2 {
                seg(h, N2 + P2 * m, P2, &f2, &mut h2);
            }
            paths.push(Path { i, o, head, h1, h2 });
        }
        Some(Kernel { peak_gain: peak_gain(ir, map, taps), paths, k1, k2, taps, f1, f2 })
    }
}

fn peak_gain(ir: &[Vec<f32>], map: &[(usize, usize, usize)], taps: usize) -> f64 {
    let n = (2 * taps).next_power_of_two().max(1024);
    let f = Fft::new(n);
    let mut sum = [vec![0f64; n / 2 + 1], vec![0f64; n / 2 + 1]];
    for &(_, o, c) in map {
        let mut re = vec![0.0 as R; n];
        let mut im = vec![0.0 as R; n];
        for (d, s) in re.iter_mut().zip(ir[c].iter().take(taps)) {
            *d = *s as R;
        }
        f.forward(&mut re, &mut im);
        for b in 0..=n / 2 {
            sum[o][b] += ((re[b] as f64).powi(2) + (im[b] as f64).powi(2)).sqrt();
        }
    }
    sum.iter().flatten().fold(0f64, |a, &b| a.max(b))
}

pub struct Convolver {
    k: Arc<Kernel>,
    wet: f64,
    dry: f64,
    pos1: usize,
    cur1: [[R; P1]; 2],
    prev1: [[R; P1]; 2],
    x1: Vec<R>,
    x1_head: usize,
    out1: [[R; P1]; 2],
    ov: [[R; N1]; 2],
    pos2: usize,
    cur2: Vec<R>,
    prev2: Vec<R>,
    x2: Vec<R>,
    x2_head: usize,
    out2: Vec<R>,
    next2: Vec<R>,
    acc1: [R; 2 * B1],
    acc: Vec<R>,
    tre: Vec<R>,
    tim: Vec<R>,
}

impl Convolver {
    pub fn new(k: Arc<Kernel>, wet: f64, dry: f64) -> Self {
        let (k1, k2) = (k.k1, k.k2);
        Convolver {
            wet,
            dry,
            pos1: 0,
            cur1: [[0.0; P1]; 2],
            prev1: [[0.0; P1]; 2],
            x1: vec![0.0; 2 * k1 * 2 * B1],
            x1_head: 0,
            out1: [[0.0; P1]; 2],
            ov: [[0.0; N1]; 2],
            pos2: 0,
            cur2: vec![0.0; if k2 > 0 { 2 * P2 } else { 0 }],
            prev2: vec![0.0; if k2 > 0 { 2 * P2 } else { 0 }],
            x2: vec![0.0; 2 * k2 * 2 * B2],
            x2_head: 0,
            out2: vec![0.0; if k2 > 0 { 2 * P2 } else { 0 }],
            next2: vec![0.0; if k2 > 0 { 2 * P2 } else { 0 }],
            acc1: [0.0; 2 * B1],
            acc: vec![0.0; 2 * 2 * B2],
            tre: vec![0.0; N2],
            tim: vec![0.0; N2],
            k,
        }
    }

    fn chunk(&mut self, x: &mut [Frame]) {
        let c = x.len();
        let p = self.pos1;
        let k2 = self.k.k2;
        let base2 = self.pos2 * P1 + p;
        for (s, f) in x.iter().enumerate() {
            for ch in 0..2 {
                let v = f[ch] as R;
                self.cur1[ch][p + s] = v;
                if k2 > 0 {
                    self.cur2[ch * P2 + base2 + s] = v;
                }
            }
        }
        for path in &self.k.paths {
            let ov = &mut self.ov[path.o];
            let src = &self.cur1[path.i][p..p + c];
            for (s, &v) in src.iter().enumerate() {
                for (d, h) in ov[s..s + P1].iter_mut().zip(path.head.iter()) {
                    *d += v * h;
                }
            }
        }
        for (s, f) in x.iter_mut().enumerate() {
            for o in 0..2 {
                let mut y = self.ov[o][s] + self.out1[o][p + s];
                if k2 > 0 {
                    y += self.out2[o * P2 + base2 + s];
                }
                f[o] = self.dry * f[o] + self.wet * y as f64;
            }
        }
        for o in 0..2 {
            self.ov[o].copy_within(c.., 0);
            self.ov[o][N1 - c..].fill(0.0);
        }
        self.pos1 += c;
        if self.pos1 == P1 {
            self.pos1 = 0;
            self.round1();
            if k2 > 0 {
                self.round2();
            }
        }
    }

    // A 64-sample block has ended: produce the short partitions' output for the next block.
    fn round1(&mut self) {
        let k = &*self.k;
        let k1 = k.k1;
        if k1 == 0 {
            return;
        }
        let (tre, tim) = (&mut self.tre[..N1], &mut self.tim[..N1]);
        self.x1_head = (self.x1_head + 1) % k1;
        for ch in 0..2 {
            tre[..P1].copy_from_slice(&self.prev1[ch]);
            tre[P1..].copy_from_slice(&self.cur1[ch]);
            tim.fill(0.0);
            k.f1.forward(tre, tim);
            let at = (ch * k1 + self.x1_head) * 2 * B1;
            self.x1[at..at + B1].copy_from_slice(&tre[..B1]);
            self.x1[at + B1..at + 2 * B1].copy_from_slice(&tim[..B1]);
        }
        self.prev1 = self.cur1;
        for o in 0..2 {
            let (ar, ai) = self.acc1.split_at_mut(B1);
            ar.fill(0.0);
            ai.fill(0.0);
            for path in k.paths.iter().filter(|p| p.o == o) {
                for s in 0..k1 {
                    let at = (path.i * k1 + (self.x1_head + k1 - s) % k1) * 2 * B1;
                    mac(ar, ai, &self.x1[at..at + 2 * B1], &path.h1[s * 2 * B1..(s + 1) * 2 * B1], B1);
                }
            }
            unfold(tre, tim, ar, ai, P1);
            k.f1.inverse(tre, tim);
            self.out1[o].copy_from_slice(&tre[P1..]);
        }
    }

    // Long partitions, one slice of work per 64-sample round.
    fn round2(&mut self) {
        let k = &*self.k;
        let k2 = k.k2;
        self.pos2 += 1;
        if self.pos2 == ROUNDS {
            self.pos2 = 0;
            std::mem::swap(&mut self.out2, &mut self.next2);
            self.x2_head = (self.x2_head + 1) % k2;
            for ch in 0..2 {
                self.tre[..P2].copy_from_slice(&self.prev2[ch * P2..(ch + 1) * P2]);
                self.tre[P2..].copy_from_slice(&self.cur2[ch * P2..(ch + 1) * P2]);
                self.tim.fill(0.0);
                k.f2.forward(&mut self.tre, &mut self.tim);
                let at = (ch * k2 + self.x2_head) * 2 * B2;
                self.x2[at..at + B2].copy_from_slice(&self.tre[..B2]);
                self.x2[at + B2..at + 2 * B2].copy_from_slice(&self.tim[..B2]);
            }
            std::mem::swap(&mut self.prev2, &mut self.cur2);
            self.acc.fill(0.0);
        } else if self.pos2 <= MAC_SLICES {
            let per = k2.div_ceil(MAC_SLICES);
            let lo = (self.pos2 - 1) * per;
            let hi = (self.pos2 * per).min(k2);
            for o in 0..2 {
                let (ar, ai) = self.acc[o * 2 * B2..(o + 1) * 2 * B2].split_at_mut(B2);
                for path in k.paths.iter().filter(|p| p.o == o) {
                    for m in lo..hi {
                        let at = (path.i * k2 + (self.x2_head + k2 - m) % k2) * 2 * B2;
                        mac(ar, ai, &self.x2[at..at + 2 * B2], &path.h2[m * 2 * B2..(m + 1) * 2 * B2], B2);
                    }
                }
            }
        } else {
            for o in 0..2 {
                let (ar, ai) = self.acc[o * 2 * B2..(o + 1) * 2 * B2].split_at(B2);
                unfold(&mut self.tre, &mut self.tim, ar, ai, P2);
                k.f2.inverse(&mut self.tre, &mut self.tim);
                self.next2[o * P2..(o + 1) * P2].copy_from_slice(&self.tre[P2..]);
            }
        }
    }
}

// acc += x * h over `bins` complex bins; x and h hold real parts then imaginary parts.
#[inline]
fn mac(ar: &mut [R], ai: &mut [R], x: &[R], h: &[R], bins: usize) {
    let (xr, xi) = x.split_at(bins);
    let (hr, hi) = h.split_at(bins);
    let (ar, ai) = (&mut ar[..bins], &mut ai[..bins]);
    let (xr, xi, hr, hi) = (&xr[..bins], &xi[..bins], &hr[..bins], &hi[..bins]);
    for b in 0..bins {
        ar[b] += xr[b] * hr[b] - xi[b] * hi[b];
        ai[b] += xr[b] * hi[b] + xi[b] * hr[b];
    }
}

// Half spectrum (p + 1 bins) to the full conjugate-symmetric one of length 2p.
#[inline]
fn unfold(re: &mut [R], im: &mut [R], ar: &[R], ai: &[R], p: usize) {
    re[..=p].copy_from_slice(&ar[..=p]);
    im[..=p].copy_from_slice(&ai[..=p]);
    for b in 1..p {
        re[2 * p - b] = ar[b];
        im[2 * p - b] = -ai[b];
    }
}

impl Stage for Convolver {
    fn process(&mut self, x: &mut [Frame]) {
        let mut i = 0;
        while i < x.len() {
            let c = (P1 - self.pos1).min(x.len() - i);
            self.chunk(&mut x[i..i + c]);
            i += c;
        }
    }

    fn reset(&mut self) {
        self.cur1 = [[0.0; P1]; 2];
        self.prev1 = [[0.0; P1]; 2];
        self.out1 = [[0.0; P1]; 2];
        self.ov = [[0.0; N1]; 2];
        for v in [&mut self.x1, &mut self.cur2, &mut self.prev2, &mut self.x2, &mut self.out2, &mut self.next2, &mut self.acc] {
            v.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn rng(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    fn direct(x: &[f64], h: &[f32]) -> Vec<f64> {
        let mut y = vec![0f64; x.len()];
        for (n, yn) in y.iter_mut().enumerate() {
            let mut a = 0f64;
            for (k, hk) in h.iter().enumerate().take(n + 1) {
                a += *hk as f64 * x[n - k];
            }
            *yn = a;
        }
        y
    }

    fn run(taps: usize, chunk: usize, len: usize) -> f64 {
        let mut seed = taps as u64 * 977 + chunk as u64;
        // Decaying noise, like a room response, normalised so the output stays near full scale.
        let h: Vec<f32> = (0..taps).map(|i| (rng(&mut seed) * (-(i as f64) / (taps as f64 / 4.0 + 1.0)).exp()) as f32).collect();
        let norm = h.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
        let h: Vec<f32> = h.iter().map(|v| (*v as f64 / norm) as f32).collect();
        let x: Vec<f64> = (0..len).map(|_| (rng(&mut seed) * 0.5) as f32 as f64).collect();
        let want = direct(&x, &h);
        let k = Arc::new(Kernel::new(&[h.clone()]).unwrap());
        let mut c = Convolver::new(k, 1.0, 0.0);
        let mut buf: Vec<Frame> = x.iter().map(|&v| [v, -v]).collect();
        for part in buf.chunks_mut(chunk) {
            c.process(part);
        }
        let mut err = 0f64;
        for (n, f) in buf.iter().enumerate() {
            err = err.max((f[0] - want[n]).abs()).max((f[1] + want[n]).abs());
        }
        20.0 * err.max(1e-20).log10()
    }

    #[test]
    fn matches_direct_convolution() {
        for (taps, chunk, len) in [
            (1, 144, 1000),
            (64, 7, 1500),
            (65, 144, 2000),
            (700, 480, 6000),
            (2048, 1, 5000),
            (2049, 144, 9000),
            (5000, 960, 14000),
            (20000, 144, 30000),
            (9000, 1920, 26000),
        ] {
            let e = run(taps, chunk, len);
            assert!(e < -200.0, "taps {taps} chunk {chunk}: error {e:.1} dBFS");
        }
    }

    #[test]
    fn first_sample_is_immediate() {
        // An impulse at input sample n appears at output sample n: no delay at any call size.
        for chunk in [1usize, 96, 144, 480] {
            let mut h = vec![0f32; 6000];
            h[0] = 0.5;
            h[5999] = 0.25;
            let k = Arc::new(Kernel::new(&[h]).unwrap());
            let mut c = Convolver::new(k, 1.0, 0.0);
            let mut buf = vec![[0.0; 2]; 8000];
            buf[37] = [1.0, 1.0];
            for part in buf.chunks_mut(chunk) {
                c.process(part);
            }
            assert!((buf[37][0] - 0.5).abs() < 1e-9, "chunk {chunk}: {}", buf[37][0]);
            assert!((buf[37 + 5999][0] - 0.25).abs() < 1e-9, "chunk {chunk}: {}", buf[37 + 5999][0]);
            assert!(buf[36][0].abs() < 1e-12 && buf[38][0].abs() < 1e-9);
        }
    }

    #[test]
    fn true_stereo_routes_channels() {
        let z = vec![0f32; 3000];
        let mut lr = z.clone();
        lr[2500] = 1.0;
        let k = Arc::new(Kernel::new(&[z.clone(), lr, z.clone(), z]).unwrap());
        let mut c = Convolver::new(k, 1.0, 0.0);
        let mut buf = vec![[0.0; 2]; 4000];
        buf[10] = [1.0, 0.0];
        for part in buf.chunks_mut(144) {
            c.process(part);
        }
        assert!((buf[2510][1] - 1.0).abs() < 1e-9 && buf[2510][0].abs() < 1e-9);
    }
}
