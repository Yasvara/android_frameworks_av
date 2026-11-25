//! Radix-2 complex FFT on split real/imaginary arrays. Tables are built once; transforms
//! allocate nothing.

/// Sample type of the convolver's frequency domain.
pub type R = f64;

pub struct Fft {
    n: usize,
    rev: Vec<u32>,
    // Twiddles per stage, concatenated: the stage with half-length h starts at h - 1.
    wr: Vec<R>,
    wi: Vec<R>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let mut wr = Vec::with_capacity(n);
        let mut wi = Vec::with_capacity(n);
        let mut h = 1;
        while h < n {
            for j in 0..h {
                let a = -std::f64::consts::PI * j as f64 / h as f64;
                wr.push(a.cos() as R);
                wi.push(a.sin() as R);
            }
            h *= 2;
        }
        Fft { n, rev, wr, wi }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn forward(&self, re: &mut [R], im: &mut [R]) {
        let n = self.n;
        let (re, im) = (&mut re[..n], &mut im[..n]);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut h = 1;
        while h < n {
            let wr = &self.wr[h - 1..2 * h - 1];
            let wi = &self.wi[h - 1..2 * h - 1];
            for (cr, ci) in re.chunks_exact_mut(2 * h).zip(im.chunks_exact_mut(2 * h)) {
                let (ar, br) = cr.split_at_mut(h);
                let (ai, bi) = ci.split_at_mut(h);
                for j in 0..h {
                    let tr = br[j] * wr[j] - bi[j] * wi[j];
                    let ti = br[j] * wi[j] + bi[j] * wr[j];
                    br[j] = ar[j] - tr;
                    bi[j] = ai[j] - ti;
                    ar[j] += tr;
                    ai[j] += ti;
                }
            }
            h *= 2;
        }
    }

    /// Inverse, scaled by 1/n.
    pub fn inverse(&self, re: &mut [R], im: &mut [R]) {
        self.forward(im, re);
        let s = 1.0 / self.n as R;
        for v in re[..self.n].iter_mut() {
            *v *= s;
        }
        for v in im[..self.n].iter_mut() {
            *v *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_bins() {
        for n in [2usize, 8, 128, 2048] {
            let f = Fft::new(n);
            let src: Vec<R> = (0..n).map(|i| ((i * 7 + 3) % 13) as R / 13.0 - 0.5).collect();
            let mut re = src.clone();
            let mut im = vec![0.0; n];
            f.forward(&mut re, &mut im);
            // DC bin equals the sum.
            let sum: f64 = src.iter().map(|&v| v as f64).sum();
            assert!((re[0] as f64 - sum).abs() < 1e-9);
            f.inverse(&mut re, &mut im);
            for i in 0..n {
                assert!((re[i] - src[i]).abs() < 1e-12 && im[i].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn single_tone() {
        let n = 128;
        let f = Fft::new(n);
        let mut re: Vec<R> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / n as f64).cos() as R).collect();
        let mut im = vec![0.0; n];
        f.forward(&mut re, &mut im);
        assert!((re[5] - 64.0).abs() < 1e-3 && (re[n - 5] - 64.0).abs() < 1e-3);
        assert!(re[6].abs() < 1e-3 && im[5].abs() < 1e-3);
    }
}
