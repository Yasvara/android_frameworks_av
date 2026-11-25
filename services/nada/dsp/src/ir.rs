//! Impulse responses: upload, checks and rate conversion. Control thread only.

use std::collections::HashMap;
use std::sync::Arc;

/// Longest upload, in frames at the file's own rate.
pub const MAX_FRAMES: usize = 1 << 19;

pub struct Ir {
    pub rate: u32,
    pub gen: u64,
    /// One vector per channel (1, 2 or 4).
    pub ch: Vec<Vec<f32>>,
}

struct Staging {
    id: String,
    rate: u32,
    frames: usize,
    crc: u32,
    data: Vec<f32>,
    filled: usize,
}

#[derive(Default)]
pub struct Store {
    pub irs: HashMap<String, Arc<Ir>>,
    staging: Option<Staging>,
    gen: u64,
}

fn name_ok(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.')
}

impl Store {
    /// `id,rate,channels,frames,crc32hex`
    pub fn begin(&mut self, v: &str) -> Result<(), String> {
        let f: Vec<&str> = v.split(',').collect();
        if f.len() != 5 || !name_ok(f[0]) {
            return Err("bad header".into());
        }
        let rate: u32 = f[1].parse().map_err(|_| "bad rate")?;
        let channels: usize = f[2].parse().map_err(|_| "bad channels")?;
        let frames: usize = f[3].parse().map_err(|_| "bad frames")?;
        let crc = u32::from_str_radix(f[4], 16).map_err(|_| "bad crc")?;
        if !(8000..=384_000).contains(&rate) || ![1, 2, 4].contains(&channels) || frames == 0 || frames > MAX_FRAMES {
            return Err("out of range".into());
        }
        self.staging = Some(Staging { id: f[0].into(), rate, frames, crc, data: vec![0.0; channels * frames], filled: 0 });
        Ok(())
    }

    /// `id,sample_offset,base64` of little-endian f32, channel after channel.
    pub fn data(&mut self, v: &str) -> Result<(), String> {
        let mut f = v.splitn(3, ',');
        let (id, off, b) = (f.next().unwrap_or(""), f.next().unwrap_or(""), f.next().unwrap_or(""));
        let st = self.staging.as_mut().filter(|s| s.id == id).ok_or("no upload")?;
        let off: usize = off.parse().map_err(|_| "bad offset")?;
        let raw = b64(b).ok_or("bad base64")?;
        if raw.len() % 4 != 0 || off != st.filled || off + raw.len() / 4 > st.data.len() {
            return Err("bad chunk".into());
        }
        for (d, s) in st.data[off..].iter_mut().zip(raw.chunks_exact(4)) {
            *d = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
        st.filled += raw.len() / 4;
        Ok(())
    }

    /// Checks and stores the upload. On any failure the previous response under that name stays.
    pub fn commit(&mut self, id: &str) -> Result<(), String> {
        let st = self.staging.take().filter(|s| s.id == id).ok_or("no upload")?;
        if st.filled != st.data.len() {
            return Err("incomplete".into());
        }
        let mut crc = Crc::new();
        for v in &st.data {
            if !v.is_finite() {
                return Err("non-finite sample".into());
            }
            crc.feed(&v.to_le_bytes());
        }
        if crc.done() != st.crc {
            return Err("crc mismatch".into());
        }
        self.gen += 1;
        let ch = st.data.chunks_exact(st.frames).map(<[f32]>::to_vec).collect();
        self.irs.insert(st.id, Arc::new(Ir { rate: st.rate, gen: self.gen, ch }));
        Ok(())
    }
}

pub struct Crc(u32);

impl Crc {
    pub fn new() -> Self {
        Crc(!0)
    }
    pub fn feed(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= b as u32;
            for _ in 0..8 {
                self.0 = (self.0 >> 1) ^ (0xEDB8_8320 & (!(self.0 & 1)).wrapping_add(1));
            }
        }
    }
    pub fn done(&self) -> u32 {
        !self.0
    }
}

impl Default for Crc {
    fn default() -> Self {
        Crc::new()
    }
}

pub fn b64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

pub(crate) fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Converts an impulse response to another rate with a Kaiser-windowed sinc (32 zero
/// crossings each side). The result is scaled so the filter keeps its gain.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let fc = ratio.min(1.0) * 0.97;
    let half = 32.0 / fc;
    let beta = 12.0;
    let norm = bessel_i0(beta);
    let n_out = (x.len() as f64 * ratio).ceil() as usize;
    let scale = fc / ratio;
    (0..n_out)
        .map(|n| {
            let t = n as f64 / ratio;
            let lo = (t - half).ceil().max(0.0) as usize;
            let hi = ((t + half).floor() as usize).min(x.len() - 1);
            let mut acc = 0f64;
            for (k, xv) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let d = t - k as f64;
                let a = std::f64::consts::PI * fc * d;
                let sinc = if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                let w = bessel_i0(beta * (1.0 - (d / half).powi(2)).max(0.0).sqrt()) / norm;
                acc += *xv as f64 * sinc * w;
            }
            (acc * scale) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_and_base64_vectors() {
        let mut c = Crc::new();
        c.feed(b"123456789");
        assert_eq!(c.done(), 0xCBF4_3926);
        assert_eq!(b64("AAAAPw==").unwrap(), 0.5f32.to_le_bytes());
        assert!(b64("a b").is_none());
    }

    #[test]
    fn upload_checks() {
        let mut s = Store::default();
        let data = [0.5f32, -0.25, 0.125, 1.0];
        let mut c = Crc::new();
        data.iter().for_each(|v| c.feed(&v.to_le_bytes()));
        let hdr = format!("room,48000,2,2,{:08x}", c.done());
        s.begin(&hdr).unwrap();
        // 0.5, -0.25 then 0.125, 1.0
        s.data("room,0,AAAAPwAAgL4=").unwrap();
        assert!(s.data("room,3,AAAAPgAAgD8=").is_err());
        s.data("room,2,AAAAPgAAgD8=").unwrap();
        s.commit("room").unwrap();
        let ir = &s.irs["room"];
        assert_eq!(ir.ch, vec![vec![0.5, -0.25], vec![0.125, 1.0]]);
        // Wrong checksum: rejected, the stored one stays.
        s.begin("room,48000,1,1,00000000").unwrap();
        s.data("room,0,AAAAPw==").unwrap();
        assert!(s.commit("room").is_err());
        assert_eq!(s.irs["room"].ch.len(), 2);
        assert!(s.begin("../x,48000,1,1,0").is_err());
        assert!(s.begin("x,48000,3,1,0").is_err());
    }

    #[test]
    fn resample_keeps_gain_and_delay() {
        // A delayed, slightly smoothed impulse: DC gain 1.
        let mut x = vec![0f32; 400];
        x[100] = 0.25;
        x[101] = 0.5;
        x[102] = 0.25;
        for (from, to) in [(44100u32, 48000u32), (48000, 44100), (48000, 96000), (96000, 48000)] {
            let y = resample(&x, from, to);
            let sum: f64 = y.iter().map(|v| *v as f64).sum();
            assert!((sum - 1.0).abs() < 2e-3, "{from}->{to}: dc {sum}");
            let peak = y.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0;
            let want = 101.0 * to as f64 / from as f64;
            assert!((peak as f64 - want).abs() <= 1.0, "{from}->{to}: peak at {peak}, want {want}");
        }
    }
}
