//! Front end of the learned tagger: the exact features the student was trained on,
//! from the tap's 48 kHz audio, for the network that runs on HTP.
//!
//! 48 kHz → 32 kHz → 16 kHz with torchaudio's sinc kernels; 16 kHz log-mel (pre-emphasis, 512-point
//! STFT of a centred 400-sample window, hop 10 ms, Kaldi mel bank, `(ln + 4.5) / 5`); 32 kHz high
//! band (512-point STFT, hop 10 ms, 16 log bands over 8 kHz, `(log10 + 6) / 3`). Both STFTs centre
//! with reflect padding, as `torch.stft` does, over the whole window on every call. Constants come
//! from `data/tagger_front.bin` (lab/haptics/tagger/export_front.py).

const FRONT: &[u8] = include_bytes!("../data/tagger_front.bin");

pub const N_FFT: usize = 512;
pub const BINS: usize = N_FFT / 2 + 1;
pub const MELS: usize = 128;
pub const BANDS: usize = 16;
const HOP16: usize = 160;
const HOP32: usize = 320;

struct Kernel {
    taps: Vec<f32>, // new × len
    len: usize,
    width: usize,
    orig: usize,
    new: usize,
}

impl Kernel {
    /// torchaudio.functional.resample: zero pad (width, width + orig), stride-orig correlation per
    /// output phase, truncated to ceil(new · n / orig).
    fn apply(&self, x: &[f32], out: &mut Vec<f32>) {
        let n_out = (self.new * x.len()).div_ceil(self.orig);
        out.clear();
        out.reserve(n_out);
        let at = |i: isize| -> f32 {
            if i < 0 || i as usize >= x.len() {
                0.0
            } else {
                x[i as usize]
            }
        };
        'outer: for j in 0.. {
            let base = (j * self.orig) as isize - self.width as isize;
            for p in 0..self.new {
                if out.len() == n_out {
                    break 'outer;
                }
                let k = &self.taps[p * self.len..(p + 1) * self.len];
                out.push(k.iter().enumerate().map(|(t, w)| w * at(base + t as isize)).sum());
            }
        }
    }
}

struct Fft {
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<u16>,
}

impl Fft {
    fn new() -> Self {
        let bits = N_FFT.trailing_zeros();
        Fft {
            cos: (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).cos() as f32).collect(),
            sin: (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).sin() as f32).collect(),
            rev: (0..N_FFT).map(|i| ((i as u32).reverse_bits() >> (32 - bits)) as u16).collect(),
        }
    }

    fn power(&self, x: &[f32; N_FFT], out: &mut [f32; BINS]) {
        let mut re = [0f32; N_FFT];
        let mut im = [0f32; N_FFT];
        for i in 0..N_FFT {
            re[i] = x[self.rev[i] as usize];
        }
        let mut size = 2;
        while size <= N_FFT {
            let half = size / 2;
            let step = N_FFT / size;
            for start in (0..N_FFT).step_by(size) {
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
        for (k, o) in out.iter_mut().enumerate() {
            *o = re[k] * re[k] + im[k] * im[k];
        }
    }
}

pub struct Front {
    k48: Kernel,
    k32: Kernel,
    win16: Vec<f32>,
    win32: Vec<f32>,
    mel: Vec<f32>, // MELS × BINS
    hi: Vec<f32>,  // BANDS × BINS
    fft: Fft,
    x32: Vec<f32>,
    x16: Vec<f32>,
}

/// Named f32 arrays in the "HTF1"/"HTG1" layout: name[8], ndim u32, dims u32 × ndim, data.
pub fn arrays(mut b: &[u8]) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    while b.len() >= 12 {
        let name = String::from_utf8_lossy(&b[..8]).trim_end_matches('\0').to_string();
        let nd = u32_at(b, 8);
        let dims: Vec<usize> = (0..nd).map(|i| u32_at(b, 12 + 4 * i)).collect();
        let n: usize = dims.iter().product();
        let at = 12 + 4 * nd;
        let data = b[at..at + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.push((name, dims, data));
        b = &b[at + 4 * n..];
    }
    out
}

impl Default for Front {
    fn default() -> Self {
        Self::new()
    }
}

impl Front {
    pub fn new() -> Self {
        assert_eq!(&FRONT[..4], b"HTF1");
        let h: Vec<usize> = FRONT[4..28].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        let a = arrays(&FRONT[28..]);
        let get = |n: &str| a.iter().find(|x| x.0 == n).unwrap_or_else(|| panic!("tagger_front.bin: no {n}"));
        let kernel = |n: &str, width, orig, new| {
            let (_, d, v) = get(n);
            Kernel { taps: v.clone(), len: d[1], width, orig, new }
        };
        Front {
            k48: kernel("k48to32", h[0], h[1], h[2]),
            k32: kernel("k32to16", h[3], h[4], h[5]),
            win16: get("win16").2.clone(),
            win32: get("win32").2.clone(),
            mel: get("mel").2.clone(),
            hi: get("hi").2.clone(),
            fft: Fft::new(),
            x32: Vec::new(),
            x16: Vec::new(),
        }
    }

    /// Centred, reflect-padded STFT power frames of `x` (as torch.stft, center=True).
    fn stft(&self, x: &[f32], hop: usize, win: &[f32], mut each: impl FnMut(usize, &[f32; BINS])) -> usize {
        let pad = N_FFT / 2;
        let frames = 1 + x.len() / hop;
        let reflect = |i: isize| -> f32 {
            let n = x.len() as isize;
            let j = if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i };
            x[j.clamp(0, n - 1) as usize]
        };
        let mut buf = [0f32; N_FFT];
        let mut p = [0f32; BINS];
        for f in 0..frames {
            let s = (f * hop) as isize - pad as isize;
            for (k, b) in buf.iter_mut().enumerate() {
                *b = reflect(s + k as isize) * win[k];
            }
            self.fft.power(&buf, &mut p);
            each(f, &p);
        }
        frames
    }

    /// 32 kHz audio → (mel: MELS × T, bands: BANDS × T), row-major, T = frames returned.
    pub fn features32(&mut self, x32: &[f32], mel: &mut Vec<f32>, bands: &mut Vec<f32>) -> usize {
        let mut x16 = std::mem::take(&mut self.x16);
        self.k32.apply(x32, &mut x16);
        let pre: Vec<f32> = x16.windows(2).map(|w| w[1] - 0.97 * w[0]).collect();
        let t16 = 1 + pre.len() / HOP16;
        let t32 = 1 + x32.len() / HOP32;
        let t = t16.min(t32);
        mel.clear();
        mel.resize(MELS * t, 0.0);
        bands.clear();
        bands.resize(BANDS * t, 0.0);
        self.stft(&pre, HOP16, &self.win16, |f, p| {
            if f < t {
                for m in 0..MELS {
                    let row = &self.mel[m * BINS..(m + 1) * BINS];
                    let e: f32 = row.iter().zip(p).map(|(w, v)| w * v).sum();
                    mel[m * t + f] = (e.max(1e-7).ln() + 4.5) / 5.0;
                }
            }
        });
        self.stft(x32, HOP32, &self.win32, |f, p| {
            if f < t {
                for k in 0..BANDS {
                    let row = &self.hi[k * BINS..(k + 1) * BINS];
                    let e: f32 = row.iter().zip(p).map(|(w, v)| w * v).sum();
                    bands[k * t + f] = ((e + 1e-8).log10() + 6.0) / 3.0;
                }
            }
        });
        self.x16 = x16;
        t
    }

    /// The tap's 48 kHz audio → features (see [`Front::features32`]).
    pub fn features48(&mut self, x48: &[f32], mel: &mut Vec<f32>, bands: &mut Vec<f32>) -> usize {
        let mut x32 = std::mem::take(&mut self.x32);
        self.k48.apply(x48, &mut x32);
        let t = self.features32(&x32, mel, bands);
        self.x32 = x32;
        t
    }
}

/// How often the tagger labels the newest audio without a request.
pub const PERIOD_MS: u32 = 40;
/// How long a request may wait for its window's audio to arrive.
const REQUEST_WAIT_MS: u64 = 25;

/// The tagger's side of a session: waits for the next window (a request, or [`PERIOD_MS`] of new
/// audio) and turns it into the network's features.
pub struct Runner {
    session: shm::Session,
    front: Front,
    win: Vec<f32>,
    seen: u32,
    last: u64,
}

impl Runner {
    /// `window48`: the network's window in 48 kHz samples (its input frames × 480).
    pub fn new(session: shm::Session, window48: usize) -> Self {
        Runner { session, front: Front::new(), win: vec![0.0; window48], seen: 0, last: 0 }
    }

    /// Blocks up to [`PERIOD_MS`]. Some((window end, frames)) when `mel` and `bands` hold a new
    /// window's features.
    pub fn next(&mut self, mel: &mut Vec<f32>, bands: &mut Vec<f32>) -> Option<(u64, usize)> {
        let requested = self.session.wait(&mut self.seen, PERIOD_MS);
        let end = match requested {
            Some(e) => {
                let t0 = std::time::Instant::now();
                while self.session.written() < e && t0.elapsed().as_millis() < REQUEST_WAIT_MS as u128 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                e.min(self.session.written())
            }
            None => {
                let w = self.session.written();
                if w < self.last + (shm::RATE * PERIOD_MS / 1000) as u64 {
                    return None; // idle, or too little new audio
                }
                w
            }
        };
        if !self.session.window(end, &mut self.win) {
            return None;
        }
        self.last = end;
        let t = self.front.features48(&self.win, mel, bands);
        Some((end, t))
    }

    pub fn publish(&self, v: &shm::Verdict) {
        self.session.publish(v);
    }
}

/// Labels the tagger outputs, in its network's order.
pub const LABELS: [&str; 12] = ["gun", "explosion", "impact", "swing", "skill", "fire", "footstep", "engine", "ui", "voice", "music", "ambience"];

/// Score at which a label counts as sounding.
pub const ON: f32 = 0.5;
/// Score at which a label is a candidate for the sound that made a hit.
const PRESENT: f32 = 0.3;
const GUN: usize = 0;
const EXPLOSION: usize = 1;
const IMPACT: usize = 2;
const SWING: usize = 3;
const SKILL: usize = 4;
const FIRE: usize = 5;
const ENGINE: usize = 7;
const MUSIC: usize = 10;

fn level(v: &shm::Verdict, k: usize) -> f32 {
    if v.levels[k].is_finite() { v.levels[k] } else { -40.0 }
}

/// How a hit plays: by the loudest sounding label against the mix, so a shot over the battle
/// music kicks and the music alone stays quiet. A gun, explosion or impact kicks (gain from its
/// level: a third at 20 dB under the mix, full at the mix), fire streams, a swing swishes, a skill
/// goes to the detector's own rules (its stream mode already smooths a flamethrower), and voice,
/// UI, footsteps, engines and ambience stay quiet. Music on top, or a skill with music under it
/// (synth music sounds like one), plays only where the catalog's matcher knows the music.
pub fn verdict(v: &shm::Verdict) -> (crate::tap2::Verdict, f32) {
    use crate::tap2::Verdict as V;
    let Some(top) = (0..12).filter(|&k| v.labels[k] >= PRESENT).max_by(|&a, &b| {
        level(v, a).total_cmp(&level(v, b)).then(v.labels[a].total_cmp(&v.labels[b]))
    }) else {
        return (V::Rules, 1.0);
    };
    let gain = 10f32.powf(level(v, top).clamp(-20.0, 0.0) / 40.0);
    match top {
        GUN | EXPLOSION | IMPACT => (V::Hit, gain),
        FIRE => (V::Stream, gain),
        SWING => (V::Swing, gain),
        SKILL if v.labels[MUSIC] < PRESENT => (V::Rules, 1.0),
        SKILL | MUSIC => (V::Music, 1.0),
        _ => (V::Quiet, 1.0),
    }
}

/// A burst is on: shots louder than any music keep coming, so they kick at once instead of
/// waiting for a verdict.
pub fn burst(v: &shm::Verdict) -> bool {
    v.labels[GUN] >= ON && (v.labels[MUSIC] < PRESENT || level(v, GUN) >= level(v, MUSIC))
}

/// Music sounds and nothing that rumbles does (an explosion, an engine): its bass is no rumble.
pub fn bed(v: &shm::Verdict) -> bool {
    v.labels[MUSIC] >= PRESENT && v.labels[EXPLOSION] < PRESENT && v.labels[ENGINE] < PRESENT
}

/// The session memory between audioserver's worker and the tagger in the vibrator HAL
/// (`vendor.yasvara.hentak.ITagger`): a header page, then a 48 kHz mono f32 ring. The worker writes
/// audio and requests; the tagger writes results behind a seqlock.
pub mod shm {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    pub const MAGIC: u32 = u32::from_le_bytes(*b"HTS1");
    pub const VERSION: u32 = 1;
    pub const RATE: u32 = 48000;
    pub const RING: usize = 1 << 17; // 2.7 s
    pub const RING_OFFSET: usize = 4096;
    pub const BYTES: usize = RING_OFFSET + RING * 4;

    #[repr(C)]
    pub struct Header {
        pub magic: u32,
        pub version: u32,
        pub rate: u32,
        pub ring: u32,
        /// Samples ever written; the newest sample is `written - 1`.
        pub written: AtomicU64,
        /// Futex word: bumped with `request_end` set to ask for that window now.
        pub request: AtomicU32,
        _pad: u32,
        pub request_end: AtomicU64,
        /// Seqlock over the result: odd while the tagger writes it.
        pub seq: AtomicU32,
        _pad2: u32,
        /// End (exclusive sample position) of the window the result describes.
        pub end: AtomicU64,
        pub labels: [AtomicU32; 12],
        pub levels: [AtomicU32; 12],
    }

    /// A result: what was sounding in the window ending at `end`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Verdict {
        pub end: u64,
        pub labels: [f32; 12],
        /// dB of each label's sources against the mix (NaN where the label isn't sounding).
        pub levels: [f32; 12],
    }

    extern "C" {
        fn syscall(num: i64, ...) -> i64;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    #[cfg(target_arch = "aarch64")]
    const SYS_FUTEX: i64 = 98;
    #[cfg(target_arch = "x86_64")]
    const SYS_FUTEX: i64 = 202;
    const FUTEX_WAIT: i64 = 0;
    const FUTEX_WAKE: i64 = 1;

    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }

    pub struct Session {
        base: *mut u8,
        /// Bytes to unmap when the session drops (a mapping handed over to it), else 0.
        owned: usize,
    }

    impl Drop for Session {
        fn drop(&mut self) {
            if self.owned > 0 {
                // SAFETY: the mapping was handed to this session, and it is the last user.
                unsafe { munmap(self.base, self.owned) };
            }
        }
    }

    // SAFETY: all shared state is atomics; the ring has one writer (the worker) and readers that
    // validate what they copied against `written`.
    unsafe impl Send for Session {}
    unsafe impl Sync for Session {}

    impl Session {
        /// # Safety
        /// `base` must map at least [`BYTES`] of the session memory, page aligned, for as long as
        /// the session lives. `init` formats it (the side that created the memory).
        pub unsafe fn new(base: *mut u8, len: usize, init: bool) -> Option<Self> {
            if base.is_null() || len < BYTES {
                return None;
            }
            let s = Session { base, owned: 0 };
            if init {
                // SAFETY: the caller hands over the whole region, not yet shared.
                unsafe { std::ptr::write_bytes(base, 0, RING_OFFSET) };
                let h = base as *mut Header;
                // SAFETY: as above.
                unsafe {
                    (*h).magic = MAGIC;
                    (*h).version = VERSION;
                    (*h).rate = RATE;
                    (*h).ring = RING as u32;
                }
            }
            let h = s.header();
            (h.magic == MAGIC && h.version == VERSION && h.ring as usize == RING).then_some(s)
        }

        /// # Safety
        /// As [`Session::new`], and `base` is an mmap of `len` bytes that the session now owns: it
        /// unmaps it when the last user drops it.
        pub unsafe fn adopt(base: *mut u8, len: usize) -> Option<Self> {
            // SAFETY: the caller's contract.
            let mut s = unsafe { Self::new(base, len, false) }?;
            s.owned = len;
            Some(s)
        }

        fn header(&self) -> &Header {
            // SAFETY: `new` checked the region holds a header.
            unsafe { &*(self.base as *const Header) }
        }

        fn ring(&self) -> *mut f32 {
            // SAFETY: the ring follows the header page inside the region.
            unsafe { self.base.add(RING_OFFSET) as *mut f32 }
        }

        pub fn written(&self) -> u64 {
            self.header().written.load(Ordering::Acquire)
        }

        /// Worker: appends 48 kHz mono audio.
        pub fn write(&self, x: &[f32]) {
            let h = self.header();
            let mut at = h.written.load(Ordering::Relaxed);
            for &v in x {
                // SAFETY: the index is masked into the ring.
                unsafe { *self.ring().add(at as usize & (RING - 1)) = v };
                at += 1;
            }
            h.written.store(at, Ordering::Release);
        }

        /// Worker: asks for the window ending at `end` (a sample position) now.
        pub fn request(&self, end: u64) {
            let h = self.header();
            h.request_end.store(end, Ordering::Release);
            h.request.fetch_add(1, Ordering::Release);
            // SAFETY: futex wake on a shared u32 the mapping keeps alive.
            unsafe { syscall(SYS_FUTEX, &h.request as *const AtomicU32, FUTEX_WAKE, i32::MAX as i64) };
        }

        /// Worker: the latest result, or None while none was published or one is being written.
        pub fn result(&self) -> Option<Verdict> {
            let h = self.header();
            let s0 = h.seq.load(Ordering::Acquire);
            if s0 == 0 || s0 & 1 == 1 {
                return None;
            }
            let v = Verdict {
                end: h.end.load(Ordering::Relaxed),
                labels: std::array::from_fn(|i| f32::from_bits(h.labels[i].load(Ordering::Relaxed))),
                levels: std::array::from_fn(|i| f32::from_bits(h.levels[i].load(Ordering::Relaxed))),
            };
            std::sync::atomic::fence(Ordering::Acquire);
            (h.seq.load(Ordering::Relaxed) == s0).then_some(v)
        }

        /// Tagger: waits up to `timeout_ms` for a request. Returns the requested end when one came
        /// since `seen` (updated), else None.
        pub fn wait(&self, seen: &mut u32, timeout_ms: u32) -> Option<u64> {
            let h = self.header();
            let cur = h.request.load(Ordering::Acquire);
            if cur == *seen {
                let ts = Timespec { sec: (timeout_ms / 1000) as i64, nsec: (timeout_ms % 1000) as i64 * 1_000_000 };
                // SAFETY: futex wait on a shared u32; returns at once if it no longer holds `cur`.
                unsafe { syscall(SYS_FUTEX, &h.request as *const AtomicU32, FUTEX_WAIT, cur as i64, &ts as *const Timespec) };
            }
            let now = h.request.load(Ordering::Acquire);
            if now == *seen {
                return None;
            }
            *seen = now;
            Some(h.request_end.load(Ordering::Acquire))
        }

        /// Tagger: copies the `out.len()` samples ending at `end`. False when they aren't all
        /// written yet or the worker has overwritten some of them meanwhile.
        pub fn window(&self, end: u64, out: &mut [f32]) -> bool {
            let n = out.len() as u64;
            if n as usize > RING || end < n || end > self.written() {
                return false;
            }
            let start = end - n;
            for (k, o) in out.iter_mut().enumerate() {
                // SAFETY: the index is masked into the ring.
                *o = unsafe { *self.ring().add((start as usize + k) & (RING - 1)) };
            }
            self.written() - start <= RING as u64
        }

        /// Tagger: publishes the result for the window ending at `end`.
        pub fn publish(&self, v: &Verdict) {
            let h = self.header();
            let s = h.seq.load(Ordering::Relaxed);
            h.seq.store(s | 1, Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::Release);
            h.end.store(v.end, Ordering::Relaxed);
            for i in 0..12 {
                h.labels[i].store(v.labels[i].to_bits(), Ordering::Relaxed);
                h.levels[i].store(v.levels[i].to_bits(), Ordering::Relaxed);
            }
            h.seq.store((s | 1) + 1, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &[u8] = include_bytes!("../data/tagger_golden.bin");

    #[test]
    fn verdicts() {
        use crate::tap2::Verdict as V;
        let v = |pairs: &[(usize, f32, f32)]| {
            let mut v = shm::Verdict { end: 0, labels: [0.0; 12], levels: [f32::NAN; 12] };
            for &(k, p, db) in pairs {
                v.labels[k] = p;
                v.levels[k] = db;
            }
            v
        };
        let at = |pairs: &[(usize, f32, f32)]| verdict(&v(pairs));
        assert_eq!(at(&[(GUN, 0.9, -2.0), (10, 0.9, -9.0)]).0, V::Hit, "a shot over the music kicks");
        assert_eq!(at(&[(GUN, 0.4, -14.0), (10, 0.95, -1.0)]).0, V::Music, "music on top: the matcher");
        assert_eq!(at(&[(SKILL, 0.4, -3.0), (10, 0.9, -8.0)]).0, V::Music, "a skill over the music: the matcher");
        assert_eq!(at(&[(SKILL, 0.6, -2.0)]).0, V::Rules, "a skill alone: the rules");
        assert_eq!(at(&[(FIRE, 0.8, -1.0)]).0, V::Stream);
        assert_eq!(at(&[(SWING, 0.7, -2.0)]).0, V::Swing);
        assert_eq!(at(&[(9, 0.9, -1.0)]).0, V::Quiet, "voice");
        assert_eq!(at(&[]).0, V::Rules);
        let (_, g) = at(&[(GUN, 0.9, -20.0)]);
        assert!((g - 0.316).abs() < 0.01, "{g}");
        assert!(burst(&v(&[(GUN, 0.8, -2.0), (MUSIC, 0.9, -9.0)])), "shots over the music");
        assert!(!burst(&v(&[(GUN, 0.6, -12.0), (MUSIC, 1.0, -1.0)])), "drums in the music");
        assert!(bed(&v(&[(MUSIC, 0.9, -1.0)])));
        assert!(!bed(&v(&[(MUSIC, 0.9, -6.0), (EXPLOSION, 0.7, -2.0)])), "an explosion rumbles over it");
    }

    #[test]
    fn session_round_trip() {
        use shm::*;
        let mut mem = vec![0u64; BYTES / 8];
        let base = mem.as_mut_ptr() as *mut u8;
        // SAFETY: `mem` outlives both views and is BYTES long.
        let worker = unsafe { Session::new(base, BYTES, true) }.unwrap();
        let tagger = unsafe { Session::new(base, BYTES, false) }.unwrap();
        assert!(worker.result().is_none());
        let x: Vec<f32> = (0..RING + 1000).map(|i| i as f32).collect();
        worker.write(&x);
        let mut w = vec![0f32; 4800];
        let end = worker.written();
        assert!(tagger.window(end, &mut w));
        assert_eq!(w[4799], (end - 1) as f32);
        assert!(!tagger.window(end + 1, &mut w), "not written yet");
        let mut seen = 0;
        assert_eq!(tagger.wait(&mut seen, 1), None);
        worker.request(end - 100);
        assert_eq!(tagger.wait(&mut seen, 1), Some(end - 100));
        let v = shm::Verdict { end, labels: [0.5; 12], levels: [-6.0; 12] };
        tagger.publish(&v);
        assert_eq!(worker.result(), Some(v));
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn matches_torch() {
        assert_eq!(&GOLDEN[..4], b"HTG1");
        let g = arrays(&GOLDEN[4..]);
        let get = |n: &str| &g.iter().find(|x| x.0 == n).unwrap().2;
        let mut f = Front::new();
        let mut x32 = Vec::new();
        f.k48.apply(get("x48"), &mut x32);
        assert_eq!(x32.len(), get("x32").len());
        assert!(max_err(&x32, get("x32")) < 1e-5, "48→32 kHz {}", max_err(&x32, get("x32")));
        let (mut mel, mut bands) = (Vec::new(), Vec::new());
        let t = f.features48(get("x48"), &mut mel, &mut bands);
        assert_eq!(t * MELS, get("mel").len());
        assert!(max_err(&mel, get("mel")) < 2e-3, "mel {}", max_err(&mel, get("mel")));
        assert!(max_err(&bands, get("hi")) < 2e-3, "bands {}", max_err(&bands, get("hi")));
    }
}
