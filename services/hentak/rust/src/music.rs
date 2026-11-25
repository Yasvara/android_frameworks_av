//! Music the game ships. The scan fingerprints each music file the way audio search engines do,
//! pairs of spectral peaks hashed with their spacing, and keeps its loudness every frame. Live,
//! the mix's own pairs vote for a track and position; while one holds, an onset no louder than
//! that track predicts there is the music's and stays quiet.

use crate::front::{Front, Pcm};
use crate::ring::Ring;
use std::collections::{HashMap, VecDeque};

/// Fingerprints run on 8 kHz mono: what survives every speaker and mix.
pub const RATE: u32 = 8000;
const N: usize = 512; // 64 ms
const HOP: usize = 128;
/// One frame, ms.
pub const FRAME_MS: u64 = 16;
const BINS: usize = N / 2 + 1;
const BIN_LO: usize = 8; // 125 Hz
const BIN_HI: usize = 208; // 3.25 kHz
const LOWPASS_HZ: f32 = 3400.0;
// Peaks: local maxima where a bin just got louder (a note or hit starting, so its frame is
// exact), clearing a masking threshold that each pick raises around itself and that decays
// frame by frame. A held note gives one peak, not one per frame.
const PEAKS: usize = 2;
/// Live, the game's own sounds take peaks too: more per frame, so the music's are still among them.
const PEAKS_LIVE: usize = 4;
const NEIGHBOURS: usize = 2;
const RISE_DB: f32 = 8.0;
const RECENT_DECAY_DB: f32 = 1.5;
const MASK_DECAY_DB: f32 = 0.15;
const MASK_SPREAD_DB: f32 = 0.05; // per bin²
const MASK_REACH: usize = 32;
const PEAK_FLOOR_DB: f32 = -80.0;
// Pairs: each peak with the latest few before it, within half a second. Live pairs reach further
// back, past peaks of the game's sounds in between, so they still contain the track's.
const FAN: usize = 2;
const FAN_LIVE: usize = 6;
const DT_MAX: u64 = 31;
const DF_MAX: i32 = 63;
// Index keys: hash (20 bits) << 32 | track << FRAME_BITS | anchor frame.
const FRAME_BITS: u32 = 20;
pub const MAX_TRACKS: usize = 1 << (32 - FRAME_BITS);
const MAX_FRAMES: u64 = 1 << FRAME_BITS; // 4.6 h
/// A hash this common says nothing.
const MAX_HITS_PER_HASH: usize = 48;
// Live: votes over the last 2 s; a track and position (±1 frame) with this many is playing.
const VOTE_FRAMES: u64 = 125;
const LOCK_VOTES: u32 = 5;
const MAX_LOCKS: usize = 3;
/// A lock no pair has backed for this long has ended (8 s).
const DROP_FRAMES: u64 = 500;
const HISTORY: usize = 256;
const GAIN_FRAMES: usize = 128;
const GAIN_EVERY: u64 = 8;
/// Track frames quieter than this say nothing about the gain.
const GAIN_REF_DB: f32 = -60.0;
/// Music gone: most of the last frames far under what the track predicts.
const GONE_FRAMES: usize = 24;
const GONE_DB: f32 = 10.0;
/// An onset needs the mix this much louder than the music predicts to be the game's.
pub const MARGIN_DB: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Print {
    pub hash: u32,
    /// Frame of the earlier peak.
    pub frame: u64,
}

fn db_byte(db: f32) -> u8 {
    ((db + 100.0) * 2.0).round().clamp(0.0, 255.0) as u8
}

fn byte_db(b: u8) -> f32 {
    b as f32 / 2.0 - 100.0
}

#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn lowpass(fc: f32, rate: f32, q: f32) -> Self {
        let w = 2.0 * std::f32::consts::PI * fc / rate;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad { b0: (1.0 - c) / 2.0 / a0, b1: (1.0 - c) / a0, b2: (1.0 - c) / 2.0 / a0, a1: -2.0 * c / a0, a2: (1.0 - alpha) / a0, z1: 0.0, z2: 0.0 }
    }

    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

struct Fft {
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<u16>,
}

impl Fft {
    fn new() -> Self {
        let bits = N.trailing_zeros();
        Fft {
            cos: (0..N / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N as f64).cos() as f32).collect(),
            sin: (0..N / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N as f64).sin() as f32).collect(),
            rev: (0..N).map(|i| ((i as u32).reverse_bits() >> (32 - bits)) as u16).collect(),
        }
    }

    fn power(&self, x: &[f32; N], out: &mut [f32; BINS]) {
        let mut re = [0f32; N];
        let mut im = [0f32; N];
        for i in 0..N {
            re[i] = x[self.rev[i] as usize];
        }
        let mut size = 2;
        while size <= N {
            let half = size / 2;
            let step = N / size;
            for start in (0..N).step_by(size) {
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
            out[k] = re[k] * re[k] + im[k] * im[k];
        }
    }
}

/// One frame out of [`Prints`].
pub struct FrameOut<'a> {
    pub frame: u64,
    /// Loudness of the frame's 16 ms, dB re full-scale square.
    pub db: f32,
    pub prints: &'a [Print],
}

/// Mono at any rate in, frames with their peak-pair hashes out.
pub struct Prints {
    lp: [Biquad; 2],
    step: f64,
    pos: f64,
    prev: f32,
    win: [f32; N],
    at: usize,
    since_hop: usize,
    hop_sq: f32,
    hann: [f32; N],
    norm: f32,
    fft: Fft,
    mask: [f32; BINS],
    /// Per bin, the recent level, decaying.
    level: [f32; BINS],
    recent: VecDeque<(u64, u8)>,
    frame: u64,
    out: Vec<Print>,
    peaks: usize,
    fan: usize,
}

impl Prints {
    /// `live`: of a mix to match against the index, not of a track to index.
    pub fn new(rate: u32, live: bool) -> Self {
        let r = rate as f32;
        let hann: [f32; N] = std::array::from_fn(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos());
        let sum: f32 = hann.iter().sum();
        Prints {
            lp: [Biquad::lowpass(LOWPASS_HZ, r, 0.5412), Biquad::lowpass(LOWPASS_HZ, r, 1.3066)],
            step: rate as f64 / RATE as f64,
            pos: 1.0,
            prev: 0.0,
            win: [0.0; N],
            at: 0,
            since_hop: 0,
            hop_sq: 0.0,
            hann,
            // A full-scale sine peaks at 0 dB.
            norm: 4.0 / (sum * sum),
            fft: Fft::new(),
            mask: [PEAK_FLOOR_DB; BINS],
            level: [PEAK_FLOOR_DB; BINS],
            recent: VecDeque::new(),
            frame: 0,
            out: Vec::new(),
            peaks: if live { PEAKS_LIVE } else { PEAKS },
            fan: if live { FAN_LIVE } else { FAN },
        }
    }

    /// Frames emitted so far.
    pub fn frames(&self) -> u64 {
        self.frame
    }

    pub fn push(&mut self, x: &[f32], mut f: impl FnMut(FrameOut<'_>)) {
        for &x in x {
            let y = self.lp[0].run(x);
            let y = self.lp[1].run(y);
            while self.pos <= 1.0 {
                let s = self.prev + (y - self.prev) * self.pos as f32;
                self.sample(s, &mut f);
                self.pos += self.step;
            }
            self.pos -= 1.0;
            self.prev = y;
        }
    }

    fn sample(&mut self, s: f32, f: &mut impl FnMut(FrameOut<'_>)) {
        self.win[self.at] = s;
        self.at = (self.at + 1) % N;
        self.hop_sq += s * s;
        self.since_hop += 1;
        if self.since_hop < HOP {
            return;
        }
        let db = 10.0 * (self.hop_sq / HOP as f32 + 1e-10).log10();
        self.since_hop = 0;
        self.hop_sq = 0.0;
        let mut x = [0f32; N];
        for (i, v) in x.iter_mut().enumerate() {
            *v = self.win[(self.at + i) % N] * self.hann[i];
        }
        let mut p = [0f32; BINS];
        self.fft.power(&x, &mut p);
        for v in p.iter_mut() {
            *v = 10.0 * (*v * self.norm + 1e-12).log10();
        }
        self.mask.iter_mut().for_each(|m| *m -= MASK_DECAY_DB);
        let mut cands: Vec<usize> = (BIN_LO..=BIN_HI)
            .filter(|&k| p[k] > PEAK_FLOOR_DB && p[k] > self.level[k] + RISE_DB && (1..=NEIGHBOURS).all(|j| p[k] > p[k - j] && p[k] >= p[k + j]))
            .collect();
        for (l, &v) in self.level.iter_mut().zip(&p) {
            *l = v.max(*l - RECENT_DECAY_DB);
        }
        cands.sort_by(|&a, &b| (p[b] - self.mask[b]).total_cmp(&(p[a] - self.mask[a])).then(a.cmp(&b)));
        let mut picked: Vec<usize> = Vec::with_capacity(self.peaks);
        for k in cands {
            if picked.len() == self.peaks {
                break;
            }
            if p[k] <= self.mask[k] {
                continue;
            }
            picked.push(k);
            for j in k.saturating_sub(MASK_REACH)..=(k + MASK_REACH).min(BINS - 1) {
                let d = j as f32 - k as f32;
                self.mask[j] = self.mask[j].max(p[k] - MASK_SPREAD_DB * d * d);
            }
        }
        picked.sort_unstable();
        let t = self.frame;
        self.out.clear();
        for &f2 in &picked {
            let mut n = 0;
            for &(t1, f1) in self.recent.iter().rev() {
                let df = f2 as i32 - f1 as i32;
                if df.abs() > DF_MAX {
                    continue;
                }
                let dt = t - t1;
                self.out.push(Print { hash: (f1 as u32) << 12 | ((df + 64) as u32) << 5 | dt as u32, frame: t1 });
                n += 1;
                if n == self.fan {
                    break;
                }
            }
        }
        for &k in &picked {
            self.recent.push_back((t, k as u8));
        }
        while self.recent.front().is_some_and(|r| t + 1 - r.0 > DT_MAX) {
            self.recent.pop_front();
        }
        self.frame += 1;
        f(FrameOut { frame: t, db, prints: &self.out });
    }
}

/// The prints and loudness of one decoded track, through the same front end as live audio.
pub struct Track {
    pub prints: Vec<Print>,
    pub env: Vec<u8>,
}

/// Feeds a track in pieces at its own rate.
pub struct TrackPrinter {
    front: Front,
    ring: Ring,
    buf: Vec<f32>,
    prints: Prints,
    track: Track,
}

impl TrackPrinter {
    pub fn new(rate: u32) -> Self {
        Self {
            front: Front::new(rate),
            ring: Ring::new(1 << 14),
            buf: vec![0.0; 1 << 14],
            prints: Prints::new(Front::out_rate(rate), false),
            track: Track { prints: Vec::new(), env: Vec::new() },
        }
    }

    pub fn push(&mut self, mono: &[f32]) {
        for c in mono.chunks(4096) {
            self.front.feed(Pcm::F32(c), 1, &self.ring);
            let n = self.ring.pop(&mut self.buf);
            let Self { prints, track, buf, .. } = self;
            prints.push(&buf[..n], |f| {
                if f.frame < MAX_FRAMES {
                    track.env.push(db_byte(f.db));
                    track.prints.extend_from_slice(f.prints);
                }
            });
        }
    }

    pub fn finish(self) -> Track {
        self.track
    }
}

/// Every music track's prints, by hash, and loudness.
pub struct Index {
    keys: Vec<u64>,
    env: Vec<Vec<u8>>,
    names: Vec<String>,
}

impl Index {
    /// Tracks past [`MAX_TRACKS`] are left out.
    pub fn build(tracks: Vec<(String, Track)>) -> Index {
        let mut keys = Vec::new();
        let (mut env, mut names) = (Vec::new(), Vec::new());
        for (i, (name, t)) in tracks.into_iter().take(MAX_TRACKS).enumerate() {
            keys.extend(t.prints.iter().map(|p| (p.hash as u64) << 32 | (i as u64) << FRAME_BITS | p.frame));
            env.push(t.env);
            names.push(name);
        }
        keys.sort_unstable();
        Index { keys, env, names }
    }

    pub fn tracks(&self) -> usize {
        self.env.len()
    }

    pub fn prints(&self) -> usize {
        self.keys.len()
    }

    pub fn name(&self, track: usize) -> &str {
        &self.names[track]
    }

    fn lookup(&self, hash: u32) -> &[u64] {
        let lo = self.keys.partition_point(|&k| k >> 32 < hash as u64);
        let hi = lo + self.keys[lo..].partition_point(|&k| k >> 32 <= hash as u64);
        &self.keys[lo..hi]
    }

    fn env_db(&self, track: usize, frame: i64) -> Option<f32> {
        usize::try_from(frame).ok().and_then(|f| self.env[track].get(f)).map(|&b| byte_db(b))
    }

    /// "HTM1", tracks (u32), per track: name length (u8), name, frames (u32), loudness bytes;
    /// keys (u32), keys (u64 LE).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"HTM1".to_vec();
        out.extend_from_slice(&(self.env.len() as u32).to_le_bytes());
        for (name, env) in self.names.iter().zip(&self.env) {
            let name = &name.as_bytes()[..name.len().min(255)];
            out.push(name.len() as u8);
            out.extend_from_slice(name);
            out.extend_from_slice(&(env.len() as u32).to_le_bytes());
            out.extend_from_slice(env);
        }
        out.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for k in &self.keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Index> {
        if b.get(..4)? != b"HTM1" {
            return None;
        }
        let u32_at = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?)) };
        let tracks = u32_at(4)? as usize;
        if tracks > MAX_TRACKS {
            return None;
        }
        let mut at = 8;
        let (mut env, mut names) = (Vec::with_capacity(tracks), Vec::with_capacity(tracks));
        for _ in 0..tracks {
            let len = *b.get(at)? as usize;
            names.push(String::from_utf8_lossy(b.get(at + 1..at + 1 + len)?).into_owned());
            at += 1 + len;
            let frames = u32_at(at)? as usize;
            env.push(b.get(at + 4..at + 4 + frames)?.to_vec());
            at += 4 + frames;
        }
        let n = u32_at(at)? as usize;
        let keys: Vec<u64> = b.get(at + 4..at + 4 + 8 * n)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        if keys.windows(2).any(|w| w[0] > w[1]) || keys.iter().any(|k| (k >> FRAME_BITS & (MAX_TRACKS as u64 - 1)) as usize >= tracks) {
            return None;
        }
        Some(Index { keys, env, names })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lock {
    pub track: usize,
    /// Track frame = live frame + offset.
    pub offset: i64,
    /// Live loudness over the track's, dB.
    pub gain: f32,
    support: u64,
}

/// Live: which indexed track is playing, where, and how loud.
pub struct Matcher {
    prints: Prints,
    /// The detector's clock (ms) at frame 0.
    start_ms: u64,
    db: VecDeque<f32>,
    votes: HashMap<(usize, i64), u32>,
    window: VecDeque<(u64, usize, i64)>,
    locks: Vec<Lock>,
    /// Locks ended: no pairs, track over, music gone.
    pub drops: [u32; 3],
}

impl Matcher {
    /// `rate` of the signal [`Matcher::push`] gets; `start_ms` on the detector's clock.
    pub fn new(rate: u32, start_ms: u64) -> Self {
        Self { prints: Prints::new(rate, true), start_ms, db: VecDeque::new(), votes: HashMap::new(), window: VecDeque::new(), locks: Vec::new(), drops: [0; 3] }
    }

    pub fn locks(&self) -> &[Lock] {
        &self.locks
    }

    /// Frames in so far.
    pub fn frames(&self) -> u64 {
        self.prints.frames()
    }

    pub fn push(&mut self, index: &Index, x: &[f32]) {
        let mut frames: Vec<(u64, f32, Vec<Print>)> = Vec::new();
        self.prints.push(x, |f| frames.push((f.frame, f.db, f.prints.to_vec())));
        for (frame, db, prints) in frames {
            self.frame(index, frame, db, &prints);
        }
    }

    fn live_db(&self, frame: i64) -> Option<f32> {
        let first = self.prints.frames() as i64 - self.db.len() as i64;
        usize::try_from(frame - first).ok().and_then(|i| self.db.get(i)).copied()
    }

    fn frame(&mut self, index: &Index, frame: u64, db: f32, prints: &[Print]) {
        self.db.push_back(db);
        if self.db.len() > HISTORY {
            self.db.pop_front();
        }
        let mut fresh: Vec<(usize, i64)> = Vec::new();
        for p in prints {
            let hits = index.lookup(p.hash);
            if hits.len() > MAX_HITS_PER_HASH {
                continue;
            }
            for &k in hits {
                let track = (k >> FRAME_BITS & (MAX_TRACKS as u64 - 1)) as usize;
                let offset = (k & (MAX_FRAMES - 1)) as i64 - p.frame as i64;
                self.window.push_back((frame, track, offset));
                *self.votes.entry((track, offset)).or_default() += 1;
                fresh.push((track, offset));
            }
        }
        while let Some(&(f, track, offset)) = self.window.front() {
            if f + VOTE_FRAMES > frame {
                break;
            }
            self.window.pop_front();
            if let Some(v) = self.votes.get_mut(&(track, offset)) {
                *v -= 1;
                if *v == 0 {
                    self.votes.remove(&(track, offset));
                }
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for (track, offset) in fresh {
            // One pair keeps a lock: a chance match right at its place is rare.
            if let Some(l) = self.locks.iter_mut().find(|l| l.track == track && (l.offset - offset).abs() <= 2) {
                l.support = frame;
                continue;
            }
            let votes: u32 = (offset - 1..=offset + 1).map(|o| self.votes.get(&(track, o)).copied().unwrap_or(0)).sum();
            if votes < LOCK_VOTES {
                continue;
            }
            let mut lock = Lock { track, offset, gain: 0.0, support: frame };
            let Some(gain) = self.gain(index, &lock) else { continue };
            lock.gain = gain;
            if self.locks.len() == MAX_LOCKS {
                let oldest = (0..self.locks.len()).min_by_key(|&i| self.locks[i].support).unwrap();
                self.locks.swap_remove(oldest);
            }
            self.locks.push(lock);
        }
        let live = frame as i64;
        let mut keep = Vec::with_capacity(self.locks.len());
        for mut l in std::mem::take(&mut self.locks) {
            // Game music loops: past its end a track goes on from its start, until pairs say otherwise.
            if index.env_db(l.track, live + l.offset).is_none() && live + l.offset >= 0 {
                l.offset -= index.env[l.track].len() as i64;
                l.support = l.support.max(frame.saturating_sub(DROP_FRAMES / 2));
            }
            let ended = [frame - l.support > DROP_FRAMES, index.env_db(l.track, live + l.offset).is_none(), self.gone(index, &l)];
            if let Some(why) = ended.iter().position(|&e| e) {
                self.drops[why] += 1;
                continue;
            }
            if frame % GAIN_EVERY == 0 {
                if let Some(g) = self.gain(index, &l) {
                    l.gain = g;
                }
            }
            keep.push(l);
        }
        self.locks = keep;
    }

    /// A low quantile of live over track loudness: the game's own sounds only add.
    fn gain(&self, index: &Index, l: &Lock) -> Option<f32> {
        let last = self.prints.frames() as i64 - 1;
        let mut d: Vec<f32> = (0..GAIN_FRAMES as i64)
            .filter_map(|i| {
                let f = last - i;
                let r = index.env_db(l.track, f + l.offset)?;
                (r > GAIN_REF_DB).then_some(self.live_db(f)? - r)
            })
            .collect();
        if d.len() < 8 {
            return None;
        }
        d.sort_by(f32::total_cmp);
        Some(d[d.len() / 4])
    }

    fn gone(&self, index: &Index, l: &Lock) -> bool {
        let last = self.prints.frames() as i64 - 1;
        let (mut heard, mut under) = (0, 0);
        for f in last + 1 - GONE_FRAMES as i64..=last {
            let (Some(r), Some(live)) = (index.env_db(l.track, f + l.offset), self.live_db(f)) else { continue };
            if r > GAIN_REF_DB {
                heard += 1;
                under += (live < r + l.gain - GONE_DB) as usize;
            }
        }
        heard >= GONE_FRAMES / 2 && under * 3 >= heard * 2
    }

    /// Whether the music playing explains an onset at `onset_ms` (detector clock): the mix around
    /// it no louder than the locked tracks predict, plus [`MARGIN_DB`].
    pub fn explains(&self, index: &Index, onset_ms: u64) -> bool {
        if self.locks.is_empty() {
            return false;
        }
        let rel = onset_ms as i64 - self.start_ms as i64;
        let k0 = (rel - FRAME_MS as i64).div_euclid(FRAME_MS as i64);
        let k1 = (rel + 20).div_euclid(FRAME_MS as i64).min(self.prints.frames() as i64 - 1);
        if k1 < k0 {
            return false;
        }
        let live = (k0..=k1).filter_map(|k| self.live_db(k)).fold(f32::MIN, f32::max);
        let music = self
            .locks
            .iter()
            .flat_map(|l| (k0 - 2..=k1 + 2).filter_map(move |k| Some(index.env_db(l.track, k + l.offset)? + l.gain)))
            .fold(f32::MIN, f32::max);
        live > f32::MIN && live <= music + MARGIN_DB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_track(seed: u64, secs: f32, rate: u32) -> Vec<f32> {
        // Notes of a random melody with a kick every 500 ms.
        let mut s = seed;
        let mut out = Vec::new();
        let note = (rate / 4) as usize;
        for _ in 0..(secs * 4.0) as usize {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = 200.0 + (s >> 40) as f32 % 2000.0;
            let g = 300.0 + (s >> 20 & 0xfff) as f32 % 1500.0;
            for i in 0..note {
                let t = i as f32 / rate as f32;
                let kick = if i < note / 2 { (-t * 40.0).exp() * (2.0 * std::f32::consts::PI * 60.0 * t).sin() } else { 0.0 };
                out.push(0.2 * (2.0 * std::f32::consts::PI * f * t).sin() + 0.1 * (2.0 * std::f32::consts::PI * g * t).sin() + 0.4 * kick);
            }
        }
        out
    }

    fn index_of(tracks: &[Vec<f32>], rate: u32) -> Index {
        Index::build(
            tracks
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut p = TrackPrinter::new(rate);
                    p.push(t);
                    (format!("t{i}"), p.finish())
                })
                .collect(),
        )
    }

    #[test]
    fn round_trip() {
        let idx = index_of(&[tone_track(1, 3.0, 48_000)], 48_000);
        let back = Index::from_bytes(&idx.to_bytes()).unwrap();
        assert_eq!(back.keys, idx.keys);
        assert_eq!(back.env, idx.env);
        assert_eq!(back.names, idx.names);
    }

    #[test]
    fn locks_on_the_playing_track_and_place() {
        let rate = 48_000;
        let tracks: Vec<Vec<f32>> = (0..4).map(|i| tone_track(i + 10, 20.0, rate)).collect();
        let idx = index_of(&tracks, rate);
        // Track 2 from 7 s in, 6 dB down.
        let live: Vec<f32> = tracks[2][7 * rate as usize..].iter().map(|x| x * 0.5).collect();
        let mut front = Front::new(rate);
        let ring = Ring::new(1 << 15);
        let mut m = Matcher::new(Front::out_rate(rate), 0);
        let mut buf = vec![0f32; 1 << 15];
        for c in live[..4 * rate as usize].chunks(960) {
            front.feed(Pcm::F32(c), 1, &ring);
            let n = ring.pop(&mut buf);
            m.push(&idx, &buf[..n]);
        }
        let l = m.locks().first().copied().expect("no lock");
        assert_eq!(l.track, 2);
        assert!((l.offset - 7000 / FRAME_MS as i64).abs() <= 1, "offset {}", l.offset);
        assert!((l.gain + 6.0).abs() < 1.5, "gain {}", l.gain);
    }
}
