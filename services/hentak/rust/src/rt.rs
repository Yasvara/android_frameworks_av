//! The audioserver side. Mixer threads hand each armed track's audio over as native-rate mono
//! (one add per channel, a ring push, a futex wake); one worker thread runs the v2 detector on it
//! and, when the game has a catalog, the onset classifier that keeps voice, music and UI quiet.
//! Events leave through the C++ sender's queue.

use crate::catalog::Catalog;
use crate::front::{Front, Pcm};
use crate::mel::{Frame, Mel};
use crate::model::{embed, CTX_MAX};
use crate::music::Matcher;
use crate::onset::{Onset, OnsetCfg};
use crate::ring::Ring;
use crate::tagger::{self, shm::Session};
use crate::tap2::{Kind, Sink, Tap2, Verdict};
use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::thread::Thread;
use std::time::Duration;

/// Kind codes of the C++ `Event`.
pub const LEVELS: i32 = 3;
const RING: usize = 1 << 16; // ~1.3 s at 48 kHz
const BATCH: usize = 256;
/// A v2 hit waits for the classifier's verdict on onsets this close to its own.
const NEAR_MS: u64 = 10;
/// ...and is decided this long after its onset at the latest (the classifier decides 15 ms after
/// an onset, on a 5 ms grid).
const VERDICT_MS: u64 = 20;
/// A hit the tagger hasn't answered this long after its onset plays by the catalog or the rules.
const TAGGER_TIMEOUT_MS: u64 = 60;
/// A tagger result this recent tells whether a burst is on.
const TAGGER_FRESH_MS: u64 = 120;
/// The catalog's thresholds while known music plays.
const MUSIC_ENTER: f32 = 0.75;
const MUSIC_STAY: f32 = 0.5;

/// ANDROID_PRIORITY_URGENT_AUDIO, as the mixer threads: at nice 0 a game's own threads hold the
/// levels back and the motor's envelope stutters.
const WORKER_NICE: i32 = -19;

extern "C" {
    fn setpriority(which: i32, who: u32, prio: i32) -> i32;
}

/// uid, kind, intensity, sharpness, track, count, levels.
pub type PostFn = extern "C" fn(i32, i32, f32, f32, usize, i32, *const f32);

pub struct Tap {
    ring: Ring,
    uid: AtomicI32,
    sens: AtomicU32,
    rate: AtomicU32,
    /// Bumped when the rate changes: the worker starts the track over.
    epoch: AtomicU32,
    scratch: UnsafeCell<[f32; BATCH]>,
}

// SAFETY: `scratch` is touched only by the track's one consuming (mixer) thread, in `feed`.
unsafe impl Sync for Tap {}

struct Shared {
    taps: Mutex<Vec<Arc<Tap>>>,
    catalogs: Mutex<HashMap<i32, Arc<Catalog>>>,
    catalog_gen: AtomicU32,
    worker: OnceLock<Thread>,
    post: AtomicPtr<()>,
    fired: AtomicU64,
    quieted: AtomicU64,
    music: AtomicU64,
    music_status: Mutex<String>,
    tagger: Mutex<Option<Arc<Session>>>,
    tagger_gen: AtomicU32,
    /// Hits settled by the tagger: kick, swish, stream, quiet, rules, music; and timed out.
    verdicts: [AtomicU64; 7],
}

static SHARED: LazyLock<Shared> = LazyLock::new(|| Shared {
    taps: Mutex::new(Vec::new()),
    catalogs: Mutex::new(HashMap::new()),
    catalog_gen: AtomicU32::new(0),
    worker: OnceLock::new(),
    post: AtomicPtr::new(std::ptr::null_mut()),
    fired: AtomicU64::new(0),
    quieted: AtomicU64::new(0),
    music: AtomicU64::new(0),
    music_status: Mutex::new(String::new()),
    tagger: Mutex::new(None),
    tagger_gen: AtomicU32::new(0),
    verdicts: Default::default(),
});

pub fn init(post: PostFn) {
    SHARED.post.store(post as *mut (), Ordering::Release);
}

fn post(uid: i32, kind: i32, intensity: f32, sharpness: f32, track: usize, levels: &[f32]) {
    let p = SHARED.post.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: only `init` stores here, and only a `PostFn`.
        let f: PostFn = unsafe { std::mem::transmute::<*mut (), PostFn>(p) };
        f(uid, kind, intensity, sharpness, track, levels.len() as i32, levels.as_ptr());
    }
}

impl Tap {
    pub fn new() -> Arc<Tap> {
        let t = Arc::new(Tap {
            ring: Ring::new(RING),
            uid: AtomicI32::new(0),
            sens: AtomicU32::new(0),
            rate: AtomicU32::new(0),
            epoch: AtomicU32::new(0),
            scratch: UnsafeCell::new([0.0; BATCH]),
        });
        SHARED.taps.lock().unwrap().push(t.clone());
        SHARED.worker.get_or_init(|| {
            std::thread::Builder::new().name("hentak".into()).spawn(worker).expect("hentak worker").thread().clone()
        });
        t
    }

    pub fn release(self: &Arc<Self>) {
        SHARED.taps.lock().unwrap().retain(|t| !Arc::ptr_eq(t, self));
    }

    /// Mixer thread: no locks, no allocation.
    pub fn feed(&self, uid: i32, sens: f32, pcm: Pcm<'_>, channels: usize, rate: u32) {
        if channels == 0 || rate < 8000 {
            return;
        }
        self.uid.store(uid, Ordering::Relaxed);
        self.sens.store(sens.to_bits(), Ordering::Relaxed);
        if self.rate.swap(rate, Ordering::Relaxed) != rate {
            self.epoch.fetch_add(1, Ordering::Release);
        }
        // SAFETY: see the Sync impl.
        let scratch = unsafe { &mut *self.scratch.get() };
        let scale = 1.0 / channels as f32;
        match pcm {
            Pcm::I16(s) => {
                for chunk in s.chunks(BATCH * channels) {
                    let n = chunk.len() / channels;
                    for (o, f) in scratch.iter_mut().zip(chunk.chunks_exact(channels)) {
                        *o = f.iter().map(|&v| v as i32).sum::<i32>() as f32 * (scale / 32768.0);
                    }
                    self.ring.push(&scratch[..n]);
                }
            }
            Pcm::F32(s) => {
                for chunk in s.chunks(BATCH * channels) {
                    let n = chunk.len() / channels;
                    for (o, f) in scratch.iter_mut().zip(chunk.chunks_exact(channels)) {
                        *o = f.iter().sum::<f32>() * scale;
                    }
                    self.ring.push(&scratch[..n]);
                }
            }
        }
        if let Some(w) = SHARED.worker.get() {
            w.unpark();
        }
    }
}

/// The tagger session (the vibrator HAL's `ITagger`), or None when it goes away.
pub fn set_tagger(session: Option<Session>) {
    *SHARED.tagger.lock().unwrap() = session.map(Arc::new);
    SHARED.tagger_gen.fetch_add(1, Ordering::Release);
}

pub fn set_catalog(uid: i32, catalog: Option<Catalog>) {
    let mut m = SHARED.catalogs.lock().unwrap();
    match catalog {
        Some(c) => m.insert(uid, Arc::new(c)),
        None => m.remove(&uid),
    };
    SHARED.catalog_gen.fetch_add(1, Ordering::Release);
}

pub fn dump() -> String {
    let m = SHARED.catalogs.lock().unwrap();
    let mut uids: Vec<_> = m
        .iter()
        .map(|(u, c)| format!("{u}:{:?}{}", c.sounds, c.music.as_ref().map(|i| format!("+{} tracks", i.tracks())).unwrap_or_default()))
        .collect();
    uids.sort();
    let v: Vec<u64> = SHARED.verdicts.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "v3 catalogs [{}] fired={} quieted={} music={} playing: {} | tagger {} kick={} swish={} stream={} quiet={} rules={} music={} late={}",
        uids.join(" "),
        SHARED.fired.load(Ordering::Relaxed),
        SHARED.quieted.load(Ordering::Relaxed),
        SHARED.music.load(Ordering::Relaxed),
        SHARED.music_status.lock().unwrap(),
        if SHARED.tagger.lock().unwrap().is_some() { "on" } else { "off" },
        v[0], v[1], v[2], v[3], v[4], v[5], v[6]
    )
}

struct Out {
    uid: i32,
    track: usize,
}

impl Sink for Out {
    fn event(&mut self, kind: Kind, intensity: f32, sharpness: f32) {
        post(self.uid, kind as i32, intensity, sharpness, 0, &[]);
    }
    fn levels(&mut self, levels: &[f32]) {
        post(self.uid, LEVELS, 0.0, 0.0, self.track, levels);
    }
}

/// The onset classifier of a catalogued game, on the decimated signal.
struct Classifier {
    catalog: Arc<Catalog>,
    /// The detector's clock (ms) when this started, so verdicts and hits share one timeline.
    start_ms: u64,
    front: Front,
    ring: Ring,
    mel: Mel,
    onset: Onset,
    hist: [Frame; 16],
    t: u64,
    pending: VecDeque<u64>,
    /// (onset ms, quiet probability) of recent onsets.
    verdicts: VecDeque<(u64, f32)>,
    hold_until: u64,
    buf: Vec<f32>,
    matcher: Option<Matcher>,
}

impl Classifier {
    fn new(catalog: Arc<Catalog>, rate: u32, start_ms: u64) -> Self {
        Self {
            matcher: catalog.music.as_ref().map(|_| Matcher::new(Front::out_rate(rate), start_ms)),
            catalog,
            start_ms,
            front: Front::new(rate),
            ring: Ring::new(8192),
            mel: Mel::new(Front::out_rate(rate)),
            onset: Onset::new(OnsetCfg::default()),
            hist: [Frame::default(); 16],
            t: 0,
            pending: VecDeque::new(),
            verdicts: VecDeque::new(),
            hold_until: 0,
            buf: vec![0.0; 8192],
        }
    }

    fn push(&mut self, x: &[f32]) {
        self.front.feed(Pcm::F32(x), 1, &self.ring);
        let n = self.ring.pop(&mut self.buf);
        if let (Some(m), Some(index)) = (self.matcher.as_mut(), self.catalog.music.as_ref()) {
            m.push(index, &self.buf[..n]);
        }
        let ctx = self.catalog.cfg.ctx.min(CTX_MAX);
        let start_ms = self.start_ms;
        let Self { mel, onset, hist, t, pending, verdicts, hold_until, catalog, buf, matcher, .. } = self;
        // A beat of the known music is no line of dialogue: it neither holds nor quiets.
        let music = |ms: u64| matches!((&*matcher, &catalog.music), (Some(m), Some(index)) if m.explains(index, ms));
        let (enter, stay) = thresholds(catalog, matcher);
        mel.push(&buf[..n], |f| {
            hist[(*t % 16) as usize] = *f;
            if onset.step(f).is_some() {
                pending.push_back(*t);
            }
            while let Some(&t0) = pending.front() {
                if *t < t0 + ctx as u64 - 1 {
                    break;
                }
                pending.pop_front();
                let mut patch = [Frame::default(); CTX_MAX];
                for (k, p) in patch.iter_mut().take(ctx).enumerate() {
                    *p = hist[((t0 + k as u64) % 16) as usize];
                }
                let before = if t0 >= 2 { hist[((t0 - 2) % 16) as usize] } else { Frame::default() };
                let p = catalog.prob(&embed(&patch[..ctx], &before, &catalog.cfg));
                let ms = start_ms + t0 * crate::mel::FRAME_MS as u64;
                let p = if music(ms) { 0.0 } else { p };
                if p > enter || (p > stay && *hold_until > ms) {
                    *hold_until = ms + catalog.hold_ms as u64;
                }
                verdicts.push_back((ms, p));
            }
            *t += 1;
        });
        let now = self.decided_until();
        while self.verdicts.front().is_some_and(|v| v.0 + 200 < now) {
            self.verdicts.pop_front();
        }
    }

    fn decided_until(&self) -> u64 {
        self.start_ms + self.t * crate::mel::FRAME_MS as u64
    }

    /// Whether a hit with this onset stays quiet.
    fn quiet(&self, onset_ms: u64) -> bool {
        let near = self.verdicts.iter().filter(|v| v.0.abs_diff(onset_ms) <= NEAR_MS).map(|v| v.1).fold(None, |a: Option<f32>, p| Some(a.map_or(p, |a| a.max(p))));
        let (enter, stay) = thresholds(&self.catalog, &self.matcher);
        near.is_some_and(|p| p > enter) || (self.hold_until > onset_ms && near.is_none_or(|p| p > stay))
    }

    /// Whether the matcher knows a track playing now.
    fn locked(&self) -> bool {
        self.matcher.as_ref().is_some_and(|m| !m.locks().is_empty())
    }

    /// Whether the game's music, playing now, makes an onset this loud there.
    fn music(&self, onset_ms: u64) -> bool {
        match (&self.matcher, &self.catalog.music) {
            (Some(m), Some(index)) => m.explains(index, onset_ms),
            _ => false,
        }
    }

    fn music_status(&self) -> String {
        let (Some(m), Some(index)) = (&self.matcher, &self.catalog.music) else { return String::new() };
        let now = m.frames() as i64;
        m.locks()
            .iter()
            .map(|l| format!("{} @ {:.1} s {:+.0} dB", index.name(l.track), (now + l.offset) as f32 * crate::music::FRAME_MS as f32 / 1000.0, l.gain))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Enter and stay thresholds: music under an onset pulls its embedding toward the quiet side.
fn thresholds(catalog: &Catalog, matcher: &Option<Matcher>) -> (f32, f32) {
    if matcher.as_ref().is_some_and(|m| !m.locks().is_empty()) {
        (MUSIC_ENTER, MUSIC_STAY)
    } else {
        (catalog.enter, catalog.stay)
    }
}

/// One track's detection, off the mixer thread: v2 alone, or v2 hits held for the catalog's
/// verdict.
pub struct Detector {
    rate: u32,
    tap: Tap2,
    classifier: Option<Classifier>,
    /// The tagger session this track feeds (48 kHz only), and the onset it was last asked about.
    tagger: Option<Arc<Session>>,
    asked: Option<u64>,
    /// When set, every decided hit: onset ms and verdict (0 fired, 1 quiet, 2 music).
    pub trace: Option<Vec<(u64, usize)>>,
}

impl Detector {
    pub fn new(rate: u32) -> Self {
        Self { rate, tap: Tap2::new(), classifier: None, tagger: None, asked: None, trace: None }
    }

    /// Feeds this track's audio to the tagger and lets its verdicts settle hits; only at the
    /// session's 48 kHz.
    pub fn set_tagger(&mut self, session: Option<Arc<Session>>) {
        self.tagger = session.filter(|_| self.rate == tagger::shm::RATE);
        self.asked = None;
    }

    pub fn set_catalog(&mut self, catalog: Option<Arc<Catalog>>) {
        let start = self.tap.now_ms();
        self.classifier = catalog.map(|c| Classifier::new(c, self.rate, start));
    }

    /// Music locks ended: no pairs, track over, music gone.
    pub fn music_drops(&self) -> [u32; 3] {
        self.classifier.as_ref().and_then(|c| c.matcher.as_ref()).map_or([0; 3], |m| m.drops)
    }

    /// The music tracks playing, where and how loud; empty when none or no catalog.
    pub fn music_status(&self) -> String {
        self.classifier.as_ref().map(|c| c.music_status()).unwrap_or_default()
    }

    /// Native-rate mono. Returns hits fired, kept quiet as voice or UI, and kept quiet as music.
    pub fn process(&mut self, x: &[f32], sens: f32, out: &mut impl Sink) -> [u64; 3] {
        if self.tagger.is_some() {
            return self.process_tagged(x, sens, out);
        }
        let mut n = [0u64; 3];
        let Some(cls) = self.classifier.as_mut() else {
            self.tap.defer = false;
            self.tap.bed = false;
            self.tap.resolve(true, out);
            self.tap.feed(sens, x, self.rate, out);
            return n;
        };
        self.tap.defer = true;
        for chunk in x.chunks((self.rate / 200).max(1) as usize) {
            self.tap.bed = cls.music(self.tap.now_ms().saturating_sub(VERDICT_MS));
            self.tap.feed(sens, chunk, self.rate, out);
            cls.push(chunk);
            if let Some(p) = self.tap.pending() {
                if cls.decided_until() >= p.onset_ms + VERDICT_MS {
                    let verdict = if cls.quiet(p.onset_ms) {
                        1
                    } else if cls.music(p.onset_ms) {
                        2
                    } else {
                        0
                    };
                    n[verdict] += 1;
                    if let Some(t) = self.trace.as_mut() {
                        t.push((p.onset_ms, verdict));
                    }
                    self.tap.resolve(verdict == 0, out);
                }
            }
        }
        n
    }

    /// With the tagger: hits wait for its verdict on the window ending 20 ms after their onset
    /// (asked for once that audio is in), except during a burst, which kicks at once. A hit it
    /// hasn't answered by TAGGER_TIMEOUT_MS plays by the catalog's verdict, or the rules.
    fn process_tagged(&mut self, x: &[f32], sens: f32, out: &mut impl Sink) -> [u64; 3] {
        let session = self.tagger.clone().unwrap();
        let per_ms = (self.rate / 1000) as u64;
        let mut n = [0u64; 3];
        for chunk in x.chunks((self.rate / 200).max(1) as usize) {
            session.write(chunk);
            let latest = session.result();
            let end = session.written();
            let fresh = latest.filter(|v| v.end + TAGGER_FRESH_MS * per_ms >= end);
            self.tap.defer = !fresh.is_some_and(|v| tagger::burst(&v));
            let heard = fresh.is_some_and(|v| tagger::bed(&v));
            if let Some(cls) = self.classifier.as_mut() {
                self.tap.bed = heard || cls.music(self.tap.now_ms().saturating_sub(VERDICT_MS));
                self.tap.feed(sens, chunk, self.rate, out);
                cls.push(chunk);
            } else {
                self.tap.bed = heard;
                self.tap.feed(sens, chunk, self.rate, out);
            }
            let Some(p) = self.tap.pending() else { continue };
            let now = self.tap.now_ms();
            let want = end - now.saturating_sub(p.onset_ms + VERDICT_MS).min(end / per_ms) * per_ms;
            if self.asked != Some(p.onset_ms) && now >= p.onset_ms + VERDICT_MS {
                session.request(want);
                self.asked = Some(p.onset_ms);
            }
            let answer = latest.filter(|v| v.end >= want && self.asked == Some(p.onset_ms));
            let (verdict, gain) = match answer {
                Some(v) => tagger::verdict(&v),
                None if now >= p.onset_ms + TAGGER_TIMEOUT_MS => {
                    SHARED.verdicts[6].fetch_add(1, Ordering::Relaxed);
                    (Verdict::Rules, 1.0)
                }
                None => continue,
            };
            if answer.is_some() {
                let k = [Verdict::Hit, Verdict::Swing, Verdict::Stream, Verdict::Quiet, Verdict::Rules, Verdict::Music].iter().position(|v| *v == verdict).unwrap();
                SHARED.verdicts[k].fetch_add(1, Ordering::Relaxed);
            }
            let decided = self.classifier.as_ref().filter(|c| c.decided_until() >= p.onset_ms + VERDICT_MS);
            let rules = |c: &Classifier| if c.quiet(p.onset_ms) { 1 } else if c.music(p.onset_ms) { 2 } else { 0 };
            let quiet = match verdict {
                Verdict::Rules => decided.map_or(0, rules),
                // Music nobody knows: every onset in it is the music's own.
                Verdict::Music => decided.filter(|c| c.locked()).map_or(2, rules),
                Verdict::Quiet => 1,
                _ => 0,
            };
            n[quiet] += 1;
            if let Some(t) = self.trace.as_mut() {
                t.push((p.onset_ms, quiet));
            }
            if quiet > 0 {
                self.tap.resolve(false, out);
            } else {
                self.tap.resolve_as(verdict, gain, out);
            }
        }
        n
    }
}

struct Track {
    epoch: u32,
    catalog_gen: u32,
    tagger_gen: u32,
    det: Detector,
}

fn worker() {
    // SAFETY: a libc call on plain integers; PRIO_PROCESS (0) with who 0 is this thread on Linux.
    unsafe { setpriority(0, 0, WORKER_NICE) };
    let mut tracks: HashMap<usize, Track> = HashMap::new();
    let mut buf = vec![0f32; 4096];
    // The one track feeding the tagger: the first 48 kHz track with audio, until it goes away.
    let mut source: Option<usize> = None;
    loop {
        std::thread::park_timeout(Duration::from_millis(20));
        let taps: Vec<Arc<Tap>> = SHARED.taps.lock().unwrap().clone();
        tracks.retain(|id, _| taps.iter().any(|t| Arc::as_ptr(t) as usize == *id));
        if source.is_some_and(|s| !tracks.contains_key(&s)) {
            source = None;
        }
        let gen = SHARED.catalog_gen.load(Ordering::Acquire);
        let tgen = SHARED.tagger_gen.load(Ordering::Acquire);
        for tap in &taps {
            let id = Arc::as_ptr(tap) as usize;
            let epoch = tap.epoch.load(Ordering::Acquire);
            let rate = tap.rate.load(Ordering::Relaxed);
            let uid = tap.uid.load(Ordering::Relaxed);
            let sens = f32::from_bits(tap.sens.load(Ordering::Relaxed));
            let tr = tracks.entry(id).or_insert_with(|| Track {
                epoch: epoch.wrapping_sub(1),
                catalog_gen: gen,
                tagger_gen: tgen.wrapping_sub(1),
                det: Detector::new(rate),
            });
            if tr.epoch != epoch {
                *tr = Track { epoch, catalog_gen: gen.wrapping_sub(1), tagger_gen: tgen.wrapping_sub(1), det: Detector::new(rate) };
                if source == Some(id) {
                    source = None;
                }
            }
            if tr.catalog_gen != gen {
                tr.catalog_gen = gen;
                tr.det.set_catalog(SHARED.catalogs.lock().unwrap().get(&uid).cloned());
            }
            if tr.tagger_gen != tgen {
                tr.tagger_gen = tgen;
                let s = if source == Some(id) { SHARED.tagger.lock().unwrap().clone() } else { None };
                tr.det.set_tagger(s);
            }
            let mut out = Out { uid, track: id };
            let mut had = false;
            loop {
                let n = tap.ring.pop(&mut buf);
                if n == 0 {
                    break;
                }
                had = true;
                let [fired, quieted, music] = tr.det.process(&buf[..n], sens, &mut out);
                SHARED.fired.fetch_add(fired, Ordering::Relaxed);
                SHARED.quieted.fetch_add(quieted, Ordering::Relaxed);
                SHARED.music.fetch_add(music, Ordering::Relaxed);
            }
            if had && source.is_none() && rate == tagger::shm::RATE {
                source = Some(id);
                tr.tagger_gen = tgen.wrapping_sub(1); // links on the next pass
            }
            let status = tr.det.music_status();
            if !status.is_empty() {
                *SHARED.music_status.lock().unwrap() = status;
            }
        }
    }
}
