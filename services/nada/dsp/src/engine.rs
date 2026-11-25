//! The chain as an audio thread sees it: a fixed row of stage positions run in place on the
//! caller's buffer.

use crate::slot::{SlotRt, SlotShared};
use crate::stages::LimStats;
use crate::Frame;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::Arc;
use std::time::Instant;

pub const TRIM: usize = 0;
pub const CONV: usize = 1;
pub const EQ: usize = 2;
pub const DYNEQ: usize = 3;
pub const BASS: usize = 4;
pub const COLOUR: usize = 5;
pub const SPATIAL: usize = 6;
pub const MBC: usize = 7;
pub const COMP: usize = 8;
pub const LOUD: usize = 9;
pub const LIM: usize = 10;
pub const STAGES: usize = 11;
pub const STAGE_NAMES: [&str; STAGES] = ["trim", "conv", "eq", "dyneq", "bass", "colour", "spatial", "mbc", "comp", "loud", "lim"];

const BLOCK: usize = 1024;
/// Histogram buckets: four per octave of nanoseconds.
pub const BUCKETS: usize = 128;

pub fn bucket(ns: u64) -> usize {
    let ns = ns.max(1);
    let oct = 63 - ns.leading_zeros() as usize;
    let sub = if oct >= 2 { ((ns >> (oct - 2)) & 3) as usize } else { 0 };
    (oct * 4 + sub).min(BUCKETS - 1)
}

/// Upper bound of a bucket, in nanoseconds.
pub fn bucket_top(b: usize) -> u64 {
    let (oct, sub) = (b / 4, b % 4);
    if oct < 2 {
        return 1u64 << (oct + 1);
    }
    (1u64 << oct) + ((sub as u64 + 1) << (oct - 2))
}

/// Process-time record of one insert. Written by the audio thread with relaxed stores.
pub struct Tele {
    pub calls: AtomicU64,
    pub frames: AtomicU64,
    pub ns_sum: AtomicU64,
    pub ns_max: AtomicU32,
    pub hist: [AtomicU32; BUCKETS],
    /// Calls that took more than half their period.
    pub late: AtomicU32,
    pub nonfinite: AtomicU32,
    pub overload: AtomicBool,
}

impl Default for Tele {
    fn default() -> Self {
        Tele {
            calls: AtomicU64::new(0),
            frames: AtomicU64::new(0),
            ns_sum: AtomicU64::new(0),
            ns_max: AtomicU32::new(0),
            hist: std::array::from_fn(|_| AtomicU32::new(0)),
            late: AtomicU32::new(0),
            nonfinite: AtomicU32::new(0),
            overload: AtomicBool::new(false),
        }
    }
}

impl Tele {
    pub fn clear(&self) {
        self.calls.store(0, Relaxed);
        self.frames.store(0, Relaxed);
        self.ns_sum.store(0, Relaxed);
        self.ns_max.store(0, Relaxed);
        self.late.store(0, Relaxed);
        self.nonfinite.store(0, Relaxed);
        self.hist.iter().for_each(|h| h.store(0, Relaxed));
    }

    /// Upper bound of the given quantile in nanoseconds (bucket resolution, at most +25 %).
    pub fn quantile(&self, q: f64) -> u64 {
        let total: u64 = self.hist.iter().map(|h| h.load(Relaxed) as u64).sum();
        if total == 0 {
            return 0;
        }
        let want = (total as f64 * q).ceil() as u64;
        let mut seen = 0u64;
        for (b, h) in self.hist.iter().enumerate() {
            seen += h.load(Relaxed) as u64;
            if seen >= want {
                return bucket_top(b);
            }
        }
        bucket_top(BUCKETS - 1)
    }
}

/// What the control thread and the audio thread share for one insert.
pub struct Shared {
    pub id: u64,
    pub io: i32,
    /// 0: FastMixer, 1: mixer thread without one.
    pub kind: i32,
    /// Sample rate the audio thread runs at; 0 until its first call.
    pub rate: AtomicU32,
    pub slots: [SlotShared; STAGES],
    /// Set by the control thread after a publish; cleared by the audio thread when idle.
    pub busy: AtomicBool,
    pub tele: Tele,
    pub lim: Arc<LimStats>,
    /// Frames the published chain holds audio back. Always 0 on a fast mixer.
    pub latency: AtomicU32,
    /// Gain the mixer applied to the loudest track, f32 bits.
    pub level: AtomicU32,
}

impl Shared {
    pub fn new(id: u64, io: i32, kind: i32) -> Self {
        Shared {
            id,
            io,
            kind,
            rate: AtomicU32::new(0),
            slots: Default::default(),
            busy: AtomicBool::new(false),
            tele: Tele::default(),
            lim: Arc::new(LimStats::default()),
            latency: AtomicU32::new(0),
            level: AtomicU32::new(1f32.to_bits()),
        }
    }

    /// Clamped to −60…0 dB; a non-finite or silent value is ignored.
    pub fn set_level(&self, gain: f32) {
        if gain.is_finite() && gain > 0.0 {
            self.level.store(gain.clamp(1e-3, 1.0).to_bits(), Relaxed);
        }
    }

    pub fn level(&self) -> f64 {
        f32::from_bits(self.level.load(Relaxed)) as f64
    }

    /// May this mix point hold audio back?
    pub fn may_delay(&self) -> bool {
        self.kind != 0
    }

    pub fn wake(&self) {
        self.busy.store(true, SeqCst);
    }
}

/// Audio-thread state of one insert.
pub struct Engine {
    pub shared: Arc<Shared>,
    rate: u32,
    slots: [SlotRt; STAGES],
    bus: Vec<Frame>,
    scratch: Vec<Frame>,
    late_run: u32,
    /// Track gain undone around the level-dependent stages, as last applied.
    level: f64,
}

/// Flush-to-zero while the chain runs, so decaying filter memory never reaches the slow
/// denormal path.
struct Ftz(#[allow(dead_code)] u64);

impl Ftz {
    #[inline]
    fn on() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let old: u64;
            // SAFETY: reads and writes the calling thread's FPCR only.
            unsafe {
                std::arch::asm!("mrs {}, fpcr", out(reg) old);
                std::arch::asm!("msr fpcr, {}", in(reg) old | (1 << 24));
            }
            Ftz(old)
        }
        #[cfg(not(target_arch = "aarch64"))]
        Ftz(0)
    }
}

impl Drop for Ftz {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: restores the value read in `on`.
        unsafe {
            std::arch::asm!("msr fpcr, {}", in(reg) self.0);
        }
    }
}

impl Engine {
    pub fn new(shared: Arc<Shared>) -> Self {
        Engine {
            shared,
            rate: 0,
            slots: Default::default(),
            bus: vec![[0.0; 2]; BLOCK],
            scratch: vec![[0.0; 2]; BLOCK],
            late_run: 0,
            level: 1.0,
        }
    }

    /// A new sample rate: stages built for the old one are dropped. Returns true when the
    /// control thread must be told. Not real-time safe; the mixer is reconfiguring anyway.
    pub fn set_rate(&mut self, rate: u32) -> bool {
        if rate == self.rate {
            return false;
        }
        self.rate = rate;
        self.shared.rate.store(rate, SeqCst);
        self.slots.iter_mut().for_each(SlotRt::clear);
        true
    }

    /// Processes `frames` interleaved frames of `ch` channels in place. With every stage off
    /// this returns after one atomic load and the buffer is untouched.
    #[inline]
    pub fn process(&mut self, buf: &mut [f32], ch: usize) {
        let sh = &*self.shared;
        if !sh.busy.load(Relaxed) {
            return;
        }
        let rate = self.rate;
        let fade = rate / 50;
        let mut live = false;
        for (rt, s) in self.slots.iter_mut().zip(sh.slots.iter()) {
            rt.poll(s, fade);
            live |= rt.live(rate);
        }
        if !live {
            sh.busy.store(false, SeqCst);
            if sh.slots.iter().any(SlotShared::has_pending) {
                sh.busy.store(true, SeqCst);
            }
            return;
        }
        let t0 = Instant::now();
        let ftz = Ftz::on();
        let frames = buf.len() / ch;
        // The level-dependent stages see the mix with the track gain taken off, so a tube or
        // a threshold acts on the music as mastered at any volume. The gain goes back on
        // before the limiter, which guards the real output. A volume change ramps over the call.
        let dynamic = (DYNEQ..=LOUD).any(|st| self.slots[st].live(rate));
        let (from, to) = (self.level, sh.level());
        let undo = dynamic && (from != 1.0 || to != 1.0);
        self.level = to;
        let mut done = 0usize;
        for part in buf.chunks_mut(BLOCK * ch) {
            let n = part.len() / ch;
            let bus = &mut self.bus[..n];
            if ch == 1 {
                for (b, s) in bus.iter_mut().zip(part.iter()) {
                    *b = [*s as f64, *s as f64];
                }
            } else {
                for (b, s) in bus.iter_mut().zip(part.chunks_exact(ch)) {
                    *b = [s[0] as f64, s[1] as f64];
                }
            }
            for (st, (rt, s)) in self.slots.iter_mut().zip(sh.slots.iter()).enumerate() {
                if undo && st == DYNEQ {
                    ramp(bus, done, frames, from, to, true);
                }
                rt.run(s, rate, bus, &mut self.scratch);
                if undo && st == LOUD {
                    ramp(bus, done, frames, from, to, false);
                }
            }
            done += n;
            let last = bus[n - 1];
            if !(last[0].is_finite() && last[1].is_finite()) {
                // Filter memory is poisoned: clear it and give silence for this block.
                self.slots.iter_mut().for_each(|rt| rt.reset(rate));
                bus.fill([0.0; 2]);
                sh.tele.nonfinite.fetch_add(1, Relaxed);
            }
            if ch == 1 {
                for (b, s) in bus.iter().zip(part.iter_mut()) {
                    *s = b[0] as f32;
                }
            } else {
                for (b, s) in bus.iter().zip(part.chunks_exact_mut(ch)) {
                    s[0] = b[0] as f32;
                    s[1] = b[1] as f32;
                }
            }
        }
        drop(ftz);
        let ns = t0.elapsed().as_nanos() as u64;
        let t = &sh.tele;
        t.calls.fetch_add(1, Relaxed);
        t.frames.fetch_add(frames as u64, Relaxed);
        t.ns_sum.fetch_add(ns, Relaxed);
        t.hist[bucket(ns)].fetch_add(1, Relaxed);
        if ns > t.ns_max.load(Relaxed) as u64 {
            t.ns_max.store(ns.min(u32::MAX as u64) as u32, Relaxed);
        }
        // More than half the period, three calls running: shed the convolver.
        let period = frames as u64 * 1_000_000_000 / rate.max(1) as u64;
        if ns * 2 > period {
            t.late.fetch_add(1, Relaxed);
            self.late_run += 1;
            if self.late_run >= 3 && self.slots[CONV].kill(&sh.slots[CONV]) {
                t.overload.store(true, Relaxed);
                self.late_run = 0;
            }
        } else {
            self.late_run = 0;
        }
    }
}

/// Multiplies by the gain (or divides, `undo`) ramped linearly from `from` to `to` over a
/// call of `total` frames, of which `x` starts at frame `at`.
#[inline]
fn ramp(x: &mut [Frame], at: usize, total: usize, from: f64, to: f64, undo: bool) {
    let step = (to - from) / total.max(1) as f64;
    for (i, f) in x.iter_mut().enumerate() {
        let g = from + step * (at + i + 1) as f64;
        let g = if undo { 1.0 / g } else { g };
        f[0] *= g;
        f[1] *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::StageBox;
    use crate::stages::Trim;

    fn eng() -> Engine {
        let mut e = Engine::new(Arc::new(Shared::new(1, 13, 0)));
        e.set_rate(48000);
        e
    }

    #[test]
    fn idle_is_bit_exact_and_untouched() {
        let mut e = eng();
        let src: Vec<f32> = (0..288).map(|i| (i as f32 * 0.37).sin() * 1.5).collect();
        let mut buf = src.clone();
        e.process(&mut buf, 2);
        assert_eq!(buf, src);
        assert_eq!(e.shared.tele.calls.load(Relaxed), 0);
    }

    #[test]
    fn unity_stage_is_bit_exact_after_fade() {
        let mut e = eng();
        e.shared.slots[TRIM].publish(StageBox { rate: 48000, stage: Some(Box::new(Trim(1.0))) });
        e.shared.wake();
        let src: Vec<f32> = (0..288).map(|i| (i as f32 * 0.11).sin()).collect();
        for _ in 0..20 {
            let mut buf = src.clone();
            e.process(&mut buf, 2);
            assert_eq!(buf, src);
        }
        assert_eq!(e.shared.tele.calls.load(Relaxed), 20);
    }

    #[test]
    fn swap_crossfades_then_hands_old_stage_back() {
        let mut e = eng();
        let sh = e.shared.clone();
        sh.slots[TRIM].publish(StageBox { rate: 48000, stage: Some(Box::new(Trim(0.5))) });
        sh.wake();
        let mut first = vec![1.0f32; 2 * 144];
        e.process(&mut first, 2);
        // Fade from dry (1.0) towards 0.5, monotonic, no step.
        assert!(first[0] > 0.99 && first[286] < first[0] && first[286] > 0.5);
        for _ in 0..10 {
            let mut b = vec![1.0f32; 2 * 144];
            e.process(&mut b, 2);
        }
        let mut done = vec![1.0f32; 2 * 144];
        e.process(&mut done, 2);
        assert!(done.iter().all(|v| *v == 0.5));
        // Switch off: fades back to dry, then the engine goes idle on its own.
        sh.slots[TRIM].publish(StageBox { rate: 48000, stage: None });
        sh.wake();
        for _ in 0..12 {
            let mut b = vec![1.0f32; 2 * 144];
            e.process(&mut b, 2);
        }
        let mut b = vec![1.0f32; 2 * 144];
        e.process(&mut b, 2);
        assert!(b.iter().all(|v| *v == 1.0));
        assert!(!sh.busy.load(Relaxed));
    }

    #[test]
    fn stage_for_another_rate_is_ignored() {
        let mut e = eng();
        e.shared.slots[TRIM].publish(StageBox { rate: 44100, stage: Some(Box::new(Trim(0.0))) });
        e.shared.wake();
        let mut b = vec![1.0f32; 2 * 4800];
        e.process(&mut b, 2);
        assert!(b.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn mono_and_wide_buffers() {
        let mut e = eng();
        e.shared.slots[TRIM].publish(StageBox { rate: 48000, stage: Some(Box::new(Trim(0.5))) });
        e.shared.wake();
        for _ in 0..3 {
            let mut b = vec![1.0f32; 2000];
            e.process(&mut b, 1);
        }
        let mut m = vec![1.0f32; 100];
        e.process(&mut m, 1);
        assert!(m.iter().all(|v| *v == 0.5));
        let mut w = vec![1.0f32; 3 * 100];
        e.process(&mut w, 3);
        assert!(w.chunks(3).all(|f| f[0] == 0.5 && f[1] == 0.5 && f[2] == 1.0));
    }

    #[test]
    fn level_dependent_stages_ignore_the_volume() {
        use crate::stages::{Comp, CompP};
        let cp = CompP { threshold: -18.0, ratio: 4.0, attack_ms: 5.0, release_ms: 100.0, makeup: 0.0 };
        let run = |vol: f32, tell: bool| -> Vec<f32> {
            let mut e = eng();
            e.shared.slots[COMP].publish(StageBox { rate: 48000, stage: Some(Box::new(Comp::new(&cp, 48000.0))) });
            e.shared.wake();
            if tell {
                e.shared.set_level(vol);
            }
            let mut last = Vec::new();
            for k in 0..40 {
                let mut b: Vec<f32> = (0..2 * 480).map(|i| vol * 0.5 * ((k * 480 + i / 2) as f32 * 0.05).sin()).collect();
                e.process(&mut b, 2);
                last = b;
            }
            last
        };
        let full = run(1.0, true);
        let quiet = run(0.05, true);
        // At −26 dB volume the compressor acts exactly as at full volume, 26 dB lower.
        for (a, b) in full.iter().zip(quiet.iter()) {
            assert!((a * 0.05 - b).abs() < 1e-6, "{a} {b}");
        }
        // Without the level it would not compress at all.
        let blind = run(0.05, false);
        let peak = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak(&blind) > 0.0249 && peak(&quiet) < 0.02);
    }

    #[test]
    fn histogram_buckets_are_ordered() {
        let mut last = 0;
        for ns in [1u64, 3, 9, 100, 1000, 1500, 30_000, 3_000_000] {
            let b = bucket(ns);
            assert!(b >= last && bucket_top(b) >= ns, "{ns}");
            last = b;
        }
    }
}
