//! The v2 detector: transients by band envelopes, for games nothing was taught or catalogued for,
//! and as the onset source a catalog vetoes. Mono mix → three bands from two one-pole filters:
//! low (< 200 Hz, the "body" a speaker can't carry), mid (200 Hz–2 kHz, most of a mobile game's
//! punch), high (> 2 kHz, crack and sparkle). A transient is the body envelope jumping well above
//! its own slow average; its loudness sets the intensity and the share of high band the
//! sharpness. A loud low band that stays up is a rumble (engines, explosions' tails).

const ONSET_RATIO: f32 = 2.0; // fast body over slow average
const ONSET_FLOOR_DB: f32 = -38.0; // body envelope
const REFRACTORY_MS: u32 = 55; // a burst of a gun at ~600 rpm still lands every shot
const REARM: f32 = 0.5; // body must fall to this share of the last peak
// Full auto: shot tails overlap, so the body never falls that far again. Within a burst a shot is
// the body rising from the trough since the last one, near the burst's level.
const BURST_MS: u32 = 400;
const BURST_REFRACTORY_MS: u32 = 45; // ~1300 rpm
const BURST_RISE: f32 = 1.8; // over the trough, ~5 dB
const BURST_LEVEL: f32 = 0.45; // of the last event's peak: a bolt click is quieter
const WINDOW_MS: u32 = 12; // a shot's crack comes before its thump
const BLAST_QUIET_MS: u32 = 400; // a blast stands alone; rapid fire stays impacts
const RUMBLE_FLOOR_DB: f32 = -26.0;
const RUMBLE_RISE: f32 = 2.5; // +8 dB over the 2 s low average: music bass never is
const RUMBLE_MAX_MS: u32 = 1500;
const RUMBLE_HOLD_MS: u32 = 160;
const RUMBLE_EVERY_MS: u32 = 110;
const RUMBLE_AFTER_HIT_MS: u32 = 150; // a run of shots is not a rumble
// Voice: vowels put their energy in 200 Hz–2 kHz with little above; shots and hits carry a crack
// above 2 kHz and a thump below 200 Hz. Shares are of the three band envelopes at the peak.
const VOICE_MID: f32 = 0.42;
const VOICE_HIGH: f32 = 0.30;
// ...and are pitched: a knife's whoosh has a voice's spectrum and no pitch. An onset that looks
// like a voice waits PITCH_WAIT_MS more and is one only if the signal below 2 kHz repeats at a
// pitch period (80-400 Hz) that strongly over its first PITCH_SPAN.
const PITCH_RATE: u32 = 4000;
const PITCH_BUF: usize = 256;
const PITCH_SPAN: usize = 128; // 32 ms
const PITCH_WAIT_MS: u32 = 20;
const VOICE_PITCH: f32 = 0.5;
/// An unpitched one is a swing: a soft swish, never a kick (the HAL kicks on rises over a quarter
/// of full level), so a knife that lands still kicks once, on its thump.
const SWISH_LEVEL: f32 = 0.22;
const SWISH_MS: u64 = 120;
/// After a swing, its own onsets swish too: any that is no thump or crack within SWISH_MS (the
/// blade's tail), and whoosh-like ones for SWISH_HOLD_MS (the next swing).
const SWISH_HOLD_MS: u64 = 500;
// A weak onset right before a strong one is its lead-in (a blade arriving, a shell's first
// crackle): it waits this long, and a stronger onset in that time plays alone. Two kicks 50 ms
// apart feel like one hit that rattles.
const WEAK_INTENSITY: f32 = 0.55;
const WEAK_WAIT_MS: u32 = 50;
const TALK_HOLD_MS: u32 = 900; // syllables of one line land within this
const TALK_SFX_MID: f32 = 0.30; // during dialogue only clearly non-vocal onsets pass
const TALK_SFX_HIGH: f32 = 0.25;
// A hit's thump: the low band leads and the body peaks within a few ms of the onset, where a
// voiced syllable with as much low swells through the whole window. A knife's hit lands right
// after its whoosh, which reads as a voice, so a thump is never one and passes a dialogue hold.
const THUMP_LOW: f32 = 0.5;
const THUMP_PEAK_MS: u32 = 3;
// Gunfire keeps a beat, speech doesn't: onsets this regular are never taken for a voice, and they
// end a dialogue hold that a voice-like shot started.
const RHYTHM_MAX_MS: u32 = 300; // 200 rpm and up
const RHYTHM_TOLERANCE: f32 = 0.2;
const RHYTHM_ONSETS: i32 = 2;
// Intensity is relative to the loudest recent event, so quiet and loud games both use the range.
const RANGE_DB: f32 = 24.0;
const REF_DECAY_DB_PER_MS: f32 = 0.001;
// Envelope. A hit starts at its intensity and decays with the sound's own body envelope:
// (body / body at the hit)^gamma, so a crack stays short and a boom rings out. It only ever
// falls: a noisy body wobbles every few ms, and followed up and down that drives the motor as
// crackle, not as a hit ringing out. Hits tail off linearly over the last 30 % of their cap.
// Rumble sits lower than hits so they stand out on it.
const SHARP_GAMMA: f32 = 1.6;
const DULL_GAMMA: f32 = 1.0;
const BLAST_GAMMA: f32 = 0.7;
const HIT_MAX_MS: u32 = 350;
const BLAST_MAX_MS: u32 = 1200;
const HIT_END: f32 = 0.03;
const EXPAND: f32 = 1.5; // intensity^EXPAND: soft steps stay soft next to shots
const RUMBLE_DEPTH: f32 = 0.6;
const RUMBLE_ATTACK: f32 = 1.0 / 30.0; // per ms
const RUMBLE_RELEASE: f32 = 1.0 / 80.0;
// A stream: onsets this dense and off the beat (fire, sparks, a flamethrower's crackle) are one
// sound. Each its own kick, the motor plays the crackle itself; as one sustained level following
// their loudness, it plays the sound. Gunfire keeps its beat and stays rounds; a blast stands
// alone.
const STREAM_ONSETS: usize = 5;
/// Peak over the trough before it: a round rises out of near silence, a crackle off a bed that
/// never falls. And a round cracks above 2 kHz (DF: a share of 0.24 and up), a flame roars below
/// it (X.Borg: 0.15 and under). Streams are made of onsets under both.
const STREAM_PROMINENCE: f32 = 3.0;
const STREAM_HIGH: f32 = 0.18;
const STREAM_WINDOW_MS: u64 = 600;
const STREAM_HOLD_MS: u64 = 200;
const STREAM_DEPTH: f32 = 0.7;
const STREAM_ATTACK: f32 = 1.0 / 40.0; // per ms
const STREAM_RELEASE: f32 = 1.0 / 150.0;
const LEVEL_STEP_MS: u32 = 5;
pub const LEVEL_BATCH: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Kind {
    Impact = 0,
    Blast = 1,
    Rumble = 2,
}

pub trait Sink {
    fn event(&mut self, kind: Kind, intensity: f32, sharpness: f32);
    /// Envelope steps, 5 ms apart; a silent batch is sent once to end the envelope.
    fn levels(&mut self, levels: &[f32]);
}

/// How a waiting hit plays, from the tagger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Gun, explosion, impact: a kick.
    Hit,
    /// A swing: a soft swish.
    Swing,
    /// Fire, a flamethrower: one sustained level.
    Stream,
    /// Voice, UI, footsteps, engines, ambience: nothing.
    Quiet,
    /// A skill, or nothing the tagger knows: the detector's own rules.
    Rules,
    /// The music, or a sound the tagger can't tell from it: the rules, but only where the
    /// catalog's matcher knows the music playing; nothing otherwise.
    Music,
}

/// A hit waiting for a catalog's verdict.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    /// Milliseconds since the detector started, at the onset.
    pub onset_ms: u64,
    blast: bool,
    intensity: f32,
    sharpness: f32,
    prominence: f32,
    low: f32,
    high: f32,
    swish: bool,
}

fn dbfs(x: f32) -> f32 {
    20.0 * x.max(1e-6).log10()
}

pub struct Tap2 {
    rate: u32,
    low_a: f32,
    high_a: f32,
    fast_atk: f32,
    fast_rel: f32,
    slow: f32,
    low_slow: f32,
    low_long_a: f32,
    low: f32,
    mid: f32,
    body: f32,
    high: f32,
    mid_env: f32,
    base: f32,
    low_env: f32,
    low_base: f32,
    low_long: f32,
    hop_left: u32,
    since_event: u32,
    rumble_ms: u32,
    since_rumble: u32,
    window_ms: u32,
    peak: f32,
    peak_low: f32,
    peak_mid: f32,
    peak_high: f32,
    peak_ms: u32,
    armed: bool,
    trough: f32,
    /// The trough the current onset rose from.
    onset_trough: f32,
    talk_ms: u32,
    since_onset: u32,
    onset_gap: u32,
    rhythm: i32,
    ref_db: f32,
    hit: f32,
    hit_peak: f32,
    hit_gamma: f32,
    hit_ms: u32,
    hit_max_ms: u32,
    /// The hit's envelope so far, falling only.
    hit_env: f32,
    rumble_level: f32,
    /// Onsets let through in the last STREAM_WINDOW_MS: ms, whether stream-like.
    recent: std::collections::VecDeque<(u64, bool)>,
    stream_target: f32,
    stream_until: u64,
    stream_level: f32,
    level_ms: u32,
    level_max: f32,
    levels: [f32; LEVEL_BATCH],
    level_count: usize,
    levels_live: bool,
    /// Hits wait for `resolve` instead of firing.
    pub defer: bool,
    /// When set, every onset let through: ms, prominence, went to a stream, low and high shares.
    pub trace: Option<Vec<(u64, f32, bool, f32, f32)>>,
    /// Known music is all there is: its bass is no rumble.
    pub bed: bool,
    /// The signal below 2 kHz at ~PITCH_RATE, newest at pitch_at - 1.
    pitch_buf: [f32; PITCH_BUF],
    pitch_at: usize,
    pitch_step: u32,
    pitch_left: u32,
    /// A voice-like onset waiting for its pitch: ms left, then the onset's numbers.
    voice_wait: u32,
    voice_hit: Option<(f32, f32, [f32; 3], u64)>,
    /// The last swing's onset, ms.
    swish_ms: u64,
    /// A weak onset waiting to see whether a stronger one follows: ms left, then its numbers.
    weak_wait: u32,
    weak: Option<(u64, f32, f32, [f32; 3])>,
    pending: Option<Pending>,
    now_ms: u64,
}

impl Default for Tap2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Tap2 {
    pub fn new() -> Self {
        Self {
            rate: 0,
            low_a: 0.0,
            high_a: 0.0,
            fast_atk: 0.0,
            fast_rel: 0.0,
            slow: 0.0,
            low_slow: 0.0,
            low_long_a: 0.0,
            low: 0.0,
            mid: 0.0,
            body: 0.0,
            high: 0.0,
            mid_env: 0.0,
            base: 0.0,
            low_env: 0.0,
            low_base: 0.0,
            low_long: 0.0,
            hop_left: 0,
            since_event: u32::MAX,
            rumble_ms: 0,
            since_rumble: u32::MAX,
            window_ms: 0,
            peak: 0.0,
            peak_low: 0.0,
            peak_mid: 0.0,
            peak_high: 0.0,
            peak_ms: 0,
            armed: true,
            trough: 0.0,
            onset_trough: 0.0,
            talk_ms: 0,
            since_onset: u32::MAX,
            onset_gap: 0,
            rhythm: 0,
            ref_db: -120.0,
            hit: 0.0,
            hit_peak: 0.0,
            hit_gamma: 1.0,
            hit_ms: 0,
            hit_max_ms: 0,
            hit_env: 0.0,
            rumble_level: 0.0,
            recent: std::collections::VecDeque::new(),
            stream_target: 0.0,
            stream_until: 0,
            stream_level: 0.0,
            level_ms: 0,
            level_max: 0.0,
            levels: [0.0; LEVEL_BATCH],
            level_count: 0,
            levels_live: false,
            defer: false,
            trace: None,
            bed: false,
            pitch_buf: [0.0; PITCH_BUF],
            pitch_at: 0,
            pitch_step: 1,
            pitch_left: 1,
            voice_wait: 0,
            voice_hit: None,
            swish_ms: 0,
            weak_wait: 0,
            weak: None,
            pending: None,
            now_ms: 0,
        }
    }

    pub fn now_ms(&self) -> u64 {
        self.now_ms
    }

    pub fn pending(&self) -> Option<Pending> {
        self.pending
    }

    /// Fires or drops the waiting hit.
    pub fn resolve(&mut self, fire: bool, sink: &mut impl Sink) {
        if let Some(p) = self.pending.take() {
            if fire {
                self.emit(p.onset_ms, p.blast, p.intensity, p.sharpness, [p.prominence, p.low, p.high], p.swish, sink);
            }
        }
    }

    /// Settles the waiting hit as the tagger heard it, over this detector's own guesses: a hit is
    /// a kick (never part of a stream), a swing a swish, fire a stream, anything else quiet.
    /// `gain` scales the intensity (the sound's level against the mix).
    pub fn resolve_as(&mut self, verdict: Verdict, gain: f32, sink: &mut impl Sink) {
        let Some(p) = self.pending.take() else { return };
        let intensity = (p.intensity * gain).clamp(0.0, 1.0);
        match verdict {
            Verdict::Hit => self.fire(p.blast, intensity, p.sharpness, sink),
            Verdict::Swing => self.emit(p.onset_ms, false, intensity, p.sharpness, [p.prominence, p.low, p.high], true, sink),
            Verdict::Stream => {
                let target = intensity.powf(EXPAND) * STREAM_DEPTH;
                self.stream_target = if self.now_ms < self.stream_until { 0.6 * self.stream_target + 0.4 * target } else { target };
                self.stream_until = self.now_ms + STREAM_HOLD_MS;
                self.since_event = 0;
            }
            Verdict::Quiet => {}
            Verdict::Rules | Verdict::Music => self.emit(p.onset_ms, p.blast, p.intensity, p.sharpness, [p.prominence, p.low, p.high], p.swish, sink),
        }
    }

    /// An onset let through: a hit of its own, or part of a stream.
    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, onset_ms: u64, blast: bool, intensity: f32, sharpness: f32, shape: [f32; 3], swish: bool, sink: &mut impl Sink) {
        if swish {
            self.stream_target = self.stream_target.max(SWISH_LEVEL * intensity.sqrt());
            self.stream_until = self.stream_until.max(self.now_ms + SWISH_MS);
            if self.since_rumble >= RUMBLE_EVERY_MS {
                sink.event(Kind::Rumble, intensity, 0.0);
                self.since_rumble = 0;
            }
            return;
        }
        let [prominence, low, high] = shape;
        while self.recent.front().is_some_and(|r| r.0 + STREAM_WINDOW_MS < onset_ms) {
            self.recent.pop_front();
        }
        // Once going, a stream rides out a short lull (two onsets keep it), a bigger crackle, and
        // a couple of gaps that happen to match; a crack or a real beat still breaks it.
        let going = self.now_ms < self.stream_until + STREAM_HOLD_MS;
        let (need, top, beat) = if going { (2, 2.0 * STREAM_PROMINENCE, RHYTHM_ONSETS + 2) } else { (STREAM_ONSETS, STREAM_PROMINENCE, RHYTHM_ONSETS) };
        let flat = prominence < top && high < STREAM_HIGH;
        self.recent.push_back((onset_ms, flat));
        let stream = !blast && flat && self.recent.iter().filter(|r| r.1).count() >= need && self.rhythm < beat;
        if let Some(t) = self.trace.as_mut() {
            t.push((onset_ms, prominence, stream, low, high));
        }
        if !stream {
            self.fire(blast, intensity, sharpness, sink);
            return;
        }
        let target = intensity.powf(EXPAND) * STREAM_DEPTH;
        self.stream_target = if self.now_ms < self.stream_until { 0.6 * self.stream_target + 0.4 * target } else { target };
        self.stream_until = self.now_ms + STREAM_HOLD_MS;
        self.since_event = 0;
        if self.since_rumble >= RUMBLE_EVERY_MS {
            sink.event(Kind::Rumble, intensity, 0.0);
            self.since_rumble = 0;
        }
    }

    fn fire(&mut self, blast: bool, intensity: f32, sharpness: f32, sink: &mut impl Sink) {
        sink.event(if blast { Kind::Blast } else { Kind::Impact }, intensity, sharpness);
        if blast {
            self.start_hit(1.0, BLAST_GAMMA, BLAST_MAX_MS);
        } else {
            self.start_hit(intensity.powf(EXPAND), if sharpness > 0.5 { SHARP_GAMMA } else { DULL_GAMMA }, HIT_MAX_MS);
        }
        self.since_event = 0;
    }

    fn start_hit(&mut self, intensity: f32, gamma: f32, max_ms: u32) {
        self.hit = intensity;
        self.hit_peak = self.body.max(1e-6);
        self.hit_gamma = gamma;
        self.hit_ms = 0;
        self.hit_max_ms = max_ms;
        self.hit_env = intensity;
    }

    pub fn flush_levels(&mut self, sink: &mut impl Sink) {
        if self.level_count == 0 {
            return;
        }
        let any = self.levels[..self.level_count].iter().any(|&l| l > 0.0);
        if any || self.levels_live {
            sink.levels(&self.levels[..self.level_count]);
        }
        self.levels_live = self.levels[self.level_count - 1] > 0.0;
        self.level_count = 0;
    }

    /// Mono samples at `rate`.
    pub fn feed(&mut self, sens: f32, x: &[f32], rate: u32, sink: &mut impl Sink) {
        if rate < 8000 {
            return;
        }
        if rate != self.rate {
            self.rate = rate;
            let r = rate as f32;
            let pi = std::f32::consts::PI;
            self.low_a = 1.0 - (-2.0 * pi * 200.0 / r).exp();
            self.high_a = 1.0 - (-2.0 * pi * 2000.0 / r).exp();
            self.fast_atk = 1.0 - (-1.0 / (0.0005 * r)).exp();
            self.fast_rel = 1.0 - (-1.0 / (0.020 * r)).exp();
            self.slow = 1.0 - (-1.0 / (0.250 * r)).exp();
            self.low_slow = 1.0 - (-1.0 / (0.080 * r)).exp();
            self.low_long_a = 1.0 - (-1.0 / (2.0 * r)).exp();
            self.hop_left = rate / 1000;
            self.pitch_step = (rate / PITCH_RATE).max(1);
            self.pitch_left = self.pitch_step;
        }
        for &x in x {
            self.low += (x - self.low) * self.low_a;
            self.mid += (x - self.mid) * self.high_a; // < 2 kHz
            let low = self.low.abs();
            let body = low + 0.6 * (self.mid - self.low).abs();
            let high = (x - self.mid).abs();
            let mid_band = (self.mid - self.low).abs();
            self.mid_env += (mid_band - self.mid_env) * if mid_band > self.mid_env { self.fast_atk } else { self.fast_rel };
            self.body += (body - self.body) * if body > self.body { self.fast_atk } else { self.fast_rel };
            self.high += (high - self.high) * if high > self.high { self.fast_atk } else { self.fast_rel };
            self.base += (self.body - self.base) * self.slow;
            self.low_env += (low - self.low_env) * if low > self.low_env { self.fast_atk } else { self.fast_rel };
            self.low_base += (self.low_env - self.low_base) * self.low_slow;
            self.low_long += (self.low_env - self.low_long) * self.low_long_a;
            self.pitch_left -= 1;
            if self.pitch_left == 0 {
                self.pitch_left = self.pitch_step;
                self.pitch_buf[self.pitch_at] = self.mid;
                self.pitch_at = (self.pitch_at + 1) % PITCH_BUF;
            }
            self.hop_left -= 1;
            if self.hop_left == 0 {
                self.hop_left = self.rate / 1000;
                self.hop(sens, sink);
            }
        }
        self.flush_levels(sink);
    }

    /// Normalised autocorrelation at the best pitch lag over the last PITCH_SPAN samples.
    fn pitch(&self) -> f32 {
        let rate = self.rate / self.pitch_step;
        let x = |k: usize| self.pitch_buf[(self.pitch_at + PITCH_BUF - 1 - k) % PITCH_BUF];
        let (lo, hi) = ((rate / 400) as usize, ((rate / 80) as usize).min(PITCH_BUF - PITCH_SPAN));
        let e0: f32 = (0..PITCH_SPAN).map(|k| x(k) * x(k)).sum();
        (lo..=hi)
            .map(|lag| {
                let (mut c, mut e1) = (0f32, 0f32);
                for k in 0..PITCH_SPAN {
                    c += x(k) * x(k + lag);
                    e1 += x(k + lag) * x(k + lag);
                }
                c / (e0 * e1).sqrt().max(1e-12)
            })
            .fold(0f32, f32::max)
    }

    /// A hit (or a swish) found; a weak one first waits for a stronger one behind it.
    #[allow(clippy::too_many_arguments)]
    fn hit(&mut self, onset_ms: u64, blast: bool, intensity: f32, sharpness: f32, shape: [f32; 3], swish: bool, sink: &mut impl Sink) {
        if let Some((ms, i, s, sh)) = self.weak.take() {
            if intensity <= i || swish {
                self.release(ms, false, i, s, sh, false, sink);
            }
        }
        if !blast && !swish && intensity < WEAK_INTENSITY {
            self.weak = Some((onset_ms, intensity, sharpness, shape));
            self.weak_wait = WEAK_WAIT_MS;
            return;
        }
        self.release(onset_ms, blast, intensity, sharpness, shape, swish, sink);
    }

    /// Lets a hit (or a swish) through: straight away, or to the catalog's verdict.
    #[allow(clippy::too_many_arguments)]
    fn release(&mut self, onset_ms: u64, blast: bool, intensity: f32, sharpness: f32, shape: [f32; 3], swish: bool, sink: &mut impl Sink) {
        if self.defer {
            // A newer onset replaces a hit still waiting.
            let [prominence, low, high] = shape;
            self.pending = Some(Pending { onset_ms, blast, intensity, sharpness, prominence, low, high, swish });
        } else {
            self.emit(onset_ms, blast, intensity, sharpness, shape, swish, sink);
        }
    }

    /// Once per millisecond.
    fn hop(&mut self, sens: f32, sink: &mut impl Sink) {
        if self.weak_wait > 0 {
            self.weak_wait -= 1;
            if self.weak_wait == 0 {
                if let Some((ms, i, s, sh)) = self.weak.take() {
                    self.release(ms, false, i, s, sh, false, sink);
                }
            }
        }
        if self.voice_wait > 0 {
            self.voice_wait -= 1;
            if self.voice_wait == 0 {
                if let Some((intensity, sharpness, shape, onset_ms)) = self.voice_hit.take() {
                    if self.pitch() > VOICE_PITCH {
                        self.talk_ms = TALK_HOLD_MS;
                    } else {
                        self.swish_ms = self.now_ms;
                        self.hit(onset_ms, false, intensity, sharpness, shape, true, sink);
                    }
                }
            }
        }
        self.now_ms += 1;
        self.since_event = self.since_event.saturating_add(1);
        self.since_onset = self.since_onset.saturating_add(1);
        self.since_rumble = self.since_rumble.saturating_add(1);
        self.talk_ms = self.talk_ms.saturating_sub(1);
        self.ref_db -= REF_DECAY_DB_PER_MS;

        // Sensitivity lowers the floor and the jump needed: 1.0 is the tuned default.
        let floor_db = ONSET_FLOOR_DB - 10.0 * sens.max(0.05).log2();
        if self.window_ms > 0 {
            if self.body > self.peak {
                self.peak = self.body;
                self.peak_low = self.low_env;
                self.peak_ms = WINDOW_MS + 1 - self.window_ms;
            }
            self.peak_mid = self.peak_mid.max(self.mid_env);
            self.peak_high = self.peak_high.max(self.high);
            self.window_ms -= 1;
            if self.window_ms == 0 {
                let total = self.peak_low + self.peak_mid + self.peak_high + 1e-6;
                let (mid, high) = (self.peak_mid / total, self.peak_high / total);
                let firing = self.rhythm >= RHYTHM_ONSETS;
                let thump = self.peak_low / total >= THUMP_LOW && self.peak_ms <= THUMP_PEAK_MS;
                let voice_like = !firing && !thump && mid > VOICE_MID && high < VOICE_HIGH;
                let sfx = firing || thump || (mid < TALK_SFX_MID && high > TALK_SFX_HIGH);
                // Inside a line, a voice-like onset is the next syllable; outside one it waits for
                // its pitch.
                // An onset that is no thump or crack while one waits for its pitch is the same sound.
                let voice = (voice_like && self.talk_ms > 0) || (!sfx && self.voice_wait > 0);
                let since_swish = self.now_ms.saturating_sub(self.swish_ms);
                let swishing = self.swish_ms > 0 && !sfx && (since_swish < SWISH_MS || (voice_like && since_swish < SWISH_HOLD_MS));
                let wait = voice_like && self.talk_ms == 0 && !swishing && self.voice_wait == 0;
                if voice && self.voice_wait == 0 {
                    self.talk_ms = TALK_HOLD_MS;
                }
                if firing {
                    self.talk_ms = 0;
                }
                // Dialogue: no event, no refractory.
                if !voice && (self.talk_ms == 0 || sfx) {
                    let peak_db = dbfs(self.peak);
                    self.ref_db = self.ref_db.max(peak_db);
                    let rel = (1.0 - (self.ref_db - peak_db) / RANGE_DB).clamp(0.0, 1.0);
                    let loud = ((peak_db - floor_db) / 30.0).clamp(0.0, 1.0);
                    let intensity = (0.15 + 0.6 * rel + 0.25 * loud).clamp(0.15, 1.0);
                    let sharpness = (high * 2.0).clamp(0.0, 1.0);
                    let blast = rel > 0.9 && loud > 0.7 && self.peak_low / total > 0.4 && self.since_event >= BLAST_QUIET_MS;
                    let shape = [self.peak / self.onset_trough.max(1e-6), self.peak_low / total, high];
                    let onset_ms = self.now_ms - WINDOW_MS as u64;
                    if swishing {
                        self.swish_ms = self.now_ms;
                        self.hit(onset_ms, false, intensity, sharpness, shape, true, sink);
                    } else if wait {
                        self.voice_wait = PITCH_WAIT_MS;
                        self.voice_hit = Some((intensity, sharpness, shape, onset_ms));
                    } else {
                        self.hit(onset_ms, blast, intensity, sharpness, shape, false, sink);
                    }
                }
                self.trough = self.body;
            }
        } else {
            if !self.armed && self.body < REARM * self.peak {
                self.armed = true;
            }
            self.trough = self.trough.min(self.body);
            let ratio = self.body / self.base.max(1e-5);
            let onset = self.armed && ratio > ONSET_RATIO / sens.sqrt() && self.since_event >= REFRACTORY_MS;
            // From the last onset, not event: a first shot taken for a voice still lets the rest in.
            let shot = self.since_onset < BURST_MS
                && self.since_onset >= BURST_REFRACTORY_MS
                && self.body > BURST_RISE / sens.sqrt() * self.trough
                && self.body > BURST_LEVEL * self.peak;
            if (onset || shot) && dbfs(self.body) > floor_db {
                self.onset_trough = self.trough;
                let gap = self.since_onset;
                let steady = gap <= RHYTHM_MAX_MS
                    && self.onset_gap > 0
                    && (gap as f32 - self.onset_gap as f32).abs() <= RHYTHM_TOLERANCE * self.onset_gap as f32;
                self.rhythm = if steady { self.rhythm + 1 } else { 0 };
                self.onset_gap = if gap <= RHYTHM_MAX_MS { gap } else { 0 };
                self.since_onset = 0;
                self.peak = self.body;
                self.peak_low = self.low_env;
                self.peak_mid = self.mid_env;
                self.peak_high = self.high;
                self.peak_ms = 0;
                self.window_ms = WINDOW_MS;
                self.armed = false;
            }
        }

        let rising = self.low_base > RUMBLE_RISE * self.low_long || (self.rumble_ms > 0 && self.rumble_ms < RUMBLE_MAX_MS);
        let mut rumble = 0.0;
        // A low voice fills the low band too: no rumble while someone talks.
        if self.talk_ms == 0
            && !self.bed
            && self.since_event >= RUMBLE_AFTER_HIT_MS
            && rising
            && self.rumble_ms < RUMBLE_MAX_MS
            && dbfs(self.low_base) > RUMBLE_FLOOR_DB - 6.0 * sens.max(0.05).log2()
        {
            self.rumble_ms += 1;
            if self.rumble_ms >= RUMBLE_HOLD_MS {
                let level = ((dbfs(self.low_base) - RUMBLE_FLOOR_DB + 12.0) / 18.0).clamp(0.0, 1.0);
                rumble = RUMBLE_DEPTH * level;
                if self.since_rumble >= RUMBLE_EVERY_MS {
                    sink.event(Kind::Rumble, level, 0.0);
                    self.since_rumble = 0;
                }
            }
        } else {
            self.rumble_ms = 0;
        }
        self.rumble_level += (rumble - self.rumble_level) * if rumble > self.rumble_level { RUMBLE_ATTACK } else { RUMBLE_RELEASE };
        if self.rumble_level < 0.005 {
            self.rumble_level = 0.0;
        }

        let want = if self.now_ms < self.stream_until { self.stream_target } else { 0.0 };
        self.stream_level += (want - self.stream_level) * if want > self.stream_level { STREAM_ATTACK } else { STREAM_RELEASE };
        if self.stream_level < 0.005 {
            self.stream_level = 0.0;
        }
        let mut env = self.rumble_level.max(self.stream_level);
        if self.hit > 0.0 {
            self.hit_ms += 1;
            let mut h = self.hit * (self.body / self.hit_peak).min(1.0).powf(self.hit_gamma);
            let fade_from = self.hit_max_ms * 7 / 10;
            if self.hit_ms > fade_from {
                h *= 1.0 - (self.hit_ms - fade_from) as f32 / (self.hit_max_ms - fade_from) as f32;
            }
            let h = h.min(self.hit_env);
            self.hit_env = h;
            if h < HIT_END || self.hit_ms >= self.hit_max_ms {
                self.hit = 0.0;
            } else {
                env = env.max(h);
            }
        }
        self.level_max = self.level_max.max(env);
        self.level_ms += 1;
        if self.level_ms >= LEVEL_STEP_MS {
            self.levels[self.level_count] = self.level_max;
            self.level_count += 1;
            self.level_ms = 0;
            self.level_max = 0.0;
            if self.level_count == LEVEL_BATCH {
                self.flush_levels(sink);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    #[derive(Default)]
    struct Rec {
        hits: usize,
        levels: Vec<f32>,
    }

    impl Sink for Rec {
        fn event(&mut self, kind: Kind, _: f32, _: f32) {
            if kind != Kind::Rumble {
                self.hits += 1;
            }
        }
        fn levels(&mut self, l: &[f32]) {
            self.levels.extend_from_slice(l);
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn run(x: &[f32]) -> Rec {
        let mut tap = Tap2::new();
        let mut rec = Rec::default();
        for c in x.chunks((RATE / 200) as usize) {
            let before = rec.levels.len();
            tap.feed(1.0, c, RATE, &mut rec);
            if rec.levels.len() == before {
                rec.levels.push(0.0);
            }
        }
        rec
    }

    /// A flamethrower: a dull roar that never falls, crackling off the beat every 50-150 ms.
    fn flame(secs: f32) -> Vec<f32> {
        let mut r = Lcg(7);
        let n = (secs * RATE as f32) as usize;
        let (mut lp, mut lp2, mut next, mut bump) = (0f32, 0f32, 0usize, 0f32);
        (0..n)
            .map(|i| {
                // Two poles at ~300 Hz: a roar, little above 2 kHz.
                lp += (r.next() * 2.0 - 1.0 - lp) * 0.04;
                lp2 += (lp - lp2) * 0.04;
                let lp = lp2 * 4.0;
                if i >= next {
                    bump = 1.3;
                    next = i + RATE as usize * (50 + (r.next() * 100.0) as usize) / 1000;
                }
                bump *= 0.9985; // ~15 ms
                lp * 0.6 * (1.0 + bump)
            })
            .collect()
    }

    /// Shots at irregular gaps, each a crack and a thump out of near silence.
    fn shots(secs: f32) -> (Vec<f32>, usize) {
        let mut r = Lcg(11);
        let n = (secs * RATE as f32) as usize;
        let mut x = vec![0f32; n];
        let mut at = RATE as usize / 10;
        let mut count = 0;
        while at + RATE as usize / 10 < n {
            for k in 0..RATE as usize / 10 {
                let t = k as f32 / RATE as f32;
                let crack = (r.next() * 2.0 - 1.0) * (-t / 0.004).exp();
                let thump = (2.0 * std::f32::consts::PI * 70.0 * t).sin() * (-t / 0.03).exp();
                x[at + k] += 0.5 * crack + 0.4 * thump;
            }
            count += 1;
            at += RATE as usize * (90 + (r.next() * 110.0) as usize) / 1000;
        }
        (x, count)
    }

    #[test]
    fn a_flame_is_one_sustained_level() {
        let rec = run(&flame(3.0));
        assert!(rec.hits <= 6, "{} kicks", rec.hits);
        // 0.5-2.5 s: driven throughout, and smooth step to step.
        let body = &rec.levels[100..500];
        assert!(body.iter().all(|&l| l > 0.1), "drive gaps in the flame");
        let jump = body.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / body.len() as f32;
        assert!(jump < 0.01, "level jumps {jump} per step");
    }

    #[test]
    fn shots_off_the_beat_stay_rounds() {
        let (x, count) = shots(3.0);
        let rec = run(&x);
        assert!(rec.hits * 10 >= count * 8, "{} of {count} shots felt", rec.hits);
    }
}
