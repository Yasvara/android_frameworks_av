//! Worker side, one per armed track: frames → onsets → decision → haptic envelope, 5 ms steps.

use crate::mel::{Frame, Mel};
use crate::model::{embed, EmbedCfg, CTX_MAX};
use crate::onset::{Onset, OnsetCfg};
use crate::verify::{Background, Policy, Verifier};

const HIST: usize = 16;
const PENDING: usize = 4;
/// About the last 400 onsets: a few minutes of play.
const BACKGROUND_WINDOW: f32 = 400.0;
// Envelope, as in v2: a hit follows the body down, (body / body at the hit)^gamma.
const GAMMA: f32 = 1.6;
const HIT_MAX_FRAMES: u32 = 70;
const HIT_END: f32 = 0.03;
const EXPAND: f32 = 1.5;

#[derive(Clone, Copy, Debug)]
pub struct Detection {
    /// Frame index of the onset and of the decision.
    pub onset: u64,
    pub decided: u64,
    pub flux: f32,
    /// Peak frame level over the patch, dB.
    pub level: f32,
    /// Log-likelihood ratio per dimension, taught class over background.
    pub score: f32,
    pub class: usize,
    pub fire: bool,
}

/// What a game was taught, and how its decisions are made.
pub struct Classifier {
    pub cfg: EmbedCfg,
    pub verifier: Verifier,
    pub policy: Policy,
}

pub struct Track {
    mel: Mel,
    onset: Onset,
    hist: [Frame; HIST],
    t: u64,
    pending: [(u64, f32); PENDING],
    npending: usize,
    classifier: Option<Classifier>,
    background: Background,
    last_fire: Option<u64>,
    /// Patch length without a classifier (teaching: collect patches for any ctx).
    pub raw_ctx: usize,
    hit: f32,
    hit_peak: f32,
    hit_frames: u32,
}

impl Track {
    pub fn new(rate: u32, onset: OnsetCfg, classifier: Option<Classifier>) -> Self {
        let dim = classifier.as_ref().map_or(1, |c| embed(&[Frame::default(); CTX_MAX][..c.cfg.ctx], &Frame::default(), &c.cfg).n);
        Self {
            mel: Mel::new(rate),
            onset: Onset::new(onset),
            hist: [Frame::default(); HIST],
            t: 0,
            pending: [(0, 0.0); PENDING],
            npending: 0,
            classifier,
            background: Background::new(dim, BACKGROUND_WINDOW),
            last_fire: None,
            raw_ctx: 4,
            hit: 0.0,
            hit_peak: 1e-6,
            hit_frames: 0,
        }
    }

    pub fn ctx(&self) -> usize {
        self.classifier.as_ref().map_or(self.raw_ctx, |c| c.cfg.ctx).min(CTX_MAX)
    }

    /// The patch and pre-onset frame of an onset at `t0`, once ctx frames are in.
    pub fn patch(&self, t0: u64, ctx: usize) -> ([Frame; CTX_MAX], Frame) {
        let mut p = [Frame::default(); CTX_MAX];
        for (k, f) in p.iter_mut().take(ctx).enumerate() {
            *f = self.hist[((t0 + k as u64) % HIST as u64) as usize];
        }
        let before = if t0 >= 2 { self.hist[((t0 - 2) % HIST as u64) as usize] } else { Frame::default() };
        (p, before)
    }

    /// Feeds decimated samples. `on_detect` gets every onset once decided; `on_level` one haptic
    /// level per 5 ms frame. `on_raw` sees each onset's patch (for teaching).
    pub fn push(
        &mut self,
        x: &[f32],
        on_detect: &mut impl FnMut(&Detection),
        on_level: &mut impl FnMut(f32),
        on_raw: &mut impl FnMut(&Self, u64, f32),
    ) {
        let mut frames = [Frame::default(); 8];
        let mut nf = 0;
        let mut rest = x;
        while !rest.is_empty() {
            let take = rest.len().min(self.mel.hop() * frames.len());
            self.mel.push(&rest[..take], |f| {
                frames[nf] = *f;
                nf += 1;
            });
            rest = &rest[take..];
            for f in &frames[..nf] {
                self.frame(f, on_detect, on_level, on_raw);
            }
            nf = 0;
        }
    }

    fn frame(
        &mut self,
        f: &Frame,
        on_detect: &mut impl FnMut(&Detection),
        on_level: &mut impl FnMut(f32),
        on_raw: &mut impl FnMut(&Self, u64, f32),
    ) {
        let t = self.t;
        self.hist[(t % HIST as u64) as usize] = *f;
        if let Some(flux) = self.onset.step(f) {
            if self.npending < PENDING {
                self.pending[self.npending] = (t, flux);
                self.npending += 1;
            }
        }
        let ctx = self.ctx();
        let mut i = 0;
        while i < self.npending {
            let (t0, flux) = self.pending[i];
            if t < t0 + ctx as u64 - 1 {
                i += 1;
                continue;
            }
            on_raw(self, t0, flux);
            let (patch, before) = self.patch(t0, ctx);
            let level = patch[..ctx].iter().fold(-120f32, |a, p| a.max(p.level));
            let mut d = Detection { onset: t0, decided: t, flux, level, score: f32::NEG_INFINITY, class: 0, fire: false };
            if let Some(c) = &self.classifier {
                let e = embed(&patch[..ctx], &before, &c.cfg);
                let (score, class) = c.verifier.score(&e, &self.background);
                self.background.observe(&e);
                let p = &c.policy;
                let in_burst = self.last_fire.is_some_and(|l| t0 - l <= p.hold_frames && t0 - l >= 10);
                let need = if in_burst { p.stay } else { p.enter };
                let gate = level >= c.verifier.classes[class].level - p.range_db;
                let spaced = self.last_fire.is_none_or(|l| t0 - l >= 10);
                d.score = score;
                d.class = class;
                d.fire = score >= need && gate && spaced;
                if d.fire {
                    self.last_fire = Some(t0);
                    let rel = (1.0 - (c.verifier.classes[class].level - level).max(0.0) / p.range_db).clamp(0.15, 1.0);
                    self.hit = rel.powf(EXPAND);
                    self.hit_peak = patch[..ctx].iter().fold(1e-6f32, |a, p| a.max(p.body));
                    self.hit_frames = 0;
                }
            }
            on_detect(&d);
            self.pending[i] = self.pending[self.npending - 1];
            self.npending -= 1;
        }
        let mut level = 0.0;
        if self.hit > 0.0 {
            self.hit_frames += 1;
            let mut h = self.hit * (f.body / self.hit_peak).min(1.0).powf(GAMMA);
            let fade_from = HIT_MAX_FRAMES * 7 / 10;
            if self.hit_frames > fade_from {
                h *= 1.0 - (self.hit_frames - fade_from) as f32 / (HIT_MAX_FRAMES - fade_from) as f32;
            }
            if h < HIT_END || self.hit_frames >= HIT_MAX_FRAMES {
                self.hit = 0.0;
            } else {
                level = h;
            }
        }
        on_level(level);
        self.t += 1;
    }
}
