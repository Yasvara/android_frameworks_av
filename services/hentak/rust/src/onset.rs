//! Onsets by spectral flux: the mean rise in dB per band over the frame 10 ms earlier, against
//! the louder of its neighbour bands (so vibrato and pitch slides do not count). Permissive on
//! purpose: the classifier decides what an onset was.

use crate::mel::{Frame, BANDS};

const LAG: usize = 2;
const FLOOR_DB: f32 = -90.0;

#[derive(Clone, Copy)]
pub struct OnsetCfg {
    /// Flux needed regardless of context, dB.
    pub min_flux: f32,
    /// Flux needed relative to its recent mean.
    pub rel: f32,
    /// Frame level needed, dB.
    pub floor_db: f32,
    /// Frames between onsets: 8 = 40 ms, faster than any gun's cycle (1500 rpm).
    pub refractory: u32,
}

impl Default for OnsetCfg {
    fn default() -> Self {
        Self { min_flux: 2.5, rel: 2.0, floor_db: -60.0, refractory: 8 }
    }
}

pub struct Onset {
    cfg: OnsetCfg,
    hist: [[f32; BANDS]; LAG],
    at: usize,
    mean: f32,
    armed: bool,
    since: u32,
}

impl Onset {
    pub fn new(cfg: OnsetCfg) -> Self {
        Self { cfg, hist: [[FLOOR_DB; BANDS]; LAG], at: 0, mean: 0.0, armed: true, since: u32::MAX / 2 }
    }

    /// Returns the flux when this frame starts an onset.
    pub fn step(&mut self, f: &Frame) -> Option<f32> {
        let reference = &self.hist[self.at];
        let mut flux = 0.0;
        for b in 0..BANDS {
            let lo = b.saturating_sub(1);
            let hi = (b + 1).min(BANDS - 1);
            let r = reference[lo].max(reference[b]).max(reference[hi]);
            flux += (f.mel[b].max(FLOOR_DB) - r).max(0.0);
        }
        flux /= BANDS as f32;
        for b in 0..BANDS {
            self.hist[self.at][b] = f.mel[b].max(FLOOR_DB);
        }
        self.at = (self.at + 1) % LAG;

        let threshold = self.cfg.min_flux.max(self.cfg.rel * self.mean);
        // 300 ms mean, frozen during the onset itself so a burst does not raise its own bar.
        if flux < threshold {
            self.mean += (flux - self.mean) * (1.0 / 60.0);
        }
        self.since = self.since.saturating_add(1);
        if !self.armed && flux < 0.5 * threshold {
            self.armed = true;
        }
        if self.armed && flux > threshold && f.level > self.cfg.floor_db && self.since >= self.cfg.refractory {
            self.armed = false;
            self.since = 0;
            return Some(flux);
        }
        None
    }
}
