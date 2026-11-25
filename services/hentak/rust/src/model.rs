//! What a taught sound looks like: log-mel patches from the onset on, matched by cosine to a few
//! prototypes learnt from one teaching take.

use crate::mel::{Frame, BANDS};

pub const CTX_MAX: usize = 8;
pub const DIM_MAX: usize = 2 * BANDS * CTX_MAX;
pub const PROTOS_MAX: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Kind {
    /// The patch itself, mean removed: spectral and temporal shape, gain-free.
    Shape,
    /// What the event added over the frame before it, in dB: background-free.
    Added,
    /// Both halves.
    Both,
}

#[derive(Clone, Copy, Debug)]
pub struct EmbedCfg {
    /// Frames from the onset frame on; the decision waits ctx - 1 frames (5 ms each).
    pub ctx: usize,
    pub kind: Kind,
}

impl Default for EmbedCfg {
    fn default() -> Self {
        Self { ctx: 4, kind: Kind::Both }
    }
}

#[derive(Clone, Copy)]
pub struct Emb {
    pub v: [f32; DIM_MAX],
    pub n: usize,
}

impl Emb {
    pub fn dot(&self, o: &Emb) -> f32 {
        self.v[..self.n].iter().zip(&o.v[..o.n]).map(|(a, b)| a * b).sum()
    }

    fn normalize(&mut self, from: usize, to: usize) {
        let s = self.v[from..to].iter().map(|x| x * x).sum::<f32>().sqrt();
        if s > 1e-9 {
            self.v[from..to].iter_mut().for_each(|x| *x /= s);
        }
    }
}

/// `patch` holds ctx frames from the onset on; `before` is the frame 10 ms ahead of the onset.
pub fn embed(patch: &[Frame], before: &Frame, cfg: &EmbedCfg) -> Emb {
    let mut e = Emb { v: [0.0; DIM_MAX], n: 0 };
    let len = BANDS * patch.len();
    if matches!(cfg.kind, Kind::Shape | Kind::Both) {
        let top = patch.iter().flat_map(|f| f.mel.iter()).fold(-200f32, |a, &b| a.max(b));
        let floor = top - 60.0;
        for (k, f) in patch.iter().enumerate() {
            for b in 0..BANDS {
                e.v[k * BANDS + b] = f.mel[b].max(floor);
            }
        }
        let mean = e.v[..len].iter().sum::<f32>() / len as f32;
        e.v[..len].iter_mut().for_each(|x| *x -= mean);
        e.normalize(0, len);
        e.n = len;
    }
    if matches!(cfg.kind, Kind::Added | Kind::Both) {
        let at = e.n;
        for (k, f) in patch.iter().enumerate() {
            for b in 0..BANDS {
                e.v[at + k * BANDS + b] = (f.mel[b] - before.mel[b].max(-90.0)).max(0.0);
            }
        }
        e.normalize(at, at + len);
        e.n = at + len;
    }
    if cfg.kind == Kind::Both {
        let r = std::f32::consts::FRAC_1_SQRT_2;
        e.v[..e.n].iter_mut().for_each(|x| *x *= r);
    }
    e
}

#[derive(Clone, Copy)]
pub struct Proto {
    pub v: Emb,
    /// Cosine needed to count as this sound.
    pub thresh: f32,
    /// Median peak level of the taught hits, dB: the player's own shot is the loudest copy.
    pub level: f32,
    pub members: u32,
}

#[derive(Clone)]
pub struct Model {
    pub cfg: EmbedCfg,
    pub protos: Vec<Proto>,
}

pub struct Sample {
    pub e: Emb,
    pub level: f32,
}

fn cluster_sim() -> f32 {
    std::env::var("CLUSTER_SIM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.80)
}
const THRESH_FLOOR: f32 = 0.70;
const THRESH_MARGIN: f32 = 0.06;

impl Model {
    /// One teaching take: the player fires for a few seconds. The sound repeated most, weighted by
    /// loudness, becomes prototype 1; other big clusters (alternating layers, a second fire mode)
    /// become the rest.
    pub fn teach(samples: &[Sample], cfg: EmbedCfg) -> Option<Model> {
        if samples.is_empty() {
            return None;
        }
        let n = samples.len();
        let cluster = cluster_sim();
        let sim: Vec<f32> = (0..n * n).map(|ij| samples[ij / n].e.dot(&samples[ij % n].e)).collect();
        let weight: Vec<f32> = samples.iter().map(|s| 10f32.powf(s.level / 20.0)).collect();
        let mut left: Vec<bool> = vec![true; n];
        let mut protos = Vec::new();
        let min_members = (n / 6).max(if n >= 6 { 3 } else { 1 });
        while protos.len() < PROTOS_MAX {
            let mut best = None;
            for i in (0..n).filter(|&i| left[i]) {
                let score: f32 = (0..n).filter(|&j| left[j] && sim[i * n + j] >= cluster).map(|j| weight[j]).sum();
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((i, score));
                }
            }
            let Some((medoid, _)) = best else { break };
            let members: Vec<usize> = (0..n).filter(|&j| left[j] && sim[medoid * n + j] >= cluster).collect();
            if members.len() < min_members && !protos.is_empty() {
                break;
            }
            let mut mean = Emb { v: [0.0; DIM_MAX], n: samples[0].e.n };
            for &j in &members {
                for d in 0..mean.n {
                    mean.v[d] += samples[j].e.v[d];
                }
                left[j] = false;
            }
            mean.normalize(0, mean.n);
            let mut sims: Vec<f32> = members.iter().map(|&j| samples[j].e.dot(&mean)).collect();
            sims.sort_by(f32::total_cmp);
            let mut levels: Vec<f32> = members.iter().map(|&j| samples[j].level).collect();
            levels.sort_by(f32::total_cmp);
            protos.push(Proto {
                v: mean,
                thresh: (sims[sims.len() / 10] - THRESH_MARGIN).max(THRESH_FLOOR),
                level: levels[levels.len() / 2],
                members: members.len() as u32,
            });
        }
        Some(Model { cfg, protos })
    }

    /// Best cosine and its prototype.
    pub fn score(&self, e: &Emb) -> (f32, usize) {
        self.protos.iter().enumerate().map(|(i, p)| (p.v.dot(e), i)).fold((-1.0, 0), |a, b| if b.0 > a.0 { b } else { a })
    }

    /// Tiny wire format for the model (system_server → audioserver): int8 per dimension.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"HTK3".to_vec();
        out.push(self.cfg.ctx as u8);
        out.push(match self.cfg.kind {
            Kind::Shape => 0,
            Kind::Added => 1,
            Kind::Both => 2,
        });
        out.push(self.protos.len() as u8);
        for p in &self.protos {
            out.extend_from_slice(&p.thresh.to_le_bytes());
            out.extend_from_slice(&p.level.to_le_bytes());
            out.extend_from_slice(&(p.v.n as u16).to_le_bytes());
            let scale = p.v.v[..p.v.n].iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-9);
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend(p.v.v[..p.v.n].iter().map(|&x| ((x / scale) * 127.0).round() as i8 as u8));
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Model> {
        if b.len() < 7 || &b[..4] != b"HTK3" {
            return None;
        }
        let kind = match b[5] {
            0 => Kind::Shape,
            1 => Kind::Added,
            2 => Kind::Both,
            _ => return None,
        };
        let cfg = EmbedCfg { ctx: (b[4] as usize).clamp(1, CTX_MAX), kind };
        let mut at = 7;
        let mut protos = Vec::new();
        let f32_at = |at: usize| b.get(at..at + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap()));
        for _ in 0..b[6].min(PROTOS_MAX as u8) {
            let thresh = f32_at(at)?;
            let level = f32_at(at + 4)?;
            let n = u16::from_le_bytes(b.get(at + 8..at + 10)?.try_into().ok()?) as usize;
            let scale = f32_at(at + 10)?;
            at += 14;
            if n > DIM_MAX {
                return None;
            }
            let mut v = Emb { v: [0.0; DIM_MAX], n };
            for (d, &q) in b.get(at..at + n)?.iter().enumerate() {
                v.v[d] = (q as i8) as f32 / 127.0 * scale;
            }
            v.normalize(0, n);
            at += n;
            protos.push(Proto { v, thresh, level, members: 0 });
        }
        Some(Model { cfg, protos })
    }
}
