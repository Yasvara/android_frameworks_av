//! Teach-and-verify by likelihood ratio: each taught sound is a diagonal Gaussian over the onset
//! embedding, scored against a background Gaussian of every onset the game makes (learnt as it
//! plays, no labels). Cosine alone ignores which dimensions a gun keeps steady; the variances
//! carry that.

use crate::model::{Emb, DIM_MAX};

/// Pseudo-samples of background variance mixed into a taught class: a take has few hits.
const SHRINK: f32 = 4.0;
const VAR_FLOOR: f32 = 1e-5;
/// Teaching keeps the take's onsets within this many dB of its loudest: the player's own gun.
pub const TEACH_RANGE_DB: f32 = 10.0;

#[derive(Clone, Copy)]
pub struct Diag {
    pub mean: [f32; DIM_MAX],
    pub var: [f32; DIM_MAX],
    pub n: usize,
    /// Median level of the taught hits, dB.
    pub level: f32,
    /// Lowest per-dimension log-likelihood a taught hit had, less a margin: anything farther out
    /// is a sound neither class has seen, whatever the ratio says.
    pub floor: f32,
    log_det: f32,
}

impl Diag {
    pub fn new(n: usize) -> Self {
        Self { mean: [0.0; DIM_MAX], var: [0.0; DIM_MAX], n, level: 0.0, floor: f32::NEG_INFINITY, log_det: 0.0 }
    }

    /// After mean and var change.
    pub fn finish(&mut self) {
        self.log_det = self.var[..self.n].iter().map(|v| v.max(VAR_FLOOR).ln()).sum();
    }

    pub fn log_like(&self, e: &Emb) -> f32 {
        let q: f32 = (0..self.n).map(|d| (e.v[d] - self.mean[d]).powi(2) / self.var[d].max(VAR_FLOOR)).sum();
        -0.5 * (q + self.log_det)
    }
}

/// Running background over all onsets: exponential, about the last `window` onsets.
pub struct Background {
    pub g: Diag,
    count: f32,
    window: f32,
}

impl Background {
    pub fn new(dim: usize, window: f32) -> Self {
        Self {
            g: Diag { mean: [0.0; DIM_MAX], var: [1.0 / dim as f32; DIM_MAX], n: dim, level: 0.0, floor: f32::NEG_INFINITY, log_det: 0.0 },
            count: 0.0,
            window,
        }
    }

    pub fn observe(&mut self, e: &Emb) {
        self.count += 1.0;
        let a = 1.0 / self.count.min(self.window);
        for d in 0..self.g.n {
            let delta = e.v[d] - self.g.mean[d];
            self.g.mean[d] += a * delta;
            self.g.var[d] += a * (delta * delta * (1.0 - a) - self.g.var[d]);
        }
        self.g.finish();
    }
}

fn floor_margin() -> f32 {
    std::env::var("FLOOR_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0)
}

pub struct Verifier {
    pub classes: Vec<Diag>,
}

/// Decision policy around the verifier, as measured on Delta Force (lab/haptics/v3/README.md).
#[derive(Clone, Copy)]
pub struct Policy {
    /// Log-likelihood ratio per dimension needed outside a burst.
    pub enter: f32,
    /// Needed within `hold_frames` of a fire: later rounds sit on the earlier ones' tails.
    pub stay: f32,
    pub hold_frames: u64,
    /// Fires only within this many dB of the taught level: the player's own gun is the loudest.
    pub range_db: f32,
}

impl Default for Policy {
    fn default() -> Self {
        Self { enter: 0.0, stay: -1.0, hold_frames: 50, range_db: 12.0 }
    }
}

impl Verifier {
    /// One class from one teaching take: (embedding, level) of each onset in it.
    pub fn teach(take: &[(Emb, f32)], bg: &Background) -> Option<Diag> {
        let top = take.iter().fold(-200f32, |a, t| a.max(t.1));
        let kept: Vec<&(Emb, f32)> = take.iter().filter(|t| t.1 >= top - TEACH_RANGE_DB).collect();
        let n = kept.first()?.0.n;
        let k = kept.len() as f32;
        let mut g = Diag { mean: [0.0; DIM_MAX], var: [0.0; DIM_MAX], n, level: 0.0, floor: f32::NEG_INFINITY, log_det: 0.0 };
        for (e, _) in &kept {
            for d in 0..n {
                g.mean[d] += e.v[d] / k;
            }
        }
        for (e, _) in &kept {
            for d in 0..n {
                g.var[d] += (e.v[d] - g.mean[d]).powi(2);
            }
        }
        for d in 0..n {
            g.var[d] = (g.var[d] + SHRINK * bg.g.var[d]) / (k + SHRINK);
        }
        let mut levels: Vec<f32> = kept.iter().map(|t| t.1).collect();
        levels.sort_by(f32::total_cmp);
        g.level = levels[levels.len() / 2];
        g.finish();
        let worst = kept.iter().map(|(e, _)| g.log_like(e) / n as f32).fold(f32::INFINITY, f32::min);
        g.floor = worst - floor_margin();
        Some(g)
    }

    /// Wire format (system_server → audioserver): ctx, kind, then per class n, level, floor,
    /// mean[n], var[n] as f32 LE.
    pub fn to_bytes(&self, cfg: &crate::model::EmbedCfg) -> Vec<u8> {
        let mut out = b"HTV3".to_vec();
        out.push(cfg.ctx as u8);
        out.push(cfg.kind as u8);
        out.push(self.classes.len() as u8);
        for c in &self.classes {
            out.extend_from_slice(&(c.n as u16).to_le_bytes());
            for v in [c.level, c.floor].iter().chain(&c.mean[..c.n]).chain(&c.var[..c.n]) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<(Verifier, crate::model::EmbedCfg)> {
        use crate::model::{EmbedCfg, Kind, CTX_MAX};
        if b.len() < 7 || &b[..4] != b"HTV3" {
            return None;
        }
        let kind = [Kind::Shape, Kind::Added, Kind::Both].into_iter().find(|k| *k as u8 == b[5])?;
        let cfg = EmbedCfg { ctx: (b[4] as usize).clamp(1, CTX_MAX), kind };
        let mut at = 7;
        let mut classes = Vec::new();
        for _ in 0..b[6] {
            let n = u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?) as usize;
            at += 2;
            if n > DIM_MAX {
                return None;
            }
            let f: Vec<f32> = b.get(at..at + 4 * (2 + 2 * n))?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            at += 4 * (2 + 2 * n);
            let mut g = Diag { mean: [0.0; DIM_MAX], var: [0.0; DIM_MAX], n, level: f[0], floor: f[1], log_det: 0.0 };
            g.mean[..n].copy_from_slice(&f[2..2 + n]);
            g.var[..n].copy_from_slice(&f[2 + n..2 + 2 * n]);
            g.finish();
            classes.push(g);
        }
        Some((Verifier { classes }, cfg))
    }

    /// Best log-likelihood ratio per dimension over the background, and its class.
    pub fn score(&self, e: &Emb, bg: &Background) -> (f32, usize) {
        let b = bg.g.log_like(e);
        let n = e.n.max(1) as f32;
        self.classes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let own = c.log_like(e) / n;
                (if own < c.floor { f32::NEG_INFINITY } else { own - b / n }, i)
            })
            .fold((f32::NEG_INFINITY, 0), |a, x| if x.0 > a.0 { x } else { a })
    }
}
