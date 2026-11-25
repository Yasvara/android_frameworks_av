//! A game's sound catalog: what its own files say voice, music and UI sound like at an onset,
//! against SFX (built by [`crate::scan`]). Live, an onset the catalog calls quiet does not
//! vibrate, and a confident one holds the rest of a voice line quiet.

use crate::model::{Emb, EmbedCfg, Kind, CTX_MAX, DIM_MAX};
use crate::music;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Label {
    Sfx = 0,
    Voice = 1,
    Music = 2,
    Ui = 3,
}

impl Label {
    pub fn from_u8(v: u8) -> Option<Self> {
        [Label::Sfx, Label::Voice, Label::Music, Label::Ui].into_iter().find(|l| *l as u8 == v)
    }

    /// Whether onsets that look like this vibrate.
    pub fn vibrates(self) -> bool {
        self == Label::Sfx
    }

    pub fn name(self) -> &'static str {
        ["sfx", "voice", "music", "ui"][self as usize]
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub e: Emb,
    /// Peak frame level over the patch, dB.
    pub level: f32,
}

/// What a game's files taught: a logistic model of "this onset is voice, music or UI, not SFX"
/// over the live path's onset embedding, and the hold thresholds that apply it.
pub struct Catalog {
    pub cfg: EmbedCfg,
    pub weights: Vec<f32>,
    pub bias: f32,
    /// An onset this likely quiet starts a voice hold and stays quiet.
    pub enter: f32,
    /// Within a hold, onsets above this stay quiet too; clearly SFX ones still fire.
    pub stay: f32,
    /// How long a quiet onset holds the next ones, ms.
    pub hold_ms: u32,
    /// Sounds per label it learnt from.
    pub sounds: [u32; 4],
    /// Of the container files it read, so an unchanged game is not scanned again.
    pub fingerprint: u64,
    /// The game's music, to tell its beats from the game's sounds.
    pub music: Option<music::Index>,
}

pub const ENTER: f32 = 0.6;
pub const STAY: f32 = 0.35;
pub const HOLD_MS: u32 = 500;

impl Catalog {
    pub fn prob(&self, e: &Emb) -> f32 {
        let n = e.n.min(self.weights.len());
        let z = self.bias + e.v[..n].iter().zip(&self.weights[..n]).map(|(a, b)| a * b).sum::<f32>();
        1.0 / (1.0 + (-z).exp())
    }

    /// Balanced logistic regression, quiet (voice, music, UI) against SFX; deterministic.
    pub fn train(hits: &[(Label, Hit)], cfg: EmbedCfg) -> Option<Catalog> {
        let dim = hits.first()?.1.e.n;
        let nq = hits.iter().filter(|h| !h.0.vibrates()).count() as f32;
        let ns = hits.len() as f32 - nq;
        if nq < 20.0 || ns < 20.0 {
            return None;
        }
        let mut w = vec![0f32; dim];
        let mut bias = 0f32;
        let mut g = vec![0f32; dim];
        for _ in 0..400 {
            g.iter_mut().for_each(|x| *x = 0.0);
            let mut gb = 0f32;
            for (l, h) in hits {
                let quiet = !l.vibrates();
                let wt = if quiet { 0.5 / nq } else { 0.5 / ns };
                let z = bias + h.e.v[..dim].iter().zip(&w).map(|(a, b)| a * b).sum::<f32>();
                let d = (1.0 / (1.0 + (-z).exp()) - quiet as u8 as f32) * wt;
                for (gk, x) in g.iter_mut().zip(&h.e.v[..dim]) {
                    *gk += d * x;
                }
                gb += d;
            }
            for (wk, gk) in w.iter_mut().zip(&g) {
                *wk -= 10.0 * (gk + 1e-4 * *wk);
            }
            bias -= 10.0 * gb;
        }
        let mut sounds = [0u32; 4];
        for (l, _) in hits {
            sounds[*l as usize] += 1;
        }
        Some(Catalog { cfg, weights: w, bias, enter: ENTER, stay: STAY, hold_ms: HOLD_MS, sounds, fingerprint: 0, music: None })
    }

    /// "HTC2", ctx, kind, dim (u16), enter, stay, bias, weights[dim] (f32 LE), hold_ms,
    /// sounds[4] (u32), fingerprint (u64), then the music index if any.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"HTC2".to_vec();
        out.push(self.cfg.ctx as u8);
        out.push(self.cfg.kind as u8);
        out.extend_from_slice(&(self.weights.len() as u16).to_le_bytes());
        for v in [self.enter, self.stay, self.bias].iter().chain(&self.weights) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.hold_ms.to_le_bytes());
        for s in self.sounds {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&self.fingerprint.to_le_bytes());
        if let Some(m) = &self.music {
            out.extend(m.to_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Catalog> {
        if b.len() < 8 || &b[..4] != b"HTC2" {
            return None;
        }
        let kind = [Kind::Shape, Kind::Added, Kind::Both].into_iter().find(|k| *k as u8 == b[5])?;
        let cfg = EmbedCfg { ctx: (b[4] as usize).clamp(1, CTX_MAX), kind };
        let dim = u16::from_le_bytes([b[6], b[7]]) as usize;
        if dim > DIM_MAX {
            return None;
        }
        let f: Vec<f32> = b.get(8..8 + 4 * (3 + dim))?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut at = 8 + 4 * (3 + dim);
        let hold_ms = u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?);
        at += 4;
        let mut sounds = [0u32; 4];
        for s in sounds.iter_mut() {
            *s = u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?);
            at += 4;
        }
        let fingerprint = u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?);
        let music = match b.get(at + 8..) {
            Some(rest) if !rest.is_empty() => Some(music::Index::from_bytes(rest)?),
            _ => None,
        };
        Some(Catalog { cfg, weights: f[3..].to_vec(), bias: f[2], enter: f[0], stay: f[1], hold_ms, sounds, fingerprint, music })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let cfg = EmbedCfg { ctx: 4, kind: Kind::Both };
        let c = Catalog { cfg, weights: vec![0.5, -1.0, 2.0], bias: 0.25, enter: 0.6, stay: 0.35, hold_ms: 700, sounds: [1, 2, 3, 4], fingerprint: 7, music: None };
        let back = Catalog::from_bytes(&c.to_bytes()).unwrap();
        assert_eq!(back.weights, vec![0.5, -1.0, 2.0]);
        assert_eq!((back.bias, back.enter, back.stay, back.hold_ms, back.sounds, back.fingerprint), (0.25, 0.6, 0.35, 700, [1, 2, 3, 4], 7));
    }
}
