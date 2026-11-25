//! Catalog scan of a game's own sound files: Wwise soundbanks and packages decoded in Rust,
//! labelled by container, the onsets in each sound embedded with the live path's front end, and
//! a [`Catalog`] trained on them.

use crate::bank;
use crate::catalog::{Catalog, Hit, Label};
use crate::front::{Front, Pcm};
use crate::mel::{Frame, Mel};
use crate::model::{embed, EmbedCfg, CTX_MAX};
use crate::music::{self, TrackPrinter};
use crate::onset::{Onset, OnsetCfg};
use crate::ring::Ring;
use crate::vorbis::Decoder;
use crate::wwise::{self, Wem};

/// Source label from a container's file or folder name, as engines and studios name them:
/// `sfx_*`, `vo_*`, `bgm_*`, `ui_*` (MLBB), `Banks* / Streamed* / Music* / VoBanks*` (miHoYo), voice
/// language folders. None when the name says nothing.
pub fn label_of_name(path: &str) -> Option<Label> {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    let has = |s: &str| name.contains(s);
    let starts = |s: &str| name.starts_with(s);
    if starts("vo") || has("voice") || has("message") || has("dialog") || starts("show") || has("_vo_") || has("chat") {
        Some(Label::Voice)
    } else if starts("bgm") || has("music") || starts("mus_") || has("_bgm") || starts("mg") {
        Some(Label::Music)
    } else if starts("ui") || has("_ui_") || has("emoji") || has("menu") || starts("taptap") {
        Some(Label::Ui)
    } else if starts("sfx") || has("skill") || has("weapon") || starts("banks") || starts("streamed") || has("battle") {
        Some(Label::Sfx)
    } else {
        None
    }
}

/// Decodes up to `max_ms` of a Wwise Vorbis sound, mono, and embeds each onset in it.
pub fn sound_hits(wem: &[u8], max_ms: u32, cfg: &EmbedCfg) -> Result<Vec<Hit>, String> {
    decode_sound(wem, Some(max_ms), false, cfg).map(|d| d.0)
}

/// One decode of a Wwise Vorbis sound: the onsets in its first `hits_ms` embedded, and with
/// `print`, the whole sound fingerprinted as music.
pub fn decode_sound(wem: &[u8], hits_ms: Option<u32>, print: bool, cfg: &EmbedCfg) -> Result<(Vec<Hit>, Option<music::Track>), String> {
    let w = Wem::parse(wem).map_err(|e| format!("{e:?}"))?;
    let h = w.headers().map_err(|e| format!("{e:?}"))?;
    let mut dec = Decoder::new(&h.ident, &h.setup)?;
    let ch = dec.channels();
    let want = hits_ms.map_or(0, |ms| (dec.rate() as u64 * ms as u64 / 1000) as usize);
    let mut head = Vec::with_capacity(want);
    let mut printer = print.then(|| TrackPrinter::new(dec.rate()));
    let mut out = vec![Vec::new(); ch];
    let mut mono = Vec::new();
    for p in w.packets(&h) {
        let p = p.map_err(|e| format!("{e:?}"))?;
        dec.decode(&p, &mut out)?;
        mono.clear();
        mono.extend((0..out[0].len()).map(|i| out.iter().map(|c| c[i]).sum::<f32>() / ch as f32));
        out.iter_mut().for_each(Vec::clear);
        let take = (want - head.len()).min(mono.len());
        head.extend_from_slice(&mono[..take]);
        match printer.as_mut() {
            Some(pr) => pr.push(&mono),
            None if head.len() >= want => break,
            None => {}
        }
    }
    let hits = if want > 0 { hits_in(&head, dec.rate(), cfg) } else { Vec::new() };
    Ok((hits, printer.map(TrackPrinter::finish)))
}

/// The live path's front end, onset detector and embedding over a whole buffer.
pub fn hits_in(pcm: &[f32], rate: u32, cfg: &EmbedCfg) -> Vec<Hit> {
    let ring = Ring::new(pcm.len() + 1024);
    let mut front = Front::new(rate);
    // Room after the sound so onsets near its end still get their whole patch.
    let mut padded = pcm.to_vec();
    padded.extend(std::iter::repeat_n(0.0, rate as usize / 20));
    front.feed(Pcm::F32(&padded), 1, &ring);
    let mut x = vec![0f32; padded.len()];
    let got = ring.pop(&mut x);
    let mut mel = Mel::new(Front::out_rate(rate));
    let mut onset = Onset::new(OnsetCfg::default());
    let ctx = cfg.ctx.min(CTX_MAX);
    const HIST: usize = 16;
    let mut hist = [Frame::default(); HIST];
    let mut t = 0usize;
    let mut pending: Vec<usize> = Vec::new();
    let mut hits = Vec::new();
    mel.push(&x[..got], |f| {
        hist[t % HIST] = *f;
        if onset.step(f).is_some() {
            pending.push(t);
        }
        pending.retain(|&t0| {
            if t < t0 + ctx - 1 {
                return true;
            }
            let mut patch = [Frame::default(); CTX_MAX];
            for (k, p) in patch.iter_mut().take(ctx).enumerate() {
                *p = hist[(t0 + k) % HIST];
            }
            let before = if t0 >= 2 { hist[(t0 - 2) % HIST] } else { Frame::default() };
            let level = patch[..ctx].iter().fold(-120f32, |a, p| a.max(p.level));
            hits.push(Hit { e: embed(&patch[..ctx], &before, cfg), level });
            false
        });
        t += 1;
    });
    hits
}

/// Every Wwise sound in one container file, with its label: package entries in a voice
/// language are voice whatever the name says.
pub fn container_sounds<'a>(path: &str, b: &'a [u8]) -> Vec<(Option<Label>, &'a [u8])> {
    let by_name = label_of_name(path);
    let mut out = Vec::new();
    if let Some(p) = bank::akpk(b) {
        let lang = |l: u32| if l != 0 { Some(Label::Voice) } else { by_name };
        for m in p.media {
            out.push((lang(m.language), m.data));
        }
        for bk in p.banks {
            for m in bank::bnk_media(bk.data) {
                out.push((lang(bk.language), m.data));
            }
        }
    } else {
        for m in bank::bnk_media(b) {
            out.push((by_name, m.data));
        }
        if out.is_empty() && wwise::codec(b).is_some() {
            out.push((by_name, b));
        }
    }
    out
}

/// One sound inside a container file.
#[derive(Clone, Copy)]
struct SoundRef {
    file: usize,
    offset: u64,
    size: u32,
    label: Label,
    key: u64,
    id: u64,
}

fn fnv(h: &mut u64, b: &[u8]) {
    for &c in b {
        *h ^= c as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn read_at(f: &std::fs::File, offset: u64, len: usize) -> Option<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut b = vec![0u8; len];
    f.read_exact_at(&mut b, offset).ok()?;
    Some(b)
}

/// (offset, size) of each medium of the soundbank at `base`, from its DIDX and DATA chunk
/// headers only.
fn bnk_refs(f: &std::fs::File, base: u64, len: u64) -> Vec<(u64, u32, u32)> {
    let (mut at, mut didx, mut data) = (0u64, None, None);
    while at + 8 <= len {
        let Some(h) = read_at(f, base + at, 8) else { break };
        let size = u32::from_le_bytes(h[4..8].try_into().unwrap()) as u64;
        match &h[..4] {
            b"DIDX" => didx = read_at(f, base + at + 8, size as usize),
            b"DATA" => data = Some(base + at + 8),
            _ => {}
        }
        if didx.is_some() && data.is_some() {
            break;
        }
        at += 8 + size;
    }
    let (Some(didx), Some(data)) = (didx, data) else { return Vec::new() };
    didx.chunks_exact(12)
        .map(|e| {
            let id = u32::from_le_bytes(e[..4].try_into().unwrap());
            let off = u32::from_le_bytes(e[4..8].try_into().unwrap()) as u64;
            let size = u32::from_le_bytes(e[8..12].try_into().unwrap());
            (data + off, size, id)
        })
        .collect()
}

/// Wwise soundbanks, packages and loose media.
fn is_container_name(p: &std::path::Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["bnk", "pck", "wem"].iter().any(|x| e.eq_ignore_ascii_case(x)))
}

pub struct Scan {
    pub catalog: Option<Catalog>,
    pub fingerprint: u64,
    pub containers: usize,
    pub sounds: usize,
    pub decoded: usize,
    pub music_tracks: usize,
}

/// Most sounds decoded per label: enough for the model, bounded time on the phone.
pub const PER_LABEL: [usize; 4] = [3000, 3000, 600, 600];
/// Most compressed music fingerprinted, about 2-4 h; a game with more keeps a fixed sample.
pub const MUSIC_BYTES: u64 = 96 << 20;
/// Shorter music (stingers) is over before a lock.
const MUSIC_MIN_MS: u64 = 3000;
/// Bumped when what a scan makes changes, so every game is scanned again.
const SCAN_VERSION: &[u8] = b"hentak-scan 2: music";

/// Scans every container file under `roots`. Returns None when the files are unchanged since
/// `previous` (a catalog's fingerprint).
pub fn scan(roots: &[std::path::PathBuf], previous: Option<u64>, threads: usize, max_ms: u32) -> Option<Scan> {
    use std::os::unix::fs::MetadataExt;
    let mut files: Vec<(std::path::PathBuf, u64, i64)> = Vec::new();
    let mut stack: Vec<std::path::PathBuf> = roots.to_vec();
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let Ok(m) = e.metadata() else { continue };
            if m.is_dir() {
                stack.push(e.path());
            } else if m.is_file() && m.len() >= 16 && is_container_name(&e.path()) {
                files.push((e.path(), m.len(), m.mtime()));
            }
        }
    }
    files.sort();
    // From metadata alone, so an unchanged game costs a directory walk and no reads; only
    // container files count, since games rewrite their caches and logs all the time.
    let mut fp = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut fp, SCAN_VERSION);
    for (path, size, mtime) in &files {
        fnv(&mut fp, path.to_string_lossy().as_bytes());
        fnv(&mut fp, &size.to_le_bytes());
        fnv(&mut fp, &mtime.to_le_bytes());
    }
    if previous == Some(fp) {
        return None;
    }
    let mut containers = Vec::new();
    for (path, size, _) in files {
        let Ok(f) = std::fs::File::open(&path) else { continue };
        let Some(magic) = read_at(&f, 0, 4) else { continue };
        if matches!(&magic[..], b"BKHD" | b"AKPK" | b"RIFF") {
            containers.push((path, size, magic));
        }
    }

    let mut refs: Vec<SoundRef> = Vec::new();
    let mut handles = Vec::new();
    for (i, (path, size, magic)) in containers.iter().enumerate() {
        let name = path.to_string_lossy();
        let by_name = label_of_name(&name);
        let Ok(f) = std::fs::File::open(path) else { continue };
        let mut key = 0xcbf2_9ce4_8422_2325u64;
        fnv(&mut key, name.as_bytes());
        let mut push = |offset: u64, size: u32, label: Option<Label>, id: u64| {
            if let Some(label) = label {
                let mut k = key;
                fnv(&mut k, &id.to_le_bytes());
                refs.push(SoundRef { file: i, offset, size, label, key: k, id });
            }
        };
        match &magic[..] {
            b"BKHD" => {
                for (off, sz, id) in bnk_refs(&f, 0, *size) {
                    push(off, sz, by_name, id as u64);
                }
            }
            b"AKPK" => {
                let Some(head) = read_at(&f, 0, 8) else { continue };
                let hlen = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize + 8;
                let Some(header) = read_at(&f, 0, hlen.min(*size as usize)) else { continue };
                let Some(entries) = bank::akpk_entries(&header) else { continue };
                for e in entries {
                    let label = if e.language != 0 { Some(Label::Voice) } else { by_name };
                    if e.bank {
                        for (off, sz, id) in bnk_refs(&f, e.offset, e.size as u64) {
                            push(off, sz, label, e.id ^ (id as u64) << 32);
                        }
                    } else {
                        push(e.offset, e.size, label, e.id);
                    }
                }
            }
            _ => push(0, *size as u32, by_name, 0),
        }
        handles.push(f);
    }
    let total = refs.len();
    // Deterministic sample: the lowest keys per label.
    refs.sort_by_key(|r| (r.label, r.key));
    // (sound, embed its onsets, fingerprint it): the sample, then music past it within budget.
    let mut picked: Vec<(SoundRef, bool, bool)> = Vec::new();
    for label in [Label::Sfx, Label::Voice, Label::Music, Label::Ui] {
        picked.extend(refs.iter().filter(|r| r.label == label).take(PER_LABEL[label as usize]).map(|r| (*r, true, false)));
    }
    let mut budget = MUSIC_BYTES;
    let mut music_refs = std::collections::HashSet::new();
    for r in refs.iter().filter(|r| r.label == Label::Music) {
        if (r.size as u64) > budget {
            break;
        }
        budget -= r.size as u64;
        music_refs.insert(r.key);
    }
    for p in picked.iter_mut() {
        p.2 = music_refs.remove(&p.0.key);
    }
    picked.extend(refs.iter().filter(|r| music_refs.contains(&r.key)).map(|r| (*r, false, true)));

    let cfg = EmbedCfg::default();
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Decoded = (usize, Label, Vec<Hit>, Option<music::Track>);
    let results: Vec<Decoded> = std::thread::scope(|sc| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                sc.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(r, hits, print)) = picked.get(i) else { break };
                        let Some(b) = read_at(&handles[r.file], r.offset, r.size as usize) else { continue };
                        if wwise::codec(&b) != Some(0xFFFF) {
                            continue;
                        }
                        if let Ok((h, t)) = decode_sound(&b, hits.then_some(max_ms), print, &cfg) {
                            out.push((i, r.label, h, t));
                        }
                    }
                    out
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
    });
    let mut results = results;
    results.sort_by_key(|d| d.0);
    let mut decoded_hits: Vec<(Label, Hit)> = Vec::new();
    let mut tracks: Vec<(String, music::Track)> = Vec::new();
    for (i, label, hits, track) in results {
        decoded_hits.extend(hits.into_iter().map(|h| (label, h)));
        if let Some(t) = track.filter(|t| t.env.len() as u64 * music::FRAME_MS >= MUSIC_MIN_MS) {
            let r = &picked[i].0;
            let file = containers[r.file].0.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            tracks.push((format!("{file}#{}", r.id), t));
        }
    }
    let decoded = picked.len();
    let music_tracks = tracks.len();
    let mut catalog = Catalog::train(&decoded_hits, cfg);
    if let Some(c) = catalog.as_mut() {
        c.fingerprint = fp;
        c.music = (!tracks.is_empty()).then(|| music::Index::build(tracks));
    }
    Some(Scan { catalog, fingerprint: fp, containers: containers.len(), sounds: total, decoded, music_tracks })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(label_of_name("x/sfx_100111.2021.bnk"), Some(Label::Sfx));
        assert_eq!(label_of_name("vo_100111.2021.bnk"), Some(Label::Voice));
        assert_eq!(label_of_name("BattleMessage_12.bnk"), Some(Label::Voice));
        assert_eq!(label_of_name("bgm_Battle2022.2021.bnk"), Some(Label::Music));
        assert_eq!(label_of_name("Audio/Android/Music10.pck"), Some(Label::Music));
        assert_eq!(label_of_name("Audio/Android/VoBanks23.pck"), Some(Label::Voice));
        assert_eq!(label_of_name("UI_Common.bnk"), Some(Label::Ui));
        assert_eq!(label_of_name("Banks28.pck"), Some(Label::Sfx));
        assert_eq!(label_of_name("Init.bnk"), None);
    }
}
