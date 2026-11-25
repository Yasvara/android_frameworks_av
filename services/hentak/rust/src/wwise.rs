//! Wwise Vorbis (.wem, codec 0xFFFF) to standard Vorbis packets, after ww2ogg (BSD, Adam
//! Gashlin; see third_party/ww2ogg/COPYING). Wwise stores Vorbis without Ogg, with 2- or 6-byte
//! packet headers, a setup header stripped of every field the encoder always writes the same way,
//! codebooks as ids into a shared library (or in a packed inline form), and audio packets that
//! drop the packet-type bit and the window flags the decoder can infer.

use crate::bits::{ilog, BitReader, BitWriter};

/// The aoTuV 6.03 library modern Wwise encoders use, and the older one.
pub const AOTUV_603: &[u8] = include_bytes!("../data/packed_codebooks_aoTuV_603.bin");
pub const LEGACY: &[u8] = include_bytes!("../data/packed_codebooks.bin");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NotRiff,
    Codec(u16),
    Layout(&'static str),
    Setup(&'static str),
    Truncated,
}

type Result<T> = std::result::Result<T, Error>;

pub struct Codebooks<'a> {
    data: &'a [u8],
    offsets: Vec<u32>,
}

impl<'a> Codebooks<'a> {
    /// File layout: codebook bytes, then a u32 LE offset per codebook plus one past the last,
    /// then the u32 LE offset of that table.
    pub fn parse(file: &'a [u8]) -> Option<Self> {
        let n = file.len();
        if n < 8 {
            return None;
        }
        let table = u32::from_le_bytes(file[n - 4..].try_into().ok()?) as usize;
        if table >= n - 4 {
            return None;
        }
        let offsets: Vec<u32> = file[table..n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Some(Self { data: &file[..table], offsets })
    }

    fn get(&self, i: usize) -> Option<&'a [u8]> {
        if i + 2 > self.offsets.len() {
            return None;
        }
        let (s, e) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        self.data.get(s..e)
    }
}

fn u16_at(b: &[u8], i: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b.get(i..i + 2).ok_or(Error::Truncated)?.try_into().unwrap()))
}

fn u32_at(b: &[u8], i: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(i..i + 4).ok_or(Error::Truncated)?.try_into().unwrap()))
}

pub struct Wem<'a> {
    pub channels: u16,
    pub rate: u32,
    pub avg_bytes_per_second: u32,
    /// Total samples per channel.
    pub samples: u32,
    data: &'a [u8],
    setup: usize,
    audio: usize,
    no_granule: bool,
    mod_packets: bool,
    blocksize0: u8,
    blocksize1: u8,
}

/// The format tag of a RIFF/WAVE .wem, so callers can tell Vorbis from PCM or ADPCM.
pub fn codec(b: &[u8]) -> Option<u16> {
    chunks(b).ok()?.into_iter().find(|c| &c.0 == b"fmt ").and_then(|c| u16_at(b, c.1).ok())
}

/// (id, body offset, body size) of each chunk.
fn chunks(b: &[u8]) -> Result<Vec<([u8; 4], usize, usize)>> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(Error::NotRiff);
    }
    let end = (u32_at(b, 4)? as usize + 8).min(b.len());
    let mut at = 12;
    let mut out = Vec::new();
    while at + 8 <= end {
        let size = u32_at(b, at + 4)? as usize;
        out.push((b[at..at + 4].try_into().unwrap(), at + 8, size));
        at += 8 + size;
    }
    Ok(out)
}

impl<'a> Wem<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Self> {
        let cs = chunks(b)?;
        let find = |id: &[u8; 4]| cs.iter().find(|c| &c.0 == id).copied();
        let (_, fmt, fmt_size) = find(b"fmt ").ok_or(Error::Layout("no fmt"))?;
        let (_, data, data_size) = find(b"data").ok_or(Error::Layout("no data"))?;
        let codec = u16_at(b, fmt)?;
        if codec != 0xFFFF {
            return Err(Error::Codec(codec));
        }
        let (vorb, vorb_size) = match find(b"vorb") {
            Some((_, o, s)) => (o, Some(s)),
            None if fmt_size == 0x42 => (fmt + 0x18, None),
            None => return Err(Error::Layout("no vorb")),
        };
        let channels = u16_at(b, fmt + 2)?;
        let rate = u32_at(b, fmt + 4)?;
        let avg_bytes_per_second = u32_at(b, fmt + 8)?;
        let samples = u32_at(b, vorb)?;
        let (no_granule, mod_packets, offsets_at, uid_at) = match vorb_size {
            None | Some(0x2A) => {
                let signal = u32_at(b, vorb + 4)?;
                (true, !matches!(signal, 0x4A | 0x4B | 0x69 | 0x70), vorb + 0x10, vorb + 0x24)
            }
            Some(0x32) | Some(0x34) => (false, false, vorb + 0x18, vorb + 0x2C),
            // 0x28 / 0x2C carry the three standard headers themselves (very old encoders).
            _ => return Err(Error::Layout("old vorb layout")),
        };
        let setup = u32_at(b, offsets_at)? as usize;
        let audio = u32_at(b, offsets_at + 4)? as usize;
        let blocksize0 = *b.get(uid_at + 4).ok_or(Error::Truncated)?;
        let blocksize1 = *b.get(uid_at + 5).ok_or(Error::Truncated)?;
        let data = b.get(data..(data + data_size).min(b.len())).ok_or(Error::Truncated)?;
        if channels == 0 || !(6..=13).contains(&blocksize0) || !(6..=13).contains(&blocksize1) {
            return Err(Error::Layout("bad header values"));
        }
        Ok(Self { channels, rate, avg_bytes_per_second, samples, data, setup, audio, no_granule, mod_packets, blocksize0, blocksize1 })
    }

    fn header_size(&self) -> usize {
        if self.no_granule {
            2
        } else {
            6
        }
    }

    /// (payload offset, payload size) of the packet whose header starts at `at`.
    fn packet(&self, at: usize) -> Result<(usize, usize)> {
        let size = u16_at(self.data, at)? as usize;
        let start = at + self.header_size();
        if start + size > self.data.len() {
            return Err(Error::Truncated);
        }
        Ok((start, size))
    }

    /// The three standard header packets. Tries each codebook library, then inline codebooks,
    /// and keeps the first that reads the stripped setup to its last byte.
    pub fn headers(&self) -> Result<Headers> {
        let mut last = Error::Setup("no codebook source");
        for source in [Source::Library(AOTUV_603), Source::Library(LEGACY), Source::Inline] {
            match self.setup_header(source) {
                Ok(h) => return Ok(h),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn setup_header(&self, source: Source) -> Result<Headers> {
        let mut id = BitWriter::new();
        id.write(1, 8);
        id.bytes(b"vorbis");
        id.write(0, 32);
        id.write(self.channels as u32, 8);
        id.write(self.rate, 32);
        id.write(0, 32);
        id.write(self.avg_bytes_per_second.wrapping_mul(8), 32);
        id.write(0, 32);
        id.write(self.blocksize0 as u32, 4);
        id.write(self.blocksize1 as u32, 4);
        id.write(1, 1);

        let mut cm = BitWriter::new();
        cm.write(3, 8);
        cm.bytes(b"vorbis");
        const VENDOR: &[u8] = b"Hentak (from Wwise)";
        cm.write(VENDOR.len() as u32, 32);
        cm.bytes(VENDOR);
        cm.write(0, 32);
        cm.write(1, 1);

        let (start, size) = self.packet(self.setup)?;
        let mut r = BitReader::new(&self.data[start..start + size]);
        let mut w = BitWriter::new();
        w.write(5, 8);
        w.bytes(b"vorbis");
        let setup = rebuild_setup(&mut r, &mut w, self.channels as u32, source)?;
        if (r.bits_read() + 7) / 8 != size {
            return Err(Error::Setup("setup not read exactly"));
        }
        if start + size != self.audio {
            return Err(Error::Setup("audio does not follow setup"));
        }
        Ok(Headers { ident: id.finish(), comment: cm.finish(), setup: w.finish(), blockflags: setup.blockflags, mode_bits: setup.mode_bits })
    }

    /// Audio packets in standard form.
    pub fn packets<'s>(&'s self, h: &'s Headers) -> Packets<'s, 'a> {
        Packets { wem: self, h, at: self.audio, prev_long: false }
    }
}

pub struct Headers {
    pub ident: Vec<u8>,
    pub comment: Vec<u8>,
    pub setup: Vec<u8>,
    blockflags: Vec<bool>,
    mode_bits: u32,
}

pub struct Packets<'s, 'a> {
    wem: &'s Wem<'a>,
    h: &'s Headers,
    at: usize,
    prev_long: bool,
}

impl Iterator for Packets<'_, '_> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let wem = self.wem;
        if self.at + wem.header_size() > wem.data.len() {
            return None;
        }
        let (start, size) = match wem.packet(self.at) {
            Ok(p) => p,
            Err(e) => {
                self.at = wem.data.len();
                return Some(Err(e));
            }
        };
        let next = start + size;
        self.at = next;
        let payload = &wem.data[start..next];
        if !wem.mod_packets || size == 0 {
            return Some(Ok(payload.to_vec()));
        }
        // First byte: mode number, then the rest of the byte. Standard form puts the packet type
        // before it and, for a long block, the previous and next window flags after it.
        let mb = self.h.mode_bits;
        let mut r = BitReader::new(payload);
        let mode = r.read(mb).unwrap_or(0) as usize;
        let rest = r.read(8 - mb).unwrap_or(0);
        let Some(&long) = self.h.blockflags.get(mode) else {
            return Some(Err(Error::Setup("mode out of range")));
        };
        let mut w = BitWriter::new();
        w.write(0, 1);
        w.write(mode as u32, mb);
        if long {
            let next_long = wem
                .packet(next)
                .ok()
                .filter(|&(_, s)| s > 0)
                .and_then(|(s, _)| self.h.blockflags.get(BitReader::new(&wem.data[s..]).read(mb)? as usize).copied())
                .unwrap_or(false);
            w.write(self.prev_long as u32, 1);
            w.write(next_long as u32, 1);
        }
        self.prev_long = long;
        w.write(rest, 8 - mb);
        w.bytes(&payload[1..]);
        Some(Ok(w.finish()))
    }
}

#[derive(Clone, Copy)]
enum Source {
    Library(&'static [u8]),
    Inline,
}

struct Setup {
    blockflags: Vec<bool>,
    mode_bits: u32,
}

macro_rules! rd {
    ($r:expr, $n:expr) => {
        $r.read($n).ok_or(Error::Setup("setup truncated"))?
    };
}

/// A copied field: read n bits, write them unchanged.
fn cp(r: &mut BitReader, w: &mut BitWriter, n: u32) -> Result<u32> {
    let v = rd!(r, n);
    w.write(v, n);
    Ok(v)
}

fn rebuild_setup(r: &mut BitReader, w: &mut BitWriter, channels: u32, source: Source) -> Result<Setup> {
    let books = cp(r, w, 8)? as usize + 1;
    match source {
        Source::Library(file) => {
            let lib = Codebooks::parse(file).ok_or(Error::Setup("bad codebook library"))?;
            for _ in 0..books {
                let id = rd!(r, 10) as usize;
                let cb = lib.get(id).ok_or(Error::Setup("codebook id out of range"))?;
                let mut cr = BitReader::new(cb);
                rebuild_codebook(&mut cr, w)?;
                // The library pads each book to whole bytes, plus a zero byte if it fills its last.
                if cr.bits_read() / 8 + 1 != cb.len() {
                    return Err(Error::Setup("codebook size mismatch"));
                }
            }
        }
        Source::Inline => {
            for _ in 0..books {
                rebuild_codebook(r, w)?;
            }
        }
    }
    // Time domain transforms: one placeholder.
    w.write(0, 6);
    w.write(0, 16);

    let floors = cp(r, w, 6)? as usize + 1;
    for _ in 0..floors {
        w.write(1, 16);
        let partitions = cp(r, w, 5)? as usize;
        let mut classes = Vec::with_capacity(partitions);
        for _ in 0..partitions {
            classes.push(cp(r, w, 4)? as usize);
        }
        let max_class = classes.iter().copied().max().map_or(0, |m| m + 1);
        let mut dims = vec![0usize; max_class];
        for d in dims.iter_mut() {
            *d = cp(r, w, 3)? as usize + 1;
            let sub = cp(r, w, 2)?;
            if sub != 0 && cp(r, w, 8)? as usize >= books {
                return Err(Error::Setup("floor masterbook"));
            }
            for _ in 0..1 << sub {
                let b = cp(r, w, 8)? as i32 - 1;
                if b >= books as i32 {
                    return Err(Error::Setup("floor subclass book"));
                }
            }
        }
        cp(r, w, 2)?;
        let rangebits = cp(r, w, 4)?;
        for &c in &classes {
            for _ in 0..dims[c] {
                cp(r, w, rangebits)?;
            }
        }
    }

    let residues = cp(r, w, 6)? as usize + 1;
    for _ in 0..residues {
        let kind = rd!(r, 2);
        if kind > 2 {
            return Err(Error::Setup("residue type"));
        }
        w.write(kind, 16);
        cp(r, w, 24)?;
        cp(r, w, 24)?;
        cp(r, w, 24)?;
        let classifications = cp(r, w, 6)? as usize + 1;
        if cp(r, w, 8)? as usize >= books {
            return Err(Error::Setup("residue classbook"));
        }
        let mut cascade = vec![0u32; classifications];
        for c in cascade.iter_mut() {
            let low = cp(r, w, 3)?;
            let high = if cp(r, w, 1)? != 0 { cp(r, w, 5)? } else { 0 };
            *c = high * 8 + low;
        }
        for &c in &cascade {
            for k in 0..8 {
                if c & (1 << k) != 0 && cp(r, w, 8)? as usize >= books {
                    return Err(Error::Setup("residue book"));
                }
            }
        }
    }

    let mappings = cp(r, w, 6)? as usize + 1;
    for _ in 0..mappings {
        w.write(0, 16);
        let submaps = if cp(r, w, 1)? != 0 { cp(r, w, 4)? + 1 } else { 1 };
        if cp(r, w, 1)? != 0 {
            let steps = cp(r, w, 8)? + 1;
            let bits = ilog(channels - 1);
            for _ in 0..steps {
                let m = cp(r, w, bits)?;
                let a = cp(r, w, bits)?;
                if m == a || m >= channels || a >= channels {
                    return Err(Error::Setup("coupling"));
                }
            }
        }
        if cp(r, w, 2)? != 0 {
            return Err(Error::Setup("mapping reserved"));
        }
        if submaps > 1 {
            for _ in 0..channels {
                if cp(r, w, 4)? >= submaps {
                    return Err(Error::Setup("mapping mux"));
                }
            }
        }
        for _ in 0..submaps {
            cp(r, w, 8)?;
            if cp(r, w, 8)? as usize >= floors {
                return Err(Error::Setup("mapping floor"));
            }
            if cp(r, w, 8)? as usize >= residues {
                return Err(Error::Setup("mapping residue"));
            }
        }
    }

    let modes = cp(r, w, 6)? as usize + 1;
    let mut blockflags = Vec::with_capacity(modes);
    for _ in 0..modes {
        blockflags.push(cp(r, w, 1)? != 0);
        w.write(0, 16);
        w.write(0, 16);
        if cp(r, w, 8)? as usize >= mappings {
            return Err(Error::Setup("mode mapping"));
        }
    }
    w.write(1, 1);
    Ok(Setup { blockflags, mode_bits: ilog(modes as u32 - 1) })
}

pub(crate) fn lookup1_values(entries: u32, dims: u32) -> u32 {
    if dims == 0 {
        return 0;
    }
    let mut r = (entries as f64).powf(1.0 / dims as f64).floor() as u32;
    let pow = |v: u32| (0..dims).try_fold(1u64, |a, _| a.checked_mul(v as u64));
    while pow(r + 1).is_some_and(|p| p <= entries as u64) {
        r += 1;
    }
    while r > 0 && pow(r).is_none_or(|p| p > entries as u64) {
        r -= 1;
    }
    r
}

/// Packed codebook (4-bit dimensions, 14-bit entries, short length fields, 1-bit lookup type)
/// to the standard form.
fn rebuild_codebook(r: &mut BitReader, w: &mut BitWriter) -> Result<()> {
    let dims = rd!(r, 4);
    let entries = rd!(r, 14);
    w.write(0x564342, 24);
    w.write(dims, 16);
    w.write(entries, 24);
    if cp(r, w, 1)? != 0 {
        cp(r, w, 5)?;
        let mut cur = 0;
        while cur < entries {
            cur += cp(r, w, ilog(entries - cur))?;
        }
        if cur > entries {
            return Err(Error::Setup("ordered lengths overrun"));
        }
    } else {
        let len_bits = rd!(r, 3);
        if len_bits == 0 || len_bits > 5 {
            return Err(Error::Setup("codeword length size"));
        }
        let sparse = cp(r, w, 1)? != 0;
        for _ in 0..entries {
            if !sparse || cp(r, w, 1)? != 0 {
                let len = rd!(r, len_bits);
                w.write(len, 5);
            }
        }
    }
    let lookup = rd!(r, 1);
    w.write(lookup, 4);
    if lookup == 1 {
        cp(r, w, 32)?;
        cp(r, w, 32)?;
        let value_bits = cp(r, w, 4)? + 1;
        cp(r, w, 1)?;
        for _ in 0..lookup1_values(entries, dims) {
            cp(r, w, value_bits)?;
        }
    }
    Ok(())
}
