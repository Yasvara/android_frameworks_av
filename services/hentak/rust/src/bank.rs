//! Wwise containers: soundbanks (.bnk: BKHD, DIDX index, DATA media) and file packages (.pck,
//! "AKPK": banks and streamed .wem files, each tagged with a language; 0 is SFX).

pub struct Media<'a> {
    pub id: u64,
    /// Language of a package entry; 0 = SFX (language-independent), else a voice language.
    pub language: u32,
    pub data: &'a [u8],
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

/// Media embedded in a soundbank; empty if `b` is not one.
pub fn bnk_media(b: &[u8]) -> Vec<Media<'_>> {
    if b.get(..4) != Some(b"BKHD") {
        return Vec::new();
    }
    let (mut didx, mut data) = (None, None);
    let mut at = 0;
    while at + 8 <= b.len() {
        let size = u32_at(b, at + 4).unwrap() as usize;
        let body = at + 8;
        match &b[at..at + 4] {
            b"DIDX" => didx = b.get(body..body + size),
            b"DATA" => data = b.get(body..body + size),
            _ => {}
        }
        at = body + size;
    }
    let (Some(didx), Some(data)) = (didx, data) else { return Vec::new() };
    didx.chunks_exact(12)
        .filter_map(|e| {
            let (id, off, size) = (u32_at(e, 0)?, u32_at(e, 4)? as usize, u32_at(e, 8)? as usize);
            Some(Media { id: id as u64, language: 0, data: data.get(off..off + size)? })
        })
        .collect()
}

/// Entries of a file package: soundbanks (`banks`) and loose media (streamed and external).
pub struct Package<'a> {
    pub banks: Vec<Media<'a>>,
    pub media: Vec<Media<'a>>,
}

/// Header: "AKPK", header size, version 1, then the sizes of the language map, the bank table,
/// the sound table and (newer packages) the external table. Table entries: id (u32; u64 for
/// externals), block size, file size, start block, language; data at start block × block size.
pub fn akpk(b: &[u8]) -> Option<Package<'_>> {
    if b.get(..4) != Some(b"AKPK") {
        return None;
    }
    // The header size counts from the version field on.
    let header = u32_at(b, 4)? as usize;
    if u32_at(b, 8)? != 1 {
        return None;
    }
    let langs = u32_at(b, 12)? as usize;
    let banks = u32_at(b, 16)? as usize;
    let sounds = u32_at(b, 20)? as usize;
    let (tables, externals) = if header == 16 + langs + banks + sounds {
        (24, 0)
    } else {
        let externals = u32_at(b, 24)? as usize;
        if header != 20 + langs + banks + sounds + externals {
            return None;
        }
        (28, externals)
    };
    let mut at = tables + langs;
    let table = |at: usize, size: usize, wide: bool| -> Vec<Media<'_>> {
        let Some(count) = u32_at(b, at) else { return Vec::new() };
        let entry = if wide { 24 } else { 20 };
        if 4 + count as usize * entry > size {
            return Vec::new();
        }
        (0..count as usize)
            .filter_map(|i| {
                let e = at + 4 + i * entry;
                let (id, rest) = if wide {
                    (u32_at(b, e)? as u64 | (u32_at(b, e + 4)? as u64) << 32, e + 8)
                } else {
                    (u32_at(b, e)? as u64, e + 4)
                };
                let block = u32_at(b, rest)? as usize;
                let size = u32_at(b, rest + 4)? as usize;
                let start = u32_at(b, rest + 8)? as usize;
                let language = u32_at(b, rest + 12)?;
                let off = start.checked_mul(block.max(1))?;
                Some(Media { id, language, data: b.get(off..off.checked_add(size)?)? })
            })
            .collect()
    };
    let bank_entries = table(at, banks, false);
    at += banks;
    let mut media = table(at, sounds, false);
    at += sounds;
    if externals > 0 {
        media.extend(table(at, externals, true));
    }
    Some(Package { banks: bank_entries, media })
}

/// A package entry located in the file, from the header alone.
pub struct Entry {
    pub id: u64,
    pub language: u32,
    pub offset: u64,
    pub size: u32,
    pub bank: bool,
}

/// The entries of a package from its header bytes (the first header size + 8 bytes).
pub fn akpk_entries(h: &[u8]) -> Option<Vec<Entry>> {
    if h.get(..4) != Some(b"AKPK") || u32_at(h, 8)? != 1 {
        return None;
    }
    let header = u32_at(h, 4)? as usize;
    let langs = u32_at(h, 12)? as usize;
    let banks = u32_at(h, 16)? as usize;
    let sounds = u32_at(h, 20)? as usize;
    let (tables, externals) = if header == 16 + langs + banks + sounds {
        (24, 0)
    } else {
        let externals = u32_at(h, 24)? as usize;
        if header != 20 + langs + banks + sounds + externals {
            return None;
        }
        (28, externals)
    };
    let mut out = Vec::new();
    let mut table = |at: usize, size: usize, wide: bool, bank: bool| -> Option<()> {
        let count = u32_at(h, at)? as usize;
        let entry = if wide { 24 } else { 20 };
        if 4 + count * entry > size {
            return None;
        }
        for i in 0..count {
            let e = at + 4 + i * entry;
            let (id, rest) = if wide { (u32_at(h, e)? as u64 | (u32_at(h, e + 4)? as u64) << 32, e + 8) } else { (u32_at(h, e)? as u64, e + 4) };
            let block = u32_at(h, rest)?.max(1) as u64;
            let size = u32_at(h, rest + 4)?;
            let start = u32_at(h, rest + 8)? as u64;
            out.push(Entry { id, language: u32_at(h, rest + 12)?, offset: start * block, size, bank });
        }
        Some(())
    };
    let mut at = tables + langs;
    table(at, banks, false, true)?;
    at += banks;
    table(at, sounds, false, false)?;
    at += sounds;
    if externals > 0 {
        table(at, externals, true, false)?;
    }
    Some(out)
}
