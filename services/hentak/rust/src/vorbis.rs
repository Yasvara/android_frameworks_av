//! Vorbis I decoder (floor 1 only, as every Wwise and modern encoder writes): identification and
//! setup headers, then audio packets to per-channel f32 PCM. Follows the Xiph specification;
//! the IMDCT runs as a DCT-IV over a quarter-size complex FFT.

use crate::bits::{ilog, BitReader};
use crate::wwise::lookup1_values;

pub type Error = &'static str;
type Result<T> = std::result::Result<T, Error>;

const FAST_BITS: u32 = 10;
const NONE: u16 = u16::MAX;

struct Codebook {
    dims: usize,
    lengths: Vec<u8>,
    /// Entry per next FAST_BITS bits, for codewords that short.
    fast: Vec<u16>,
    /// (bit-reversed codeword, length, entry) for longer ones.
    long: Vec<(u32, u8, u16)>,
    /// The only used entry of a single-entry book.
    single: Option<u16>,
    /// dims values per entry, when the book has a lookup table.
    values: Vec<f32>,
}

fn float32_unpack(x: u32) -> f32 {
    let mantissa = (x & 0x1f_ffff) as f64;
    let exponent = ((x & 0x7fe0_0000) >> 21) as i32;
    let v = mantissa * 2f64.powi(exponent - 788);
    (if x & 0x8000_0000 != 0 { -v } else { v }) as f32
}

/// Codewords per the specification's tree fill (libvorbis `_make_words`), bit-reversed for the
/// LSB-first reader.
fn make_words(lengths: &[u8]) -> Result<Vec<u32>> {
    let mut marker = [0u32; 33];
    let mut words = vec![0u32; lengths.len()];
    let mut used = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let len = len as usize;
        if len == 0 {
            continue;
        }
        used += 1;
        let mut entry = marker[len];
        if len < 32 && (entry >> len) != 0 {
            return Err("overpopulated codebook");
        }
        words[i] = entry;
        for j in (1..=len).rev() {
            if marker[j] & 1 != 0 {
                if j == 1 {
                    marker[1] += 1;
                } else {
                    marker[j] = marker[j - 1] << 1;
                }
                break;
            }
            marker[j] += 1;
        }
        for j in len + 1..33 {
            if (marker[j] >> 1) == entry {
                entry = marker[j];
                marker[j] = marker[j - 1] << 1;
            } else {
                break;
            }
        }
    }
    if !(used == 1 && marker[2] == 2) {
        for (i, &m) in marker.iter().enumerate().skip(1) {
            if m & (u32::MAX >> (32 - i)) != 0 {
                return Err("underpopulated codebook");
            }
        }
    }
    for (w, &len) in words.iter_mut().zip(lengths) {
        let mut r = 0;
        for j in 0..len as u32 {
            r = (r << 1) | ((*w >> j) & 1);
        }
        *w = r;
    }
    Ok(words)
}

impl Codebook {
    fn read(r: &mut BitReader) -> Result<Self> {
        if r.read(24) != Some(0x564342) {
            return Err("codebook sync");
        }
        let dims = r.read(16).ok_or("eof")? as usize;
        let entries = r.read(24).ok_or("eof")? as usize;
        if entries == 0 || entries > 1 << 16 || dims == 0 {
            return Err("codebook size");
        }
        let mut lengths = vec![0u8; entries];
        if r.bit().ok_or("eof")? {
            let mut len = r.read(5).ok_or("eof")? + 1;
            let mut cur = 0;
            while cur < entries {
                let n = r.read(ilog((entries - cur) as u32)).ok_or("eof")? as usize;
                if cur + n > entries || len > 32 {
                    return Err("ordered lengths");
                }
                lengths[cur..cur + n].fill(len as u8);
                cur += n;
                len += 1;
            }
        } else {
            let sparse = r.bit().ok_or("eof")?;
            for l in lengths.iter_mut() {
                if !sparse || r.bit().ok_or("eof")? {
                    *l = r.read(5).ok_or("eof")? as u8 + 1;
                }
            }
        }
        let lookup = r.read(4).ok_or("eof")?;
        let mut values = Vec::new();
        if lookup == 1 || lookup == 2 {
            let min = float32_unpack(r.read(32).ok_or("eof")?);
            let delta = float32_unpack(r.read(32).ok_or("eof")?);
            let bits = r.read(4).ok_or("eof")? + 1;
            let sequence = r.bit().ok_or("eof")?;
            let count = if lookup == 1 { lookup1_values(entries as u32, dims as u32) as usize } else { entries * dims };
            let mult: Vec<u32> = (0..count).map(|_| r.read(bits).ok_or("eof")).collect::<Result<_>>()?;
            values = vec![0f32; entries * dims];
            for e in 0..entries {
                let mut last = 0f32;
                let mut div = 1usize;
                for d in 0..dims {
                    let m = if lookup == 1 {
                        let v = mult[(e / div) % count];
                        div = div.saturating_mul(count);
                        v
                    } else {
                        mult[e * dims + d]
                    };
                    let v = m as f32 * delta + min + last;
                    if sequence {
                        last = v;
                    }
                    values[e * dims + d] = v;
                }
            }
        } else if lookup != 0 {
            return Err("lookup type");
        }

        let words = make_words(&lengths)?;
        let used: Vec<usize> = (0..entries).filter(|&e| lengths[e] > 0).collect();
        let single = (used.len() == 1).then(|| used[0] as u16);
        let mut fast = vec![NONE; 1 << FAST_BITS];
        let mut long = Vec::new();
        for &e in &used {
            let (w, len) = (words[e], lengths[e] as u32);
            if len <= FAST_BITS {
                for k in 0..1u32 << (FAST_BITS - len) {
                    fast[(w | (k << len)) as usize] = e as u16;
                }
            } else {
                long.push((w, len as u8, e as u16));
            }
        }
        long.sort_by_key(|x| x.1);
        Ok(Self { dims, lengths, fast, long, single, values })
    }

    #[inline]
    fn decode(&self, r: &mut BitReader) -> Option<usize> {
        if let Some(e) = self.single {
            return r.skip(self.lengths[e as usize] as u32).then_some(e as usize);
        }
        let e = self.fast[r.peek(FAST_BITS) as usize];
        if e != NONE {
            return r.skip(self.lengths[e as usize] as u32).then_some(e as usize);
        }
        let p = r.peek(32);
        for &(w, len, e) in &self.long {
            let mask = if len >= 32 { u32::MAX } else { (1u32 << len) - 1 };
            if p & mask == w {
                return r.skip(len as u32).then_some(e as usize);
            }
        }
        None
    }

    #[inline]
    fn vector(&self, r: &mut BitReader) -> Option<&[f32]> {
        let e = self.decode(r)?;
        self.values.get(e * self.dims..(e + 1) * self.dims)
    }
}

struct Floor1 {
    partition_class: Vec<u8>,
    class_dims: Vec<u8>,
    class_sub: Vec<u8>,
    class_master: Vec<u8>,
    sub_books: Vec<[i16; 8]>,
    multiplier: i32,
    xs: Vec<i32>,
    /// Indices of xs in ascending x.
    order: Vec<usize>,
    low: Vec<usize>,
    high: Vec<usize>,
}

impl Floor1 {
    fn read(r: &mut BitReader, books: usize) -> Result<Self> {
        let partitions = r.read(5).ok_or("eof")? as usize;
        let partition_class: Vec<u8> = (0..partitions).map(|_| r.read(4).map(|v| v as u8).ok_or("eof")).collect::<Result<_>>()?;
        let classes = partition_class.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        let (mut class_dims, mut class_sub, mut class_master, mut sub_books) = (vec![0u8; classes], vec![0u8; classes], vec![0u8; classes], vec![[-1i16; 8]; classes]);
        for c in 0..classes {
            class_dims[c] = r.read(3).ok_or("eof")? as u8 + 1;
            class_sub[c] = r.read(2).ok_or("eof")? as u8;
            if class_sub[c] != 0 {
                class_master[c] = r.read(8).ok_or("eof")? as u8;
                if class_master[c] as usize >= books {
                    return Err("floor masterbook");
                }
            }
            for k in 0..1usize << class_sub[c] {
                sub_books[c][k] = r.read(8).ok_or("eof")? as i16 - 1;
                if sub_books[c][k] >= books as i16 {
                    return Err("floor subclass book");
                }
            }
        }
        let multiplier = r.read(2).ok_or("eof")? as i32 + 1;
        let rangebits = r.read(4).ok_or("eof")?;
        let mut xs = vec![0, 1 << rangebits];
        for &c in &partition_class {
            for _ in 0..class_dims[c as usize] {
                xs.push(r.read(rangebits).ok_or("eof")? as i32);
            }
        }
        if xs.len() > 65 {
            return Err("floor values");
        }
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by_key(|&i| xs[i]);
        let (mut low, mut high) = (vec![0usize; xs.len()], vec![0usize; xs.len()]);
        for i in 2..xs.len() {
            let (mut lo, mut hi) = (0usize, 1usize);
            for j in 0..i {
                if xs[j] < xs[i] && xs[j] > xs[lo] {
                    lo = j;
                }
                if xs[j] > xs[i] && xs[j] < xs[hi] {
                    hi = j;
                }
            }
            low[i] = lo;
            high[i] = hi;
        }
        Ok(Self { partition_class, class_dims, class_sub, class_master, sub_books, multiplier, xs, order, low, high })
    }

    /// Fills `out` (n/2) with the curve; false when the channel is unused or the packet ended.
    fn decode(&self, r: &mut BitReader, books: &[Codebook], out: &mut [f32]) -> bool {
        if r.bit() != Some(true) {
            return false;
        }
        let range = [256, 128, 86, 64][self.multiplier as usize - 1];
        let bits = ilog(range as u32 - 1);
        let mut y = [0i32; 65];
        let Some(y0) = r.read(bits) else { return false };
        let Some(y1) = r.read(bits) else { return false };
        y[0] = y0 as i32;
        y[1] = y1 as i32;
        let mut at = 2;
        for &c in &self.partition_class {
            let c = c as usize;
            let cdim = self.class_dims[c] as usize;
            let cbits = self.class_sub[c] as u32;
            let csub = (1i32 << cbits) - 1;
            let mut cval = 0i32;
            if cbits > 0 {
                let Some(v) = books[self.class_master[c] as usize].decode(r) else { return false };
                cval = v as i32;
            }
            for _ in 0..cdim {
                let book = self.sub_books[c][(cval & csub) as usize];
                cval >>= cbits;
                y[at] = if book >= 0 {
                    let Some(v) = books[book as usize].decode(r) else { return false };
                    v as i32
                } else {
                    0
                };
                at += 1;
            }
        }

        // Amplitude values to final ys.
        let n = self.xs.len();
        let mut fy = [0i32; 65];
        let mut step2 = [false; 65];
        fy[0] = y[0];
        fy[1] = y[1];
        step2[0] = true;
        step2[1] = true;
        for i in 2..n {
            let (lo, hi) = (self.low[i], self.high[i]);
            let predicted = render_point(self.xs[lo], fy[lo], self.xs[hi], fy[hi], self.xs[i]);
            let val = y[i];
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom { highroom } else { lowroom } * 2;
            if val != 0 {
                step2[lo] = true;
                step2[hi] = true;
                step2[i] = true;
                fy[i] = if val >= room {
                    if highroom > lowroom {
                        val - lowroom + predicted
                    } else {
                        predicted - val + highroom - 1
                    }
                } else if val & 1 != 0 {
                    predicted - (val + 1) / 2
                } else {
                    predicted + val / 2
                };
            } else {
                fy[i] = predicted;
            }
        }

        // Curve synthesis in x order.
        let half = out.len() as i32;
        let (mut lx, mut ly) = (0i32, fy[self.order[0]] * self.multiplier);
        let mut hy = ly;
        let mut hx = 0;
        for &i in &self.order[1..] {
            if step2[i] {
                hy = fy[i] * self.multiplier;
                hx = self.xs[i];
                render_line(lx, ly, hx, hy, out);
                lx = hx;
                ly = hy;
            }
        }
        if hx < half {
            render_line(hx, hy, half, hy, out);
        }
        true
    }
}

fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let off = dy.abs() * (x - x0) / adx;
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, v: &mut [f32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    if adx <= 0 {
        return;
    }
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let ady = dy.abs() - base.abs() * adx;
    let (mut y, mut err) = (y0, 0);
    let n = v.len() as i32;
    let put = |v: &mut [f32], x: i32, y: i32| {
        if x < n {
            v[x as usize] = INVERSE_DB[y.clamp(0, 255) as usize];
        }
    };
    put(v, x0, y);
    for x in x0 + 1..x1 {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        put(v, x, y);
    }
}

struct Residue {
    kind: u16,
    begin: usize,
    end: usize,
    partition: usize,
    classifications: usize,
    classbook: usize,
    books: Vec<[i16; 8]>,
}

impl Residue {
    fn read(r: &mut BitReader, kind: u16, nbooks: usize) -> Result<Self> {
        let begin = r.read(24).ok_or("eof")? as usize;
        let end = r.read(24).ok_or("eof")? as usize;
        let partition = r.read(24).ok_or("eof")? as usize + 1;
        let classifications = r.read(6).ok_or("eof")? as usize + 1;
        let classbook = r.read(8).ok_or("eof")? as usize;
        if classbook >= nbooks {
            return Err("residue classbook");
        }
        let mut cascade = vec![0u32; classifications];
        for c in cascade.iter_mut() {
            let low = r.read(3).ok_or("eof")?;
            let high = if r.bit().ok_or("eof")? { r.read(5).ok_or("eof")? } else { 0 };
            *c = high * 8 + low;
        }
        let mut books = vec![[-1i16; 8]; classifications];
        for (c, &bits) in cascade.iter().enumerate() {
            for (k, b) in books[c].iter_mut().enumerate() {
                if bits & (1 << k) != 0 {
                    *b = r.read(8).ok_or("eof")? as i16;
                    if *b as usize >= nbooks {
                        return Err("residue book");
                    }
                }
            }
        }
        Ok(Self { kind, begin, end, partition, classifications, classbook, books })
    }

    /// Adds the residue of the channels in `vecs` (each `size` long) for which `decode` is set.
    fn decode(&self, r: &mut BitReader, books: &[Codebook], vecs: &mut [&mut [f32]], decode: &[bool], size: usize, scratch: &mut Vec<f32>) {
        if self.kind == 2 {
            if !decode.iter().any(|&d| d) {
                return;
            }
            let ch = vecs.len();
            scratch.clear();
            scratch.resize(size * ch, 0.0);
            self.decode_format(r, books, &mut [scratch.as_mut_slice()], &[true], size * ch, 1);
            for (i, v) in scratch.iter().enumerate() {
                vecs[i % ch][i / ch] += v;
            }
        } else {
            self.decode_format(r, books, vecs, decode, size, self.kind);
        }
    }

    fn decode_format(&self, r: &mut BitReader, books: &[Codebook], vecs: &mut [&mut [f32]], decode: &[bool], size: usize, format: u16) {
        let begin = self.begin.min(size);
        let end = self.end.min(size);
        let classbook = &books[self.classbook];
        let per_word = classbook.dims;
        let to_read = (end - begin) / self.partition;
        if to_read == 0 {
            return;
        }
        let ch = vecs.len();
        let mut classes = vec![0u8; ch * (to_read + per_word)];
        for pass in 0..8 {
            let mut p = 0;
            while p < to_read {
                if pass == 0 {
                    for j in 0..ch {
                        if !decode[j] {
                            continue;
                        }
                        let Some(mut word) = classbook.decode(r) else { return };
                        for i in (0..per_word).rev() {
                            classes[j * (to_read + per_word) + p + i] = (word % self.classifications) as u8;
                            word /= self.classifications;
                        }
                    }
                }
                let mut i = 0;
                while i < per_word && p < to_read {
                    for j in 0..ch {
                        if !decode[j] {
                            continue;
                        }
                        let class = classes[j * (to_read + per_word) + p] as usize;
                        let book = self.books[class][pass];
                        if book < 0 {
                            continue;
                        }
                        let book = &books[book as usize];
                        if book.values.is_empty() {
                            return;
                        }
                        let off = begin + p * self.partition;
                        let v = &mut vecs[j][off..off + self.partition];
                        if format == 0 {
                            let step = self.partition / book.dims;
                            for k in 0..step {
                                let Some(e) = book.vector(r) else { return };
                                for (d, &x) in e.iter().enumerate() {
                                    v[k + d * step] += x;
                                }
                            }
                        } else {
                            let mut k = 0;
                            while k < self.partition {
                                let Some(e) = book.vector(r) else { return };
                                for &x in e {
                                    if k < self.partition {
                                        v[k] += x;
                                    }
                                    k += 1;
                                }
                            }
                        }
                    }
                    i += 1;
                    p += 1;
                }
            }
        }
    }
}

struct Mapping {
    coupling: Vec<(usize, usize)>,
    mux: Vec<usize>,
    submaps: Vec<(usize, usize)>, // (floor, residue)
}

struct Mode {
    long: bool,
    mapping: usize,
}

/// n-point inverse MDCT: y[i] = sum_k x[k] cos(2π/n (i + 1/2 + n/4)(k + 1/2)).
struct Imdct {
    n: usize,
    pre: Vec<(f32, f32)>,
    post: Vec<(f32, f32)>,
    fft: Fft,
}

impl Imdct {
    fn new(n: usize) -> Self {
        let m = n / 2;
        let p = m / 2;
        let pre = (0..p).map(|j| {
            let a = -std::f64::consts::PI * (j as f64 + 0.25) / m as f64;
            (a.cos() as f32, a.sin() as f32)
        });
        let post = (0..p).map(|k| {
            let a = -std::f64::consts::PI * k as f64 / m as f64;
            (a.cos() as f32, a.sin() as f32)
        });
        Self { n, pre: pre.collect(), post: post.collect(), fft: Fft::new(p) }
    }

    /// `x` has n/2 coefficients; `y` gets n samples.
    fn run(&self, x: &[f32], y: &mut [f32]) {
        let (n, m) = (self.n, self.n / 2);
        let p = m / 2;
        // c[k] = (-1)^k DCT-IV(r)[k], r[j] = (-1)^j x[m - 1 - j]; y[n/4 + k] = c[k].
        let r = |j: usize| if j & 1 == 0 { x[m - 1 - j] } else { -x[m - 1 - j] };
        let mut re = vec![0f32; p];
        let mut im = vec![0f32; p];
        for j in 0..p {
            let (a, b) = (r(2 * j), r(m - 1 - 2 * j));
            let (c, s) = self.pre[j];
            re[j] = a * c - b * s;
            im[j] = a * s + b * c;
        }
        self.fft.run(&mut re, &mut im);
        let q = n / 4;
        for k in 0..p {
            let (c, s) = self.post[k];
            let ur = re[k] * c - im[k] * s;
            let ui = re[k] * s + im[k] * c;
            // DCT-IV: C[2k] = Re u, C[m-1-2k] = -Im u; then the (-1)^index sign.
            let even = ur;
            let odd_idx = m - 1 - 2 * k;
            let odd = -ui;
            y[q + 2 * k] = even; // index 2k is even
            y[q + odd_idx] = if odd_idx & 1 == 0 { odd } else { -odd };
        }
        for i in 0..q {
            y[i] = -y[m - 1 - i];
        }
        for i in 3 * q..n {
            y[i] = y[3 * m - 1 - i];
        }
    }
}

/// Radix-2 complex FFT, forward (e^{-2πi jk/n}).
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<u32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        let cos = (0..n / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / n as f64).cos() as f32).collect();
        let sin = (0..n / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / n as f64).sin() as f32).collect();
        let rev = (0..n as u32).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (32 - bits) }).collect();
        Self { n, cos, sin, rev }
    }

    fn run(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (c, s) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }
}

pub struct Decoder {
    channels: usize,
    rate: u32,
    blocksize: [usize; 2],
    books: Vec<Codebook>,
    floors: Vec<Floor1>,
    residues: Vec<Residue>,
    mappings: Vec<Mapping>,
    modes: Vec<Mode>,
    mode_bits: u32,
    imdct: [Imdct; 2],
    /// Windowed output of the last block, per channel.
    prev: Vec<Vec<f32>>,
    prev_n: usize,
    have_prev: bool,
    scratch: Vec<f32>,
}

fn header(r: &mut BitReader, kind: u32) -> Result<()> {
    if r.read(8) != Some(kind) {
        return Err("header type");
    }
    for &c in b"vorbis" {
        if r.read(8) != Some(c as u32) {
            return Err("header magic");
        }
    }
    Ok(())
}

impl Decoder {
    pub fn new(ident: &[u8], setup: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(ident);
        header(&mut r, 1)?;
        if r.read(32) != Some(0) {
            return Err("vorbis version");
        }
        let channels = r.read(8).ok_or("eof")? as usize;
        let rate = r.read(32).ok_or("eof")?;
        r.read(32);
        r.read(32);
        r.read(32);
        let b0 = r.read(4).ok_or("eof")?;
        let b1 = r.read(4).ok_or("eof")?;
        if channels == 0 || rate == 0 || !(6..=13).contains(&b0) || !(6..=13).contains(&b1) || b0 > b1 {
            return Err("identification values");
        }
        let blocksize = [1usize << b0, 1usize << b1];

        let mut r = BitReader::new(setup);
        header(&mut r, 5)?;
        let nbooks = r.read(8).ok_or("eof")? as usize + 1;
        let books: Vec<Codebook> = (0..nbooks).map(|_| Codebook::read(&mut r)).collect::<Result<_>>()?;
        let times = r.read(6).ok_or("eof")? + 1;
        for _ in 0..times {
            if r.read(16) != Some(0) {
                return Err("time domain type");
            }
        }
        let nfloors = r.read(6).ok_or("eof")? as usize + 1;
        let mut floors = Vec::with_capacity(nfloors);
        for _ in 0..nfloors {
            if r.read(16) != Some(1) {
                return Err("floor type 0 not supported");
            }
            floors.push(Floor1::read(&mut r, nbooks)?);
        }
        let nres = r.read(6).ok_or("eof")? as usize + 1;
        let mut residues = Vec::with_capacity(nres);
        for _ in 0..nres {
            let kind = r.read(16).ok_or("eof")? as u16;
            if kind > 2 {
                return Err("residue type");
            }
            residues.push(Residue::read(&mut r, kind, nbooks)?);
        }
        let nmaps = r.read(6).ok_or("eof")? as usize + 1;
        let mut mappings = Vec::with_capacity(nmaps);
        for _ in 0..nmaps {
            if r.read(16) != Some(0) {
                return Err("mapping type");
            }
            let submaps = if r.bit().ok_or("eof")? { r.read(4).ok_or("eof")? as usize + 1 } else { 1 };
            let mut coupling = Vec::new();
            if r.bit().ok_or("eof")? {
                let steps = r.read(8).ok_or("eof")? as usize + 1;
                let bits = ilog(channels as u32 - 1);
                for _ in 0..steps {
                    let m = r.read(bits).ok_or("eof")? as usize;
                    let a = r.read(bits).ok_or("eof")? as usize;
                    if m == a || m >= channels || a >= channels {
                        return Err("coupling");
                    }
                    coupling.push((m, a));
                }
            }
            if r.read(2) != Some(0) {
                return Err("mapping reserved");
            }
            let mut mux = vec![0usize; channels];
            if submaps > 1 {
                for m in mux.iter_mut() {
                    *m = r.read(4).ok_or("eof")? as usize;
                    if *m >= submaps {
                        return Err("mapping mux");
                    }
                }
            }
            let mut subs = Vec::with_capacity(submaps);
            for _ in 0..submaps {
                r.read(8);
                let f = r.read(8).ok_or("eof")? as usize;
                let res = r.read(8).ok_or("eof")? as usize;
                if f >= nfloors || res >= nres {
                    return Err("submap");
                }
                subs.push((f, res));
            }
            mappings.push(Mapping { coupling, mux, submaps: subs });
        }
        let nmodes = r.read(6).ok_or("eof")? as usize + 1;
        let mut modes = Vec::with_capacity(nmodes);
        for _ in 0..nmodes {
            let long = r.bit().ok_or("eof")?;
            r.read(16);
            r.read(16);
            let mapping = r.read(8).ok_or("eof")? as usize;
            if mapping >= nmaps {
                return Err("mode mapping");
            }
            modes.push(Mode { long, mapping });
        }
        if r.bit() != Some(true) {
            return Err("setup framing");
        }
        Ok(Self {
            channels,
            rate,
            blocksize,
            books,
            floors,
            residues,
            mappings,
            modes,
            mode_bits: ilog(nmodes as u32 - 1),
            imdct: [Imdct::new(blocksize[0]), Imdct::new(blocksize[1])],
            prev: vec![Vec::new(); channels],
            prev_n: 0,
            have_prev: false,
            scratch: Vec::new(),
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Decodes one audio packet and appends the samples it completes to `out[ch]`. Returns how
    /// many samples per channel were appended (0 for the first packet).
    pub fn decode(&mut self, packet: &[u8], out: &mut [Vec<f32>]) -> Result<usize> {
        let mut r = BitReader::new(packet);
        if r.bit() != Some(false) {
            return Err("not an audio packet");
        }
        let mode = r.read(self.mode_bits).ok_or("eof")? as usize;
        let mode = self.modes.get(mode).ok_or("mode")?;
        let long = mode.long as usize;
        let n = self.blocksize[long];
        let (prev_long, next_long) = if mode.long { (r.bit().unwrap_or(false), r.bit().unwrap_or(false)) } else { (false, false) };
        let map = &self.mappings[mode.mapping];
        let half = n / 2;

        let mut floor = vec![vec![0f32; half]; self.channels];
        let mut used = vec![false; self.channels];
        for ch in 0..self.channels {
            let f = map.submaps[map.mux[ch]].0;
            used[ch] = self.floors[f].decode(&mut r, &self.books, &mut floor[ch]);
        }
        let mut decode = used.clone();
        for &(m, a) in &map.coupling {
            if decode[m] || decode[a] {
                decode[m] = true;
                decode[a] = true;
            }
        }
        let mut res = vec![vec![0f32; half]; self.channels];
        for (s, &(_, ri)) in map.submaps.iter().enumerate() {
            let chs: Vec<usize> = (0..self.channels).filter(|&c| map.mux[c] == s).collect();
            let flags: Vec<bool> = chs.iter().map(|&c| decode[c]).collect();
            let mut taken: Vec<&mut [f32]> = Vec::with_capacity(chs.len());
            // Disjoint mutable borrows of the submap's channels.
            let mut rest: &mut [Vec<f32>] = &mut res;
            let mut base = 0;
            for &c in &chs {
                let (head, tail) = rest.split_at_mut(c - base + 1);
                taken.push(head[c - base].as_mut_slice());
                rest = tail;
                base = c + 1;
            }
            self.residues[ri].decode(&mut r, &self.books, &mut taken, &flags, half, &mut self.scratch);
        }
        for &(m, a) in map.coupling.iter().rev() {
            for i in 0..half {
                let (mv, av) = (res[m][i], res[a][i]);
                let (nm, na) = if mv > 0.0 {
                    if av > 0.0 {
                        (mv, mv - av)
                    } else {
                        (mv + av, mv)
                    }
                } else if av > 0.0 {
                    (mv, mv + av)
                } else {
                    (mv - av, mv)
                };
                res[m][i] = nm;
                res[a][i] = na;
            }
        }

        let window = window(n, self.blocksize[0], mode.long, prev_long, next_long);
        let mut produced = 0;
        let mut y = vec![0f32; n];
        for ch in 0..self.channels {
            if used[ch] {
                for i in 0..half {
                    res[ch][i] *= floor[ch][i];
                }
                self.imdct[long].run(&res[ch], &mut y);
                for (v, w) in y.iter_mut().zip(&window) {
                    *v *= w;
                }
            } else {
                y.fill(0.0);
            }
            if self.have_prev {
                let pn = self.prev_n;
                let count = pn / 4 + n / 4;
                let prev = &self.prev[ch];
                let o = &mut out[ch];
                o.reserve(count);
                for s in 0..count {
                    let t = s as isize - (pn / 4) as isize;
                    let pi = (3 * pn / 4) as isize + t;
                    let ci = (n / 4) as isize + t;
                    let a = if pi >= 0 && (pi as usize) < pn { prev[pi as usize] } else { 0.0 };
                    let b = if ci >= 0 && (ci as usize) < n { y[ci as usize] } else { 0.0 };
                    o.push(a + b);
                }
                produced = count;
            }
            std::mem::swap(&mut self.prev[ch], &mut y);
            y.resize(n, 0.0);
        }
        self.prev_n = n;
        self.have_prev = true;
        Ok(produced)
    }
}

fn slope(i: usize, start: usize, len: usize, rising: bool) -> f32 {
    let x = (i - start) as f64 + 0.5;
    let p = std::f64::consts::FRAC_PI_2;
    let inner = x / len as f64 * p + if rising { 0.0 } else { p };
    (p * inner.sin().powi(2)).sin() as f32
}

fn window(n: usize, short: usize, long: bool, prev_long: bool, next_long: bool) -> Vec<f32> {
    let center = n / 2;
    let (ls, le, ln) = if long && !prev_long { (n / 4 - short / 4, n / 4 + short / 4, short / 2) } else { (0, center, n / 2) };
    let (rs, re, rn) = if long && !next_long { (3 * n / 4 - short / 4, 3 * n / 4 + short / 4, short / 2) } else { (center, n, n / 2) };
    (0..n)
        .map(|i| {
            if i < ls {
                0.0
            } else if i < le {
                slope(i, ls, ln, true)
            } else if i < rs {
                1.0
            } else if i < re {
                slope(i, rs, rn, false)
            } else {
                0.0
            }
        })
        .collect()
}

/// floor1_inverse_dB_table: 256 steps of 35/64 dB up to 1.0.
static INVERSE_DB: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = 10f64.powf((i as f64 - 255.0) * (35.0 / 64.0) / 20.0) as f32;
    }
    t
});

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_imdct(x: &[f32]) -> Vec<f32> {
        let n = x.len() * 2;
        (0..n)
            .map(|i| {
                x.iter()
                    .enumerate()
                    .map(|(k, &v)| v as f64 * (2.0 * std::f64::consts::PI / n as f64 * (i as f64 + 0.5 + n as f64 / 4.0) * (k as f64 + 0.5)).cos())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn imdct_matches_definition() {
        for n in [64usize, 256, 2048] {
            let x: Vec<f32> = (0..n / 2).map(|k| ((k * 7919 % 113) as f32 / 56.0 - 1.0) / (1.0 + k as f32 * 0.01)).collect();
            let want = direct_imdct(&x);
            let mut got = vec![0f32; n];
            Imdct::new(n).run(&x, &mut got);
            let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |a, b| a.max(b.abs()));
            assert!(err < 1e-4 * scale.max(1.0), "n {n}: max error {err} of {scale}");
        }
    }

    #[test]
    fn inverse_db_ends() {
        assert!((INVERSE_DB[0] - 1.0649863e-07).abs() < 1e-12);
        assert!((INVERSE_DB[1] - 1.1341951e-07).abs() < 1e-12);
        assert_eq!(INVERSE_DB[255], 1.0);
    }

    #[test]
    fn princeton_windows_sum_to_one() {
        // Overlapping slopes are power complementary: w(i)^2 + w(i + n/2)^2 = 1 for a short block.
        let w = window(256, 256, false, false, false);
        for i in 0..128 {
            assert!((w[i] * w[i] + w[i + 128] * w[i + 128] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn codewords_fill_the_tree() {
        // Lengths 2,4,4,4,4,2,3,3 from the specification's example.
        let w = make_words(&[2, 4, 4, 4, 4, 2, 3, 3]).unwrap();
        let msb: Vec<u32> = w.iter().zip([2u32, 4, 4, 4, 4, 2, 3, 3]).map(|(&c, l)| c.reverse_bits() >> (32 - l)).collect();
        assert_eq!(msb, vec![0b00, 0b0100, 0b0101, 0b0110, 0b0111, 0b10, 0b110, 0b111]);
    }
}
