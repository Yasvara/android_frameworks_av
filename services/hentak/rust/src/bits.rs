//! LSB-first bit packing, as Vorbis and Wwise's stripped setup use it.

pub struct BitReader<'a> {
    b: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitReader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    /// Up to 32 bits; None past the end (the packet ended).
    #[inline]
    pub fn read(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if self.pos + n as usize > self.b.len() * 8 {
            self.pos = self.b.len() * 8;
            return None;
        }
        let v = self.peek_raw(n);
        self.pos += n as usize;
        Some(v)
    }

    #[inline]
    pub fn bit(&mut self) -> Option<bool> {
        self.read(1).map(|v| v != 0)
    }

    /// The next n ≤ 32 bits without consuming them, zero-filled past the end.
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        self.peek_raw(n)
    }

    #[inline]
    fn peek_raw(&self, n: u32) -> u32 {
        let byte = self.pos >> 3;
        let shift = (self.pos & 7) as u32;
        let mut w = 0u64;
        for i in 0..5 {
            if let Some(&c) = self.b.get(byte + i) {
                w |= (c as u64) << (8 * i);
            }
        }
        ((w >> shift) & ((1u64 << n) - 1)) as u32
    }

    /// Consumes n bits a peek already looked at; false past the end.
    #[inline]
    pub fn skip(&mut self, n: u32) -> bool {
        if self.pos + n as usize > self.b.len() * 8 {
            self.pos = self.b.len() * 8;
            return false;
        }
        self.pos += n as usize;
        true
    }

    pub fn bits_read(&self) -> usize {
        self.pos
    }

    pub fn bits_left(&self) -> usize {
        self.b.len() * 8 - self.pos
    }
}

#[derive(Default)]
pub struct BitWriter {
    pub out: Vec<u8>,
    bits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write(&mut self, v: u32, n: u32) {
        for i in 0..n {
            if self.bits == 0 {
                self.out.push(0);
            }
            if (v >> i) & 1 != 0 {
                *self.out.last_mut().unwrap() |= 1 << self.bits;
            }
            self.bits = (self.bits + 1) & 7;
        }
    }

    pub fn bytes(&mut self, b: &[u8]) {
        for &c in b {
            self.write(c as u32, 8);
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

pub fn ilog(v: u32) -> u32 {
    32 - v.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = BitWriter::new();
        w.write(5, 3);
        w.write(0x1234, 13);
        w.write(1, 1);
        w.write(0xdead_beef, 32);
        let b = w.finish();
        let mut r = BitReader::new(&b);
        assert_eq!(r.read(3), Some(5));
        assert_eq!(r.read(13), Some(0x1234));
        assert_eq!(r.bit(), Some(true));
        assert_eq!(r.read(32), Some(0xdead_beef));
        assert_eq!(r.read(8), None);
    }
}
