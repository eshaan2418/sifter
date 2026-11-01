//! Posting lists in blocks of 128 documents.
//!
//! A term's postings are stored as
//!
//! ```text
//! u32 num_blocks
//! num_blocks x BlockHeader (24 bytes, fixed size, so cursors can skip blocks
//!                           without decoding them)
//! packed block data
//! ```
//!
//! Each block bitpacks its doc-id deltas and (tf - 1) values with the
//! smallest bit widths that fit. The header also records the block's max tf
//! and min document length, which give block-max WAND a safe upper bound on
//! any score inside the block.

pub const BLOCK: usize = 128;
pub const HEADER_SIZE: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    pub last_doc: u32,
    pub data_offset: u32, // relative to the start of block data
    pub max_tf: u32,
    pub min_len: u32,
    pub first_doc: u32,
    pub n: u8,
    pub doc_bits: u8,
    pub tf_bits: u8,
}

impl BlockHeader {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.last_doc.to_le_bytes());
        out.extend_from_slice(&self.data_offset.to_le_bytes());
        out.extend_from_slice(&self.max_tf.to_le_bytes());
        out.extend_from_slice(&self.min_len.to_le_bytes());
        out.extend_from_slice(&self.first_doc.to_le_bytes());
        out.extend_from_slice(&[self.n, self.doc_bits, self.tf_bits, 0]);
    }

    pub fn read(b: &[u8]) -> BlockHeader {
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        BlockHeader {
            last_doc: u(0),
            data_offset: u(4),
            max_tf: u(8),
            min_len: u(12),
            first_doc: u(16),
            n: b[20],
            doc_bits: b[21],
            tf_bits: b[22],
        }
    }
}

fn bits_needed(v: u32) -> u8 {
    (32 - v.leading_zeros()) as u8
}

/// Packs `vals` with `bits` bits each, little-endian within a u64 stream.
pub fn pack(vals: &[u32], bits: u8, out: &mut Vec<u8>) {
    if bits == 0 {
        return;
    }
    let mut acc: u64 = 0;
    let mut filled = 0u32;
    for &v in vals {
        acc |= (v as u64) << filled;
        filled += bits as u32;
        while filled >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            filled -= 8;
        }
    }
    if filled > 0 {
        out.push(acc as u8);
    }
}

pub fn unpack(data: &[u8], n: usize, bits: u8, out: &mut [u32]) -> usize {
    if bits == 0 {
        out[..n].fill(0);
        return 0;
    }
    let mask = if bits == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits) - 1
    };
    let mut acc: u64 = 0;
    let mut have = 0u32;
    let mut pos = 0;
    for slot in out.iter_mut().take(n) {
        while have < bits as u32 {
            acc |= (data[pos] as u64) << have;
            pos += 1;
            have += 8;
        }
        *slot = (acc & mask) as u32;
        acc >>= bits;
        have -= bits as u32;
    }
    (n * bits as usize).div_ceil(8)
}

/// Encodes a term's postings. `docs` strictly increasing; `lens[doc]` is the
/// document length table of the segment.
pub fn encode(docs: &[u32], tfs: &[u32], lens: &[u32], out: &mut Vec<u8>) {
    let nblocks = docs.len().div_ceil(BLOCK);
    out.extend_from_slice(&(nblocks as u32).to_le_bytes());
    let mut headers = Vec::with_capacity(nblocks);
    let mut data = Vec::new();
    let mut deltas = [0u32; BLOCK];
    let mut tfm1 = [0u32; BLOCK];
    for (bi, chunk) in docs.chunks(BLOCK).enumerate() {
        let base = bi * BLOCK;
        let n = chunk.len();
        let prev = if bi == 0 { 0 } else { docs[base - 1] };
        for i in 0..n {
            deltas[i] = chunk[i] - if i == 0 { prev } else { chunk[i - 1] };
            tfm1[i] = tfs[base + i] - 1;
        }
        let doc_bits = deltas[..n]
            .iter()
            .map(|&d| bits_needed(d))
            .max()
            .unwrap_or(0);
        let tf_bits = tfm1[..n].iter().map(|&t| bits_needed(t)).max().unwrap_or(0);
        let header = BlockHeader {
            last_doc: chunk[n - 1],
            data_offset: data.len() as u32,
            max_tf: tfs[base..base + n].iter().copied().max().unwrap(),
            min_len: chunk.iter().map(|&d| lens[d as usize]).min().unwrap(),
            first_doc: chunk[0],
            n: n as u8,
            doc_bits,
            tf_bits,
        };
        pack(&deltas[..n], doc_bits, &mut data);
        pack(&tfm1[..n], tf_bits, &mut data);
        headers.push(header);
    }
    for h in &headers {
        h.write(out);
    }
    out.extend_from_slice(&data);
}

/// Forward cursor over one term's postings.
pub struct Cursor<'a> {
    headers: &'a [u8],
    data: &'a [u8],
    nblocks: usize,
    block: usize,
    loaded: usize, // block whose docs/tfs are decoded, or usize::MAX
    docs: [u32; BLOCK],
    tfs: [u32; BLOCK],
    pos: usize,
    pub doc: u32, // current doc, or u32::MAX when exhausted
}

impl<'a> Cursor<'a> {
    pub fn new(postings: &'a [u8]) -> Cursor<'a> {
        let nblocks = u32::from_le_bytes(postings[0..4].try_into().unwrap()) as usize;
        let headers = &postings[4..4 + nblocks * HEADER_SIZE];
        let data = &postings[4 + nblocks * HEADER_SIZE..];
        let mut c = Cursor {
            headers,
            data,
            nblocks,
            block: 0,
            loaded: usize::MAX,
            docs: [0; BLOCK],
            tfs: [0; BLOCK],
            pos: 0,
            doc: 0,
        };
        if nblocks == 0 {
            c.doc = u32::MAX;
        } else {
            c.load(0);
            c.doc = c.docs[0];
        }
        c
    }

    pub fn header(&self, b: usize) -> BlockHeader {
        BlockHeader::read(&self.headers[b * HEADER_SIZE..])
    }

    pub fn num_blocks(&self) -> usize {
        self.nblocks
    }

    pub fn current_block(&self) -> usize {
        self.block
    }

    fn load(&mut self, b: usize) {
        let h = self.header(b);
        let n = h.n as usize;
        let d = &self.data[h.data_offset as usize..];
        let used = unpack(d, n, h.doc_bits, &mut self.docs);
        unpack(&d[used..], n, h.tf_bits, &mut self.tfs);
        let mut prev = if b == 0 {
            0
        } else {
            self.header(b - 1).last_doc
        };
        for i in 0..n {
            prev += self.docs[i];
            self.docs[i] = prev;
            self.tfs[i] += 1;
        }
        self.block = b;
        self.loaded = b;
        self.pos = 0;
    }

    pub fn tf(&self) -> u32 {
        self.tfs[self.pos]
    }

    pub fn next(&mut self) {
        let n = self.header(self.block).n as usize;
        self.pos += 1;
        if self.pos < n {
            self.doc = self.docs[self.pos];
        } else if self.block + 1 < self.nblocks {
            self.load(self.block + 1);
            self.doc = self.docs[0];
        } else {
            self.doc = u32::MAX;
        }
    }

    /// Moves to the first block whose last doc >= target, without decoding.
    /// Returns that block's header, or None if past the end.
    pub fn shallow_seek(&mut self, target: u32) -> Option<BlockHeader> {
        let mut b = self.block;
        while b < self.nblocks && self.header(b).last_doc < target {
            b += 1;
        }
        if b >= self.nblocks {
            return None;
        }
        self.block = b;
        Some(self.header(b))
    }

    /// Advances to the first doc >= target (u32::MAX if none).
    pub fn seek(&mut self, target: u32) {
        if self.doc >= target {
            return;
        }
        let Some(h) = self.shallow_seek(target) else {
            self.doc = u32::MAX;
            return;
        };
        if self.loaded != self.block {
            self.load(self.block);
        }
        let n = h.n as usize;
        while self.pos < n && self.docs[self.pos] < target {
            self.pos += 1;
        }
        self.doc = self.docs[self.pos];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(docs: &[u32], tfs: &[u32]) {
        let max_doc = docs.last().copied().unwrap_or(0) as usize;
        let lens: Vec<u32> = (0..=max_doc).map(|d| (d % 97) as u32 + 1).collect();
        let mut buf = Vec::new();
        encode(docs, tfs, &lens, &mut buf);
        let mut c = Cursor::new(&buf);
        for (i, (&d, &t)) in docs.iter().zip(tfs).enumerate() {
            assert_eq!(c.doc, d, "doc {i}");
            assert_eq!(c.tf(), t, "tf {i}");
            c.next();
        }
        assert_eq!(c.doc, u32::MAX);
    }

    #[test]
    fn pack_unpack_all_widths() {
        for bits in 0..=32u8 {
            let max = if bits == 0 {
                0
            } else if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let vals: Vec<u32> = (0..100u32)
                .map(|i| if bits == 0 { 0 } else { max - i.min(max) })
                .collect();
            let mut buf = Vec::new();
            pack(&vals, bits, &mut buf);
            buf.extend_from_slice(&[0; 8]);
            let mut out = [0u32; 128];
            unpack(&buf, vals.len(), bits, &mut out);
            assert_eq!(&out[..vals.len()], &vals[..], "bits={bits}");
        }
    }

    #[test]
    fn roundtrips_various_shapes() {
        roundtrip(&[5], &[1]);
        roundtrip(&(0..128).collect::<Vec<_>>(), &vec![1; 128]);
        roundtrip(
            &(0..129).collect::<Vec<_>>(),
            &(1..=129).collect::<Vec<_>>(),
        );
        let sparse: Vec<u32> = (0..1000).map(|i| i * 4093 + (i % 7)).collect();
        let tfs: Vec<u32> = (0..1000).map(|i| (i * 31 % 50) + 1).collect();
        roundtrip(&sparse, &tfs);
        // Wide gaps and large tfs (doc ids kept small enough that the test's
        // dense length table stays a few MB).
        roundtrip(&[0, 1, 1 << 20], &[1, 1_000_000, 3]);
    }

    #[test]
    fn seek_and_block_metadata() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let tfs: Vec<u32> = (0..1000).map(|i| i % 9 + 1).collect();
        let lens: Vec<u32> = (0..3000).map(|d| d % 50 + 10).collect();
        let mut buf = Vec::new();
        encode(&docs, &tfs, &lens, &mut buf);
        let mut c = Cursor::new(&buf);
        assert_eq!(c.num_blocks(), 8);
        c.seek(301);
        assert_eq!(c.doc, 303);
        assert_eq!(c.tf(), 101 % 9 + 1); // doc 303 is entry 101
        c.seek(303);
        assert_eq!(c.doc, 303); // seek never moves backwards
        c.seek(2000);
        assert_eq!(c.doc, 2001);
        c.seek(3000);
        assert_eq!(c.doc, u32::MAX);
        let c = Cursor::new(&buf);
        let h = c.header(0);
        assert_eq!((h.first_doc, h.last_doc, h.n), (0, 381, 128));
        assert_eq!(h.max_tf, 9);
        assert_eq!(h.min_len, 10);
    }
}
