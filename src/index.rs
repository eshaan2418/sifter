//! Segments, the index directory, and merging.
//!
//! A segment file (`seg_<n>.sft`) is immutable:
//!
//! ```text
//! "SIFT" u32 version
//! postings for every term (see postings.rs), back to back
//! doc lengths: u32 per doc
//! doc ids:     u32 len + bytes per doc
//! term dict:   per term: u16 len + bytes, u32 df, u64 postings offset, u32 postings len
//! footer:      u64 lens_off, u64 ids_off, u64 dict_off, u32 n_terms, u32 n_docs, u64 total_len, "TFIS"
//! ```
//!
//! Terms are sorted, so lookups binary-search an offsets table built at open.
//! `meta.json` lists the live segments and is replaced atomically on commit.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::analysis::analyze_into;
use crate::postings;

const MAGIC: &[u8; 4] = b"SIFT";
const FOOTER_MAGIC: &[u8; 4] = b"TFIS";
const VERSION: u32 = 1;
const FOOTER: usize = 8 * 3 + 4 + 4 + 8 + 4;

/// In-memory postings for one segment being built.
#[derive(Default)]
struct Buffer {
    terms: BTreeMap<String, (Vec<u32>, Vec<u32>)>,
    lens: Vec<u32>,
    ids: Vec<String>,
    scratch: Vec<String>,
}

impl Buffer {
    fn add(&mut self, id: &str, text: &str) {
        let doc = self.lens.len() as u32;
        self.scratch.clear();
        analyze_into(text, &mut self.scratch);
        self.scratch.sort_unstable();
        let mut i = 0;
        while i < self.scratch.len() {
            let mut j = i + 1;
            while j < self.scratch.len() && self.scratch[j] == self.scratch[i] {
                j += 1;
            }
            let e = self.terms.entry(self.scratch[i].clone()).or_default();
            e.0.push(doc);
            e.1.push((j - i) as u32);
            i = j;
        }
        self.lens.push(self.scratch.len() as u32);
        self.ids.push(id.to_string());
    }
}

/// Writes a segment from term -> (docs, tfs) postings.
fn write_segment<'a>(
    path: &Path,
    terms: impl Iterator<Item = (&'a str, &'a [u32], &'a [u32])>,
    lens: &[u32],
    ids: &[String],
) -> io::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let mut dict = Vec::new();
    let mut n_terms = 0u32;
    for (term, docs, tfs) in terms {
        let off = out.len() as u64;
        postings::encode(docs, tfs, lens, &mut out);
        let len = (out.len() as u64 - off) as u32;
        dict.extend_from_slice(&(term.len() as u16).to_le_bytes());
        dict.extend_from_slice(term.as_bytes());
        dict.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        dict.extend_from_slice(&off.to_le_bytes());
        dict.extend_from_slice(&len.to_le_bytes());
        n_terms += 1;
    }
    let lens_off = out.len() as u64;
    for &l in lens {
        out.extend_from_slice(&l.to_le_bytes());
    }
    let ids_off = out.len() as u64;
    for id in ids {
        out.extend_from_slice(&(id.len() as u32).to_le_bytes());
        out.extend_from_slice(id.as_bytes());
    }
    let dict_off = out.len() as u64;
    out.extend_from_slice(&dict);
    out.extend_from_slice(&lens_off.to_le_bytes());
    out.extend_from_slice(&ids_off.to_le_bytes());
    out.extend_from_slice(&dict_off.to_le_bytes());
    out.extend_from_slice(&n_terms.to_le_bytes());
    out.extend_from_slice(&(lens.len() as u32).to_le_bytes());
    out.extend_from_slice(&lens.iter().map(|&l| l as u64).sum::<u64>().to_le_bytes());
    out.extend_from_slice(FOOTER_MAGIC);
    let tmp = path.with_extension("tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(&out)?;
    f.sync_all()?;
    fs::rename(&tmp, path)
}

/// A read-only, memory-mapped segment.
pub struct Segment {
    pub name: String,
    mmap: Mmap,
    term_offsets: Vec<usize>, // dict entry start per term, for binary search
    lens_off: usize,
    ids: Vec<(usize, usize)>,
    pub num_docs: u32,
    pub total_len: u64,
}

pub struct TermInfo<'a> {
    pub df: u32,
    pub postings: &'a [u8],
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

impl Segment {
    pub fn open(path: &Path) -> io::Result<Segment> {
        let file = File::open(path)?;
        // SAFETY: segment files are immutable once renamed into place and are
        // only ever replaced by new files, never modified in place.
        let mmap = unsafe { Mmap::map(&file)? };
        let b = &mmap[..];
        if b.len() < 8 + FOOTER || &b[..4] != MAGIC || &b[b.len() - 4..] != FOOTER_MAGIC {
            return Err(bad("not a sifter segment"));
        }
        let f = &b[b.len() - FOOTER..];
        let u64at = |i: usize| u64::from_le_bytes(f[i..i + 8].try_into().unwrap());
        let lens_off = u64at(0) as usize;
        let ids_off = u64at(8) as usize;
        let dict_off = u64at(16) as usize;
        let n_terms = u32::from_le_bytes(f[24..28].try_into().unwrap()) as usize;
        let num_docs = u32::from_le_bytes(f[28..32].try_into().unwrap());
        let total_len = u64at(32);
        let dict_end = b.len() - FOOTER;
        if !(lens_off <= ids_off && ids_off <= dict_off && dict_off <= dict_end) {
            return Err(bad("corrupt segment footer"));
        }
        let mut term_offsets = Vec::with_capacity(n_terms);
        let mut p = dict_off;
        for _ in 0..n_terms {
            if p + 2 > dict_end {
                return Err(bad("truncated term dictionary"));
            }
            term_offsets.push(p);
            let tl = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
            p += 2 + tl + 4 + 8 + 4;
        }
        let mut ids = Vec::with_capacity(num_docs as usize);
        let mut p = ids_off;
        for _ in 0..num_docs {
            let l = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
            ids.push((p + 4, l));
            p += 4 + l;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        Ok(Segment {
            name,
            mmap,
            term_offsets,
            lens_off,
            ids,
            num_docs,
            total_len,
        })
    }

    fn entry(&self, i: usize) -> (&str, TermInfo<'_>) {
        let b = &self.mmap[..];
        let p = self.term_offsets[i];
        let tl = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
        let term = std::str::from_utf8(&b[p + 2..p + 2 + tl]).unwrap_or("");
        let q = p + 2 + tl;
        let df = u32::from_le_bytes(b[q..q + 4].try_into().unwrap());
        let off = u64::from_le_bytes(b[q + 4..q + 12].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(b[q + 12..q + 16].try_into().unwrap()) as usize;
        (
            term,
            TermInfo {
                df,
                postings: &b[off..off + len],
            },
        )
    }

    pub fn term(&self, t: &str) -> Option<TermInfo<'_>> {
        let i = self.term_offsets.binary_search_by(|&p| {
            let b = &self.mmap[..];
            let tl = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
            b[p + 2..p + 2 + tl].cmp(t.as_bytes())
        });
        i.ok().map(|i| self.entry(i).1)
    }

    pub fn doc_len(&self, doc: u32) -> u32 {
        let p = self.lens_off + doc as usize * 4;
        u32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap())
    }

    pub fn doc_id(&self, doc: u32) -> &str {
        let (p, l) = self.ids[doc as usize];
        std::str::from_utf8(&self.mmap[p..p + l]).unwrap_or("")
    }

    pub fn num_terms(&self) -> usize {
        self.term_offsets.len()
    }

    pub fn size_bytes(&self) -> usize {
        self.mmap.len()
    }

    fn terms(&self) -> impl Iterator<Item = (&str, TermInfo<'_>)> {
        (0..self.term_offsets.len()).map(|i| self.entry(i))
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Meta {
    segments: Vec<String>,
    next_segment: u64,
}

/// Collection-wide statistics used for scoring.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub num_docs: u64,
    pub avg_len: f64,
}

/// An index directory: a set of segments plus a writer buffer.
pub struct Index {
    dir: PathBuf,
    meta: Meta,
    pub segments: Vec<Segment>,
    buffer: Buffer,
    /// Documents buffered before a segment is flushed.
    pub flush_every: usize,
}

impl Index {
    pub fn create(dir: &Path) -> io::Result<Index> {
        fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta.json");
        let meta: Meta = match fs::read(&meta_path) {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| bad(&e.to_string()))?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Meta::default(),
            Err(e) => return Err(e),
        };
        let mut segments = Vec::new();
        for s in &meta.segments {
            segments.push(Segment::open(&dir.join(s))?);
        }
        Ok(Index {
            dir: dir.to_path_buf(),
            meta,
            segments,
            buffer: Buffer::default(),
            flush_every: 50_000,
        })
    }

    pub fn open(dir: &Path) -> io::Result<Index> {
        if !dir.join("meta.json").exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no index at {}", dir.display()),
            ));
        }
        Index::create(dir)
    }

    pub fn add(&mut self, id: &str, text: &str) -> io::Result<()> {
        self.buffer.add(id, text);
        if self.buffer.lens.len() >= self.flush_every {
            self.flush()?;
        }
        Ok(())
    }

    fn new_segment_path(&mut self) -> (String, PathBuf) {
        let name = format!("seg_{:06}.sft", self.meta.next_segment);
        self.meta.next_segment += 1;
        let path = self.dir.join(&name);
        (name, path)
    }

    /// Writes buffered documents as a new segment (not yet committed).
    pub fn flush(&mut self) -> io::Result<()> {
        if self.buffer.lens.is_empty() {
            return Ok(());
        }
        let buf = std::mem::take(&mut self.buffer);
        let (name, path) = self.new_segment_path();
        write_segment(
            &path,
            buf.terms
                .iter()
                .map(|(t, (d, f))| (t.as_str(), &d[..], &f[..])),
            &buf.lens,
            &buf.ids,
        )?;
        self.segments.push(Segment::open(&path)?);
        self.meta.segments.push(name);
        Ok(())
    }

    /// Flushes and atomically publishes the segment list.
    pub fn commit(&mut self) -> io::Result<()> {
        self.flush()?;
        let tmp = self.dir.join("meta.json.tmp");
        let mut f = File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(&self.meta).map_err(|e| bad(&e.to_string()))?)?;
        f.sync_all()?;
        fs::rename(&tmp, self.dir.join("meta.json"))?;
        if let Ok(d) = File::open(&self.dir) {
            d.sync_all()?;
        }
        Ok(())
    }

    /// Merges all segments into one (k-way merge of term dictionaries;
    /// doc ids are renumbered by concatenating segments in order).
    pub fn merge_all(&mut self) -> io::Result<()> {
        self.commit()?;
        if self.segments.len() <= 1 {
            return Ok(());
        }
        let mut bases = Vec::new();
        let mut base = 0u32;
        let mut lens = Vec::new();
        let mut ids = Vec::new();
        for s in &self.segments {
            bases.push(base);
            base += s.num_docs;
            for d in 0..s.num_docs {
                lens.push(s.doc_len(d));
                ids.push(s.doc_id(d).to_string());
            }
        }
        // Merge sorted term iterators.
        let mut merged: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
        {
            let mut iters: Vec<_> = self.segments.iter().map(|s| s.terms().peekable()).collect();
            loop {
                let next = iters
                    .iter_mut()
                    .filter_map(|it| it.peek().map(|(t, _)| t.to_string()))
                    .min();
                let Some(term) = next else { break };
                let mut docs = Vec::new();
                let mut tfs = Vec::new();
                for (si, it) in iters.iter_mut().enumerate() {
                    if it.peek().is_some_and(|(t, _)| *t == term) {
                        let (_, info) = it.next().unwrap();
                        let mut c = postings::Cursor::new(info.postings);
                        while c.doc != u32::MAX {
                            docs.push(bases[si] + c.doc);
                            tfs.push(c.tf());
                            c.next();
                        }
                    }
                }
                merged.push((term, docs, tfs));
            }
        }
        let (name, path) = self.new_segment_path();
        write_segment(
            &path,
            merged.iter().map(|(t, d, f)| (t.as_str(), &d[..], &f[..])),
            &lens,
            &ids,
        )?;
        let old: Vec<String> = std::mem::take(&mut self.meta.segments);
        self.segments.clear();
        self.segments.push(Segment::open(&path)?);
        self.meta.segments.push(name);
        self.commit()?;
        for s in old {
            // Best effort: readers keep their mmap of a deleted segment.
            let _ = fs::remove_file(self.dir.join(s));
        }
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        let n: u64 = self.segments.iter().map(|s| s.num_docs as u64).sum();
        let total: u64 = self.segments.iter().map(|s| s.total_len).sum();
        Stats {
            num_docs: n,
            avg_len: if n == 0 { 0.0 } else { total as f64 / n as f64 },
        }
    }

    /// Document frequency of an analyzed term across segments.
    pub fn df(&self, term: &str) -> u64 {
        self.segments
            .iter()
            .filter_map(|s| s.term(term))
            .map(|t| t.df as u64)
            .sum()
    }

    pub fn num_docs(&self) -> u64 {
        self.stats().num_docs
    }
}
