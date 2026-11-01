//! BM25 ranking with two top-k evaluators: exhaustive scoring (the
//! reference) and block-max WAND (Ding & Suel, SIGIR 2011), which skips
//! whole blocks whose score upper bound cannot beat the current k-th score.
//!
//! Both evaluators compute every document's score with the same arithmetic,
//! summing term contributions in query order, so their scores are
//! bit-identical. Ranking is by score descending, then by global doc number
//! ascending; block-max WAND prunes only when a document provably cannot
//! enter the top k under that order, so both return exactly the same hits.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::analysis::analyze;
use crate::index::{Index, Segment};
use crate::postings::Cursor;

#[derive(Clone, Copy, Debug)]
pub struct Bm25 {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25 {
    /// Anserini's BM25 defaults (k1 = 0.9, b = 0.4), as used for the BEIR
    /// "BM25 flat" baselines in Pyserini.
    fn default() -> Self {
        Bm25 { k1: 0.9, b: 0.4 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Exhaustive,
    BlockMaxWand,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    /// Global document number (segment order, then local id).
    pub doc: u64,
}

/// Counters for one query (used by benchmarks and tests).
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchStats {
    pub scored_docs: u64,
    pub skipped_blocks: u64,
}

struct Term {
    weight: f32, // idf * query term frequency
}

/// Lucene's BM25 idf.
pub fn idf(num_docs: u64, df: u64) -> f32 {
    (1.0 + (num_docs as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln() as f32
}

#[inline]
fn norm(p: Bm25, dl: u32, avg: f32) -> f32 {
    p.k1 * (1.0 - p.b + p.b * dl as f32 / avg)
}

#[inline]
fn contribution(weight: f32, tf: u32, norm: f32) -> f32 {
    let tf = tf as f32;
    weight * (tf / (tf + norm))
}

#[derive(PartialEq)]
struct Entry {
    score: f32,
    doc: u64,
}
impl Eq for Entry {}
// Min-heap on (score, then larger doc = worse).
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc.cmp(&o.doc))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct TopK {
    k: usize,
    heap: BinaryHeap<Entry>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }
    /// Would a document with this score (and a larger doc number than any
    /// seen so far) make the cut? Ties lose to earlier documents.
    fn beats(&self, score: f32) -> bool {
        self.heap.len() < self.k || score > self.heap.peek().unwrap().score
    }
    fn push(&mut self, score: f32, doc: u64) {
        if self.beats(score) {
            self.heap.push(Entry { score, doc });
            if self.heap.len() > self.k {
                self.heap.pop();
            }
        }
    }
    /// Scores <= this cannot enter (once full).
    fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap.peek().unwrap().score
        }
    }
}

pub struct Searcher<'a> {
    pub index: &'a Index,
    pub params: Bm25,
}

impl<'a> Searcher<'a> {
    pub fn new(index: &'a Index, params: Bm25) -> Self {
        Searcher { index, params }
    }

    /// Analyzed query terms with multiplicities, in first-occurrence order.
    fn query_terms(&self, q: &str) -> Vec<(String, Term)> {
        let stats = self.index.stats();
        let mut out: Vec<(String, Term)> = Vec::new();
        for t in analyze(q) {
            if let Some(e) = out.iter_mut().find(|(s, _)| *s == t) {
                e.1.weight += idf(stats.num_docs, self.index.df(&t));
                continue;
            }
            let df = self.index.df(&t);
            if df == 0 {
                continue;
            }
            out.push((
                t,
                Term {
                    weight: idf(stats.num_docs, df),
                },
            ));
        }
        out
    }

    pub fn search(&self, q: &str, k: usize, algo: Algorithm) -> (Vec<Hit>, SearchStats) {
        let terms = self.query_terms(q);
        let avg = self.index.stats().avg_len as f32;
        let mut top = TopK::new(k.max(1));
        let mut st = SearchStats::default();
        let mut base = 0u64;
        for seg in &self.index.segments {
            match algo {
                Algorithm::Exhaustive => self.exhaustive(seg, &terms, avg, base, &mut top, &mut st),
                Algorithm::BlockMaxWand => self.bmw(seg, &terms, avg, base, &mut top, &mut st),
            }
            base += seg.num_docs as u64;
        }
        let mut hits: Vec<(f32, u64)> = top.heap.into_iter().map(|e| (e.score, e.doc)).collect();
        hits.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let hits = hits
            .into_iter()
            .map(|(score, doc)| {
                let (seg, local) = self.locate(doc);
                Hit {
                    id: seg.doc_id(local).to_string(),
                    score,
                    doc,
                }
            })
            .collect();
        (hits, st)
    }

    fn locate(&self, mut doc: u64) -> (&Segment, u32) {
        for s in &self.index.segments {
            if doc < s.num_docs as u64 {
                return (s, doc as u32);
            }
            doc -= s.num_docs as u64;
        }
        unreachable!("doc out of range")
    }

    fn exhaustive(
        &self,
        seg: &Segment,
        terms: &[(String, Term)],
        avg: f32,
        base: u64,
        top: &mut TopK,
        st: &mut SearchStats,
    ) {
        let mut scores = vec![0f32; seg.num_docs as usize];
        let mut hit = vec![false; seg.num_docs as usize];
        for (t, term) in terms {
            let Some(info) = seg.term(t) else { continue };
            let mut c = Cursor::new(info.postings);
            while c.doc != u32::MAX {
                let d = c.doc as usize;
                scores[d] += contribution(
                    term.weight,
                    c.tf(),
                    norm(self.params, seg.doc_len(c.doc), avg),
                );
                hit[d] = true;
                c.next();
            }
        }
        for d in 0..seg.num_docs as usize {
            if hit[d] {
                st.scored_docs += 1;
                top.push(scores[d], base + d as u64);
            }
        }
    }

    fn bmw(
        &self,
        seg: &Segment,
        terms: &[(String, Term)],
        avg: f32,
        base: u64,
        top: &mut TopK,
        st: &mut SearchStats,
    ) {
        struct C<'b> {
            cur: Cursor<'b>,
            weight: f32,
            order: usize,        // position in the query, for summation order
            ub: f32,             // max over all blocks
            last_docs: Vec<u32>, // per block, for block lookups without decoding
            block_ub: Vec<f32>,  // per block upper bound
        }
        let p = self.params;
        let mut cs: Vec<C> = Vec::new();
        for (i, (t, term)) in terms.iter().enumerate() {
            let Some(info) = seg.term(t) else { continue };
            let cur = Cursor::new(info.postings);
            let n = cur.num_blocks();
            let mut last_docs = Vec::with_capacity(n);
            let mut block_ub = Vec::with_capacity(n);
            for b in 0..n {
                let h = cur.header(b);
                last_docs.push(h.last_doc);
                // Max tf with the shortest document in the block bounds every score in it.
                block_ub.push(contribution(term.weight, h.max_tf, norm(p, h.min_len, avg)));
            }
            let ub = block_ub.iter().copied().fold(0f32, f32::max);
            cs.push(C {
                cur,
                weight: term.weight,
                order: i,
                ub,
                last_docs,
                block_ub,
            });
        }
        // Block containing (or after) `target` for cursor c, searching from its
        // current block. Returns None past the last block.
        fn block_of(c: &C, target: u32) -> Option<usize> {
            let from = c.cur.current_block();
            let k = from + c.last_docs[from..].partition_point(|&l| l < target);
            (k < c.last_docs.len()).then_some(k)
        }
        let mut contribs: Vec<(usize, f32)> = Vec::with_capacity(cs.len());
        // Cursors stay put (each holds ~1 KB of decode buffers); we keep a
        // permutation sorted by current doc instead of moving them.
        let mut ord: Vec<usize> = (0..cs.len()).collect();
        ord.sort_by_key(|&i| cs[i].cur.doc);
        loop {
            // Exhausted cursors (doc = u32::MAX) sort to the end.
            if ord.is_empty() || cs[ord[0]].cur.doc == u32::MAX {
                return;
            }
            let theta = top.threshold();
            // Pivot: first position where the prefix of upper bounds beats theta.
            let mut acc = 0f32;
            let mut pivot = None;
            for (pos, &i) in ord.iter().enumerate() {
                if cs[i].cur.doc == u32::MAX {
                    break;
                }
                acc += cs[i].ub;
                if acc > theta {
                    pivot = Some(pos);
                    break;
                }
            }
            let Some(mut pi) = pivot else { return };
            let d = cs[ord[pi]].cur.doc;
            while pi + 1 < ord.len() && cs[ord[pi + 1]].cur.doc == d {
                pi += 1;
            }
            // Block-max check on the blocks that would contain d.
            let mut block_sum = 0f32;
            let mut next_boundary = u32::MAX;
            for &i in &ord[..=pi] {
                let c = &cs[i];
                if let Some(b) = block_of(c, d) {
                    block_sum += c.block_ub[b];
                    next_boundary = next_boundary.min(c.last_docs[b].saturating_add(1));
                }
            }
            if block_sum <= theta {
                // No doc in [d, next_boundary) can beat theta: skip past it,
                // but not past the next term's current doc.
                st.skipped_blocks += 1;
                let mut target = next_boundary;
                if pi + 1 < ord.len() {
                    target = target.min(cs[ord[pi + 1]].cur.doc);
                }
                let target = target.max(d + 1);
                for &i in &ord[..=pi] {
                    cs[i].cur.seek(target);
                }
            } else if cs[ord[0]].cur.doc == d {
                // Score d exactly, summing in query order.
                contribs.clear();
                let nrm = norm(p, seg.doc_len(d), avg);
                for &i in &ord[..=pi] {
                    let c = &cs[i];
                    contribs.push((c.order, contribution(c.weight, c.cur.tf(), nrm)));
                }
                contribs.sort_unstable_by_key(|x| x.0);
                let score: f32 = contribs.iter().fold(0f32, |a, x| a + x.1);
                st.scored_docs += 1;
                top.push(score, base + d as u64);
                for &i in &ord[..=pi] {
                    cs[i].cur.next();
                }
            } else {
                // Move the lagging cursors up to the pivot.
                for &i in &ord[..pi] {
                    if cs[i].cur.doc < d {
                        cs[i].cur.seek(d);
                    }
                }
            }
            // Only a prefix moved; restore order with an insertion sort on
            // the (small, nearly sorted) index permutation.
            for a in 1..ord.len() {
                let mut j = a;
                while j > 0 && cs[ord[j - 1]].cur.doc > cs[ord[j]].cur.doc {
                    ord.swap(j - 1, j);
                    j -= 1;
                }
            }
        }
    }
}
