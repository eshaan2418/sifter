# sifter design notes

## Segment layout

An index directory holds immutable segment files `seg_NNNNNN.sft` plus a
`meta.json` listing the live segments. `meta.json` is replaced atomically
(write to a temp file, fsync, rename), so a crash leaves either the old or the
new set of segments, never a mix.

```text
"SIFT" magic
postings for every term, back to back
doc length table   (u32 per doc)
external ids       (offsets + bytes)
term dictionary    (sorted terms -> df, postings offset, postings length)
footer             (section offsets, doc count, total length, "TFIS")
```

Segments are opened with `mmap`; nothing is deserialized up front except the
footer. Term lookup is a binary search over the sorted dictionary.

## Postings

Each term's postings are split into blocks of 128 documents. A fixed-size
24-byte header per block stores `first_doc`, `last_doc`, the data offset, the
bit widths and two numbers used for pruning: the block's maximum term
frequency and its minimum document length. Doc-id deltas and `tf - 1` are
bitpacked at the smallest width that fits the block.

Because headers are fixed size and stored before the data, a cursor can skip
any number of blocks by reading headers only, and decode a block only when it
needs a document inside it.

## Scoring

BM25 with Lucene's idf, `ln(1 + (N - df + 0.5) / (df + 0.5))`, and the
statistics (`N`, `df`, average length) taken over the whole index, not per
segment, so results don't change when segments are merged.

Two evaluators share the same scoring arithmetic:

- **Exhaustive**: walk every posting of every query term, accumulate into a
  dense per-segment score array, then push hits into a size-k heap.
- **Block-max WAND**: cursors sorted by current doc; the pivot is the first
  cursor where the running sum of per-term upper bounds exceeds the heap
  threshold. Before scoring the pivot doc, the per-block bounds of the blocks
  that would contain it are summed; if they can't beat the threshold the
  cursors jump to the next block boundary.

The per-block bound is `contribution(weight, max_tf, norm(min_len))`. BM25's
term contribution increases with tf and decreases with document length, so
this bounds every document in the block.

### Exactness

Both evaluators sum term contributions in query order, so a document gets a
bit-identical `f32` score from either. Ties are broken by global doc number
(lower wins), and BMW prunes only on strict `<=` threshold comparisons, so the
two return identical hit lists. `tests/search.rs` checks this on 1,800 random
queries over random corpora and parameter sets.

### Why BMW is slower here

BMW scores 62x fewer documents than exhaustive on the 1.15M-document
corpus, but every skip decision costs header reads, binary searches and a
re-sort of the cursor order. The exhaustive path is a tight loop over decoded
blocks writing into a contiguous array, which the CPU handles very well at
these sizes. An early profile was dominated by `memmove` from swapping cursor
structs (each holds ~1 KB of decode buffers); sorting an index permutation
instead cut BMW latency by about 2.4x, but not enough to win. The likely next steps
would be SIMD block decoding and a cheaper pivot selection.

## Analysis

Tokenizer: runs of alphanumerics, with internal apostrophes kept so that
possessives can be stripped. Then lowercase, drop a trailing `'s`, drop
Lucene's English stop words, and apply the original Porter (1980) stemmer.
One known difference from Lucene's StandardTokenizer: `3.5x` splits at the
dot.
