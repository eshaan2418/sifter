//! sifter: a full-text search engine with BM25 ranking, block-max WAND
//! top-k retrieval and segment-based indexing.

pub mod analysis;
pub mod eval;
pub mod index;
pub mod postings;
pub mod search;

pub use index::Index;
pub use search::{Algorithm, Bm25, Hit, Searcher};
