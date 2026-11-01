//! Block-max WAND must return exactly the exhaustive top k; segment
//! merging and reopening must not change any result.

use sifter::{Algorithm, Bm25, Index, Searcher};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// Zipf-ish vocabulary so some terms are very common and some rare.
fn word(rng: &mut Rng) -> String {
    let r = rng.below(1000) as f64 / 1000.0;
    let rank = (r.powf(3.0) * 400.0) as u64;
    format!("w{rank}")
}

fn build(dir: &std::path::Path, rng: &mut Rng, docs: usize, flush_every: usize) -> Index {
    let mut idx = Index::create(dir).unwrap();
    idx.flush_every = flush_every;
    for d in 0..docs {
        let len = 1 + rng.below(60) as usize;
        let text: Vec<String> = (0..len).map(|_| word(rng)).collect();
        idx.add(&format!("doc{d}"), &text.join(" ")).unwrap();
    }
    idx.commit().unwrap();
    idx
}

fn random_query(rng: &mut Rng) -> String {
    let n = 1 + rng.below(7) as usize;
    (0..n).map(|_| word(rng)).collect::<Vec<_>>().join(" ")
}

#[test]
fn bmw_matches_exhaustive_exactly() {
    let mut rng = Rng(0x5eed);
    let mut checked = 0;
    for trial in 0..60 {
        let dir = tempdir(&format!("bmw{trial}"));
        let docs = 50 + rng.below(3000) as usize;
        let flush = 1 + rng.below(1500) as usize;
        let idx = build(&dir, &mut rng, docs, flush);
        for params in [
            Bm25::default(),
            Bm25 { k1: 1.2, b: 0.75 },
            Bm25 { k1: 2.0, b: 1.0 },
        ] {
            let s = Searcher::new(&idx, params);
            for _ in 0..10 {
                let q = random_query(&mut rng);
                let k = [1, 3, 10, 50, 200][rng.below(5) as usize];
                let (a, _) = s.search(&q, k, Algorithm::Exhaustive);
                let (b, _) = s.search(&q, k, Algorithm::BlockMaxWand);
                assert_eq!(a, b, "trial {trial} query {q:?} k={k} params={params:?}");
                checked += 1;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
    assert_eq!(checked, 1800);
}

#[test]
fn bmw_prunes_on_selective_queries() {
    let mut rng = Rng(42);
    let dir = tempdir("prune");
    let idx = build(&dir, &mut rng, 20000, 100000);
    let s = Searcher::new(&idx, Bm25::default());
    let (_, ex) = s.search("w0 w1 w350", 10, Algorithm::Exhaustive);
    let (_, bm) = s.search("w0 w1 w350", 10, Algorithm::BlockMaxWand);
    assert!(
        bm.scored_docs * 5 < ex.scored_docs,
        "bmw scored {} vs {}",
        bm.scored_docs,
        ex.scored_docs
    );
    assert!(bm.skipped_blocks > 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn merge_and_reopen_preserve_results() {
    let mut rng = Rng(7);
    let dir = tempdir("merge");
    let mut idx = build(&dir, &mut rng, 4000, 700);
    assert!(idx.segments.len() >= 5);
    let queries: Vec<String> = (0..50).map(|_| random_query(&mut rng)).collect();
    let before: Vec<_> = {
        let s = Searcher::new(&idx, Bm25::default());
        queries
            .iter()
            .map(|q| s.search(q, 20, Algorithm::Exhaustive).0)
            .collect()
    };
    idx.merge_all().unwrap();
    assert_eq!(idx.segments.len(), 1);
    let s = Searcher::new(&idx, Bm25::default());
    for (q, want) in queries.iter().zip(&before) {
        let got = s.search(q, 20, Algorithm::BlockMaxWand).0;
        // Same documents and scores; doc numbers are unchanged because merge
        // concatenates segments in order.
        assert_eq!(&got, want, "{q}");
    }
    drop(idx);
    let idx = Index::open(&dir).unwrap();
    let s = Searcher::new(&idx, Bm25::default());
    assert_eq!(
        s.search(&queries[0], 20, Algorithm::Exhaustive).0,
        before[0]
    );
    // Only the merged segment file remains on disk.
    let segs = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "sft")
        })
        .count();
    assert_eq!(segs, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bm25_idf_and_ordering() {
    let dir = tempdir("small");
    let mut idx = Index::create(&dir).unwrap();
    idx.add("a", "the quick brown fox").unwrap();
    idx.add("b", "quick quick quick").unwrap();
    idx.add("c", "a lazy dog sleeps").unwrap();
    idx.commit().unwrap();
    let s = Searcher::new(&idx, Bm25::default());
    let (hits, _) = s.search("quick", 10, Algorithm::Exhaustive);
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["b", "a"]
    );
    assert!(s.search("zebra", 10, Algorithm::BlockMaxWand).0.is_empty());
    assert!(
        s.search("the of and", 10, Algorithm::Exhaustive)
            .0
            .is_empty()
    ); // all stop words
    // Lucene idf: ln(1 + (N - df + 0.5) / (df + 0.5)).
    assert!((sifter::search::idf(3, 2) - (1.0f64 + 1.5 / 2.5).ln() as f32).abs() < 1e-7);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_corrupt_segment() {
    let dir = tempdir("corrupt");
    let mut idx = Index::create(&dir).unwrap();
    idx.add("a", "hello world").unwrap();
    idx.commit().unwrap();
    drop(idx);
    let seg = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "sft"))
        .unwrap();
    let mut data = std::fs::read(&seg).unwrap();
    let n = data.len();
    data[n - 2] ^= 0xff; // footer magic
    std::fs::write(&seg, data).unwrap();
    assert!(Index::open(&dir).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

fn tempdir(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("sifter-test-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&p).ok();
    p
}
