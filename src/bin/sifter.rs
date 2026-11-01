//! sifter CLI.
//!
//!   sifter index  <dir> <corpus.jsonl>          BEIR-style {"_id","title","text"} lines
//!   sifter search <dir> <query> [-k N] [--bmw]
//!   sifter merge  <dir>
//!   sifter eval   <dir> <queries.jsonl> <qrels.tsv> [--k1 X --b Y] [--bmw]

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use sifter::eval::{mrr_at, ndcg_at, parse_qrels, recall_at};
use sifter::{Algorithm, Bm25, Index, Searcher};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  sifter index <dir> <corpus.jsonl>\n  sifter search <dir> <query> [-k N] [--bmw]\n  sifter merge <dir>\n  sifter eval <dir> <queries.jsonl> <qrels.tsv> [--k1 X] [--b Y] [--bmw]"
    );
    ExitCode::from(2)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return usage();
    }
    let dir = Path::new(&args[2]);
    let params = Bm25 {
        k1: flag(&args, "--k1")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.9),
        b: flag(&args, "--b")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.4),
    };
    // Exhaustive scoring is the default: it was faster on every corpus
    // measured (see README). --bmw selects block-max WAND.
    let algo = if args.iter().any(|a| a == "--bmw") {
        Algorithm::BlockMaxWand
    } else {
        Algorithm::Exhaustive
    };
    let res = match args[1].as_str() {
        "index" if args.len() >= 4 => index(dir, &args[3]),
        "search" if args.len() >= 4 => search(
            dir,
            &args[3],
            flag(&args, "-k").and_then(|v| v.parse().ok()).unwrap_or(10),
            params,
            algo,
        ),
        "merge" => merge(dir),
        "eval" if args.len() >= 5 => eval(dir, &args[3], &args[4], params, algo),
        _ => return usage(),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sifter: {e}");
            ExitCode::FAILURE
        }
    }
}

type R = Result<(), Box<dyn std::error::Error>>;

fn index(dir: &Path, corpus: &str) -> R {
    let t0 = Instant::now();
    let mut idx = Index::create(dir)?;
    let f = std::io::BufReader::new(std::fs::File::open(corpus)?);
    let mut n = 0u64;
    for line in f.lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let id = v["_id"].as_str().unwrap_or_default();
        let text = format!(
            "{} {}",
            v["title"].as_str().unwrap_or(""),
            v["text"].as_str().unwrap_or("")
        );
        idx.add(id, &text)?;
        n += 1;
    }
    idx.commit()?;
    let bytes: usize = idx.segments.iter().map(|s| s.size_bytes()).sum();
    eprintln!(
        "indexed {n} docs into {} segment(s), {} bytes, in {:.2}s",
        idx.segments.len(),
        bytes,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn search(dir: &Path, q: &str, k: usize, p: Bm25, algo: Algorithm) -> R {
    let idx = Index::open(dir)?;
    let s = Searcher::new(&idx, p);
    let t0 = Instant::now();
    let (hits, st) = s.search(q, k, algo);
    let el = t0.elapsed();
    for (i, h) in hits.iter().enumerate() {
        println!("{:>3}. {:<14} {:.4}", i + 1, h.id, h.score);
    }
    eprintln!(
        "{:.3} ms, {} docs scored, {} blocks skipped",
        el.as_secs_f64() * 1000.0,
        st.scored_docs,
        st.skipped_blocks
    );
    Ok(())
}

fn merge(dir: &Path) -> R {
    let mut idx = Index::open(dir)?;
    let before = idx.segments.len();
    let t0 = Instant::now();
    idx.merge_all()?;
    eprintln!(
        "merged {before} segments into {} in {:.2}s",
        idx.segments.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn eval(dir: &Path, queries: &str, qrels: &str, p: Bm25, algo: Algorithm) -> R {
    let idx = Index::open(dir)?;
    let s = Searcher::new(&idx, p);
    let qrels = parse_qrels(&std::fs::read_to_string(qrels)?);
    let mut texts: HashMap<String, String> = HashMap::new();
    for line in std::io::BufReader::new(std::fs::File::open(queries)?).lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        texts.insert(
            v["_id"].as_str().unwrap_or_default().to_string(),
            v["text"].as_str().unwrap_or("").to_string(),
        );
    }
    let mut qids: Vec<&String> = qrels.keys().collect();
    qids.sort();
    let (mut ndcg, mut recall, mut mrr, mut n) = (0.0, 0.0, 0.0, 0);
    let mut lat = Vec::new();
    let mut lat10 = Vec::new();
    let mut scored10 = 0u64;
    for qid in qids {
        let Some(text) = texts.get(qid) else { continue };
        let t0 = Instant::now();
        let (hits, _) = s.search(text, 100, algo);
        lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = Instant::now();
        let (_, st10) = s.search(text, 10, algo);
        scored10 += st10.scored_docs;
        lat10.push(t1.elapsed().as_secs_f64() * 1000.0);
        let ranked: Vec<String> = hits.into_iter().map(|h| h.id).collect();
        let r = &qrels[qid];
        ndcg += ndcg_at(&ranked, r, 10);
        recall += recall_at(&ranked, r, 100);
        mrr += mrr_at(&ranked, r, 10);
        n += 1;
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lat10.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &Vec<f64>, p: f64| v[((v.len() - 1) as f64 * p) as usize];
    println!(
        "{{\"engine\": \"sifter\", \"algorithm\": \"{algo:?}\", \"queries\": {n}, \"ndcg@10\": {:.4}, \"recall@100\": {:.4}, \"mrr@10\": {:.4}, \"k1\": {}, \"b\": {}, \"p50_ms_k100\": {:.3}, \"p99_ms_k100\": {:.3}, \"p50_ms_k10\": {:.3}, \"p99_ms_k10\": {:.3}, \"mean_scored_docs_k10\": {:.1}}}",
        ndcg / n as f64,
        recall / n as f64,
        mrr / n as f64,
        p.k1,
        p.b,
        q(&lat, 0.5),
        q(&lat, 0.99),
        q(&lat10, 0.5),
        q(&lat10, 0.99),
        scored10 as f64 / n as f64
    );
    Ok(())
}
