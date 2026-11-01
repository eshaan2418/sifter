//! The same BEIR evaluation with tantivy as the engine, for comparison.
//!
//!   cargo run --release --example tantivy_baseline -- corpus.jsonl queries.jsonl qrels.tsv
//!
//! Analysis mirrors sifter's: simple tokenizer, lowercase, the Lucene English
//! stop list, English stemming (tantivy ships Snowball/Porter2, not the
//! original Porter). Scoring is tantivy's BM25 (k1 = 1.2, b = 0.75).

use std::collections::HashMap;
use std::io::BufRead;
use std::time::Instant;

use sifter::eval::{mrr_at, ndcg_at, parse_qrels, recall_at};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, TantivyDocument, doc};

const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

fn main() -> tantivy::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (corpus, queries, qrels) = (&args[1], &args[2], &args[3]);
    let mut sb = Schema::builder();
    let id = sb.add_text_field("id", STRING | STORED);
    let body = sb.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("en")
                .set_index_option(IndexRecordOption::WithFreqs),
        ),
    );
    let schema = sb.build();
    let dir = std::env::temp_dir().join(format!("tantivy-baseline-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let index = Index::create_in_dir(&dir, schema)?;
    index.tokenizers().register(
        "en",
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(64))
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(STOP.iter().map(|s| s.to_string())))
            .filter(Stemmer::new(Language::English))
            .build(),
    );
    let t0 = Instant::now();
    let mut w = index.writer(256 << 20)?;
    for line in std::io::BufReader::new(std::fs::File::open(corpus)?).lines() {
        let v: serde_json::Value = serde_json::from_str(&line?).unwrap();
        let text = format!(
            "{} {}",
            v["title"].as_str().unwrap_or(""),
            v["text"].as_str().unwrap_or("")
        );
        w.add_document(doc!(id => v["_id"].as_str().unwrap_or(""), body => text))?;
    }
    w.commit()?;
    w.wait_merging_threads()?;
    let index_s = t0.elapsed().as_secs_f64();
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(&index, vec![body]);

    let qrels = parse_qrels(&std::fs::read_to_string(qrels)?);
    let mut texts: HashMap<String, String> = HashMap::new();
    for line in std::io::BufReader::new(std::fs::File::open(queries)?).lines() {
        let v: serde_json::Value = serde_json::from_str(&line?).unwrap();
        texts.insert(
            v["_id"].as_str().unwrap_or("").to_string(),
            v["text"].as_str().unwrap_or("").to_string(),
        );
    }
    let mut qids: Vec<&String> = qrels.keys().filter(|q| texts.contains_key(*q)).collect();
    qids.sort();
    let (mut ndcg, mut recall, mut mrr) = (0.0, 0.0, 0.0);
    let mut lat100 = Vec::new();
    let mut lat10 = Vec::new();
    for qid in &qids {
        // Strip query-syntax characters so every query parses as plain terms.
        let clean: String = texts[*qid]
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        let (q, _) = parser.parse_query_lenient(&clean);
        let t = Instant::now();
        let top = searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?;
        lat100.push(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        let _ = searcher.search(&q, &TopDocs::with_limit(10).order_by_score())?;
        lat10.push(t.elapsed().as_secs_f64() * 1000.0);
        let ranked: Vec<String> = top
            .iter()
            .map(|(_, a)| {
                let d: TantivyDocument = searcher.doc(*a).unwrap();
                d.get_first(id)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        let r = &qrels[*qid];
        ndcg += ndcg_at(&ranked, r, 10);
        recall += recall_at(&ranked, r, 100);
        mrr += mrr_at(&ranked, r, 10);
    }
    let n = qids.len() as f64;
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p) as usize]
    };
    println!(
        "{{\"engine\": \"tantivy\", \"queries\": {}, \"ndcg@10\": {:.4}, \"recall@100\": {:.4}, \"mrr@10\": {:.4}, \"index_s\": {:.3}, \"p50_ms_k100\": {:.3}, \"p99_ms_k100\": {:.3}, \"p50_ms_k10\": {:.3}, \"p99_ms_k10\": {:.3}}}",
        n as usize,
        ndcg / n,
        recall / n,
        mrr / n,
        index_s,
        pct(&mut lat100, 0.5),
        pct(&mut lat100, 0.99),
        pct(&mut lat10, 0.5),
        pct(&mut lat10, 0.99)
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
