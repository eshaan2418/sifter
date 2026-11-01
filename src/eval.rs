//! Retrieval metrics as computed by trec_eval / pytrec_eval (the BEIR
//! harness): nDCG@k with linear gain and graded relevance, Recall@k, MRR@k.

use std::collections::HashMap;

pub type Qrels = HashMap<String, HashMap<String, i32>>;

/// BEIR qrels TSV: `query-id \t corpus-id \t score` with a header line.
pub fn parse_qrels(tsv: &str) -> Qrels {
    let mut q: Qrels = HashMap::new();
    for line in tsv.lines().skip(1) {
        let mut f = line.split('\t');
        let (Some(qid), Some(did), Some(rel)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        if let Ok(r) = rel.trim().parse::<i32>() {
            q.entry(qid.to_string())
                .or_default()
                .insert(did.to_string(), r);
        }
    }
    q
}

pub fn ndcg_at(ranked: &[String], rels: &HashMap<String, i32>, k: usize) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| rels.get(d).copied().unwrap_or(0).max(0) as f64 / ((i + 2) as f64).log2())
        .sum();
    let mut ideal: Vec<i32> = rels.values().copied().filter(|&r| r > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &r)| r as f64 / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

pub fn recall_at(ranked: &[String], rels: &HashMap<String, i32>, k: usize) -> f64 {
    let relevant = rels.values().filter(|&&r| r > 0).count();
    if relevant == 0 {
        return 0.0;
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|d| rels.get(*d).copied().unwrap_or(0) > 0)
        .count();
    found as f64 / relevant as f64
}

pub fn mrr_at(ranked: &[String], rels: &HashMap<String, i32>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .position(|d| rels.get(d).copied().unwrap_or(0) > 0)
        .map_or(0.0, |i| 1.0 / (i + 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rels(v: &[(&str, i32)]) -> HashMap<String, i32> {
        v.iter().map(|(d, r)| (d.to_string(), *r)).collect()
    }
    fn ranked(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ndcg_matches_hand_computation() {
        let r = rels(&[("a", 2), ("b", 1), ("z", 0)]);
        assert!((ndcg_at(&ranked(&["a", "b"]), &r, 10) - 1.0).abs() < 1e-12);
        // DCG = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3)
        let want = (1.0 + 2.0 / 3f64.log2()) / (2.0 + 1.0 / 3f64.log2());
        assert!((ndcg_at(&ranked(&["b", "a"]), &r, 10) - want).abs() < 1e-12);
        assert_eq!(ndcg_at(&ranked(&["z", "q"]), &r, 10), 0.0);
        // Only the top k count.
        assert_eq!(ndcg_at(&ranked(&["x", "a"]), &r, 1), 0.0);
    }

    #[test]
    fn recall_and_mrr() {
        let r = rels(&[("a", 1), ("b", 1), ("c", 0)]);
        assert_eq!(recall_at(&ranked(&["x", "a", "c"]), &r, 100), 0.5);
        assert_eq!(mrr_at(&ranked(&["x", "c", "b"]), &r, 10), 1.0 / 3.0);
        assert_eq!(mrr_at(&ranked(&["x"]), &r, 10), 0.0);
        let q = parse_qrels("query-id\tcorpus-id\tscore\n1\td1\t1\n1\td2\t0\n2\td3\t2\n");
        assert_eq!(q.len(), 2);
        assert_eq!(q["1"]["d2"], 0);
    }
}
