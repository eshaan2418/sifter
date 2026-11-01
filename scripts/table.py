#!/usr/bin/env python3
"""Markdown tables from bench/results/beir.json."""
import json

d = json.load(open("bench/results/beir.json"))
ds = d["datasets"]
print("| Dataset | docs | sifter (k1 .9, b .4) | sifter (k1 1.2, b .75) | tantivy (k1 1.2, b .75) |")
print("|---|---:|---:|---:|---:|")
for n in ["scifact", "nfcorpus", "fiqa"]:
    e = ds[n]
    a, b, t = e["sifter exhaustive k1=0.9 b=0.4"], e["sifter exhaustive k1=1.2 b=0.75"], e["tantivy k1=1.2 b=0.75"]
    print(f"| {n} | {e['docs']:,} | {a['ndcg@10']:.4f} / {a['recall@100']:.4f} | {b['ndcg@10']:.4f} / {b['recall@100']:.4f} | {t['ndcg@10']:.4f} / {t['recall@100']:.4f} |")
print()
print("| Corpus | docs | sifter exhaustive k=10 | sifter BMW k=10 | tantivy k=10 | sifter exhaustive k=100 | sifter BMW k=100 | tantivy k=100 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for n in ["scifact", "nfcorpus", "fiqa", "fiqa20"]:
    e = ds[n]
    x, w, t = e["sifter exhaustive k1=1.2 b=0.75"], e["sifter bmw k1=1.2 b=0.75"], e["tantivy k1=1.2 b=0.75"]
    f = lambda v: f"{v:.3f} ms"
    print(f"| {n} | {e['docs']:,} | {f(x['p50_ms_k10'])} | {f(w['p50_ms_k10'])} | {f(t['p50_ms_k10'])} | {f(x['p50_ms_k100'])} | {f(w['p50_ms_k100'])} | {f(t['p50_ms_k100'])} |")
print()
print("| Corpus | sifter index time | sifter index size | sifter peak RSS | tantivy index time |")
print("|---|---:|---:|---:|---:|")
for n in ["scifact", "nfcorpus", "fiqa", "fiqa20"]:
    e = ds[n]
    print(f"| {n} | {e['index_s']:.2f} s | {e['index_bytes']/1e6:.1f} MB | {e['index_rss_bytes']/1e6:.0f} MB | {e['tantivy k1=1.2 b=0.75']['index_s']:.2f} s |")
print(f"\n{d['env']['machine']}, {d['env']['os']}, {d['env']['rustc']}; latency = median over {d['env']['runs']} runs of per-run p50.")
