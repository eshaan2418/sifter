#!/usr/bin/env python3
"""Documents scored per query (k=10): exhaustive vs block-max WAND.

Deterministic counters, so one run each. Writes bench/results/pruning.json.
"""
import json, os, shutil, subprocess, tempfile

SIFTER = "target/release/sifter"
out = {}
for name, corpus in [("fiqa", "data/fiqa/corpus.jsonl"), ("fiqa20", "data/fiqa20.jsonl")]:
    d = tempfile.mkdtemp(prefix=f"sifter-prune-{name}-")
    subprocess.run([SIFTER, "index", d, corpus], check=True, capture_output=True)
    subprocess.run([SIFTER, "merge", d], check=True, capture_output=True)
    q, r = "data/fiqa/queries.jsonl", "data/fiqa/qrels/test.tsv"
    row = {}
    for label, extra in [("exhaustive", []), ("bmw", ["--bmw"])]:
        o = json.loads(subprocess.run([SIFTER, "eval", d, q, r, "--k1", "1.2", "--b", "0.75", *extra],
                                      check=True, capture_output=True, text=True).stdout)
        row[label] = {"mean_scored_docs_k10": o["mean_scored_docs_k10"], "queries": o["queries"]}
    shutil.rmtree(d)
    out[name] = row
    print(name, row, flush=True)
os.makedirs("bench/results", exist_ok=True)
json.dump(out, open("bench/results/pruning.json", "w"), indent=2)
