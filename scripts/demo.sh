#!/bin/sh
set -eu
rm -rf /tmp/sifter-demo
./target/release/sifter index /tmp/sifter-demo data/scifact/corpus.jsonl
./target/release/sifter search /tmp/sifter-demo "vitamin D deficiency increases risk of respiratory infections" -k 5
./target/release/sifter eval /tmp/sifter-demo data/scifact/queries.jsonl data/scifact/qrels/test.tsv | python3 -c 'import json,sys; d=json.load(sys.stdin); print("SciFact test: nDCG@10 %.4f  Recall@100 %.4f  p50 %.3f ms (k=10)" % (d["ndcg@10"], d["recall@100"], d["p50_ms_k10"]))'
