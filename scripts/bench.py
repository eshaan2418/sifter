#!/usr/bin/env python3
"""BEIR relevance + latency for sifter and tantivy; writes bench/results/.

Relevance (nDCG@10, Recall@100) is deterministic; latency is the median of
per-run p50/p99 over RUNS runs. fiqa20 (FiQA replicated 20x, 1.15M docs)
measures latency at scale only (its ids do not match the qrels).
"""
import json, os, platform, shutil, statistics, subprocess, sys, tempfile, time

RUNS = int(os.environ.get("RUNS", "5"))
SIFTER = "target/release/sifter"
TANTIVY = "target/release/examples/tantivy_baseline"
DATA = "data"


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def rss_and_out(cmd):
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], check=True, capture_output=True, text=True)
    rss = next(int(l.split()[0]) for l in p.stderr.splitlines() if "maximum resident set size" in l)
    return p.stdout, p.stderr, rss


def make_fiqa20():
    out = f"{DATA}/fiqa20.jsonl"
    if not os.path.exists(out):
        with open(f"{DATA}/fiqa/corpus.jsonl") as f, open(out, "w") as g:
            docs = [json.loads(l) for l in f]
            for r in range(20):
                for d in docs:
                    g.write(json.dumps({**d, "_id": f"{d['_id']}#{r}"}) + "\n")
    return out


def main():
    results = {"env": {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "machine": run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip() if sys.platform == "darwin" else platform.processor(),
        "os": platform.platform(), "rustc": run(["rustc", "--version"]).stdout.strip(), "runs": RUNS,
    }, "datasets": {}}
    sets = [("scifact", f"{DATA}/scifact/corpus.jsonl"), ("nfcorpus", f"{DATA}/nfcorpus/corpus.jsonl"),
            ("fiqa", f"{DATA}/fiqa/corpus.jsonl"), ("fiqa20", make_fiqa20())]
    for name, corpus in sets:
        qset = "fiqa" if name == "fiqa20" else name
        queries, qrels = f"{DATA}/{qset}/queries.jsonl", f"{DATA}/{qset}/qrels/test.tsv"
        d = tempfile.mkdtemp(prefix=f"sifter-{name}-")
        _, err, rss = rss_and_out([SIFTER, "index", d, corpus])
        run([SIFTER, "merge", d])
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".sft"))
        line = next(l for l in err.splitlines() if l.startswith("indexed "))
        entry = {"docs": int(line.split()[1]), "index_s": float(line.split()[-1].rstrip("s")),
                 "index_rss_bytes": rss, "index_bytes": size}
        configs = {
            "sifter exhaustive k1=0.9 b=0.4": [],
            "sifter exhaustive k1=1.2 b=0.75": ["--k1", "1.2", "--b", "0.75"],
            "sifter bmw k1=1.2 b=0.75": ["--k1", "1.2", "--b", "0.75", "--bmw"],
        }
        for label, extra in configs.items():
            outs = [json.loads(run([SIFTER, "eval", d, queries, qrels, *extra]).stdout) for _ in range(RUNS)]
            entry[label] = {**outs[0], **{k: statistics.median(o[k] for o in outs) for k in outs[0] if k.startswith("p")}}
            print(name, label, {k: entry[label][k] for k in ("ndcg@10", "recall@100", "p50_ms_k10", "p50_ms_k100")}, flush=True)
        outs = [json.loads(run([TANTIVY, corpus, queries, qrels]).stdout) for _ in range(RUNS)]
        entry["tantivy k1=1.2 b=0.75"] = {**outs[0], **{k: statistics.median(o[k] for o in outs) for k in outs[0] if k.startswith("p") or k == "index_s"}}
        print(name, "tantivy", {k: entry["tantivy k1=1.2 b=0.75"][k] for k in ("ndcg@10", "recall@100", "p50_ms_k10", "p50_ms_k100", "index_s")}, flush=True)
        shutil.rmtree(d)
        results["datasets"][name] = entry
    os.makedirs("bench/results", exist_ok=True)
    with open("bench/results/beir.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
