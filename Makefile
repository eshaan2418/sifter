NPROC := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
J ?= $(shell n=$(NPROC); h=$$((n / 2)); [ $$h -lt 2 ] && h=2; echo $$h)
export CARGO_BUILD_JOBS ?= $(J)
DATASETS := scifact nfcorpus fiqa

.PHONY: build test lint data eval bench demo clean

build:
	cargo build --release --bins --examples

test:
	cargo test --release

lint:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

# BEIR datasets (~64 MB) into ./data
data:
	mkdir -p data
	for d in $(DATASETS); do [ -d data/$$d ] || (cd data && curl -sSLO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$$d.zip && unzip -q $$d.zip && rm $$d.zip); done

eval: build data
	for d in $(DATASETS); do rm -rf /tmp/sifter-$$d; ./target/release/sifter index /tmp/sifter-$$d data/$$d/corpus.jsonl && ./target/release/sifter eval /tmp/sifter-$$d data/$$d/queries.jsonl data/$$d/qrels/test.tsv; done

bench: build data
	python3 scripts/bench.py

demo: build data
	./scripts/demo.sh

clean:
	cargo clean
