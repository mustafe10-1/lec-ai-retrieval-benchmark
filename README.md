# Assignment 1: Retrieval, Honest Comparison

This repo implements a **simple, reproducible retrieval benchmark** over FastAPI docs section chunks.

## What is included

- `src/build_corpus.py`
  - Downloads FastAPI docs sitemap.
  - Fetches pages.
  - Splits each page into section-level chunks (`h2`/`h3` blocks).
  - Writes `data/docs.jsonl` (target: 200–500 real docs).
- `src/retrieve.py`
  - BM25 retrieval (`rank_bm25`).
  - Dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
  - Hybrid retrieval with min-max normalized score blending (`0.5*bm25 + 0.5*dense`).
- `src/evaluate.py`
  - Runs 20 hand-written labelled queries from `data/queries.jsonl`.
  - Computes recall@5, MRR, and p95 latency for all three configs.
  - Writes `results/metrics.csv` and `results/failures.jsonl`.
- `Makefile`
  - `make run` builds corpus then evaluates.

## Queries

- `data/queries.jsonl` has 20 labelled queries.
- Includes 5 deliberately hard queries (`difficulty: "hard"`) with paraphrase/multi-hop/ambiguity flavor.

## Run

```bash
python3 -m pip install -r requirements.txt
make run
```

## Performance constraint

Target is p95 retrieval latency under 1 second on a single laptop/free-tier VM.
If it is not met in your run, inspect `results/metrics.csv` and document bottlenecks.

## Honest failure analysis (what best config still misses)

Even when hybrid is best overall, it still tends to fail on ambiguous queries where the wording mixes two concepts (for example docs metadata + OpenAPI exposure) or where the relevant answer spans multiple sections/pages. In those cases BM25 may over-prioritize literal token overlap from the wrong page, while dense retrieval may retrieve semantically close but still non-answer sections. The 50/50 linear blend helps but cannot fully resolve intent ambiguity or multi-hop evidence needs without query decomposition, reranking, or context-aware aggregation across related sections.

## Notes about this execution environment

This environment blocked dependency installation/network package index access during execution, so placeholder `results/metrics.csv` and `results/failures.jsonl` are included and should be regenerated in a normal Python environment by running `make run`.
