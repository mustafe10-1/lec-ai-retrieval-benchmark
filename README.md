# Assignment 1: Retrieval, Honest Comparison

This repository implements a small retrieval benchmark over a real technical corpus: section-level chunks from the official FastAPI documentation.

The goal is not to build a polished demo. The goal is to make a defensible comparison between retrieval configurations on the same corpus and the same labelled queries.

## Why I picked this assignment

I picked the retrieval assignment because retrieval quality is central to production RAG systems. In an AI product, the model can only answer well if the right context is retrieved first, so I wanted to test retrieval as an engineering problem rather than just use a vector database blindly.

I used FastAPI documentation because it is a real technical corpus, not synthetic data. It is also relevant to backend AI engineering because FastAPI is commonly used for Python API services. The documentation has a mix of exact technical terms, such as `Depends`, `APIRouter`, and `OAuth2`, as well as broader conceptual pages, which makes it useful for comparing keyword, dense, and hybrid retrieval.

## What the system does

The pipeline has three stages:

1. `src/build_corpus.py`
   - downloads the FastAPI documentation sitemap;
   - fetches documentation pages;
   - splits pages into section-level chunks using headings;
   - writes `data/docs.jsonl`.

2. `src/retrieve.py`
   - implements BM25 retrieval using `rank_bm25`;
   - implements dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`;
   - implements hybrid retrieval using a 50/50 blend of min-max normalised BM25 and dense scores.

3. `src/evaluate.py`
   - runs the same 20 labelled queries against all three retrieval configurations;
   - computes recall@5, MRR, and p95 retrieval latency;
   - writes `results/metrics.csv` and `results/failures.jsonl`.

## Corpus

The final run produced:

- 450 section-level documents in `data/docs.jsonl`;
- all documents came from the FastAPI documentation;
- each document stores a URL, title, and text chunk.

I used section-level chunks rather than whole pages because whole documentation pages often contain several different concepts. Section chunks make the retrieval task closer to a realistic RAG setup, where a system needs to retrieve a focused passage rather than an entire long page.

## Queries

The benchmark uses 20 hand-written labelled queries in `data/queries.jsonl`.

The query set includes:

- normal documentation lookup queries;
- exact technical keyword queries;
- 5 deliberately hard queries involving paraphrase, ambiguity, or multi-hop intent.

Each query is labelled with the documentation URL substring that should appear in the top-k results.

## How to run

```bash
python3 -m pip install -r requirements.txt
make run
