# Assignment 1: Retrieval, Honest Comparison

## Why I picked this assignment

Retrieval quality is the load-bearing layer of every RAG system. Before the model ever sees a prompt, the retriever decides what context it gets. I wanted to benchmark this as an engineering problem — measurable, repeatable, comparable — rather than just wire up a vector database and call it done.

## Why FastAPI documentation

FastAPI docs are real technical prose, not synthetic data. They combine exact-match terms (`Depends`, `APIRouter`, `OAuth2PasswordBearer`) with conceptual explanation, making the corpus useful for stressing both keyword and semantic retrieval. The documentation is also well-structured with headings, which makes section-level chunking natural. Section chunks bring the task closer to a realistic RAG setup where a system retrieves a focused passage, not a full page.

The final corpus: **500 section-level documents** from the FastAPI documentation at `fastapi.tiangolo.com`.

## What the system does

Three pipeline stages:

1. `src/build_corpus.py` — fetches the FastAPI sitemap, extracts required pages first (guaranteeing every query label is satisfiable), then fills from the sitemap to 500 sections. Writes `data/docs.jsonl`. Fails with a clear error if any required URL is missing after build.

2. `src/retrieve.py` — implements BM25 (`rank_bm25`), dense cosine similarity (`all-MiniLM-L6-v2`), and hybrid blends with configurable alpha (dense weight).

3. `src/evaluate.py` — runs 20 labelled queries across five configs, computes recall@5, MRR, and p95 latency. Writes `results/metrics.csv`, `results/category_metrics.csv`, and `results/failures.jsonl`.

## Decisions made and alternatives ruled out

**Chunking at section level.** Whole pages are 5–20x larger and would swamp BM25 with off-topic text. Sentence-level chunks would split mid-explanation and hurt recall. Section headings (`h2`/`h3`) are the natural unit.

**`all-MiniLM-L6-v2` for dense.** Small, fast, no API key required. Larger models (e.g. `bge-large`) would improve MRR but break the "single laptop" constraint. A future run could swap the model in one line.

**Min-max normalisation before blending.** BM25 scores are unbounded; cosine scores are in `[-1, 1]`. Without normalisation a fixed alpha is meaningless. Min-max per query is the simplest choice that makes the blend interpretable.

**Alpha sweep instead of single 50/50.** A fixed 50/50 blend is arbitrary. Running `alpha ∈ {0.25, 0.50, 0.75}` turns the hybrid into a question — "how much BM25 helps" — rather than an assumption.

**Alternatives ruled out:** reranking (adds a cross-encoder dependency, not required); BM25 with stopword removal (marginal on a technical corpus); chunking by token count (loses natural section boundaries).

## Queries

Twenty hand-written queries in `data/queries.jsonl`. Each has a list of acceptable `relevant_url_substrings`: a result counts as a hit if its source URL contains any of the listed substrings. Fifteen are normal-difficulty lookups; five are deliberately hard (paraphrased, indirect, or multi-concept). Hard examples:

- **q16:** asks about "one auth helper reused in many endpoints" — the answer lives on the `dependencies/` or `security/get-current-user/` pages, but the phrasing mentions neither.
- **q19:** asks about "422 for mixed path/body inputs" — acceptable answers span `body-multiple-params/`, `body/`, and `handling-errors/`.

## Headline results

All p95 latencies are well under 1 second (max 34 ms). Dense is the best overall config.

| config | recall@5 | MRR | p95 latency |
|---|---|---|---|
| bm25 | 0.85 | 0.69 | 2 ms |
| dense | **1.00** | **0.83** | 17 ms |
| hybrid α=0.25 | 0.90 | 0.75 | 26 ms |
| hybrid α=0.50 | 1.00 | 0.79 | 26 ms |
| hybrid α=0.75 | 1.00 | 0.81 | 34 ms |

BM25 misses 3 queries; every hybrid with α ≥ 0.50 recovers them all. Adding any BM25 weight raises latency without improving MRR past the pure-dense baseline.

## Category-level finding

| config | category | n | recall@5 | MRR |
|---|---|---|---|---|
| dense | normal | 15 | 1.00 | 0.88 |
| dense | hard | 5 | 1.00 | **0.67** |
| hybrid α=0.75 | normal | 15 | 1.00 | 0.82 |
| hybrid α=0.75 | hard | 5 | 1.00 | **0.77** |

Dense achieves perfect recall on both groups but its MRR on hard queries (0.67) is notably worse than on normal queries (0.88). The right answer is always somewhere in the top 5, but it often lands at rank 2 or 3 because the paraphrase distance degrades similarity scores enough to let tangentially related sections edge ahead.

Interestingly, `hybrid α=0.75` closes the hard-query gap (MRR 0.77 vs 0.67 for dense) while matching dense on normal queries' recall. The BM25 component appears to anchor the result set with lexically exact sections, preventing the semantic model from drifting on indirect phrasings.

## Where the best config still loses

Dense is still the best config overall (highest MRR, 0.83), but it consistently underperforms on **paraphrased indirect queries** relative to direct lookups. The hard-query MRR of 0.67 corresponds to a noticeably lower average reciprocal rank than the normal-query MRR of 0.88, meaning the right answer is often ranked lower on indirect queries. The failure pattern is predictable: paraphrases that share no surface tokens with the target page title push the right section to position 2–3 while surface-similar but off-target sections float to rank 1. For example, q17 ("docs UI blocked in production but schema should still exist") consistently surfaces `schema-extra-example/` sections before `metadata/` because "schema" appears literally in those titles. The model understands the concept but the wrong section wins on a word-level feature the embedding partially encodes.

## A concrete failure

**Query q02 (BM25):** "How can I define path parameters with Python type hints?"
- Expected: any URL containing `/tutorial/path-params/` or `/tutorial/path-params-numeric-validations/`
- Returned: `/python-types/` (ranks 1–4), `/tutorial/query-params-str-validations/` (rank 5)

BM25 matches "Python type hints" literally against the `/python-types/` documentation page — a page about Python type annotations in general — because the tokens `python` and `type` appear densely there. It never reaches the path-params page, where the phrase "type hints" appears only once in context. Dense retrieval has no such failure: it understands that the query is asking about FastAPI path parameters and returns the correct page at rank 1.

## One thing to do differently

With another week: query-type-aware routing. The alpha sweep already shows that `hybrid α=0.75` beats dense on hard queries while dense beats every hybrid on normal queries. A simple classifier on query length and presence of indirect phrasing could switch between "dense-only" and "dense-heavy hybrid" at runtime, combining both advantages. The evaluation harness already produces per-query results, so fitting a two-class router would take one afternoon.

## How to run

```bash
python3 -m pip install -r requirements.txt
make run
```

`make run` rebuilds the corpus from the live FastAPI documentation and then runs all five retrieval configs. Results are written to `results/metrics.csv`, `results/category_metrics.csv`, and `results/failures.jsonl`. The corpus and query labels are also committed to the repo so `make evaluate` can be run alone without re-fetching.
