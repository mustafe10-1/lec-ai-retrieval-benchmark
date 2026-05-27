from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrieve import RetrieverSuite

QUERIES_PATH = Path("data/queries.jsonl")
METRICS_PATH = Path("results/metrics.csv")
CATEGORY_METRICS_PATH = Path("results/category_metrics.csv")
FAILURES_PATH = Path("results/failures.jsonl")

CONFIGS = ["bm25", "dense", "hybrid_a0.25", "hybrid_a0.50", "hybrid_a0.75"]


def load_queries() -> list[dict]:
    return [json.loads(line) for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()]


def get_relevant_substrings(q: dict) -> list[str]:
    if "relevant_url_substrings" in q:
        return q["relevant_url_substrings"]
    return [q["relevant_url_substring"]]


def evaluate_config(
    retriever: RetrieverSuite,
    config: str,
    queries: list[dict],
    top_k: int = 5,
) -> tuple[dict, list[dict], list[dict]]:
    per_query: list[dict] = []
    failures: list[dict] = []
    id_to_doc = {d["doc_id"]: d for d in retriever.docs}

    for q in queries:
        results, latency_ms = retriever.timed_search(config, q["query"], top_k)

        ranked_ids = [r.doc_id for r in results]
        ranked_docs = [id_to_doc[rid] for rid in ranked_ids]  # rank order preserved
        relevant_substrings = get_relevant_substrings(q)

        hit_positions = [
            i for i, doc in enumerate(ranked_docs, start=1)
            if any(sub in doc["source_url"] for sub in relevant_substrings)
        ]
        hit = bool(hit_positions)
        recall = 1.0 if hit else 0.0
        rr = 1.0 / hit_positions[0] if hit else 0.0

        per_query.append({
            "config": config,
            "query_id": q["query_id"],
            "difficulty": q["difficulty"],
            "recall": recall,
            "reciprocal_rank": rr,
            "latency_ms": latency_ms,
        })

        if not hit:
            failures.append({
                "config": config,
                "query_id": q["query_id"],
                "query": q["query"],
                "difficulty": q["difficulty"],
                "expected_url_substrings": relevant_substrings,
                "top5_doc_ids": ranked_ids,
                "top5_urls": [doc["source_url"] for doc in ranked_docs],
            })

    metrics = {
        "config": config,
        "recall@5": float(np.mean([p["recall"] for p in per_query])),
        "mrr": float(np.mean([p["reciprocal_rank"] for p in per_query])),
        "p95_latency_ms": float(np.percentile([p["latency_ms"] for p in per_query], 95)),
    }
    return metrics, per_query, failures


def compute_category_metrics(all_per_query: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_per_query:
        grouped[(row["config"], row["difficulty"])].append(row)

    rows = []
    for config in CONFIGS:
        for difficulty in ("normal", "hard"):
            items = grouped.get((config, difficulty), [])
            if not items:
                continue
            rows.append({
                "config": config,
                "category": difficulty,
                "n_queries": len(items),
                "recall@5": float(np.mean([i["recall"] for i in items])),
                "mrr": float(np.mean([i["reciprocal_rank"] for i in items])),
            })
    return rows


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    queries = load_queries()
    retriever = RetrieverSuite()

    all_metrics: list[dict] = []
    all_per_query: list[dict] = []
    all_failures: list[dict] = []

    for config in CONFIGS:
        metrics, per_query, failures = evaluate_config(retriever, config, queries)
        all_metrics.append(metrics)
        all_per_query.extend(per_query)
        all_failures.extend(failures)

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "recall@5", "mrr", "p95_latency_ms"])
        writer.writeheader()
        writer.writerows(all_metrics)

    category_rows = compute_category_metrics(all_per_query)
    with CATEGORY_METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "category", "n_queries", "recall@5", "mrr"])
        writer.writeheader()
        writer.writerows(category_rows)

    with FAILURES_PATH.open("w", encoding="utf-8") as f:
        for row in all_failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {METRICS_PATH}, {CATEGORY_METRICS_PATH}, and {FAILURES_PATH}")
    print("\nmetrics.csv:")
    for m in all_metrics:
        print(f"  {m['config']:15s}  recall@5={m['recall@5']:.3f}  mrr={m['mrr']:.3f}  p95={m['p95_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
