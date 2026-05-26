from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from retrieve import RetrieverSuite

QUERIES_PATH = Path("data/queries.jsonl")
METRICS_PATH = Path("results/metrics.csv")
FAILURES_PATH = Path("results/failures.jsonl")


def load_queries() -> list[dict]:
    return [json.loads(line) for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()]


def evaluate_config(retriever: RetrieverSuite, config: str, queries: list[dict], top_k: int = 5) -> tuple[dict, list[dict]]:
    recalls = []
    reciprocal_ranks = []
    latencies = []
    failures = []

    for q in queries:
        results, latency_ms = retriever.timed_search(config, q["query"], top_k)
        latencies.append(latency_ms)

        ranked_ids = [r.doc_id for r in results]
        ranked_docs = [d for d in retriever.docs if d["doc_id"] in ranked_ids]
        rel = q["relevant_url_substring"]

        hit_positions = []
        for i, doc_id in enumerate(ranked_ids, start=1):
            doc = next(d for d in retriever.docs if d["doc_id"] == doc_id)
            if rel in doc["source_url"]:
                hit_positions.append(i)

        hit = len(hit_positions) > 0
        recalls.append(1.0 if hit else 0.0)
        reciprocal_ranks.append(1.0 / hit_positions[0] if hit else 0.0)

        if not hit:
            failures.append(
                {
                    "config": config,
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "difficulty": q["difficulty"],
                    "expected_url_substring": rel,
                    "top5_doc_ids": ranked_ids,
                    "top5_urls": [doc["source_url"] for doc in ranked_docs],
                }
            )

    metrics = {
        "config": config,
        "recall@5": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
    return metrics, failures


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    queries = load_queries()
    retriever = RetrieverSuite()

    all_metrics = []
    all_failures = []
    for config in ["bm25", "dense", "hybrid"]:
        metrics, failures = evaluate_config(retriever, config, queries)
        all_metrics.append(metrics)
        all_failures.extend(failures)

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "recall@5", "mrr", "p95_latency_ms"])
        writer.writeheader()
        writer.writerows(all_metrics)

    with FAILURES_PATH.open("w", encoding="utf-8") as f:
        for row in all_failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {METRICS_PATH} and {FAILURES_PATH}")


if __name__ == "__main__":
    main()
