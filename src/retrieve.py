from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("data/docs.jsonl")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float


class RetrieverSuite:
    def __init__(self, docs_path: Path = DOCS_PATH) -> None:
        self.docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines()]
        self.doc_texts = [f"{d['page_title']} {d['section_title']} {d['text']}" for d in self.docs]
        self.doc_ids = [d["doc_id"] for d in self.docs]

        tokenized = [tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        dense = self.model.encode(self.doc_texts, convert_to_numpy=True, show_progress_bar=False)
        self.doc_embeddings = dense / np.linalg.norm(dense, axis=1, keepdims=True)

    def bm25_search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        scores = np.array(self.bm25.get_scores(tokenize(query)))
        idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievedDoc(self.doc_ids[i], float(scores[i])) for i in idx]

    def dense_search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        q = q / np.linalg.norm(q)
        scores = self.doc_embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievedDoc(self.doc_ids[i], float(scores[i])) for i in idx]

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[RetrievedDoc]:
        """alpha is the dense weight; (1-alpha) is the BM25 weight."""
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        q_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        q_vec = q_vec / np.linalg.norm(q_vec)
        dense_scores = self.doc_embeddings @ q_vec

        bm25_norm = min_max(bm25_scores)
        dense_norm = min_max(dense_scores)
        blended = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        idx = np.argsort(blended)[::-1][:top_k]
        return [RetrievedDoc(self.doc_ids[i], float(blended[i])) for i in idx]

    def timed_search(self, config: str, query: str, top_k: int = 5) -> tuple[list[RetrievedDoc], float]:
        start = time.perf_counter()
        if config == "bm25":
            out = self.bm25_search(query, top_k)
        elif config == "dense":
            out = self.dense_search(query, top_k)
        elif config == "hybrid" or config.startswith("hybrid_a"):
            alpha = 0.5
            if config.startswith("hybrid_a"):
                alpha = float(config[len("hybrid_a"):])
            out = self.hybrid_search(query, top_k, alpha=alpha)
        else:
            raise ValueError(f"Unknown config: {config!r}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return out, elapsed_ms


def min_max(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
