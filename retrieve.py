"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from rerank import score_query_passage_pairs
from utils import (
    ARTIFACTS_DIR,
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_CANDIDATE_CHUNKS,
    K_EVAL,
)


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Search the offline-built FAISS index using cosine similarity
    (inner product over L2-normalized embeddings).
    """
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    faiss_index, page_ids = load_index(artifacts_dir)
    chunk_texts = _load_chunk_texts(artifacts_dir)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Search extra rows because several chunks may map to the same page_id.
    search_k = min(faiss_index.ntotal, max(1000, CROSS_ENCODER_CANDIDATE_CHUNKS))

    scores, neighbor_indices = faiss_index.search(query_vectors, search_k)

    query_candidates: List[List[tuple[int, float]]] = []
    rerank_pairs: List[tuple[str, str]] = []
    rerank_offsets = [0]

    for query, score_row, index_row in zip(queries, scores, neighbor_indices):
        candidates = []
        seen_chunks: set[int] = set()

        for score, idx in zip(score_row, index_row):
            if idx < 0:
                continue

            chunk_index = int(idx)
            if chunk_index in seen_chunks:
                continue
            seen_chunks.add(chunk_index)

            candidates.append((chunk_index, float(score)))
            if len(candidates) >= CROSS_ENCODER_CANDIDATE_CHUNKS:
                break

        query_candidates.append(candidates)
        rerank_pairs.extend(
            (query, chunk_texts[chunk_index]) for chunk_index, _ in candidates
        )
        rerank_offsets.append(len(rerank_pairs))

    rerank_scores = score_query_passage_pairs(
        rerank_pairs,
        batch_size=CROSS_ENCODER_BATCH_SIZE,
    )

    ranked: List[List[int]] = []

    for query_index, candidates in enumerate(query_candidates):
        if not candidates:
            ranked.append([])
            continue

        passage_scores = rerank_scores[
            rerank_offsets[query_index] : rerank_offsets[query_index + 1]
        ]

        page_to_scores: dict[int, list[tuple[float, float]]] = {}
        for (chunk_index, embedding_score), rerank_score in zip(candidates, passage_scores):
            pid = page_ids[chunk_index]
            page_to_scores.setdefault(pid, []).append(
                (float(rerank_score), embedding_score)
            )

        page_final_scores = []

        for pid, chunk_scores in page_to_scores.items():
            chunk_scores.sort(reverse=True)
            best_rerank_score, best_embedding_score = chunk_scores[0]

            # Cross-encoder is the main relevance signal; the embedding score is
            # just a stable tie-breaker for near-equal reranker scores.
            page_final_scores.append((pid, best_rerank_score, best_embedding_score))

        page_final_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        ids = [pid for pid, _, _ in page_final_scores[:top_k]]
        ranked.append(ids)

    return ranked


def _load_chunk_texts(artifacts_dir: Optional[Path] = None) -> List[str]:
    root = artifacts_dir or ARTIFACTS_DIR

    meta = json.loads((root / "index_meta.json").read_text(encoding="utf-8"))
    chunk_texts = meta.get("chunk_texts")
    if chunk_texts is None:
        raise ValueError(
            "index_meta.json does not contain chunk_texts required for "
            "cross-encoder reranking. Rebuild artifacts with: python main.py"
        )

    if len(chunk_texts) != len(meta.get("page_ids", [])):
        raise ValueError("index_meta.json has inconsistent chunk_texts/page_ids lengths")

    return [str(text) for text in chunk_texts]
