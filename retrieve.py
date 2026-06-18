"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from sentence_transformers import CrossEncoder

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


# Load the CrossEncoder model.
_reranker = None

def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

    return _reranker

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

    faiss_index, page_ids, chunk_texts = load_index(
    artifacts_dir)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Search extra rows because several chunks may map to the same page_id.
    search_k = min(faiss_index.ntotal, 200)

    scores, neighbor_indices = faiss_index.search(query_vectors, search_k)

    ranked: List[List[int]] = []

    reranker = get_reranker()

    for q_idx, (score_row, index_row) in enumerate(
        zip(scores, neighbor_indices)
    ):
        # Building (query, chunk) pairs for CrossEncoder reranking.
        pairs = []
        candidate_indices = []

        for idx in index_row:
            if idx < 0:
                continue

            pairs.append(
                (
                    queries[q_idx],
                    chunk_texts[int(idx)]
                )
            )

            candidate_indices.append(int(idx))

        ce_scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        page_to_scores = {}

        for ce_score, idx in zip(
            ce_scores,
            candidate_indices
        ):
            pid = page_ids[idx]

            if pid not in page_to_scores:
                page_to_scores[pid] = []

            page_to_scores[pid].append(float(ce_score))

        page_final_scores = []

        # Aggregating chunk-level scores into page-level scores.
        for pid, chunk_scores in page_to_scores.items():
            chunk_scores.sort(reverse=True)
            # Use the strongest chunk as the main signal and
            # adding small contributions from additional relevant chunks.
            if len(chunk_scores) >= 3:
              final_score = (chunk_scores[0]+ 0.1 * chunk_scores[1] + 0.05 * chunk_scores[2]
    )
            elif len(chunk_scores) == 2:
              final_score = (chunk_scores[0]  + 0.1 * chunk_scores[1]
    )
            else:
               final_score = chunk_scores[0]


            page_final_scores.append(
                (pid, final_score)
            )
        # Rank pages by their aggregated relevance score.
        page_final_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        ids = [
            pid
            for pid, _ in page_final_scores[:top_k]
        ]

        ranked.append(ids)

    return ranked