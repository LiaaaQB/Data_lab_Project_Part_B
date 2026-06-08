"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


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
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Search extra rows because several chunks may map to the same page_id.
    search_k = min(faiss_index.ntotal, 200)

    scores, neighbor_indices = faiss_index.search(query_vectors, search_k)

    ranked: List[List[int]] = []

    for score_row, index_row in zip(scores, neighbor_indices):
        page_to_scores: dict[int, list[float]] = {}

        for score, idx in zip(score_row, index_row):
            if idx < 0:
                continue

            pid = page_ids[int(idx)]

            if pid not in page_to_scores:
                page_to_scores[pid] = []

            page_to_scores[pid].append(float(score))

        page_final_scores = []

        for pid, chunk_scores in page_to_scores.items():
            chunk_scores.sort(reverse=True)

            # Main score = best chunk.
            # Small bonus = multiple good chunks from same page.
            final_score = chunk_scores[0] + 0.1 * sum(chunk_scores[:3])

            page_final_scores.append((pid, final_score))

        page_final_scores.sort(key=lambda x: x[1], reverse=True)

        ids = [pid for pid, _ in page_final_scores[:top_k]]
        ranked.append(ids)

    return ranked