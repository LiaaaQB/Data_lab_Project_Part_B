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

    faiss_index, page_ids = load_faiss_index(artifacts_dir)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Search extra rows because several chunks may map to the same page_id.
    search_k = min(faiss_index.ntotal, max(top_k * 4, top_k))
    _, neighbor_indices = faiss_index.search(query_vectors, search_k)

    ranked: List[List[int]] = []
    for row in neighbor_indices:
        seen: set[int] = set()
        ids: List[int] = []
        for idx in row:
            if idx < 0:
                continue
            pid = page_ids[int(idx)]
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= top_k:
                break
        ranked.append(ids)
    return ranked
