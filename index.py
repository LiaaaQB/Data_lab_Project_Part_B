"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
FAISS_INDEX_NAME = "index.faiss"


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for indexing and retrieval. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return faiss


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus and persist artifacts.

    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    For multi-chunk pipelines, store chunk metadata in index_meta.json and
    aggregate to page_id in retrieve.py.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    # Normalized embeddings + inner product are equivalent to cosine similarity.
    faiss = _import_faiss()
    dimension = int(vectors.shape[1])
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(faiss_index, str(out_dir / FAISS_INDEX_NAME))

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
        "dimension": dimension,
        "faiss_index": FAISS_INDEX_NAME,
        "faiss_metric": "inner_product",
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return vectors, page_ids


def load_index_old(   # The original load_index function, kept for reference but not used in the main pipeline
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids


def load_index( # The new load_index function that loads the FAISS index and page_id mapping, used in the main pipeline
    artifacts_dir: Optional[Path] = None,
) -> Tuple[Any, List[int]]:
    """Load the persisted FAISS index and its row-to-page mapping."""
    root = artifacts_dir or ARTIFACTS_DIR
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    index_name = str(meta.get("faiss_index", FAISS_INDEX_NAME))

    faiss = _import_faiss()
    faiss_index = faiss.read_index(str(root / index_name))
    if faiss_index.ntotal != len(page_ids):
        raise ValueError(
            "FAISS index and metadata are inconsistent: "
            f"{faiss_index.ntotal} vectors but {len(page_ids)} page IDs"
        )
    return faiss_index, page_ids
