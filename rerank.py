"""Candidate reranking utilities.

The cross-encoder is used only after MiniLM embeddings and FAISS have already
selected candidate chunks. It does not create corpus or query embeddings.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from embed import get_device
from utils import CROSS_ENCODER_MODEL_NAME

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=get_device())
    return _cross_encoder


def score_query_passages(
    query: str,
    passages: Sequence[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Return cross-encoder relevance scores for one query against passages."""
    if not passages:
        return np.zeros((0,), dtype=np.float32)

    model = get_cross_encoder()
    pairs = [(query, passage) for passage in passages]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(scores, dtype=np.float32)


def score_query_passage_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Return cross-encoder relevance scores for pre-built query/passage pairs."""
    if not pairs:
        return np.zeros((0,), dtype=np.float32)

    model = get_cross_encoder()
    scores = model.predict(
        list(pairs),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(scores, dtype=np.float32)
