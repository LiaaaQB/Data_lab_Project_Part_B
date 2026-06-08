"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from streamlit import title

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record):
    page_id = int(record["page_id"])
    title = record.get("title", "")
    text = f"Title: {title}\nContent: {' '.join(chunk_words)}"

    words = text.split()

    CHUNK_SIZE = 150
    OVERLAP = 50

    chunks = []
    chunk_id = 0

    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):

        chunk_words = words[start:start + CHUNK_SIZE]

        if len(chunk_words) < 20:
            continue

        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=chunk_id,
                text=" ".join(chunk_words)
            )
        )

        chunk_id += 1

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
