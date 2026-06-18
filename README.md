# Section B — Retrieval pipeline

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).

## Build index (offline, not timed — your machine only)

The index is built offline and stored under the artifacts/ directory.

Run:
```bash
python main.py
```
During index construction the system:

1. Loads Wikipedia pages from the corpus.
2. Splits pages into overlapping text chunks.
3. Generates embeddings using: sentence-transformers/all-MiniLM-L6-v2
4. Builds a FAISS inner-product index over L2-normalized embeddings.
5. Saves all required retrieval artifacts.

Generated artifacts:

artifacts/
├── index.faiss
├── index_meta.json
├── index_vectors.npy
└── chunk_texts.json

These artifacts are included in the repository and are used directly during evaluation.
## Retrieval Pipeline

At query time:

1. Query embeddings are generated using:
sentence-transformers/all-MiniLM-L6-v2
2. FAISS retrieves the top candidate chunks from the offline index.
3. Candidate chunks are reranked using:
cross-encoder/ms-marco-MiniLM-L-6-v2
4. Chunk scores are aggregated at the page level.
5. The top-ranked page IDs are returned.

## Public self-test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

## Submit

The repository contains:

Source code
Required artifacts under artifacts/
Retrieval pipeline implementation
This README

No index rebuilding is required during grading. Evaluation uses the submitted artifacts directly.
