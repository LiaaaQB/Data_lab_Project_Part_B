"""Build a small FAISS index for quick local testing."""
from __future__ import annotations

import argparse
from pathlib import Path

from index import build_index
from retrieve import search_batch

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_TOP_K = 5
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "testing_artifacts"
DEFAULT_QUERIES = [
    "basketball championship",
    "scientific research and technology",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a small corpus sample."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of Wikipedia entries to index (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Output directory for the test index.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Retrieval query. Repeat this option to test multiple queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results per query (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def test_retrieval(
    queries: list[str],
    *,
    top_k: int,
    artifacts_dir: Path,
) -> None:
    ranked_results = search_batch(
        queries,
        top_k=top_k,
        artifacts_dir=artifacts_dir,
    )

    assert len(ranked_results) == len(queries)
    for query, page_ids in zip(queries, ranked_results):
        assert len(page_ids) <= top_k
        assert len(page_ids) == len(set(page_ids))
        assert all(isinstance(page_id, int) for page_id in page_ids)

        print(f"\nQuery: {query}")
        print(f"Top page IDs: {page_ids}")

    print("\nRetrieval test passed.")


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    _, page_ids = build_index(
        artifacts_dir=args.artifacts_dir,
        max_entries=args.sample_size,
    )
    print(f"Built test index with {len(page_ids)} vectors.")
    print(f"Artifacts: {args.artifacts_dir.resolve()}")

    queries = args.queries or DEFAULT_QUERIES
    test_retrieval(
        queries,
        top_k=args.top_k,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
