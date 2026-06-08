"""Build a small FAISS index for quick local testing."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TextIO

from eval import evaluate_run, load_query_file
from index import build_index
from retrieve import search_batch
from utils import K_EVAL, PUBLIC_QUERIES_PATH

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_TOP_K = 5
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "testing_artifacts"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "past_tests.txt"
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
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to append printed test output.",
    )
    return parser.parse_args()


def log(message: str = "", *, file: TextIO) -> None:
    print(message)
    print(message, file=file)


def test_retrieval(
    queries: list[str],
    *,
    top_k: int,
    artifacts_dir: Path,
    log_file: TextIO,
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

        log(f"\nQuery: {query}", file=log_file)
        log(f"Top page IDs: {page_ids}", file=log_file)

    log("\nRetrieval test passed.", file=log_file)


def evaluate_retrieval(
    *,
    indexed_page_ids: list[int],
    artifacts_dir: Path,
    log_file: TextIO,
) -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    indexed_ids = set(indexed_page_ids)
    covered_rows = [
        row
        for row in rows
        if row["relevant_page_ids"].issubset(indexed_ids)
    ]

    if not covered_rows:
        log(
            "\nEvaluation skipped: no public query is fully covered by the sample.",
            file=log_file,
        )
        return

    queries = [row["query"] for row in covered_rows]
    ground_truth = [row["relevant_page_ids"] for row in covered_rows]
    stats = evaluate_run(
        queries,
        ground_truth,
        lambda batch: search_batch(
            batch,
            top_k=K_EVAL,
            artifacts_dir=artifacts_dir,
        ),
        k=K_EVAL,
    )

    log("\nSample evaluation", file=log_file)
    log(f"Covered public queries: {len(covered_rows)}/{len(rows)}", file=log_file)
    log(f"Mean NDCG@{K_EVAL}: {stats['mean_ndcg@10']:.4f}", file=log_file)
    log("Note: this diagnostic score uses a reduced corpus.", file=log_file)


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    public_rows = load_query_file(PUBLIC_QUERIES_PATH)
    required_page_ids = {
        page_id
        for row in public_rows
        for page_id in row["relevant_page_ids"]
    }
    if args.sample_size < len(required_page_ids):
        raise ValueError(
            f"--sample-size must be at least {len(required_page_ids)} "
            "to include all public-query answers"
        )

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log("\n" + "=" * 72, file=log_file)
        log(f"Test run: {timestamp}", file=log_file)
        log(
            f"sample_size={args.sample_size}, top_k={args.top_k}, "
            f"artifacts_dir={args.artifacts_dir.resolve()}",
            file=log_file,
        )

        _, page_ids = build_index(
            artifacts_dir=args.artifacts_dir,
            max_entries=args.sample_size,
            required_page_ids=required_page_ids,
        )
        log(f"Built test index with {len(page_ids)} vectors.", file=log_file)
        log(f"Artifacts: {args.artifacts_dir.resolve()}", file=log_file)

        queries = args.queries or DEFAULT_QUERIES
        test_retrieval(
            queries,
            top_k=args.top_k,
            artifacts_dir=args.artifacts_dir,
            log_file=log_file,
        )
        evaluate_retrieval(
            indexed_page_ids=page_ids,
            artifacts_dir=args.artifacts_dir,
            log_file=log_file,
        )


if __name__ == "__main__":
    main()
