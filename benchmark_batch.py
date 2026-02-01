"""
Benchmark batch ranking implementations.

Compares:
- batch_rank: Current ThreadPoolExecutor-based implementation
- batch_rank_vectorized: New fused matrix operation implementation

Usage:
    uv run python benchmark_batch.py
    uv run python benchmark_batch.py --domain earth_science --num-queries 200
"""

import argparse
import time
from functools import cache

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from ranking_evolved.bm25_freeform_fast import BM25, Corpus, LuceneTokenizer


@cache
def _load_bright(domain: str):
    """Load BRIGHT documents and examples."""
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    return documents, examples


def benchmark_method(
    bm25: BM25,
    queries: list[list[str]],
    top_k: int | None,
    method: str,
    num_runs: int = 3,
) -> tuple[float, float, list]:
    """
    Benchmark a ranking method.

    Args:
        bm25: BM25 instance
        queries: List of tokenized queries
        top_k: Number of top results
        method: "current" or "vectorized"
        num_runs: Number of runs for averaging

    Returns:
        (mean_time, std_time, results)
    """
    times = []
    results = None

    for run in range(num_runs):
        start = time.perf_counter()
        if method == "current":
            results = bm25.batch_rank(queries, top_k)
        else:
            results = bm25.batch_rank_vectorized(queries, top_k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), results


def verify_correctness(
    results1: list[tuple[np.ndarray, np.ndarray]],
    results2: list[tuple[np.ndarray, np.ndarray]],
    top_k: int | None = None,
    tol: float = 1e-6,
) -> tuple[bool, str]:
    """
    Verify that two result sets are identical.

    Args:
        results1: Results from method 1
        results2: Results from method 2
        top_k: If set, only compare top_k results
        tol: Tolerance for floating point comparison

    Returns:
        (is_correct, message)
    """
    if len(results1) != len(results2):
        return False, f"Length mismatch: {len(results1)} vs {len(results2)}"

    for i, ((idx1, scores1), (idx2, scores2)) in enumerate(zip(results1, results2)):
        # Compare indices
        compare_k = top_k if top_k else len(idx1)
        idx1_top = idx1[:compare_k]
        idx2_top = idx2[:compare_k]

        if not np.array_equal(idx1_top, idx2_top):
            # Check if scores are the same but order differs due to ties
            scores1_top = scores1[:compare_k]
            scores2_top = scores2[:compare_k]
            if not np.allclose(np.sort(scores1_top)[::-1], np.sort(scores2_top)[::-1], atol=tol):
                return False, f"Query {i}: Indices differ and scores don't match"
            # Indices differ but scores match - likely tie-breaking difference
            # This is acceptable

        # Compare scores for top results
        scores1_top = scores1[:compare_k]
        scores2_top = scores2[:compare_k]
        if not np.allclose(scores1_top, scores2_top, atol=tol):
            max_diff = np.max(np.abs(scores1_top - scores2_top))
            return False, f"Query {i}: Score mismatch (max diff: {max_diff})"

    return True, "All results match"


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch ranking methods")
    parser.add_argument(
        "--domain",
        type=str,
        default="biology",
        help="BRIGHT domain to use (default: biology)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to benchmark (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k results to return (default: 100)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for averaging (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42)",
    )
    args = parser.parse_args()

    print(f"Loading BRIGHT {args.domain}...")
    documents, examples = _load_bright(args.domain)

    # Initialize tokenizer
    tokenizer = LuceneTokenizer()

    # Tokenize documents
    print("Tokenizing documents...")
    doc_tokens = [
        tokenizer(doc["content"])
        for doc in tqdm(documents, desc="Tokenizing", unit="doc")
    ]
    doc_ids = [doc["id"] for doc in documents]

    # Build corpus
    print("Building corpus index...")
    corpus = Corpus(doc_tokens, ids=doc_ids)
    print(f"Corpus: {corpus.vocab_size:,} terms, {corpus.N:,} documents")

    # Create BM25 scorer
    bm25 = BM25(corpus)

    # Prepare queries
    all_queries_raw = [ex["query"] for ex in examples]

    # Sample queries if needed
    rng = np.random.default_rng(args.seed)
    num_queries = min(args.num_queries, len(all_queries_raw))
    indices = rng.choice(len(all_queries_raw), size=num_queries, replace=False)
    sampled_queries_raw = [all_queries_raw[i] for i in indices]

    # Tokenize queries
    print(f"Tokenizing {num_queries} queries...")
    queries = [tokenizer(q) for q in sampled_queries_raw]

    print(f"\n{'='*60}")
    print(f"Benchmark Configuration:")
    print(f"  Domain: {args.domain}")
    print(f"  Documents: {corpus.N:,}")
    print(f"  Vocabulary: {corpus.vocab_size:,}")
    print(f"  Queries: {num_queries}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Runs: {args.num_runs}")
    print(f"{'='*60}\n")

    # Warmup
    print("Warming up...")
    _ = bm25.batch_rank(queries[:5], args.top_k)
    _ = bm25.batch_rank_vectorized(queries[:5], args.top_k)

    # Benchmark current implementation
    print(f"\nBenchmarking batch_rank (ThreadPoolExecutor)...")
    mean1, std1, results1 = benchmark_method(
        bm25, queries, args.top_k, "current", args.num_runs
    )
    print(f"  Time: {mean1:.3f}s ± {std1:.3f}s")
    print(f"  Throughput: {num_queries / mean1:.1f} queries/sec")

    # Benchmark vectorized implementation
    print(f"\nBenchmarking batch_rank_vectorized (fused matrix ops)...")
    mean2, std2, results2 = benchmark_method(
        bm25, queries, args.top_k, "vectorized", args.num_runs
    )
    print(f"  Time: {mean2:.3f}s ± {std2:.3f}s")
    print(f"  Throughput: {num_queries / mean2:.1f} queries/sec")

    # Verify correctness
    print(f"\nVerifying correctness...")
    is_correct, message = verify_correctness(results1, results2, args.top_k)
    if is_correct:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    speedup = mean1 / mean2 if mean2 > 0 else float("inf")
    if speedup > 1:
        print(f"  Speedup: {speedup:.2f}x faster")
    else:
        print(f"  Slowdown: {1/speedup:.2f}x slower")
    print(f"  Correctness: {'PASS' if is_correct else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
