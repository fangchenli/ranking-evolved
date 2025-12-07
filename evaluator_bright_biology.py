"""
Evaluator for BM25 variants on the BRIGHT biology split.

Implements the evaluate(program_path) entrypoint expected by OpenEvolve, using
all metrics from ranking_evolved.metrics (precision/recall@k, AP/MAP, NDCG, and MRR).
"""

import argparse
import importlib.util
import json
import random
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from datasets import load_dataset

from ranking_evolved.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ranking_evolved.bm25 import Corpus


def _load_candidate(program_path: str) -> Tuple[type, Callable[[str], list[str]]]:
    """Dynamically load a BM25 implementation and tokenizer from a file path."""
    spec = importlib.util.spec_from_file_location("candidate_bm25", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "BM25"):
        raise AttributeError("Candidate module must define a BM25 class.")
    if not hasattr(module, "tokenize"):
        raise AttributeError("Candidate module must define a tokenize(text: str) -> list[str] function.")

    return module.BM25, module.tokenize


@lru_cache(maxsize=1)
def _bright_biology() -> tuple[Corpus, list[str], list[list[int]]]:
    """Load BRIGHT biology documents, queries, and gold doc IDs."""
    documents = load_dataset("xlangai/BRIGHT", "documents", split="biology")
    examples = load_dataset("xlangai/BRIGHT", "examples", split="biology")

    corpus = Corpus.from_huggingface_dataset(documents)
    queries = [example["query"] for example in examples]
    gold_ids = [example["gold_ids"] for example in examples]
    return corpus, queries, gold_ids


def evaluate(program_path: str, k: int = 10) -> dict[str, float]:
    """Evaluate a BM25 implementation against BRIGHT biology with multiple metrics."""
    return evaluate_with_sampling(program_path, k=k, sample_queries=None, seed=42)


def evaluate_with_sampling(
    program_path: str, k: int = 10, sample_queries: int | None = None, seed: int | None = 42
) -> dict[str, float]:
    """
    Evaluate a BM25 implementation with optional query subsampling for speed.

    Args:
        program_path: Path to the BM25 implementation.
        k: Cutoff for @k metrics.
        sample_queries: If set, randomly sample this many queries (with seed) for evaluation.
        seed: Seed for reproducible sampling.
    """
    corpus, raw_queries, gold_id_lists = _bright_biology()

    if sample_queries is not None and sample_queries < len(raw_queries):
        rng = random.Random(seed)
        indices = rng.sample(range(len(raw_queries)), sample_queries)
        raw_queries = [raw_queries[i] for i in indices]
        gold_id_lists = [gold_id_lists[i] for i in indices]

    BM25Impl, tokenize_fn = _load_candidate(program_path)
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    bm25 = BM25Impl(corpus)

    all_relevant = []
    all_retrieved = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    rr_scores = []
    ap_scores = []

    try:
        for raw_query, gold in zip(raw_queries, gold_indices):
            query_tokens = tokenize_fn(raw_query)
            ranked_indices, _ = bm25.rank(query_tokens)

            relevant = np.array(gold, dtype=int)
            retrieved = np.array(ranked_indices, dtype=int)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

            precision_scores.append(precision_at_k(relevant, retrieved, k))
            recall_scores.append(recall_at_k(relevant, retrieved, k))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))
            rr_scores.append(reciprocal_rank(relevant, retrieved))
            ap_scores.append(average_precision(relevant, retrieved))

        if not all_relevant:
            raise ValueError("No queries were evaluated.")

        metrics = {
            "precision_at_k": float(np.mean(precision_scores)),
            "recall_at_k": float(np.mean(recall_scores)),
            "ndcg_at_k": float(np.mean(ndcg_scores)),
            "reciprocal_rank": float(np.mean(rr_scores)),
            "mean_average_precision": mean_average_precision(all_relevant, all_retrieved),
            "mean_reciprocal_rank": mean_reciprocal_rank(all_relevant, all_retrieved),
            "k": k,
            "queries": len(all_relevant),
        }

        metrics["combined_score"] = float(
            np.mean(
                [
                    metrics["ndcg_at_k"],
                    metrics["mean_average_precision"],
                    metrics["mean_reciprocal_rank"],
                    metrics["precision_at_k"],
                    metrics["recall_at_k"],
                ]
            )
        )
        metrics["error"] = 0.0
        return metrics
    except Exception as exc:  # noqa: BLE001
        # Penalize failed evaluations but keep the run alive
        return {
            "combined_score": 0.0,
            "error": 1.0,
            "error_message": str(exc),
            "k": k,
            "queries": len(all_relevant),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 on BRIGHT biology.")
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics.")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=32,
        help="Randomly sample this many queries for speed (default: 32; use 0 or omit to disable sampling).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for query sampling (default: 42)."
    )
    args = parser.parse_args()

    sample_queries = args.sample_queries if args.sample_queries and args.sample_queries > 0 else None
    results = evaluate_with_sampling(
        args.program_path, k=args.k, sample_queries=sample_queries, seed=args.seed
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
