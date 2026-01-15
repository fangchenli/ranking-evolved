"""
Evaluator for BM25 variants on BRIGHT.

Implements the evaluate(program_path) entrypoint expected by OpenEvolve, using
all metrics from ranking_evolved.metrics (precision/recall@k, AP/MAP, NDCG, and MRR).
Supports optional query subsampling via BRIGHT_SAMPLE_QUERIES/BRIGHT_SAMPLE_SEED env vars,
and a --domain flag (or BRIGHT_DOMAIN env) to select a split or run across all splits.
"""

import argparse
import importlib.util
import json
import random
from collections.abc import Callable
from functools import cache

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

BRIGHT_SPLITS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]


def _load_candidate(
    program_path: str,
) -> tuple[type, Callable[[str], list[str]], type]:
    """Dynamically load a BM25 implementation, tokenizer, and Corpus from a file path."""
    spec = importlib.util.spec_from_file_location("candidate_bm25", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "BM25"):
        raise AttributeError("Candidate module must define a BM25 class.")
    if not hasattr(module, "tokenize"):
        raise AttributeError(
            "Candidate module must define a tokenize(text: str) -> list[str] function."
        )
    if not hasattr(module, "Corpus"):
        raise AttributeError("Candidate module must define a Corpus class.")

    return module.BM25, module.tokenize, module.Corpus


@cache
def _bright_raw(domain: str):
    """Load raw BRIGHT documents and examples for a given domain."""
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    return documents, examples


def evaluate(program_path: str, k: int = 10) -> dict[str, float]:
    """Evaluate a BM25 implementation against BRIGHT biology with multiple metrics."""
    return evaluate_with_sampling(program_path, k=k, sample_queries=None, seed=42, domain="biology")


def evaluate_with_sampling(
    program_path: str,
    k: int = 10,
    sample_queries: int | None = None,
    seed: int | None = 42,
    domain: str = "biology",
) -> dict[str, float]:
    """
    Evaluate a BM25 implementation with optional query subsampling for speed and domain selection.

    Args:
        program_path: Path to the BM25 implementation.
        k: Cutoff for @k metrics.
        sample_queries: If set, randomly sample this many queries (with seed) for evaluation.
        seed: Seed for reproducible sampling.
        domain: BRIGHT split name (e.g., biology) or "all" to aggregate across splits.
    """
    BM25Impl, tokenize_fn, CorpusCls = _load_candidate(program_path)

    def _eval_single(split: str) -> dict[str, float]:
        documents, examples = _bright_raw(split)
        corpus = CorpusCls.from_huggingface_dataset(documents)
        raw_queries = [example["query"] for example in examples]
        gold_id_lists = [example["gold_ids"] for example in examples]

        if sample_queries is not None and sample_queries < len(raw_queries):
            rng = random.Random(seed)
            indices = rng.sample(range(len(raw_queries)), sample_queries)
            raw_queries = [raw_queries[i] for i in indices]
            gold_id_lists = [gold_id_lists[i] for i in indices]

        gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

        bm25 = BM25Impl(corpus)

        all_relevant = []
        all_retrieved = []
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rr_scores = []
        ap_scores = []

        for raw_query, gold in zip(raw_queries, gold_indices, strict=False):
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

    if domain == "all":
        domain_results = {}
        combined_scores = []
        macro_accumulators = {
            "precision_at_k": [],
            "recall_at_k": [],
            "ndcg_at_k": [],
            "mean_average_precision": [],
            "mean_reciprocal_rank": [],
            "combined_score": [],
        }

        for split in BRIGHT_SPLITS:
            metrics = _eval_single(split)
            domain_results[split] = metrics
            if metrics.get("error", 1.0) == 0.0:
                for key in macro_accumulators:
                    macro_accumulators[key].append(metrics[key])

        macro_metrics = {
            f"macro_{key}": float(np.mean(values)) if values else 0.0
            for key, values in macro_accumulators.items()
        }
        macro_metrics["domains_evaluated"] = len(domain_results)
        macro_metrics["combined_score"] = macro_metrics.get("macro_combined_score", 0.0)
        macro_metrics["domains"] = domain_results
        macro_metrics["error"] = 0.0
        return macro_metrics

    return _eval_single(domain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 on BRIGHT biology.")
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics.")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Randomly sample this many queries for speed (default: 0; use 0 or omit to disable sampling).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="biology",
        help="BRIGHT split to evaluate (e.g., biology). Use 'all' to evaluate every split.",
    )
    args = parser.parse_args()

    sample_queries = (
        args.sample_queries if args.sample_queries and args.sample_queries > 0 else None
    )
    results = evaluate_with_sampling(
        args.program_path,
        k=args.k,
        sample_queries=sample_queries,
        seed=args.seed,
        domain=args.domain,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
