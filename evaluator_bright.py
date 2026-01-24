"""
Evaluator for BM25 variants on BRIGHT.

Implements the evaluate(program_path) entrypoint expected by OpenEvolve, using
all metrics from ranking_evolved.metrics (precision/recall@k, AP/MAP, NDCG, and MRR).

Supports:
- Optional query subsampling via --sample-queries flag
- Domain selection via --domain flag (single domain or "all")
- Tokenizer selection via --tokenizer flag (simple or lucene)
- Query mode selection via --query-mode flag (unique, sum_all, saturated)
- k3 parameter for saturated query mode via --k3 flag

For OpenEvolve: The default evaluate() function is called with just program_path.
"""

import argparse
import importlib.util
import json
import os
import random
from collections.abc import Callable
from functools import cache

import numpy as np
from tqdm import tqdm

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

# Default evaluation settings (can be overridden via env vars for OpenEvolve)
DEFAULT_DOMAIN = os.environ.get("BRIGHT_DOMAIN", "biology")
DEFAULT_SAMPLE_QUERIES = int(os.environ.get("BRIGHT_SAMPLE_QUERIES", "0")) or None
DEFAULT_SEED = int(os.environ.get("BRIGHT_SEED", "42"))
DEFAULT_K = int(os.environ.get("BRIGHT_K", "10"))
DEFAULT_TOKENIZER = os.environ.get("BRIGHT_TOKENIZER", "simple")  # simple or lucene
DEFAULT_QUERY_MODE = os.environ.get("BRIGHT_QUERY_MODE", "unique")  # unique, sum_all, saturated
DEFAULT_K3 = float(os.environ.get("BRIGHT_K3", "2.0"))


def _load_candidate(
    program_path: str,
) -> tuple[type, type, Callable[[str], list[str]], type | None, type | None]:
    """
    Dynamically load a BM25 implementation from a file path.

    Returns:
        Tuple of (BM25, Corpus, tokenize, LuceneTokenizer or None, BM25Config or None)
    """
    spec = importlib.util.spec_from_file_location("candidate_bm25", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Required exports
    if not hasattr(module, "BM25"):
        raise AttributeError("Candidate module must define a BM25 class.")
    if not hasattr(module, "tokenize"):
        raise AttributeError(
            "Candidate module must define a tokenize(text: str) -> list[str] function."
        )
    if not hasattr(module, "Corpus"):
        raise AttributeError("Candidate module must define a Corpus class.")

    # Optional exports (for advanced features)
    lucene_tokenizer = getattr(module, "LuceneTokenizer", None)
    bm25_config = getattr(module, "BM25Config", None)
    bm25_unified = getattr(module, "BM25Unified", None)

    return module.BM25, module.Corpus, module.tokenize, lucene_tokenizer, bm25_config, bm25_unified


@cache
def _bright_raw(domain: str):
    """Load raw BRIGHT documents and examples for a given domain."""
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    return documents, examples


def evaluate(program_path: str, k: int = DEFAULT_K) -> dict[str, float]:
    """
    Evaluate a BM25 implementation against BRIGHT.

    This is the main entrypoint for OpenEvolve. Settings can be configured via
    environment variables:
    - BRIGHT_DOMAIN: Domain to evaluate (default: biology)
    - BRIGHT_SAMPLE_QUERIES: Number of queries to sample (default: 0 = all)
    - BRIGHT_SEED: Random seed (default: 42)
    - BRIGHT_K: Cutoff for @k metrics (default: 10)
    - BRIGHT_TOKENIZER: Tokenizer to use (simple or lucene, default: simple)
    - BRIGHT_QUERY_MODE: Query term mode (unique, sum_all, saturated, default: unique)
    - BRIGHT_K3: k3 parameter for saturated mode (default: 2.0)

    Args:
        program_path: Path to the BM25 implementation file.
        k: Cutoff for @k metrics.

    Returns:
        Dictionary with evaluation metrics. On error, returns combined_score=0.0 and error=1.0.
    """
    try:
        return evaluate_with_options(
            program_path,
            k=k,
            sample_queries=DEFAULT_SAMPLE_QUERIES,
            seed=DEFAULT_SEED,
            domain=DEFAULT_DOMAIN,
            tokenizer=DEFAULT_TOKENIZER,
            query_mode=DEFAULT_QUERY_MODE,
            k3=DEFAULT_K3,
        )
    except Exception as e:
        # Return error dict so OpenEvolve knows this candidate failed
        return {
            "combined_score": 0.0,
            "ndcg_at_k": 0.0,
            "error": 1.0,
            "error_message": str(e),
        }


def evaluate_with_options(
    program_path: str,
    k: int = 10,
    sample_queries: int | None = None,
    seed: int = 42,
    domain: str = "biology",
    tokenizer: str = "simple",
    query_mode: str = "unique",
    k3: float = 2.0,
) -> dict[str, float]:
    """
    Evaluate a BM25 implementation with full configuration options.

    Args:
        program_path: Path to the BM25 implementation file.
        k: Cutoff for @k metrics.
        sample_queries: If set, randomly sample this many queries.
        seed: Seed for reproducible sampling.
        domain: BRIGHT split name or "all" to aggregate across splits.
        tokenizer: "simple" or "lucene".
        query_mode: "unique", "sum_all", or "saturated".
        k3: k3 parameter for saturated query mode.

    Returns:
        Dictionary with evaluation metrics.
    """
    BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls, BM25ConfigCls, BM25UnifiedCls = (
        _load_candidate(program_path)
    )

    # Select tokenizer
    if tokenizer == "lucene" and LuceneTokenizerCls is not None:
        tokenize_fn = LuceneTokenizerCls()
    # else use simple tokenize_fn from module

    def _eval_single(split: str) -> dict[str, float]:
        print(f"Loading {domain} dataset...")
        documents, examples = _bright_raw(split)

        # Tokenize documents
        doc_tokens = [
            tokenize_fn(doc["content"])
            for doc in tqdm(documents, desc=f"Tokenizing {domain}", unit="doc")
        ]
        doc_ids = [doc["id"] for doc in documents]

        # Build corpus index
        print(f"Building {domain} index...")
        corpus = CorpusCls(doc_tokens, ids=doc_ids)
        # Force computation of cached properties before query loop
        _ = corpus.vocabulary_size
        _ = corpus.idf_array
        _ = corpus.term_doc_matrix
        print(f"Index built: {corpus.vocabulary_size:,} terms, {len(corpus):,} docs")

        raw_queries = [example["query"] for example in examples]
        gold_id_lists = [example["gold_ids"] for example in examples]

        if sample_queries is not None and sample_queries < len(raw_queries):
            rng = random.Random(seed)
            indices = rng.sample(range(len(raw_queries)), sample_queries)
            raw_queries = [raw_queries[i] for i in indices]
            gold_id_lists = [gold_id_lists[i] for i in indices]

        gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

        # Create BM25 scorer
        # Try to use BM25Unified with config if available and query_mode != unique
        if BM25UnifiedCls is not None and BM25ConfigCls is not None and query_mode != "unique":
            config = BM25ConfigCls(
                idf="evolved",
                tf="evolved",
                query_mode=query_mode,
                k1=0.9,
                b=0.4,
                k3=k3,
            )
            bm25 = BM25UnifiedCls(corpus, config)
        else:
            # Use base BM25 class (for evolution, this is the target)
            bm25 = BM25Impl(corpus)

        all_relevant = []
        all_retrieved = []
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rr_scores = []
        ap_scores = []

        for raw_query, gold in tqdm(
            zip(raw_queries, gold_indices, strict=False),
            total=len(raw_queries),
            desc=f"Evaluating {domain}",
            unit="query",
        ):
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
            "documents": len(corpus),
            "tokenizer": tokenizer,
            "query_mode": query_mode,
        }

        # Combined score for optimization (weights can be tuned)
        metrics["combined_score"] = float(
            0.4 * metrics["ndcg_at_k"]
            + 0.3 * metrics["mean_average_precision"]
            + 0.2 * metrics["mean_reciprocal_rank"]
            + 0.05 * metrics["precision_at_k"]
            + 0.05 * metrics["recall_at_k"]
        )
        metrics["error"] = 0.0
        return metrics

    if domain == "all":
        domain_results = {}
        macro_accumulators = {
            "precision_at_k": [],
            "recall_at_k": [],
            "ndcg_at_k": [],
            "mean_average_precision": [],
            "mean_reciprocal_rank": [],
            "combined_score": [],
        }

        for split in BRIGHT_SPLITS:
            try:
                metrics = _eval_single(split)
                domain_results[split] = metrics
                if metrics.get("error", 1.0) == 0.0:
                    for key in macro_accumulators:
                        macro_accumulators[key].append(metrics[key])
            except Exception as e:
                domain_results[split] = {"error": 1.0, "message": str(e)}

        macro_metrics = {
            f"macro_{key}": float(np.mean(values)) if values else 0.0
            for key, values in macro_accumulators.items()
        }
        macro_metrics["domains_evaluated"] = sum(
            1 for d in domain_results.values() if d.get("error", 1.0) == 0.0
        )
        macro_metrics["combined_score"] = macro_metrics.get("macro_combined_score", 0.0)
        macro_metrics["ndcg_at_k"] = macro_metrics.get("macro_ndcg_at_k", 0.0)
        macro_metrics["domains"] = domain_results
        macro_metrics["error"] = 0.0 if macro_metrics["domains_evaluated"] > 0 else 1.0
        return macro_metrics

    return _eval_single(domain)


# Legacy function for backwards compatibility
def evaluate_with_sampling(
    program_path: str,
    k: int = 10,
    sample_queries: int | None = None,
    seed: int | None = 42,
    domain: str = "biology",
) -> dict[str, float]:
    """Legacy function - use evaluate_with_options instead."""
    return evaluate_with_options(
        program_path,
        k=k,
        sample_queries=sample_queries,
        seed=seed or 42,
        domain=domain,
        tokenizer="simple",
        query_mode="unique",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 on BRIGHT dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation on biology domain
  python evaluator_bright.py src/ranking_evolved/bm25.py

  # Use Lucene tokenizer with saturated query mode
  python evaluator_bright.py src/ranking_evolved/bm25.py --tokenizer lucene --query-mode saturated

  # Evaluate across all domains with sampling
  python evaluator_bright.py src/ranking_evolved/bm25.py --domain all --sample-queries 20

  # Fast iteration during development
  python evaluator_bright.py src/ranking_evolved/bm25.py --sample-queries 10 --domain biology
""",
    )
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics (default: 10).")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Randomly sample this many queries (default: 0 = use all).",
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
        help="BRIGHT split to evaluate (e.g., biology). Use 'all' for all splits.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["simple", "lucene"],
        default="simple",
        help="Tokenizer to use (default: simple).",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        choices=["unique", "sum_all", "saturated"],
        default="unique",
        help="Query term handling mode (default: unique).",
    )
    parser.add_argument(
        "--k3",
        type=float,
        default=2.0,
        help="k3 parameter for saturated query mode (default: 2.0).",
    )
    args = parser.parse_args()

    sample_queries = args.sample_queries if args.sample_queries > 0 else None

    results = evaluate_with_options(
        args.program_path,
        k=args.k,
        sample_queries=sample_queries,
        seed=args.seed,
        domain=args.domain,
        tokenizer=args.tokenizer,
        query_mode=args.query_mode,
        k3=args.k3,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
