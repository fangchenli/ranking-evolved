"""
Unified Evaluator for BM25 variants on IR benchmarks.

Evaluates on ALL benchmarks for comprehensive assessment:
- BRIGHT: 12 reasoning-intensive retrieval domains
- BEIR: 17 heterogeneous IR datasets
- TREC DL: 2 Deep Learning Track datasets (DL19, DL20)

Total: 31 datasets, reporting 31 nDCG@10 + 31 Recall@100 scores.

Metrics:
- nDCG@10: Primary ranking quality metric
- Recall@100: Coverage metric (standardized across all datasets)

Combined score for optimization = (avg_nDCG@10 + avg_Recall@100) / 2

Run with:
    python evaluator.py src/ranking_evolved/bm25_classic.py
    python evaluator.py src/ranking_evolved/bm25_classic.py --sample-queries 20

For OpenEvolve, configure via environment variables:
    EVAL_SAMPLE_QUERIES=20 (for faster iteration)
    EVAL_TOKENIZER=simple (or lucene)
"""

import argparse
import importlib.util
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ranking_evolved.datasets import (
    BEIR_DATASETS,
    BRIGHT_SPLITS,
    TREC_DL_DATASETS,
    BEIRLoader,
    BRIGHTLoader,
    EvalDataset,
    TRECDLLoader,
)
from ranking_evolved.metrics import ndcg_at_k, recall_at_k

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_QUERIES = int(os.environ.get("EVAL_SAMPLE_QUERIES", "0")) or None
DEFAULT_SEED = int(os.environ.get("EVAL_SEED", "42"))
DEFAULT_TOKENIZER = os.environ.get(
    "EVAL_TOKENIZER", "lucene"
)  # Lucene to match BRIGHT/BEIR baselines

# Standard cutoffs
NDCG_K = 10
RECALL_K = 100  # Standardized recall cutoff for all datasets


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    sample_queries: int | None = DEFAULT_SAMPLE_QUERIES
    seed: int = DEFAULT_SEED
    tokenizer: str = DEFAULT_TOKENIZER
    beir_data_dir: str = "datasets/beir"
    trec_dl_data_dir: str = "datasets/trec_dl"
    # Which benchmarks to include
    include_bright: bool = True
    include_beir: bool = True
    include_trec_dl: bool = True
    # Subset of datasets (None = all)
    bright_datasets: list[str] | None = None
    beir_datasets: list[str] | None = None
    trec_dl_datasets: list[str] | None = None


# =============================================================================
# Candidate Loading
# =============================================================================


def load_candidate(
    program_path: str,
) -> tuple[type, type, Callable[[str], list[str]], type | None]:
    """
    Dynamically load a BM25 implementation from a file path.

    Returns:
        Tuple of (BM25, Corpus, tokenize, LuceneTokenizer or None)
    """
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

    lucene_tokenizer = getattr(module, "LuceneTokenizer", None)

    return module.BM25, module.Corpus, module.tokenize, lucene_tokenizer


# =============================================================================
# Dataset Loaders
# =============================================================================


def get_all_datasets(config: EvalConfig) -> list[tuple[str, str, Any]]:
    """
    Get all datasets to evaluate on.

    Returns:
        List of (benchmark_name, dataset_name, loader) tuples
    """
    datasets = []

    if config.include_bright:
        bright_loader = BRIGHTLoader()
        ds_list = config.bright_datasets or BRIGHT_SPLITS
        for ds in ds_list:
            datasets.append(("bright", ds, bright_loader))

    if config.include_beir:
        beir_loader = BEIRLoader(data_dir=config.beir_data_dir)
        ds_list = config.beir_datasets or BEIR_DATASETS
        for ds in ds_list:
            datasets.append(("beir", ds, beir_loader))

    if config.include_trec_dl:
        trec_loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
        ds_list = config.trec_dl_datasets or TREC_DL_DATASETS
        for ds in ds_list:
            datasets.append(("trec_dl", ds, trec_loader))

    return datasets


def list_datasets(benchmark: str | None = None) -> list[str]:
    """List available datasets for a benchmark (or all if None)."""
    if benchmark == "bright":
        return BRIGHT_SPLITS
    elif benchmark == "beir":
        return BEIR_DATASETS
    elif benchmark == "trec_dl":
        return TREC_DL_DATASETS
    elif benchmark is None:
        return BRIGHT_SPLITS + BEIR_DATASETS + TREC_DL_DATASETS
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# =============================================================================
# Evaluation Logic
# =============================================================================


def compute_metrics(
    all_relevant: list[np.ndarray],
    all_retrieved: list[np.ndarray],
) -> dict[str, float]:
    """
    Compute nDCG@10 and Recall@100.

    Args:
        all_relevant: List of relevant doc indices per query
        all_retrieved: List of retrieved doc indices per query

    Returns:
        Dictionary with ndcg@10 and recall@100
    """
    ndcg_scores = [ndcg_at_k(rel, ret, NDCG_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)]
    recall_scores = [
        recall_at_k(rel, ret, RECALL_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)
    ]

    return {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@100": float(np.mean(recall_scores)),
    }


def evaluate_on_dataset(
    bm25,
    dataset: EvalDataset,
    tokenize_fn: Callable[[str], list[str]],
    id_to_idx: dict[str, int],
    sample_queries: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """
    Evaluate BM25 on a single dataset.

    Returns:
        Dictionary with ndcg@10 and recall@100
    """
    query_ids = dataset.query_ids
    queries = dataset.queries

    # Sample queries if requested
    if sample_queries and sample_queries < len(queries):
        rng = random.Random(seed)
        indices = rng.sample(range(len(queries)), sample_queries)
        query_ids = [query_ids[i] for i in indices]
        queries = [queries[i] for i in indices]

    all_relevant = []
    all_retrieved = []

    for qid, query_text in zip(query_ids, queries, strict=False):
        query_tokens = tokenize_fn(query_text)
        ranked_indices, _ = bm25.rank(query_tokens)

        relevant_doc_ids = dataset.get_relevant_docs(qid)
        if not relevant_doc_ids:
            continue

        relevant_indices = [id_to_idx[doc_id] for doc_id in relevant_doc_ids if doc_id in id_to_idx]

        if not relevant_indices:
            continue

        all_relevant.append(np.array(relevant_indices, dtype=int))
        all_retrieved.append(np.array(ranked_indices, dtype=int))

    if not all_relevant:
        return {"ndcg@10": 0.0, "recall@100": 0.0, "queries_evaluated": 0}

    metrics = compute_metrics(all_relevant, all_retrieved)
    metrics["queries_evaluated"] = len(all_relevant)

    return metrics


# =============================================================================
# Main Evaluation Functions
# =============================================================================


def evaluate(program_path: str) -> dict[str, float]:
    """
    Evaluate a BM25 implementation on ALL benchmarks - OpenEvolve entrypoint.

    Evaluates on 31 datasets (12 BRIGHT + 17 BEIR + 2 TREC DL).

    Settings via environment variables:
    - EVAL_SAMPLE_QUERIES: Number of queries to sample per dataset (0 = all)
    - EVAL_SEED: Random seed
    - EVAL_TOKENIZER: simple or lucene

    Returns:
        Dictionary with per-dataset metrics and combined_score.
        Format: {
            "combined_score": float,  # (avg_ndcg@10 + avg_recall@100) / 2
            "avg_ndcg@10": float,
            "avg_recall@100": float,
            "bright_biology_ndcg@10": float,
            "bright_biology_recall@100": float,
            ... (all 31 datasets)
            "error": 0.0 or 1.0
        }
    """
    try:
        config = EvalConfig()
        return evaluate_with_config(program_path, config)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "avg_ndcg@10": 0.0,
            "avg_recall@100": 0.0,
            "error": 1.0,
            "error_message": str(e),
        }


def evaluate_with_config(
    program_path: str,
    config: EvalConfig,
) -> dict[str, float]:
    """
    Evaluate with explicit configuration on all benchmarks.

    Args:
        program_path: Path to BM25 implementation
        config: Evaluation configuration

    Returns:
        Dictionary with all metrics
    """
    # Load candidate
    BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)

    if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
        tokenize_fn = LuceneTokenizerCls()

    # Get all datasets to evaluate
    all_datasets = get_all_datasets(config)

    # Collect all metrics
    results: dict[str, float] = {}
    all_ndcg = []
    all_recall = []

    datasets_evaluated = 0
    datasets_failed = 0

    for benchmark, ds_name, loader in all_datasets:
        metric_prefix = f"{benchmark}_{ds_name}"

        try:
            # Load dataset
            dataset = loader.load(ds_name)

            # Tokenize corpus
            doc_tokens = [tokenize_fn(text) for text in dataset.corpus]
            corpus = CorpusCls(doc_tokens, ids=dataset.corpus_ids)

            # Build index mapping
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dataset.corpus_ids)}

            # Create BM25 scorer
            bm25 = BM25Impl(corpus)

            # Evaluate
            metrics = evaluate_on_dataset(
                bm25=bm25,
                dataset=dataset,
                tokenize_fn=tokenize_fn,
                id_to_idx=id_to_idx,
                sample_queries=config.sample_queries,
                seed=config.seed,
            )

            # Store per-dataset metrics
            ndcg = metrics["ndcg@10"]
            recall = metrics["recall@100"]

            results[f"{metric_prefix}_ndcg@10"] = ndcg
            results[f"{metric_prefix}_recall@100"] = recall

            all_ndcg.append(ndcg)
            all_recall.append(recall)
            datasets_evaluated += 1

        except Exception as e:
            # Mark dataset as failed
            results[f"{metric_prefix}_ndcg@10"] = 0.0
            results[f"{metric_prefix}_recall@100"] = 0.0
            results[f"{metric_prefix}_error"] = str(e)
            datasets_failed += 1

    # Compute aggregate metrics
    avg_ndcg = float(np.mean(all_ndcg)) if all_ndcg else 0.0
    avg_recall = float(np.mean(all_recall)) if all_recall else 0.0

    # Combined score: average of the two averages
    combined_score = (avg_ndcg + avg_recall) / 2.0

    results["avg_ndcg@10"] = avg_ndcg
    results["avg_recall@100"] = avg_recall
    results["combined_score"] = combined_score

    # Metadata
    results["datasets_evaluated"] = float(datasets_evaluated)
    results["datasets_failed"] = float(datasets_failed)
    results["error"] = 0.0 if datasets_evaluated > 0 else 1.0

    return results


def evaluate_single_dataset(
    program_path: str,
    benchmark: str,
    dataset_name: str,
    config: EvalConfig | None = None,
) -> dict[str, float]:
    """
    Evaluate on a single dataset (for debugging/testing).

    Args:
        program_path: Path to BM25 implementation
        benchmark: 'bright', 'beir', or 'trec_dl'
        dataset_name: Dataset name within benchmark
        config: Optional configuration

    Returns:
        Dictionary with ndcg@10, recall@100, and combined_score
    """
    if config is None:
        config = EvalConfig()

    # Load candidate
    BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)
    if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
        tokenize_fn = LuceneTokenizerCls()

    # Get loader
    if benchmark == "bright":
        loader = BRIGHTLoader()
    elif benchmark == "beir":
        loader = BEIRLoader(data_dir=config.beir_data_dir)
    elif benchmark == "trec_dl":
        loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Load and evaluate
    dataset = loader.load(dataset_name)
    doc_tokens = [tokenize_fn(text) for text in dataset.corpus]
    corpus = CorpusCls(doc_tokens, ids=dataset.corpus_ids)
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dataset.corpus_ids)}
    bm25 = BM25Impl(corpus)

    metrics = evaluate_on_dataset(
        bm25=bm25,
        dataset=dataset,
        tokenize_fn=tokenize_fn,
        id_to_idx=id_to_idx,
        sample_queries=config.sample_queries,
        seed=config.seed,
    )

    metrics["combined_score"] = (metrics["ndcg@10"] + metrics["recall@100"]) / 2.0
    metrics["benchmark"] = benchmark
    metrics["dataset"] = dataset_name
    metrics["error"] = 0.0

    return metrics


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 on ALL IR benchmarks (31 datasets).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation on all 31 datasets
  python evaluator.py src/ranking_evolved/bm25_classic.py

  # Fast iteration with query sampling
  python evaluator.py src/ranking_evolved/bm25_classic.py --sample-queries 20

  # Evaluate single dataset (for debugging)
  python evaluator.py src/ranking_evolved/bm25_classic.py --single bright biology

  # Only BRIGHT datasets
  python evaluator.py src/ranking_evolved/bm25_classic.py --only-bright

Benchmarks included:
  BRIGHT   - 12 reasoning-intensive retrieval domains
  BEIR     - 17 heterogeneous IR datasets
  TREC DL  - 2 Deep Learning Track datasets (DL19, DL20)

Metrics reported:
  nDCG@10    - Normalized Discounted Cumulative Gain at 10
  Recall@100 - Recall at 100 (standardized across all datasets)

Combined score = (avg_nDCG@10 + avg_Recall@100) / 2
""",
    )
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Randomly sample this many queries per dataset (default: 0 = use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["simple", "lucene"],
        default="simple",
        help="Tokenizer to use (default: simple).",
    )
    parser.add_argument(
        "--single",
        nargs=2,
        metavar=("BENCHMARK", "DATASET"),
        help="Evaluate single dataset: --single bright biology",
    )
    parser.add_argument(
        "--only-bright",
        action="store_true",
        help="Only evaluate on BRIGHT datasets.",
    )
    parser.add_argument(
        "--only-beir",
        action="store_true",
        help="Only evaluate on BEIR datasets.",
    )
    parser.add_argument(
        "--only-trec-dl",
        action="store_true",
        help="Only evaluate on TREC DL datasets.",
    )
    parser.add_argument(
        "--beir-data-dir",
        type=str,
        default="datasets/beir",
        help="Directory for BEIR data.",
    )
    parser.add_argument(
        "--trec-dl-data-dir",
        type=str,
        default="datasets/trec_dl",
        help="Directory for TREC DL data.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress during evaluation.",
    )
    args = parser.parse_args()

    # Single dataset mode
    if args.single:
        benchmark, dataset = args.single
        config = EvalConfig(
            sample_queries=args.sample_queries if args.sample_queries > 0 else None,
            seed=args.seed,
            tokenizer=args.tokenizer,
            beir_data_dir=args.beir_data_dir,
            trec_dl_data_dir=args.trec_dl_data_dir,
        )
        results = evaluate_single_dataset(args.program_path, benchmark, dataset, config)
        print(json.dumps(results, indent=2))
        return

    # Full evaluation
    config = EvalConfig(
        sample_queries=args.sample_queries if args.sample_queries > 0 else None,
        seed=args.seed,
        tokenizer=args.tokenizer,
        beir_data_dir=args.beir_data_dir,
        trec_dl_data_dir=args.trec_dl_data_dir,
        include_bright=not (args.only_beir or args.only_trec_dl),
        include_beir=not (args.only_bright or args.only_trec_dl),
        include_trec_dl=not (args.only_bright or args.only_beir),
    )

    if args.verbose:
        print(f"Evaluating {args.program_path}")
        print(f"  BRIGHT: {config.include_bright}")
        print(f"  BEIR: {config.include_beir}")
        print(f"  TREC DL: {config.include_trec_dl}")
        print(f"  Sample queries: {config.sample_queries or 'all'}")
        print()

    results = evaluate_with_config(args.program_path, config)

    # Print summary
    if args.verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Combined Score: {results['combined_score']:.4f}")
        print(f"  avg_nDCG@10:    {results['avg_ndcg@10']:.4f}")
        print(f"  avg_Recall@100: {results['avg_recall@100']:.4f}")
        print(
            f"Datasets: {int(results['datasets_evaluated'])} evaluated, "
            f"{int(results['datasets_failed'])} failed"
        )
        print("=" * 60)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
