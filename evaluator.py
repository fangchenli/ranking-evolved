"""
Unified evaluator for multiple IR benchmarks.

Supports:
- BRIGHT (12 domains)
- BEIR (17 datasets)
- DL19/DL20 (future)

For OpenEvolve: Configure via environment variables:
    EVAL_BRIGHT_DOMAINS=biology,earth_science  # Comma-separated, or "all"
    EVAL_BEIR_DATASETS=scifact,nfcorpus        # Comma-separated, or "all"
    EVAL_SAMPLE_QUERIES=20                      # Sample N queries (0 = all)
    EVAL_TOKENIZER=lucene                       # simple or lucene
    EVAL_K=10                                   # Cutoff for @k metrics

Run with:
    uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
        --bright biology,earth_science \
        --beir scifact,nfcorpus
"""

import argparse
import json
import os

import numpy as np

from evaluator_beir import BEIR_DATASETS
from evaluator_beir import evaluate_with_options as _evaluate_beir
from evaluator_bright import BRIGHT_SPLITS
from evaluator_bright import evaluate_with_options as _evaluate_bright

# Default evaluation settings (can be overridden via env vars for OpenEvolve)
DEFAULT_BRIGHT_DOMAINS = os.environ.get("EVAL_BRIGHT_DOMAINS", "")
DEFAULT_BEIR_DATASETS = os.environ.get("EVAL_BEIR_DATASETS", "")
DEFAULT_SAMPLE_QUERIES = int(os.environ.get("EVAL_SAMPLE_QUERIES", "0")) or None
DEFAULT_SEED = int(os.environ.get("EVAL_SEED", "42"))
DEFAULT_K = int(os.environ.get("EVAL_K", "10"))
DEFAULT_TOKENIZER = os.environ.get("EVAL_TOKENIZER", "lucene")


def _parse_list(value: str, all_values: list[str]) -> list[str]:
    """Parse comma-separated list or 'all'."""
    if not value:
        return []
    if value.lower() == "all":
        return all_values
    return [v.strip() for v in value.split(",") if v.strip()]


def evaluate(program_path: str, k: int = DEFAULT_K) -> dict[str, float]:
    """
    Evaluate a BM25 implementation against multiple benchmarks.

    This is the main entrypoint for OpenEvolve. Settings are configured via
    environment variables (see module docstring).

    Args:
        program_path: Path to the BM25 implementation file.
        k: Cutoff for @k metrics.

    Returns:
        Dictionary with evaluation metrics. On error, returns combined_score=0.0 and error=1.0.
    """
    try:
        bright_domains = _parse_list(DEFAULT_BRIGHT_DOMAINS, BRIGHT_SPLITS)
        beir_datasets = _parse_list(DEFAULT_BEIR_DATASETS, BEIR_DATASETS)

        return evaluate_with_options(
            program_path,
            bright_domains=bright_domains if bright_domains else None,
            beir_datasets=beir_datasets if beir_datasets else None,
            sample_queries=DEFAULT_SAMPLE_QUERIES,
            seed=DEFAULT_SEED,
            tokenizer=DEFAULT_TOKENIZER,
            k=k,
        )
    except Exception as e:
        return {
            "combined_score": 0.0,
            "ndcg_at_k": 0.0,
            "error": 1.0,
            "error_message": str(e),
        }


def evaluate_with_options(
    program_path: str,
    bright_domains: list[str] | None = None,
    beir_datasets: list[str] | None = None,
    sample_queries: int | None = None,
    seed: int = 42,
    tokenizer: str = "lucene",
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate a BM25 implementation with full configuration options.

    Args:
        program_path: Path to the BM25 implementation file.
        bright_domains: List of BRIGHT domains to evaluate (None = skip BRIGHT).
        beir_datasets: List of BEIR datasets to evaluate (None = skip BEIR).
        sample_queries: If set, randomly sample this many queries per dataset.
        seed: Seed for reproducible sampling.
        tokenizer: "simple" or "lucene".
        k: Cutoff for @k metrics.

    Returns:
        Dictionary with evaluation metrics from all benchmarks.
    """
    results: dict = {
        "error": 0.0,
        "k": k,
        "tokenizer": tokenizer,
    }

    all_ndcg_scores = []
    total_datasets = 0

    # Evaluate BRIGHT domains
    if bright_domains:
        bright_results = {}
        bright_ndcg_scores = []

        for domain in bright_domains:
            if domain not in BRIGHT_SPLITS:
                print(f"Warning: Unknown BRIGHT domain '{domain}', skipping.")
                continue

            try:
                print(f"Evaluating BRIGHT/{domain}...")
                metrics = _evaluate_bright(
                    program_path,
                    k=k,
                    sample_queries=sample_queries,
                    seed=seed,
                    domain=domain,
                    tokenizer=tokenizer,
                )
                bright_results[domain] = metrics
                if metrics.get("error", 1.0) == 0.0:
                    bright_ndcg_scores.append(metrics["ndcg_at_k"])
                    all_ndcg_scores.append(metrics["ndcg_at_k"])
                    total_datasets += 1
            except Exception as e:
                print(f"  Error: {e}")
                bright_results[domain] = {"error": 1.0, "message": str(e)}

        results["bright"] = {
            "domains": bright_results,
            "macro_ndcg_at_k": float(np.mean(bright_ndcg_scores)) if bright_ndcg_scores else 0.0,
            "domains_evaluated": len(bright_ndcg_scores),
        }

    # Evaluate BEIR datasets
    if beir_datasets:
        beir_results = {}
        beir_ndcg_scores = []

        for dataset in beir_datasets:
            if dataset not in BEIR_DATASETS:
                print(f"Warning: Unknown BEIR dataset '{dataset}', skipping.")
                continue

            try:
                print(f"Evaluating BEIR/{dataset}...")
                metrics = _evaluate_beir(
                    program_path,
                    k=k,
                    sample_queries=sample_queries,
                    seed=seed,
                    dataset=dataset,
                    tokenizer=tokenizer,
                )
                beir_results[dataset] = metrics
                if metrics.get("error", 1.0) == 0.0:
                    beir_ndcg_scores.append(metrics["ndcg_at_k"])
                    all_ndcg_scores.append(metrics["ndcg_at_k"])
                    total_datasets += 1
            except Exception as e:
                print(f"  Error: {e}")
                beir_results[dataset] = {"error": 1.0, "message": str(e)}

        results["beir"] = {
            "datasets": beir_results,
            "macro_ndcg_at_k": float(np.mean(beir_ndcg_scores)) if beir_ndcg_scores else 0.0,
            "datasets_evaluated": len(beir_ndcg_scores),
        }

    # Compute combined score (weighted average of nDCG across all datasets)
    if all_ndcg_scores:
        results["ndcg_at_k"] = float(np.mean(all_ndcg_scores))
        results["combined_score"] = results["ndcg_at_k"]
        results["total_datasets"] = total_datasets
    else:
        results["ndcg_at_k"] = 0.0
        results["combined_score"] = 0.0
        results["total_datasets"] = 0
        results["error"] = 1.0

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluator for BRIGHT and BEIR benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python evaluator.py src/ranking_evolved/bm25_classic.py --bright biology
  python evaluator.py src/ranking_evolved/bm25_classic.py --beir scifact

  # Multiple datasets (comma-separated)
  python evaluator.py src/ranking_evolved/bm25_classic.py --bright biology,earth_science
  python evaluator.py src/ranking_evolved/bm25_classic.py --beir scifact,nfcorpus,fiqa

  # Mix of benchmarks
  python evaluator.py src/ranking_evolved/bm25_classic.py \\
      --bright biology,earth_science \\
      --beir scifact,nfcorpus

  # All datasets in a benchmark
  python evaluator.py src/ranking_evolved/bm25_classic.py --bright all
  python evaluator.py src/ranking_evolved/bm25_classic.py --beir all

  # Fast iteration with sampling
  python evaluator.py src/ranking_evolved/bm25_classic.py \\
      --bright biology --beir scifact --sample-queries 20

Available BRIGHT domains:
  biology, earth_science, economics, psychology, robotics, stackoverflow,
  sustainable_living, pony, leetcode, aops, theoremqa_theorems, theoremqa_questions

Available BEIR datasets:
  scifact, nfcorpus, arguana, scidocs, fiqa, webis-touche2020, trec-covid,
  quora, cqadupstack, robust04, trec-news, hotpotqa, nq, fever, climate-fever,
  dbpedia-entity, bioasq
""",
    )
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument(
        "--bright",
        type=str,
        default="",
        help="BRIGHT domains (comma-separated, or 'all').",
    )
    parser.add_argument(
        "--beir",
        type=str,
        default="",
        help="BEIR datasets (comma-separated, or 'all').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for @k metrics (default: 10).",
    )
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
        default="lucene",
        help="Tokenizer to use (default: lucene).",
    )
    args = parser.parse_args()

    # Parse domain/dataset lists
    bright_domains = _parse_list(args.bright, BRIGHT_SPLITS)
    beir_datasets = _parse_list(args.beir, BEIR_DATASETS)

    if not bright_domains and not beir_datasets:
        parser.error("At least one of --bright or --beir must be specified.")

    sample_queries = args.sample_queries if args.sample_queries > 0 else None

    results = evaluate_with_options(
        args.program_path,
        bright_domains=bright_domains if bright_domains else None,
        beir_datasets=beir_datasets if beir_datasets else None,
        sample_queries=sample_queries,
        seed=args.seed,
        tokenizer=args.tokenizer,
        k=args.k,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
