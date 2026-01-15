"""
BRIGHT Benchmark Runner for BM25 Variants.

This module provides a reusable benchmarking framework for evaluating BM25
implementations on the BRIGHT dataset. It supports multiple BM25 variants,
external library baselines, and comprehensive metric reporting.

Usage:
    python -m benchmarks.bright_benchmark --domains biology earth_science --k 10

    Or programmatically:
        benchmark = BrightBenchmark(domains=["biology"], k=10)
        benchmark.add_variant("classic", create_bm25_classic)
        results = benchmark.run()
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import (
    BM25,
    Corpus,
    create_bm25_atire,
    create_bm25_classic,
    create_bm25_lucene,
    create_bm25_plus,
    create_bm25_query_side,
    create_bm25l,
    tokenize,
)
from ranking_evolved.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

if TYPE_CHECKING:
    pass


# All available BRIGHT dataset splits
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


class BM25Factory(Protocol):
    """Protocol for BM25 factory functions."""

    def __call__(self, corpus: Corpus, k1: float, b: float) -> BM25:
        """Create a BM25 scorer for the given corpus."""
        ...


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    variant: str
    domain: str
    k: int
    k1: float
    b: float
    num_queries: int
    num_documents: int

    # Primary metrics
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    map: float
    mrr: float

    # Combined score (average of all metrics)
    combined_score: float

    # Timing
    total_time_sec: float
    avg_query_time_ms: float

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "variant": self.variant,
            "domain": self.domain,
            "k": self.k,
            "k1": self.k1,
            "b": self.b,
            "num_queries": self.num_queries,
            "num_documents": self.num_documents,
            "ndcg_at_k": self.ndcg_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "map": self.map,
            "mrr": self.mrr,
            "combined_score": self.combined_score,
            "total_time_sec": self.total_time_sec,
            "avg_query_time_ms": self.avg_query_time_ms,
            "timestamp": self.timestamp,
        }


@cache
def load_bright_data(domain: str) -> tuple:
    """
    Load BRIGHT dataset for a specific domain.

    Args:
        domain: BRIGHT split name (e.g., "biology").

    Returns:
        Tuple of (documents dataset, examples dataset).
    """
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    return documents, examples


class BrightBenchmark:
    """
    Reusable benchmark runner for BM25 variants on BRIGHT dataset.

    This class manages the benchmark lifecycle including:
    - Loading and caching dataset splits
    - Running multiple BM25 variants
    - Computing comprehensive metrics
    - Saving results in multiple formats

    Example:
        >>> benchmark = BrightBenchmark(domains=["biology"], k=10)
        >>> benchmark.add_variant("classic", create_bm25_classic)
        >>> benchmark.add_variant("lucene", create_bm25_lucene)
        >>> results = benchmark.run()
        >>> benchmark.save_results("benchmarks/results")
    """

    def __init__(
        self,
        domains: list[str] | None = None,
        k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize benchmark runner.

        Args:
            domains: List of BRIGHT splits to evaluate. Defaults to all splits.
            k: Cutoff for @k metrics (default: 10).
            k1: BM25 k1 parameter (default: 1.5).
            b: BM25 b parameter (default: 0.75).
        """
        self.domains = domains or BRIGHT_SPLITS
        self.k = k
        self.k1 = k1
        self.b = b

        self._variants: dict[str, BM25Factory] = {}
        self._results: list[BenchmarkResult] = []

    def add_variant(self, name: str, factory: BM25Factory) -> None:
        """
        Add a BM25 variant to benchmark.

        Args:
            name: Name for the variant (e.g., "classic", "lucene").
            factory: Factory function that creates BM25 scorer.
        """
        self._variants[name] = factory

    def add_default_variants(self) -> None:
        """Add all standard BM25 variants."""
        self.add_variant("evolved", lambda c, k1, b: BM25(c, k1=k1, b=b))
        self.add_variant("classic", create_bm25_classic)
        self.add_variant("lucene", create_bm25_lucene)
        self.add_variant("bm25l", create_bm25l)
        self.add_variant("bm25+", create_bm25_plus)
        self.add_variant("atire", create_bm25_atire)
        self.add_variant("query_side", create_bm25_query_side)

    def add_external_baseline(
        self,
        name: str,
        scorer_class: type,
    ) -> None:
        """
        Add an external library baseline.

        Args:
            name: Name for the baseline.
            scorer_class: Baseline class with from_corpus() and rank() methods.
        """

        # External baselines use a different interface
        def factory(corpus: Corpus, k1: float, b: float) -> BM25:
            return scorer_class.from_corpus(corpus, k1=k1, b=b)

        self._variants[name] = factory

    def _evaluate_variant(
        self,
        variant_name: str,
        domain: str,
    ) -> BenchmarkResult:
        """
        Evaluate a single variant on a single domain.

        Args:
            variant_name: Name of the variant.
            domain: BRIGHT split name.

        Returns:
            BenchmarkResult with all metrics.
        """
        # Load data
        documents, examples = load_bright_data(domain)

        # Build corpus
        corpus = Corpus.from_huggingface_dataset(documents)

        # Create scorer
        factory = self._variants[variant_name]
        bm25 = factory(corpus, self.k1, self.b)

        # Prepare queries and gold labels
        queries = [example["query"] for example in examples]
        gold_id_lists = [example["gold_ids"] for example in examples]
        gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

        # Run evaluation
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rr_scores = []
        ap_scores = []
        all_relevant = []
        all_retrieved = []

        start_time = time.perf_counter()

        for query_text, gold in zip(queries, gold_indices, strict=False):
            query_tokens = tokenize(query_text)
            ranked_indices, _ = bm25.rank(query_tokens)

            relevant = np.array(gold, dtype=np.int64)
            retrieved = np.array(ranked_indices, dtype=np.int64)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

            precision_scores.append(precision_at_k(relevant, retrieved, self.k))
            recall_scores.append(recall_at_k(relevant, retrieved, self.k))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, self.k))
            rr_scores.append(reciprocal_rank(relevant, retrieved))
            ap_scores.append(average_precision(relevant, retrieved))

        total_time = time.perf_counter() - start_time

        # Compute aggregate metrics
        ndcg = float(np.mean(ndcg_scores))
        precision = float(np.mean(precision_scores))
        recall = float(np.mean(recall_scores))
        map_score = mean_average_precision(all_relevant, all_retrieved)
        mrr = mean_reciprocal_rank(all_relevant, all_retrieved)

        combined = float(np.mean([ndcg, precision, recall, map_score, mrr]))

        return BenchmarkResult(
            variant=variant_name,
            domain=domain,
            k=self.k,
            k1=self.k1,
            b=self.b,
            num_queries=len(queries),
            num_documents=len(corpus),
            ndcg_at_k=ndcg,
            precision_at_k=precision,
            recall_at_k=recall,
            map=map_score,
            mrr=mrr,
            combined_score=combined,
            total_time_sec=total_time,
            avg_query_time_ms=(total_time / len(queries)) * 1000,
        )

    def run(self, verbose: bool = True) -> list[BenchmarkResult]:
        """
        Run benchmark for all variants and domains.

        Args:
            verbose: Print progress to stdout.

        Returns:
            List of BenchmarkResult objects.
        """
        if not self._variants:
            raise ValueError("No variants added. Call add_variant() first.")

        self._results = []
        total = len(self._variants) * len(self.domains)
        count = 0

        for variant_name in self._variants:
            for domain in self.domains:
                count += 1
                if verbose:
                    print(f"[{count}/{total}] Evaluating {variant_name} on {domain}...")

                result = self._evaluate_variant(variant_name, domain)
                self._results.append(result)

                if verbose:
                    print(
                        f"  NDCG@{self.k}: {result.ndcg_at_k:.4f}, "
                        f"MAP: {result.map:.4f}, "
                        f"MRR: {result.mrr:.4f}, "
                        f"Time: {result.total_time_sec:.2f}s"
                    )

        return self._results

    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame.

        Returns:
            DataFrame with all benchmark results.
        """
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self._results])

    def get_summary_table(self, metric: str = "ndcg_at_k") -> str:
        """
        Generate a markdown summary table for a specific metric.

        Args:
            metric: Metric to summarize (default: "ndcg_at_k").

        Returns:
            Markdown-formatted table string.
        """

        df = self.get_results_dataframe()
        pivot = df.pivot(index="domain", columns="variant", values=metric)

        # Add average row
        pivot.loc["**average**"] = pivot.mean()

        # Format as markdown
        lines = [f"## {metric} Results\n"]
        lines.append("| Domain | " + " | ".join(pivot.columns) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(pivot.columns)) + " |")

        for idx, row in pivot.iterrows():
            values = " | ".join([f"{v:.4f}" for v in row])
            lines.append(f"| {idx} | {values} |")

        return "\n".join(lines)

    def save_results(self, output_dir: str | Path) -> None:
        """
        Save benchmark results to disk.

        Creates:
        - results.json: Full results as JSON
        - summary.md: Markdown summary tables
        - results.csv: CSV for further analysis

        Args:
            output_dir: Directory to save results.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = output_dir / f"results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in self._results], f, indent=2)

        # Save CSV
        df = self.get_results_dataframe()
        csv_path = output_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Save markdown summary
        md_path = output_dir / f"summary_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write("# BRIGHT Benchmark Results\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Parameters: k={self.k}, k1={self.k1}, b={self.b}\n\n")

            for metric in ["ndcg_at_k", "map", "mrr", "precision_at_k", "recall_at_k"]:
                f.write(self.get_summary_table(metric))
                f.write("\n\n")

        print(f"Results saved to {output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {csv_path.name}")
        print(f"  - {md_path.name}")

    def compare(
        self,
        baseline: str = "classic",
        metrics: list[str] | None = None,
    ) -> str:
        """
        Compare variants against a baseline.

        Args:
            baseline: Name of baseline variant.
            metrics: List of metrics to compare (default: all).

        Returns:
            Markdown-formatted comparison table.
        """

        metrics = metrics or ["ndcg_at_k", "map", "mrr"]
        df = self.get_results_dataframe()

        lines = [f"## Comparison vs {baseline}\n"]

        for metric in metrics:
            pivot = df.pivot(index="domain", columns="variant", values=metric)

            if baseline not in pivot.columns:
                continue

            # Compute deltas
            for col in pivot.columns:
                if col != baseline:
                    pivot[f"{col}_delta"] = pivot[col] - pivot[baseline]

            lines.append(f"\n### {metric}\n")

            # Filter to just delta columns
            delta_cols = [c for c in pivot.columns if "_delta" in c]
            if delta_cols:
                delta_df = pivot[delta_cols]
                delta_df.columns = [c.replace("_delta", "") for c in delta_cols]

                lines.append("| Domain | " + " | ".join(delta_df.columns) + " |")
                lines.append("| --- | " + " | ".join(["---"] * len(delta_df.columns)) + " |")

                for idx, row in delta_df.iterrows():
                    values = " | ".join([f"{v:+.4f}" if v != 0 else "0.0000" for v in row])
                    lines.append(f"| {idx} | {values} |")

        return "\n".join(lines)


def main() -> None:
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark BM25 variants on BRIGHT dataset.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="BRIGHT splits to evaluate (default: all).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for @k metrics (default: 10).",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to run (default: all). Options: evolved, classic, lucene, bm25l, bm25+, atire",
    )
    parser.add_argument(
        "--include-gensim",
        action="store_true",
        help="Include Gensim baseline (requires gensim package).",
    )
    args = parser.parse_args()

    # Create benchmark
    benchmark = BrightBenchmark(
        domains=args.domains,
        k=args.k,
        k1=args.k1,
        b=args.b,
    )

    # Add variants
    if args.variants:
        variant_map = {
            "evolved": lambda c, k1, b: BM25(c, k1=k1, b=b),
            "classic": create_bm25_classic,
            "lucene": create_bm25_lucene,
            "bm25l": create_bm25l,
            "bm25+": create_bm25_plus,
            "atire": create_bm25_atire,
            "query_side": create_bm25_query_side,
        }
        for name in args.variants:
            if name in variant_map:
                benchmark.add_variant(name, variant_map[name])
            else:
                print(f"Warning: Unknown variant '{name}', skipping.")
    else:
        benchmark.add_default_variants()

    # Optionally add Gensim baseline
    if args.include_gensim:
        try:
            from benchmarks.baselines.gensim_bm25 import GensimBM25Baseline

            benchmark.add_external_baseline("gensim", GensimBM25Baseline)
        except ImportError:
            print("Warning: Gensim not available, skipping gensim baseline.")

    # Run benchmark
    print("\nRunning BRIGHT benchmark")
    print(f"Domains: {benchmark.domains}")
    print(f"Variants: {list(benchmark._variants.keys())}")
    print(f"Parameters: k={args.k}, k1={args.k1}, b={args.b}")
    print("-" * 60)

    benchmark.run(verbose=True)

    # Save results
    benchmark.save_results(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(benchmark.get_summary_table("ndcg_at_k"))


if __name__ == "__main__":
    main()
