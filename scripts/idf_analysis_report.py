"""
Generate a simple IDF analysis report and histogram comparison.

Usage (example):
    uv run python scripts/idf_analysis_report.py --domain biology

This loads the BRIGHT split, computes classic BM25 IDF and the current
biology-tuned IDF, prints summary stats, and saves a histogram plot
to idf_histogram.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25_biology import Corpus
from ranking_evolved.idf_analysis import (
    compare_idf,
    compute_idf,
    idf_histogram_data,
)


def idf_bm25(df: np.ndarray, N: int) -> np.ndarray:
    """Classic BM25 IDF."""
    return np.log((N - df + 0.5) / (df + 0.5) + 1.0)


def idf_current(df: np.ndarray, N: int) -> np.ndarray:
    """Biology-tuned IDF from bm25_biology.py."""
    raw = np.log((N + 0.5) / (df + 0.5))
    idf = np.minimum(np.maximum(raw, 0.0), 8.0)
    return idf


def idf_psychology(df: np.ndarray, N: int) -> np.ndarray:
    """Psychology-tuned IDF inspired by the psych-focused variant."""
    base = np.log(np.maximum((N + 0.5) / (df + 0.5), 1e-9))
    idf = np.clip(base + 0.56, 0.055, 4.85)
    df_ratio = df / max(N, 1.0)
    mid_mask = (df_ratio > 0.015) & (df_ratio < 0.19)
    idf[mid_mask] += 0.055
    ultra_rare = df_ratio < 0.0018
    idf[ultra_rare] *= 0.965
    very_common = df_ratio > 0.35
    idf[very_common] *= 0.94
    informative = (df_ratio >= 0.012) & (df_ratio <= 0.21)
    idf[informative] *= 1.012
    return idf


def load_corpus(domain: str) -> Corpus:
    docs = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    return Corpus.from_huggingface_dataset(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="IDF analysis for BRIGHT.")
    parser.add_argument("--domain", default="biology", help="BRIGHT split to load (default: biology).")
    parser.add_argument(
        "--output", default="idf_histogram.png", help="Path to save histogram plot (default: idf_histogram.png)."
    )
    parser.add_argument(
        "--idf",
        choices=["biology", "psychology"],
        default="biology",
        help="Which tuned IDF to compare against classic BM25.",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.domain)

    idf_a = compute_idf(corpus, idf_bm25)
    tuned = idf_current if args.idf == "biology" else idf_psychology
    idf_b = compute_idf(corpus, tuned)

    cmp = compare_idf(corpus, idf_bm25, tuned)
    hist_a = idf_histogram_data(idf_a)
    hist_b = idf_histogram_data(idf_b)

    print(f"Domain: {args.domain}")
    print(f"Classic BM25 IDF: mean={hist_a['mean']:.4f}, std={hist_a['std']:.4f}, min={hist_a['min']:.4f}, max={hist_a['max']:.4f}")
    print(f"Tuned IDF ({args.idf}): mean={hist_b['mean']:.4f}, std={hist_b['std']:.4f}, min={hist_b['min']:.4f}, max={hist_b['max']:.4f}")
    print("Delta stats    :", cmp["stats"])

    # Plot histograms.
    plt.figure(figsize=(8, 5))
    plt.hist(hist_a["values"], bins=50, alpha=0.5, label="Classic BM25")
    plt.hist(hist_b["values"], bins=50, alpha=0.5, label=f"Tuned IDF ({args.idf})")
    plt.xlabel("IDF value")
    plt.ylabel("Count")
    plt.title(f"IDF Distribution Comparison ({args.domain}, {args.idf})")
    plt.legend()
    out_path = Path(args.output)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    main()
