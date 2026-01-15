"""
Baseline BM25 implementations from external libraries.

This package provides wrappers around popular IR library implementations
of BM25 for comparison benchmarking.
"""

from benchmarks.baselines.gensim_bm25 import GensimBM25Baseline

__all__ = ["GensimBM25Baseline"]
