"""
Compatibility shim exposing the current default BM25 (biology/general-tuned).

If you want alternative kernels, use bm25_biology.py, bm25_psychology.py, or bm25_classic.py directly.
"""

from ranking_evolved.bm25_biology import BM25, Corpus, tokenize  # noqa: F401

__all__ = ["BM25", "Corpus", "tokenize"]
