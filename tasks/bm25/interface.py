"""Formal contract a BM25 candidate program must satisfy.

The evaluator and the YAML system_message both refer to this contract; this
file is the single source of truth. The framework loads candidate programs by
file path, so we cannot enforce subclassing — the Protocol is structural.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BM25Candidate(Protocol):
    """A candidate retrieval program loaded by the framework.

    The module produced by a candidate must expose the four names below at
    module level (BM25, Corpus, tokenize, LuceneTokenizer). BM25 must be
    constructible from a Corpus and provide rank() and score().
    """

    def rank(self, query: str, top_k: int | None = None) -> tuple[list[int], list[float]]:
        """Return (doc_indices, scores) sorted by score descending."""
        ...

    def score(self, query: str, doc_index: int) -> float:
        """Return the relevance score for a single (query, document) pair."""
        ...
