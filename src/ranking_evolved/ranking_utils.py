"""
Shared utilities for optimized BM25 ranking.

This module provides reusable components for fast batch ranking:
1. Fused scoring - eliminates loop over query terms
2. Parallel batch ranking - ThreadPoolExecutor for query parallelism
3. Efficient top-k - np.argpartition for O(n) selection

Usage:
    from ranking_evolved.ranking_utils import (
        score_candidates_fused,
        batch_rank_parallel,
        select_top_k,
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

# Default number of workers for parallel query processing
DEFAULT_NUM_WORKERS = 32

# Minimum queries before enabling parallelism
MIN_QUERIES_FOR_PARALLEL = 10


# =============================================================================
# Protocol for Corpus (duck typing)
# =============================================================================


class CorpusProtocol(Protocol):
    """Protocol defining required corpus attributes for ranking utilities."""

    N: int
    tf_matrix: csr_matrix
    idf_array: NDArray[np.float64]
    norm_array: NDArray[np.float64]
    _posting_lists: dict[int, NDArray[np.int64]]

    def get_term_id(self, term: str) -> int | None: ...


# =============================================================================
# Fused Scoring
# =============================================================================


def score_candidates_fused(
    query_term_ids: list[int],
    candidate_docs: NDArray[np.int64],
    tf_matrix: csr_matrix,
    idf_array: NDArray[np.float64],
    norm_array: NDArray[np.float64],
    k1: float = 0.9,
    epsilon: float = 1e-9,
) -> NDArray[np.float64]:
    """
    Score candidates using fused matrix operation (no term loop).

    Instead of looping over query terms, this extracts all TF rows at once
    and computes scores in a single vectorized operation.

    Args:
        query_term_ids: List of term IDs in the query
        candidate_docs: Array of document indices to score
        tf_matrix: Sparse term-document matrix (vocab_size, N_docs)
        idf_array: IDF values for each term (vocab_size,)
        norm_array: Length normalization for each doc (N_docs,)
        k1: TF saturation parameter (default 0.9 for Lucene)
        epsilon: Small value for numerical stability

    Returns:
        Scores for candidate documents (len(candidate_docs),)
    """
    if len(candidate_docs) == 0 or len(query_term_ids) == 0:
        return np.array([], dtype=np.float64)

    # Extract all TF rows at once: (num_terms, num_candidates)
    tf_rows = tf_matrix[query_term_ids, :][:, candidate_docs].toarray()

    # IDF values: (num_terms, 1) for broadcasting
    idf_values = idf_array[query_term_ids][:, np.newaxis]

    # Norms for candidates: (num_candidates,)
    norms = norm_array[candidate_docs]

    # Vectorized saturation: tf / (tf + k1 * norm)
    # Shape: (num_terms, num_candidates)
    saturated = tf_rows / (tf_rows + k1 * norms + epsilon)

    # Sum over terms: (num_candidates,)
    scores = np.sum(idf_values * saturated, axis=0)

    return scores


# =============================================================================
# Efficient Top-K Selection
# =============================================================================


def select_top_k(
    scores: NDArray[np.float64],
    top_k: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Select top-k documents efficiently.

    Uses np.argpartition for O(n) selection when k << n,
    falling back to full sort when k is large.

    Args:
        scores: Score array for all documents (N,)
        top_k: Number of top results (None for all)

    Returns:
        (sorted_indices, sorted_scores) in descending order
    """
    n = len(scores)

    if top_k is not None and top_k < n:
        # O(n) partition + O(k log k) sort of top-k
        top_k_indices = np.argpartition(-scores, top_k)[:top_k]
        sorted_top_k = top_k_indices[np.argsort(-scores[top_k_indices])]
        return sorted_top_k.astype(np.int64), scores[sorted_top_k]
    else:
        # Full sort O(n log n)
        sorted_indices = np.argsort(-scores).astype(np.int64)
        return sorted_indices, scores[sorted_indices]


# =============================================================================
# Candidate Retrieval from Inverted Index
# =============================================================================


def get_candidates_from_posting_lists(
    query_term_ids: list[int],
    posting_lists: dict[int, NDArray[np.int64]],
) -> NDArray[np.int64]:
    """
    Get candidate documents from inverted index.

    Returns union of posting lists for all query terms.

    Args:
        query_term_ids: Term IDs in query
        posting_lists: Inverted index (term_id -> doc_ids)

    Returns:
        Sorted array of candidate document indices
    """
    candidate_set: set[int] = set()
    for term_id in query_term_ids:
        posting_list = posting_lists.get(term_id, np.array([], dtype=np.int64))
        candidate_set.update(posting_list.tolist())

    return np.array(sorted(candidate_set), dtype=np.int64)


# =============================================================================
# Single Query Ranking (Fused)
# =============================================================================


def rank_single_fused(
    query: list[str],
    corpus: CorpusProtocol,
    k1: float = 0.9,
    epsilon: float = 1e-9,
    top_k: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Rank documents for a single query using fused scoring.

    Args:
        query: Tokenized query (list of terms)
        corpus: Corpus with tf_matrix, idf_array, norm_array, _posting_lists
        k1: TF saturation parameter
        epsilon: Numerical stability
        top_k: Number of top results (None for all)

    Returns:
        (sorted_indices, sorted_scores) in descending order
    """
    if not query:
        return (
            np.arange(corpus.N, dtype=np.int64),
            np.zeros(corpus.N, dtype=np.float64),
        )

    # Get term IDs
    unique_terms = list(set(query))
    query_term_ids = [
        corpus.get_term_id(t)
        for t in unique_terms
        if corpus.get_term_id(t) is not None
    ]

    if not query_term_ids:
        return (
            np.arange(corpus.N, dtype=np.int64),
            np.zeros(corpus.N, dtype=np.float64),
        )

    # Get candidates from inverted index
    candidate_docs = get_candidates_from_posting_lists(
        query_term_ids, corpus._posting_lists
    )

    # Fused scoring
    candidate_scores = score_candidates_fused(
        query_term_ids,
        candidate_docs,
        corpus.tf_matrix,
        corpus.idf_array,
        corpus.norm_array,
        k1=k1,
        epsilon=epsilon,
    )

    # Build full score array
    all_scores = np.zeros(corpus.N, dtype=np.float64)
    all_scores[candidate_docs] = candidate_scores

    # Select top-k
    return select_top_k(all_scores, top_k)


# =============================================================================
# Parallel Batch Ranking
# =============================================================================


def batch_rank_parallel(
    queries: list[list[str]],
    corpus: CorpusProtocol,
    k1: float = 0.9,
    epsilon: float = 1e-9,
    top_k: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    min_queries_for_parallel: int = MIN_QUERIES_FOR_PARALLEL,
) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """
    Batch rank using fused scoring + parallel processing.

    Args:
        queries: List of tokenized queries
        corpus: Corpus with required attributes
        k1: TF saturation parameter
        epsilon: Numerical stability
        top_k: Number of top results per query
        num_workers: Number of parallel workers
        min_queries_for_parallel: Minimum queries before enabling parallelism

    Returns:
        List of (sorted_indices, sorted_scores) tuples
    """
    if not queries:
        return []

    def rank_single(query: list[str]) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        return rank_single_fused(query, corpus, k1, epsilon, top_k)

    # For small batches, run sequentially
    if len(queries) < min_queries_for_parallel:
        return [rank_single(query) for query in queries]

    # For larger batches, parallelize
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(rank_single, queries))

    return results


# =============================================================================
# Mixin Class for BM25 implementations
# =============================================================================


class BatchRankMixin:
    """
    Mixin providing optimized batch ranking methods.

    Requires the class to have:
    - self.corpus: CorpusProtocol
    - self.k1: float (or Config.k1)
    - self.epsilon: float (or Config.epsilon)
    """

    corpus: CorpusProtocol

    def _get_k1(self) -> float:
        """Get k1 parameter (override if needed)."""
        if hasattr(self, "k1"):
            return self.k1
        # Try Config class
        try:
            from ranking_evolved.bm25_freeform_fast import Config

            return Config.k1
        except ImportError:
            return 0.9

    def _get_epsilon(self) -> float:
        """Get epsilon parameter (override if needed)."""
        if hasattr(self, "epsilon"):
            return self.epsilon
        try:
            from ranking_evolved.bm25_freeform_fast import Config

            return Config.epsilon
        except ImportError:
            return 1e-9

    def _rank_single_fused(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank using fused matrix operations for a single query."""
        return rank_single_fused(
            query,
            self.corpus,
            k1=self._get_k1(),
            epsilon=self._get_epsilon(),
            top_k=top_k,
        )

    def batch_rank_vectorized(
        self,
        queries: list[list[str]],
        top_k: int | None = None,
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """
        Batch rank using fused matrix operations + parallel processing.

        Optimizations:
        1. Fused query term computation (no loop over terms)
        2. Uses np.argpartition for efficient top-k selection
        3. Inverted index for candidate filtering
        4. Parallel processing across queries
        """
        return batch_rank_parallel(
            queries,
            self.corpus,
            k1=self._get_k1(),
            epsilon=self._get_epsilon(),
            top_k=top_k,
        )


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Core functions
    "score_candidates_fused",
    "select_top_k",
    "get_candidates_from_posting_lists",
    "rank_single_fused",
    "batch_rank_parallel",
    # Mixin class
    "BatchRankMixin",
    # Constants
    "DEFAULT_NUM_WORKERS",
    "MIN_QUERIES_FOR_PARALLEL",
]
