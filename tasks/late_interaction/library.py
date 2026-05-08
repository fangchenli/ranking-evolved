"""Reference late-interaction retrieval kernels."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore


@dataclass(frozen=True)
class SearchDiagnostics:
    """Basic diagnostics for exact-search runs."""

    documents_scored: int
    query_tokens_used: int
    document_tokens_loaded: int


def exact_maxsim_score(
    query_tokens: NDArray[np.floating],
    doc_tokens: NDArray[np.floating],
) -> float:
    """Compute exact ColBERT-style MaxSim score for one query/document pair.

    ``score(q, d) = sum_i max_j dot(q_i, d_j)``

    Empty queries or empty documents score ``0.0``.
    """

    query = np.asarray(query_tokens, dtype=np.float32)
    doc = np.asarray(doc_tokens, dtype=np.float32)
    if query.ndim != 2 or doc.ndim != 2:
        raise ValueError("query_tokens and doc_tokens must be 2D arrays")
    if query.shape[1] != doc.shape[1]:
        raise ValueError("query and document embedding dimensions must match")
    if query.shape[0] == 0 or doc.shape[0] == 0:
        return 0.0

    similarities = query @ doc.T
    return float(np.max(similarities, axis=1).sum(dtype=np.float32))


def exact_maxsim_scores(
    query_tokens: NDArray[np.floating],
    docs: TokenEmbeddingStore,
) -> NDArray[np.float32]:
    """Score one query against every document using exact MaxSim."""

    scores = np.empty(len(docs), dtype=np.float32)
    for doc_index in range(len(docs)):
        scores[doc_index] = exact_maxsim_score(query_tokens, docs.get(doc_index))
    return scores


def rank_exact_maxsim(
    queries: TokenEmbeddingStore,
    docs: TokenEmbeddingStore,
    top_k: int = 100,
) -> dict[str, list[tuple[str, float]]]:
    """Rank all documents for each query with exact MaxSim.

    Ties are broken deterministically by document ID.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query_index, query_id in enumerate(queries.ids):
        scores = exact_maxsim_scores(queries.get(query_index), docs)
        rankings[query_id] = top_k_from_scores(docs.ids, scores, top_k)
    return rankings


def top_k_from_scores(
    doc_ids: list[str],
    scores: NDArray[np.floating],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return top-k ``(doc_id, score)`` pairs with deterministic tie-breaking."""

    if len(doc_ids) != len(scores):
        raise ValueError("doc_ids and scores must have identical lengths")
    limit = min(top_k, len(doc_ids))
    if limit == 0:
        return []

    # heapq.nsmallest over (-score, doc_id) avoids sorting every document when top_k is small.
    keyed = ((-float(score), doc_id) for doc_id, score in zip(doc_ids, scores, strict=True))
    best = heapq.nsmallest(limit, keyed)
    return [(doc_id, -neg_score) for neg_score, doc_id in best]


class ExactMaxSimRetriever:
    """Simple exact MaxSim retriever used as the Phase 1 correctness anchor."""

    def __init__(self) -> None:
        self._docs: TokenEmbeddingStore | None = None
        self.last_diagnostics: dict[str, SearchDiagnostics] = {}

    def build(self, docs: TokenEmbeddingStore) -> None:
        self._docs = docs

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        if self._docs is None:
            raise RuntimeError("build(docs) must be called before search()")

        rankings: dict[str, list[tuple[str, float]]] = {}
        diagnostics: dict[str, SearchDiagnostics] = {}
        for query_index, query_id in enumerate(queries.ids):
            query_tokens = queries.get(query_index)
            scores = exact_maxsim_scores(query_tokens, self._docs)
            rankings[query_id] = top_k_from_scores(self._docs.ids, scores, top_k)
            diagnostics[query_id] = SearchDiagnostics(
                documents_scored=len(self._docs),
                query_tokens_used=int(query_tokens.shape[0]),
                document_tokens_loaded=int(self._docs.total_tokens),
            )
        self.last_diagnostics = diagnostics
        return rankings
