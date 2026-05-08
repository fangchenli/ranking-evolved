"""
Freeform late-interaction retrieval seed — maximum freedom for inventing a
new retrieval method over fixed contextualized token embeddings.

This file IS the program RankEvolve mutates. The evaluator requires a class
named `LateInteractionRetriever` with two methods:
    build(docs: TokenEmbeddingStore) -> None
    search(queries: TokenEmbeddingStore, top_k: int) -> dict[query_id, list[(doc_id, score)]]
Everything else is evolvable. Default behavior: exact ColBERT MaxSim
(bit-identical to `tasks/late_interaction/library.py:ExactMaxSimRetriever`),
expressed as a degenerate two-stage pipeline (`_candidate_pool` returns
the full corpus; the rerank loop scores every candidate exactly).

Design directives (also in the system prompt — repeated here as inline
hints for human readers and the LLM):
  - Prefer fundamental, elegant solutions. A small principled change to the
    scoring or candidate-generation rule is worth more than three local
    heuristic tweaks.
  - No magic constants without justification. Every constant should have a
    clear interpretation (probability, budget, bin count) — not a number
    that happens to work on one dataset.
  - No dataset-specific branches or quality-floor gates. Improve the
    algorithm; don't hide bad cases behind switches.
  - If progress stagnates, restructure the entire file. Replace the scoring
    function, redesign `build()` to precompute different artifacts, change
    the `search()` pipeline shape — as long as the public surface stays.
  - Budgets must scale with corpus size. A constant `rerank_n = K * top_k`
    that is fine on a 24K-doc corpus may capture 14% of a 171K-doc corpus
    and lose recall catastrophically. Express budgets as functions of N
    (e.g. `min(N, max(top_k, alpha * sqrt(N)))`).
"""

from __future__ import annotations

# Import _runtime first so BLAS pinning takes effect before numpy is loaded.
from tasks.late_interaction import _runtime  # noqa: F401

import heapq

import numpy as np
from numpy.typing import NDArray

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore


# -----------------------------------------------------------------------------
# Config — EVOLVE: add hyperparameters for your retrieval method here
# -----------------------------------------------------------------------------
# Empty by default. Add fields like `n_centroids: int = 1024`,
# `nprobe: int = 16`, `rerank_budget: int = 256`, etc. — anything your method
# needs. Constants live here, not buried inside functions, so they're
# inspectable in one place.


class Config:
    pass


# -----------------------------------------------------------------------------
# Per-pair score — EVOLVE: the core relevance kernel
# -----------------------------------------------------------------------------
# This is the function called per (query, document) pair when scoring all
# documents brute-force. EVOLVE this when:
#   - introducing token-level pruning (drop low-IDF query tokens, etc.)
#   - introducing approximation (centroid-only upper bound, residual
#     refinement, learned weighting)
#   - vectorizing across docs (eliminate this function and replace the
#     inner loop in `search()` with a batched matmul)
#
# Default: exact ColBERT MaxSim — score(q,d) = sum_i max_j dot(q_i, d_j).


def _maxsim_score(query: NDArray[np.floating], doc: NDArray[np.floating]) -> float:
    if query.shape[0] == 0 or doc.shape[0] == 0:
        return 0.0
    similarities = query @ doc.T
    return float(np.max(similarities, axis=1).sum(dtype=np.float32))


# -----------------------------------------------------------------------------
# Top-k — EVOLVE only if you have a reason (e.g. heap-based early exit)
# -----------------------------------------------------------------------------
# Stable, deterministic tie-break by doc_id. Most evolved methods keep this
# unchanged; if you're doing partial scoring with bounds, you may want a
# heap-with-early-termination variant — replace this then.


def _top_k_from_scores(
    doc_ids: list[str],
    scores: NDArray[np.floating],
    top_k: int,
) -> list[tuple[str, float]]:
    limit = min(top_k, len(doc_ids))
    if limit == 0:
        return []
    keyed = ((-float(score), doc_id) for doc_id, score in zip(doc_ids, scores, strict=True))
    best = heapq.nsmallest(limit, keyed)
    return [(doc_id, -neg_score) for neg_score, doc_id in best]


# -----------------------------------------------------------------------------
# Candidate pool — EVOLVE: the cheap shortlist stage
# -----------------------------------------------------------------------------
# Given a query and the full doc store, return the indices to be scored
# exactly. Default: every document (exact MaxSim baseline). The point of
# this seam is so the LLM has ONE obvious place to introduce pruning
# without restructuring `search()` from scratch.
#
# Suggested budget that scales with corpus size:
#
#   def _candidate_pool(query, docs, top_k, rerank_n=None):
#       N = len(docs)
#       if rerank_n is None:
#           rerank_n = min(N, max(top_k, int(8 * np.sqrt(N))))
#       # ... pruning logic that returns ~rerank_n indices ...
#
# Keep budgets RELATIVE to N, not absolute. `rerank_n = 24 * top_k` on a
# 171K-doc corpus is only 14% coverage; on a 24K-doc corpus it covers
# everything. Constant absolute budgets break recall on the largest
# corpora and skew the per-dataset metrics.


def _candidate_pool(
    query: NDArray[np.floating],
    docs: TokenEmbeddingStore,
    top_k: int,
) -> NDArray[np.int32]:
    """Default: full corpus (exact MaxSim baseline)."""
    return np.arange(len(docs), dtype=np.int32)


# -----------------------------------------------------------------------------
# Retriever (PUBLIC SURFACE — class name + method signatures are contractual)
# -----------------------------------------------------------------------------


class LateInteractionRetriever:
    """Freeform late-interaction retriever — the seed for evolution.

    Initial behavior: brute-force exact MaxSim over every document, expressed
    as a degenerate two-stage pipeline (`_candidate_pool` returns all docs;
    `_score_candidates` does exact MaxSim). Evolved descendants are free to
    replace any internal detail — only the `build()` / `search()` signatures
    and the returned dict shape are contractual.
    """

    def __init__(self) -> None:
        self._docs: TokenEmbeddingStore | None = None
        # Plain dicts (not dataclasses) so this module can be loaded via
        # importlib.util.spec_from_file_location without the dataclass
        # machinery's `cls.__module__` lookup failing.
        self.last_diagnostics: dict[str, dict[str, int]] = {}

    def build(self, docs: TokenEmbeddingStore) -> None:
        """
        EVOLVE: this is where you precompute index structures.

        Today: just stores a reference to the doc store — no precomputation.
        Evolved methods will likely build:
          - centroids (k-means over doc tokens)
          - inverted lists (which docs touch each centroid)
          - product-quantized residuals
          - per-document summary vectors (mean/max-pool)
          - upper-bound score caches for cheap pruning
          - ANN graphs (HNSW-style) over centroids or summaries

        Build time IS measured (`build_time_ms` in metrics) but is reported
        separately from search latency, so amortize aggressively if it pays
        off in the search path.
        """
        self._docs = docs

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        EVOLVE: this is the timed retrieval path.

        Default shape: two stages.
          1. `_candidate_pool(query, docs, top_k)` returns indices to score.
             Today: every document. Replace with a centroid-probe / posting-
             list / ANN lookup that scales with corpus size.
          2. Rerank pool with `_maxsim_score`. Today: exact MaxSim. Replace
             or augment with a cheaper approximation if you keep recall.

        Whatever you do, return rankings as `{query_id: [(doc_id, score), ...]}`
        with at most `top_k` entries per query. Tie-breaking should remain
        deterministic (doc_id ASC on equal scores) so metrics are stable
        across runs.
        """
        if self._docs is None:
            raise RuntimeError("build(docs) must be called before search()")
        docs = self._docs
        n_docs = len(docs)

        rankings: dict[str, list[tuple[str, float]]] = {}
        diagnostics: dict[str, dict[str, int]] = {}

        for q_idx, query_id in enumerate(queries.ids):
            q_tokens = np.asarray(queries.get(q_idx), dtype=np.float32)

            # Stage 1: candidate selection (default = full corpus).
            cand_idx = _candidate_pool(q_tokens, docs, top_k)
            cand_idx = np.asarray(cand_idx, dtype=np.int32)

            # Stage 2: exact MaxSim on the shortlist.
            cand_scores = np.empty(cand_idx.shape[0], dtype=np.float32)
            doc_tokens_loaded = 0
            cand_doc_ids: list[str] = []
            for i, d_idx in enumerate(cand_idx.tolist()):
                d_tokens = np.asarray(docs.get(int(d_idx)), dtype=np.float32)
                doc_tokens_loaded += int(d_tokens.shape[0])
                cand_scores[i] = _maxsim_score(q_tokens, d_tokens)
                cand_doc_ids.append(docs.ids[int(d_idx)])

            rankings[query_id] = _top_k_from_scores(cand_doc_ids, cand_scores, top_k)

            # Diagnostics flow through to the evaluator — fields here become
            # `<dataset>_documents_scored`, `<dataset>_query_tokens_used`, etc.
            # in the metrics dict. EVOLVE: add fields that help interpret
            # what your method did per query (centroids probed, candidates
            # reranked, early-exit fraction, etc.).
            #
            # `corpus_coverage` is what the per-dataset feedback uses to spot
            # candidates that are scoring most of the corpus — those will
            # not be faster than the seed, no matter how clever the indexing.
            documents_scored = int(cand_idx.shape[0])
            corpus_coverage = (documents_scored / n_docs) if n_docs else 0.0
            diagnostics[query_id] = {
                "documents_scored": documents_scored,
                "corpus_size": int(n_docs),
                "corpus_coverage": float(corpus_coverage),
                "query_tokens_used": int(q_tokens.shape[0]),
                "document_tokens_loaded": int(doc_tokens_loaded),
            }

        self.last_diagnostics = diagnostics
        return rankings


__all__ = ["LateInteractionRetriever", "Config"]
