"""
Composable late-interaction retrieval seed — four explicit pipeline stages.

This is an alternative to `freeform.py` for the same evaluator. The four
stages map onto the design space the LLM is supposed to explore (mirrors
ColBERTv2 / PLAID / XTR / FastPLAID):

    encode_query(q_tokens)           -> QueryRep      (default: identity)
    encode_doc(d_tokens)             -> DocRep        (default: identity)
    Index.from_doc_reps(doc_reps)    -> Index         (default: stores list)
    candidate_pool(q, index, top_k)  -> ndarray[int]  (default: arange(N))
    score(q, d_rep)                  -> float         (default: exact MaxSim)

The seed wires those stubs into a degenerate pipeline that is bit-identical
to exact MaxSim. Mutations should target ONE stage at a time so the change
is interpretable in the per-step replay:

  - encode_query: prune low-saliency query tokens, scale by IDF, project
  - encode_doc:   per-token quantization, residuals over centroids
  - Index:        centroid posting lists, summary vectors, ANN graph
  - candidate_pool: probe centroids, intersect postings, take top-K by upper bound
  - score:        cheap upper bound first, exact only when needed

Public surface (the evaluator only sees these — DO NOT change names/sigs):
    class LateInteractionRetriever:
        def build(docs: TokenEmbeddingStore) -> None
        def search(queries: TokenEmbeddingStore, top_k: int)
            -> dict[str, list[tuple[str, float]]]

Budgets MUST scale with corpus size. A constant `rerank_n = K * top_k` is
fine on a 24K-doc corpus but captures only ~14% of a 171K-doc corpus and
collapses recall on the largest dataset. Use `min(N, max(top_k, alpha *
sqrt(N)))` or a similar size-aware rule, and report `corpus_coverage` in
the per-query diagnostics so the per-dataset feedback can flag candidates
that touch most of the corpus.
"""

from __future__ import annotations

# Import _runtime first so BLAS pinning takes effect before numpy is loaded.
from tasks.late_interaction import _runtime  # noqa: F401

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore


# -----------------------------------------------------------------------------
# Config — EVOLVE: hyperparameters for any of the four stages
# -----------------------------------------------------------------------------


class Config:
    """Add interpretable hyperparameters here as the algorithm evolves.

    Examples (uncomment / replace as needed):
        n_centroids: int = 1024            # k-means centroids over doc tokens
        nprobe: int = 16                   # how many centroids to probe per query token
        rerank_alpha: float = 8.0          # rerank_n = min(N, max(top_k, alpha*sqrt(N)))
        query_token_keep: float = 1.0      # fraction of query tokens to keep by saliency
    """


# -----------------------------------------------------------------------------
# Stage representations — EVOLVE: replace identity with whatever your method needs
# -----------------------------------------------------------------------------


class QueryRep:
    """What the index sees of a query. Default: the raw token matrix.

    Plain class (not dataclass) so this module loads via
    importlib.util.spec_from_file_location without the dataclass machinery's
    `cls.__module__` lookup failing.
    """

    __slots__ = ("tokens",)

    def __init__(self, tokens: NDArray[np.float32]) -> None:
        self.tokens = tokens


class DocRep:
    """What the index sees of a document. Default: the raw token matrix."""

    __slots__ = ("tokens",)

    def __init__(self, tokens: NDArray[np.float32]) -> None:
        self.tokens = tokens


def encode_query(q_tokens: NDArray[np.floating]) -> QueryRep:
    """Default: pass-through. EVOLVE to: prune by saliency, project, weight."""
    return QueryRep(tokens=np.asarray(q_tokens, dtype=np.float32))


def encode_doc(d_tokens: NDArray[np.floating]) -> DocRep:
    """Default: pass-through. EVOLVE to: quantize, take residuals over centroids."""
    return DocRep(tokens=np.asarray(d_tokens, dtype=np.float32))


# -----------------------------------------------------------------------------
# Index — EVOLVE: precompute structures here so search() is fast
# -----------------------------------------------------------------------------


class Index:
    """Container for whatever was precomputed in `build()`.

    Default: just stores the doc reps in order. EVOLVE to add:
      - centroids: NDArray[float32]                 # (n_centroids, dim)
      - postings: list[NDArray[int32]]              # docs touching each centroid
      - summary_vectors: NDArray[float32]           # (n_docs, dim) mean/max-pooled
      - upper_bound_cache: NDArray[float32]         # cheap per-doc score ceiling
      - ann_graph: Any                              # HNSW over summaries

    Plain class (not dataclass) for importlib safety.
    """

    def __init__(self, doc_reps: list[DocRep] | None = None) -> None:
        self.doc_reps: list[DocRep] = list(doc_reps) if doc_reps else []
        self.n_docs: int = len(self.doc_reps)
        self.extras: dict[str, Any] = {}

    @classmethod
    def from_doc_reps(cls, doc_reps: list[DocRep]) -> "Index":
        return cls(doc_reps=doc_reps)


# -----------------------------------------------------------------------------
# Candidate pool — EVOLVE: cheap shortlist of doc indices to score exactly
# -----------------------------------------------------------------------------
# Default: returns the full corpus (matches exact MaxSim). The whole point
# of evolving this seed is for the LLM to replace this stub with a
# centroid-probe / posting-list / ANN lookup that scales with corpus size.


def candidate_pool(
    query: QueryRep,
    index: Index,
    top_k: int,
) -> NDArray[np.int32]:
    """Default: every document. Replace with a SIZE-AWARE shortlist."""
    return np.arange(index.n_docs, dtype=np.int32)


# -----------------------------------------------------------------------------
# Score — EVOLVE: relevance kernel
# -----------------------------------------------------------------------------
# Default: exact ColBERT MaxSim. Replace when introducing a cheap bound or
# a different similarity (cosine + saturation, weighted MaxSim, etc.).


def score(query: QueryRep, doc: DocRep) -> float:
    q = query.tokens
    d = doc.tokens
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    similarities = q @ d.T
    return float(np.max(similarities, axis=1).sum(dtype=np.float32))


# -----------------------------------------------------------------------------
# Top-k — leave alone unless you have a specific reason
# -----------------------------------------------------------------------------


def _top_k_from_scores(
    doc_ids: list[str],
    scores: NDArray[np.floating],
    top_k: int,
) -> list[tuple[str, float]]:
    limit = min(top_k, len(doc_ids))
    if limit == 0:
        return []
    keyed = ((-float(s), doc_id) for doc_id, s in zip(doc_ids, scores, strict=True))
    best = heapq.nsmallest(limit, keyed)
    return [(doc_id, -neg_score) for neg_score, doc_id in best]


# -----------------------------------------------------------------------------
# Retriever (PUBLIC SURFACE — class name + method signatures are contractual)
# -----------------------------------------------------------------------------


class LateInteractionRetriever:
    """Composable late-interaction retriever — four explicit stages.

    Default behavior is bit-identical to exact MaxSim because every stage is
    an identity. Mutations should target ONE stage at a time.
    """

    def __init__(self) -> None:
        self.cfg = Config()
        self._docs: TokenEmbeddingStore | None = None
        self._index: Index | None = None
        # Plain dict (not dataclass) for importlib safety.
        self.last_diagnostics: dict[str, dict[str, int]] = {}

    def build(self, docs: TokenEmbeddingStore) -> None:
        self._docs = docs
        n_docs = len(docs)
        if n_docs == 0:
            self._index = Index.from_doc_reps([])
            return
        doc_reps = [encode_doc(np.asarray(docs.get(i), dtype=np.float32)) for i in range(n_docs)]
        self._index = Index.from_doc_reps(doc_reps)

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        if self._docs is None or self._index is None:
            raise RuntimeError("build(docs) must be called before search()")
        docs = self._docs
        index = self._index
        n_docs = index.n_docs

        rankings: dict[str, list[tuple[str, float]]] = {}
        diagnostics: dict[str, dict[str, int]] = {}

        for q_idx, query_id in enumerate(queries.ids):
            q_tokens = np.asarray(queries.get(q_idx), dtype=np.float32)
            q_rep = encode_query(q_tokens)

            cand_idx = np.asarray(candidate_pool(q_rep, index, top_k), dtype=np.int32)

            cand_scores = np.empty(cand_idx.shape[0], dtype=np.float32)
            doc_tokens_loaded = 0
            cand_doc_ids: list[str] = []
            for i, d_idx in enumerate(cand_idx.tolist()):
                d_rep = index.doc_reps[int(d_idx)]
                doc_tokens_loaded += int(d_rep.tokens.shape[0])
                cand_scores[i] = score(q_rep, d_rep)
                cand_doc_ids.append(docs.ids[int(d_idx)])

            rankings[query_id] = _top_k_from_scores(cand_doc_ids, cand_scores, top_k)

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


__all__ = ["LateInteractionRetriever", "Config", "QueryRep", "DocRep", "Index"]
