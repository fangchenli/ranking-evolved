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
    """
    Progressive-Prefix PCA-SimHash (PP-CP-SimHash) candidate generation + exact MaxSim rerank.

    One coherent principle:
      Use *two resolutions* of the same PCA-SimHash code (fine and coarse).
      Probe fine buckets first for precision; if they do not yield enough hits
      to fill the rerank budget, *back off to a shorter prefix* (coarser hash)
      to recover recall. This is an adaptive "progressive widening" rule that
      does not branch on dataset identity and scales with corpus size via the
      rerank budget.

    Mechanically:
      - Build postings for B_fine bits and also for B_coarse (< B_fine) prefix bits.
      - Query-time:
          1) fine multi-probe votes
          2) if fine-hit-docs < coarse_trigger_hit_frac * rerank_n:
               add coarse-prefix multi-probe votes (down-weighted)
      - Union with a dense mean-vector backstop, then exact MaxSim rerank.

    All constants are interpretable budgets or probabilities.
    """

    # SimHash structure (interpretable: number of independent hash tables and bits).
    lsh_n_tables: int = 6
    lsh_n_bits: int = 12  # fine resolution: 4096 buckets / table

    # Coarse prefix bits (interpretable: prefix length). Smaller => larger buckets => higher recall.
    lsh_coarse_bits: int = 8  # coarse resolution: 256 buckets / table

    # When fine LSH yields too few hit docs relative to the rerank budget, widen to coarse.
    coarse_trigger_hit_frac: float = 0.50  # "need at least half the rerank budget from fine votes"

    # Coarse votes are less specific; down-weight their evidence.
    coarse_vote_weight: float = 0.50  # "a coarse collision counts as half a fine collision"

    # Multi-probe parameters (interpretable: least-reliable bits considered; max flip order).
    multiprobe_k_uncertain: int = 6
    multiprobe_max_flips: int = 2

    # Index footprint control (interpretable: max tokens per doc to contribute to the index).
    index_tokens_per_doc: int = 64

    # Query-time hashing budget (interpretable: max query tokens used for LSH lookup).
    query_tokens_for_lsh: int = 40

    # Dense backstop pool size (interpretable: O(sqrt(N)) candidates from a cheap dense scan).
    mean_pool_sqrt_factor: float = 8.0

    # Exact rerank budget (interpretable: always rerank >= 4*top_k, and also >= c*sqrt(N)).
    rerank_topk_mult: int = 4
    rerank_sqrt_factor: float = 32.0

    # Keep at most this multiple of rerank_n from the LSH stage before unioning with mean-pool.
    lsh_keep_mult: int = 6


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
    """Centered PCA-SimHash candidate generation + exact MaxSim rerank."""

    def __init__(self) -> None:
        self._docs: TokenEmbeddingStore | None = None
        self._cfg = Config()

        # SimHash index artifacts
        self._hyperplanes: NDArray[np.float32] | None = None  # (T, B_fine, dim) unit hyperplanes
        self._center: NDArray[np.float32] | None = None  # (dim,) global mean for centering before hashing

        # Two-resolution postings:
        #   fine:   B_fine bits
        #   coarse: B_coarse prefix bits (lower bits of the fine code)
        self._postings: list[dict[int, NDArray[np.int32]]] | None = None  # per-table: fine_code -> doc_idxs
        self._bucket_idf: list[dict[int, np.float32]] | None = None  # per-table: fine_code -> IDF weight
        self._postings_coarse: list[dict[int, NDArray[np.int32]]] | None = None  # per-table: coarse_code -> doc_idxs
        self._bucket_idf_coarse: list[dict[int, np.float32]] | None = None  # per-table: coarse_code -> IDF weight

        # Dense backstop artifact
        self._doc_mean: NDArray[np.float32] | None = None  # (N, dim), L2-normalized

        # Plain dicts (not dataclasses) so this module can be loaded via importlib.
        self.last_diagnostics: dict[str, dict[str, int]] = {}

    @staticmethod
    def _l2_normalize_1d(vec: NDArray[np.floating]) -> NDArray[np.float32]:
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n <= 0.0:
            return np.zeros_like(v, dtype=np.float32)
        return v / n

    @staticmethod
    def _select_top_tokens_by_norm(tokens: NDArray[np.floating], m: int) -> NDArray[np.float32]:
        x = np.asarray(tokens, dtype=np.float32)
        if x.shape[0] <= m:
            return x
        # Token "energy" (L2 norm) is a deterministic salience proxy for contextual embeddings.
        norms2 = np.einsum("ij,ij->i", x, x, dtype=np.float32)
        idx = np.argpartition(norms2, -m)[-m:]
        return x[idx]

    @staticmethod
    def _simhash_codes_and_margins(
        tokens: NDArray[np.floating],
        hyperplanes: NDArray[np.float32],  # (B, dim)
        center: NDArray[np.float32] | None = None,  # (dim,)
    ) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
        """
        SimHash with optional mean-centering:
          code bits = sign((token - center) · hyperplane)

        Returns:
          codes: (n_tok,) uint32
          margins: (n_tok, B) float32 = abs((token-center) · hyperplane) (bit reliability)
        """
        x = np.asarray(tokens, dtype=np.float32)
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, hyperplanes.shape[0]), dtype=np.float32)

        if center is not None:
            c = np.asarray(center, dtype=np.float32)
            if c.ndim == 1 and c.shape[0] == x.shape[1]:
                x = x - c  # broadcast

        proj = (x @ hyperplanes.T).astype(np.float32, copy=False)  # (n_tok, B)
        signs = (proj >= 0.0)
        packed = np.packbits(signs, axis=1, bitorder="little")  # (n_tok, ceil(B/8)) uint8

        codes = packed[:, 0].astype(np.uint32)
        shift = 8
        for b in range(1, packed.shape[1]):
            codes |= packed[:, b].astype(np.uint32) << shift
            shift += 8

        B = int(hyperplanes.shape[0])
        if B < 32:
            codes &= (np.uint32(1) << np.uint32(B)) - np.uint32(1)

        return codes, np.abs(proj).astype(np.float32, copy=False)

    @staticmethod
    def _simhash_codes(
        tokens: NDArray[np.floating],
        hyperplanes: NDArray[np.float32],
        center: NDArray[np.float32] | None = None,
    ) -> NDArray[np.uint32]:
        codes, _ = LateInteractionRetriever._simhash_codes_and_margins(tokens, hyperplanes, center=center)
        return codes

    def build(self, docs: TokenEmbeddingStore) -> None:
        self._docs = docs
        n_docs = len(docs)

        # Infer embedding dimensionality from the first non-empty doc.
        dim = None
        for i in range(n_docs):
            t = np.asarray(docs.get(i), dtype=np.float32)
            if t.ndim == 2 and t.shape[0] > 0:
                dim = int(t.shape[1])
                break
        if dim is None:
            # Degenerate corpus: everything empty.
            self._hyperplanes = np.zeros((self._cfg.lsh_n_tables, self._cfg.lsh_n_bits, 1), dtype=np.float32)
            self._center = np.zeros((1,), dtype=np.float32)
            self._postings = [dict() for _ in range(self._cfg.lsh_n_tables)]
            self._bucket_idf = [dict() for _ in range(self._cfg.lsh_n_tables)]
            self._postings_coarse = [dict() for _ in range(self._cfg.lsh_n_tables)]
            self._bucket_idf_coarse = [dict() for _ in range(self._cfg.lsh_n_tables)]
            self._doc_mean = np.zeros((n_docs, 1), dtype=np.float32)
            return

        # -----------------------
        # Pass 1: compute dense doc means + a global centering vector
        # -----------------------
        doc_mean = np.zeros((n_docs, dim), dtype=np.float32)
        has = np.zeros((n_docs,), dtype=np.bool_)
        sum_mu = np.zeros((dim,), dtype=np.float32)
        n_nonempty = 0

        for d_idx in range(n_docs):
            dt = np.asarray(docs.get(d_idx), dtype=np.float32)
            if dt.ndim != 2 or dt.shape[0] == 0:
                continue
            has[d_idx] = True
            mu = dt.mean(axis=0, dtype=np.float32)  # unnormalized mean
            sum_mu += mu
            n_nonempty += 1
            doc_mean[d_idx] = self._l2_normalize_1d(mu)

        if n_nonempty > 0:
            center = (sum_mu / float(n_nonempty)).astype(np.float32, copy=False)
        else:
            center = np.zeros((dim,), dtype=np.float32)
        self._center = center
        self._doc_mean = doc_mean

        # -----------------------
        # PCA hyperplanes (data-adaptive SimHash)
        # -----------------------
        # PCA is done on doc_mean (already normalized), centered by its empirical mean.
        # This yields orthonormal hyperplanes aligned with corpus variance directions.
        rng = np.random.default_rng(0)

        if n_nonempty > 0:
            X = doc_mean[has].astype(np.float32, copy=False)
            m = X.mean(axis=0, dtype=np.float32)
            Xc = (X - m).astype(np.float32, copy=False)
            # Covariance in float64 for stability; dim is small.
            C = (Xc.T.astype(np.float64) @ Xc.astype(np.float64)) / float(max(Xc.shape[0], 1))
            eigvals, eigvecs = np.linalg.eigh(C)  # ascending
            order = np.argsort(eigvals)[::-1]
            V = eigvecs[:, order].astype(np.float32, copy=False)  # (dim, dim)
        else:
            # Fallback: if everything is empty, use random orthonormal-ish hyperplanes.
            V = rng.standard_normal((dim, dim), dtype=np.float32)
            V, _ = np.linalg.qr(V.astype(np.float64))
            V = V.astype(np.float32, copy=False)

        T = int(self._cfg.lsh_n_tables)
        B = int(self._cfg.lsh_n_bits)
        hyper = np.empty((T, B, dim), dtype=np.float32)
        for t in range(T):
            # Choose a block of PCA directions; wrap if T*B > dim.
            start = (t * B) % max(dim, 1)
            idx = (start + np.arange(B, dtype=np.int32)) % max(dim, 1)
            H = V[:, idx].T.astype(np.float32, copy=False)  # (B, dim)

            # Deterministic per-table scrambling (sign flip + permutation) to decorrelate tables.
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(B,), replace=True)
            perm = rng.permutation(B)
            H = (H * signs[:, None])[perm]

            # Ensure unit hyperplanes.
            norms = np.linalg.norm(H, axis=1, keepdims=True).astype(np.float32, copy=False)
            norms = np.maximum(norms, np.float32(1e-6))
            hyper[t] = (H / norms).astype(np.float32, copy=False)

        self._hyperplanes = hyper

        # -----------------------
        # Pass 2: build two-resolution postings using centered PCA-hyperplanes
        # -----------------------
        postings_fine: list[dict[int, list[int]]] = [dict() for _ in range(T)]
        postings_coarse: list[dict[int, list[int]]] = [dict() for _ in range(T)]

        B_fine = int(self._cfg.lsh_n_bits)
        B_coarse = int(min(max(getattr(self._cfg, "lsh_coarse_bits", B_fine), 1), B_fine))
        coarse_mask = int((1 << B_coarse) - 1) if B_coarse < 31 else -1  # B is small in our configs

        for d_idx in range(n_docs):
            dt = np.asarray(docs.get(d_idx), dtype=np.float32)
            if dt.ndim != 2 or dt.shape[0] == 0:
                continue

            dt_sel = self._select_top_tokens_by_norm(dt, int(self._cfg.index_tokens_per_doc))

            for t in range(T):
                H = self._hyperplanes[t]
                codes_fine = self._simhash_codes(dt_sel, H, center=self._center)
                if codes_fine.shape[0] == 0:
                    continue

                # Index a doc once per bucket (set), not once per token, to cap postings blow-up.
                uniq_f = np.unique(codes_fine).astype(np.uint32, copy=False)

                for code in uniq_f.tolist():
                    code_i = int(code)
                    bucket = postings_fine[t].get(code_i)
                    if bucket is None:
                        postings_fine[t][code_i] = [d_idx]
                    else:
                        bucket.append(d_idx)

                # Coarse prefix index: map fine codes to their lower-bit prefix, then unique again.
                if B_coarse < B_fine:
                    uniq_c = np.unique((uniq_f & np.uint32(coarse_mask)).astype(np.uint32, copy=False))
                    for code in uniq_c.tolist():
                        code_i = int(code)
                        bucket = postings_coarse[t].get(code_i)
                        if bucket is None:
                            postings_coarse[t][code_i] = [d_idx]
                        else:
                            bucket.append(d_idx)

        def _freeze_and_idf(
            postings: list[dict[int, list[int]]],
        ) -> tuple[list[dict[int, NDArray[np.int32]]], list[dict[int, np.float32]]]:
            frozen_out: list[dict[int, NDArray[np.int32]]] = []
            idf_out: list[dict[int, np.float32]] = []
            n_docs_f2 = float(max(n_docs, 1))
            for t in range(T):
                table: dict[int, NDArray[np.int32]] = {}
                idf_table: dict[int, np.float32] = {}
                for code, lst in postings[t].items():
                    arr = np.asarray(lst, dtype=np.int32)
                    table[int(code)] = arr
                    df = float(arr.size)
                    idf_table[int(code)] = np.float32(np.log((n_docs_f2 + 1.0) / (df + 1.0)))
                frozen_out.append(table)
                idf_out.append(idf_table)
            return frozen_out, idf_out

        frozen_fine, idf_fine = _freeze_and_idf(postings_fine)
        frozen_coarse, idf_coarse = _freeze_and_idf(postings_coarse)

        self._postings = frozen_fine
        self._bucket_idf = idf_fine
        self._postings_coarse = frozen_coarse
        self._bucket_idf_coarse = idf_coarse

    def _candidate_pool(
        self,
        q_tokens: NDArray[np.floating],
        top_k: int,
    ) -> tuple[NDArray[np.int32], dict[str, float]]:
        """
        Return (candidate_doc_indices, diag_fields_for_query).
        """
        if (
            self._docs is None
            or self._hyperplanes is None
            or self._center is None
            or self._postings is None
            or self._bucket_idf is None
            or self._postings_coarse is None
            or self._bucket_idf_coarse is None
            or self._doc_mean is None
        ):
            return np.arange(0, 0, dtype=np.int32), {}

        docs = self._docs
        n_docs = len(docs)
        if n_docs == 0:
            return np.arange(0, 0, dtype=np.int32), {}

        q = np.asarray(q_tokens, dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            return np.arange(0, 0, dtype=np.int32), {}

        # Rerank budget scales with both requested depth and corpus size.
        rerank_n = int(
            min(
                n_docs,
                max(
                    int(self._cfg.rerank_topk_mult * top_k),
                    int(self._cfg.rerank_sqrt_factor * np.sqrt(n_docs)),
                ),
            )
        )
        rerank_n = max(rerank_n, int(top_k))

        # -----------------------
        # Dense backstop (mean scan)
        # -----------------------
        q_mean = self._l2_normalize_1d(q.mean(axis=0, dtype=np.float32))
        if float(np.linalg.norm(q_mean)) > 0.0:
            mean_scores = (self._doc_mean @ q_mean).astype(np.float32, copy=False)
        else:
            mean_scores = np.zeros((n_docs,), dtype=np.float32)

        mean_pool = int(min(n_docs, max(int(top_k), int(self._cfg.mean_pool_sqrt_factor * np.sqrt(n_docs)))))
        if mean_pool > 0:
            mean_top = np.argpartition(mean_scores, -mean_pool)[-mean_pool:].astype(np.int32, copy=False)
        else:
            mean_top = np.arange(0, 0, dtype=np.int32)

        # -----------------------
        # Progressive-prefix PCA-SimHash voting:
        #   1) fine LSH votes
        #   2) if too few fine hits, widen to coarse prefix votes (down-weighted)
        # -----------------------
        q_sel = self._select_top_tokens_by_norm(q, int(self._cfg.query_tokens_for_lsh))

        counts = np.zeros((n_docs,), dtype=np.float32)

        T = int(self._cfg.lsh_n_tables)
        B_fine = int(self._cfg.lsh_n_bits)
        B_coarse = int(min(max(getattr(self._cfg, "lsh_coarse_bits", B_fine), 1), B_fine))

        k_uncertain_f = int(min(max(self._cfg.multiprobe_k_uncertain, 0), B_fine))
        max_flips_f = int(max(self._cfg.multiprobe_max_flips, 0))

        # Cache per-table code+margin so coarse backoff can reuse the already-computed projections.
        base_codes_by_t: list[NDArray[np.uint32]] = []
        margins_by_t: list[NDArray[np.float32]] = []

        buckets_probed_fine = 0
        buckets_probed_coarse = 0
        used_coarse = 0

        for t in range(T):
            H = self._hyperplanes[t]
            base_codes, margins = self._simhash_codes_and_margins(q_sel, H, center=self._center)
            base_codes_by_t.append(base_codes)
            margins_by_t.append(margins)
            if base_codes.shape[0] == 0:
                continue

            probe_codes: list[int] = []
            for code_u32, marg_row in zip(base_codes.tolist(), margins, strict=True):
                code = int(code_u32)
                probe_codes.append(code)

                if k_uncertain_f <= 0 or max_flips_f <= 0:
                    continue

                bit_idx = np.argpartition(marg_row, k_uncertain_f - 1)[:k_uncertain_f].astype(np.int32, copy=False)

                # 1-flip neighbors
                for bi in bit_idx.tolist():
                    probe_codes.append(code ^ (1 << int(bi)))

                # 2-flip neighbors
                if max_flips_f >= 2 and bit_idx.shape[0] >= 2:
                    lst = bit_idx.tolist()
                    for i in range(len(lst) - 1):
                        mi = 1 << int(lst[i])
                        for j in range(i + 1, len(lst)):
                            probe_codes.append(code ^ mi ^ (1 << int(lst[j])))

            for code in np.unique(np.asarray(probe_codes, dtype=np.uint32)).tolist():
                code_i = int(code)
                arr = self._postings[t].get(code_i)
                buckets_probed_fine += 1
                if arr is None or arr.size == 0:
                    continue
                w = float(self._bucket_idf[t].get(code_i, np.float32(0.0)))
                if w <= 0.0:
                    w = 1.0
                np.add.at(counts, arr, np.float32(w))

        hit_idx_fine = np.flatnonzero(counts).astype(np.int32, copy=False)
        lsh_hits_fine = int(hit_idx_fine.shape[0])

        # Coarse prefix backoff if fine votes don't populate enough unique docs.
        trigger_frac = float(getattr(self._cfg, "coarse_trigger_hit_frac", 0.5))
        if not np.isfinite(trigger_frac):
            trigger_frac = 0.5
        trigger_frac = float(np.clip(trigger_frac, 0.0, 1.0))
        trigger_hits = int(trigger_frac * float(rerank_n))

        if B_coarse < B_fine and lsh_hits_fine < trigger_hits:
            used_coarse = 1
            coarse_mask = np.uint32((1 << B_coarse) - 1) if B_coarse < 31 else np.uint32(0xFFFFFFFF)

            k_uncertain_c = int(min(max(self._cfg.multiprobe_k_uncertain, 0), B_coarse))
            max_flips_c = int(max(self._cfg.multiprobe_max_flips, 0))
            coarse_w = float(getattr(self._cfg, "coarse_vote_weight", 0.5))
            if not np.isfinite(coarse_w):
                coarse_w = 0.5
            coarse_w = float(np.clip(coarse_w, 0.0, 1.0))

            for t in range(T):
                base_codes = base_codes_by_t[t]
                margins = margins_by_t[t]
                if base_codes.shape[0] == 0:
                    continue

                base_c = (base_codes & coarse_mask).astype(np.uint32, copy=False)
                marg_c = margins[:, :B_coarse].astype(np.float32, copy=False)

                probe_codes: list[int] = []
                for code_u32, marg_row in zip(base_c.tolist(), marg_c, strict=True):
                    code = int(code_u32)
                    probe_codes.append(code)

                    if k_uncertain_c <= 0 or max_flips_c <= 0:
                        continue

                    bit_idx = np.argpartition(marg_row, k_uncertain_c - 1)[:k_uncertain_c].astype(np.int32, copy=False)

                    for bi in bit_idx.tolist():
                        probe_codes.append(code ^ (1 << int(bi)))

                    if max_flips_c >= 2 and bit_idx.shape[0] >= 2:
                        lst = bit_idx.tolist()
                        for i in range(len(lst) - 1):
                            mi = 1 << int(lst[i])
                            for j in range(i + 1, len(lst)):
                                probe_codes.append(code ^ mi ^ (1 << int(lst[j])))

                for code in np.unique(np.asarray(probe_codes, dtype=np.uint32)).tolist():
                    code_i = int(code)
                    arr = self._postings_coarse[t].get(code_i)
                    buckets_probed_coarse += 1
                    if arr is None or arr.size == 0:
                        continue
                    w = float(self._bucket_idf_coarse[t].get(code_i, np.float32(0.0)))
                    if w <= 0.0:
                        w = 1.0
                    np.add.at(counts, arr, np.float32(w * coarse_w))

        hit_idx = np.flatnonzero(counts).astype(np.int32, copy=False)
        lsh_hits = int(hit_idx.shape[0])

        # Keep only the strongest LSH hits before unioning with mean-top.
        lsh_keep = int(min(n_docs, max(rerank_n, int(self._cfg.lsh_keep_mult * rerank_n))))
        if lsh_hits > lsh_keep:
            hit_scores = counts[hit_idx].astype(np.float32, copy=False)
            top_local = np.argpartition(hit_scores, -lsh_keep)[-lsh_keep:]
            hit_idx = hit_idx[top_local]
            hit_scores = counts[hit_idx].astype(np.float32, copy=False)
            order = np.lexsort((hit_idx, -hit_scores))
            lsh_top = hit_idx[order].astype(np.int32, copy=False)
        else:
            lsh_top = hit_idx

        # Union (small) then trim to rerank_n with a combined pre-score:
        # pre = (IDF-weighted bucket evidence) + (mean cosine similarity).
        union = np.unique(np.concatenate([lsh_top, mean_top])).astype(np.int32, copy=False)
        if union.shape[0] > rerank_n:
            # Conservative combination: let sparse bucket evidence dominate, and only
            # use the dense mean similarity as a positive backstop (never as a strong
            # negative filter).
            pre = counts[union].astype(np.float32, copy=False) + np.maximum(np.float32(0.0), mean_scores[union])
            order = np.lexsort((union, -pre))
            union = union[order[:rerank_n]].astype(np.int32, copy=False)

        diag = {
            "rerank_n": float(rerank_n),
            "mean_pool": float(mean_pool),
            "lsh_query_tokens": float(q_sel.shape[0]),
            "lsh_buckets_probed_fine": float(buckets_probed_fine),
            "lsh_hit_docs_fine": float(lsh_hits_fine),
            "lsh_used_coarse": float(used_coarse),
            "lsh_buckets_probed_coarse": float(buckets_probed_coarse),
            "lsh_hit_docs_total": float(lsh_hits),
            "candidates_returned": float(union.shape[0]),
        }
        return union, diag

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        if self._docs is None:
            raise RuntimeError("build(docs) must be called before search()")
        docs = self._docs
        n_docs = len(docs)

        rankings: dict[str, list[tuple[str, float]]] = {}
        diagnostics: dict[str, dict[str, int]] = {}

        for q_idx, query_id in enumerate(queries.ids):
            q_tokens = np.asarray(queries.get(q_idx), dtype=np.float32)

            # Stage 1: cheap candidate generation.
            cand_idx, cand_diag = self._candidate_pool(q_tokens, top_k)
            cand_idx = np.asarray(cand_idx, dtype=np.int32)

            # Stage 2: exact MaxSim on the shortlist (full query tokens).
            cand_scores = np.empty(cand_idx.shape[0], dtype=np.float32)
            doc_tokens_loaded = 0
            cand_doc_ids: list[str] = []
            for i, d_idx in enumerate(cand_idx.tolist()):
                d_tokens = np.asarray(docs.get(int(d_idx)), dtype=np.float32)
                doc_tokens_loaded += int(d_tokens.shape[0])
                cand_scores[i] = _maxsim_score(q_tokens, d_tokens)
                cand_doc_ids.append(docs.ids[int(d_idx)])

            rankings[query_id] = _top_k_from_scores(cand_doc_ids, cand_scores, top_k)

            documents_scored = int(cand_idx.shape[0])
            corpus_coverage = (documents_scored / n_docs) if n_docs else 0.0

            row: dict[str, int] = {
                "documents_scored": documents_scored,
                "corpus_size": int(n_docs),
                "corpus_coverage": float(corpus_coverage),
                "query_tokens_used": int(q_tokens.shape[0]),
                "document_tokens_loaded": int(doc_tokens_loaded),
            }
            # Extra diagnostics (kept numeric; evaluator will surface them).
            for k, v in cand_diag.items():
                row[k] = float(v)
            diagnostics[query_id] = row

        self.last_diagnostics = diagnostics
        return rankings


__all__ = ["LateInteractionRetriever", "Config"]
