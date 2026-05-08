"""FastPLAID as an evaluator-importable `LateInteractionRetriever` program.

This program goes through the **lower-level** `fast_plaid.search.FastPlaid`
API (not `pylate.indexes.FastPlaid`) so the per-call SqliteDict lookup that
PyLate's wrapper performs is excluded from the timed region — the doc_id
mapping happens once during `build()` and is dereferenced after `search()`
returns. This matches how exact MaxSim and any evolved candidate work: the
timed region is the scoring kernel, not the doc_id bookkeeping.

Use the same evaluator CLI you'd use for any program:

    EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \\
        --program tasks/late_interaction/programs/fastplaid.py \\
        --datasets beir_scifact

Build-time work (kmeans, 4-bit residual quantization, IVF posting lists)
happens entirely in `build()`, which the evaluator times under
`build_time_ms`. `search()` only invokes `FastPlaid.search` and the
post-conversion to `dict[str, list[tuple[str, float]]]`.

Constructor args (with sensible defaults that mirror PyLate's `FastPlaid`):

    nbits:           4   (4-bit residual quantization)
    kmeans_niters:   4
    n_ivf_probe:     8
    n_full_scores:   8192

Device: by default reads from `tasks.late_interaction._runtime.DEVICE`. Pass
`device="cpu"` or `device="cuda"` to override.
"""
from __future__ import annotations

# IMPORTANT: import _runtime first so BLAS pinning takes effect before
# numpy/torch are imported by the FastPLAID dependency chain.
from tasks.late_interaction import _runtime  # noqa: F401

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore


@dataclass(frozen=True)
class FastPlaidParams:
    nbits: int = 4
    kmeans_niters: int = 4
    n_ivf_probe: int = 8
    n_full_scores: int = 8192
    # n_processes=1 is the fairness invariant; enforced unconditionally.
    # batch_size affects FastPLAID's internal batching for one search call;
    # we always pass a single query, so the value just bounds the kernel.
    batch_size: int = 1 << 18


class LateInteractionRetriever:
    """Protocol implementation backed by FastPLAID.

    Use:

        retriever = LateInteractionRetriever()  # device from _runtime.DEVICE
        retriever.build(cache.docs)
        rankings = retriever.search(one_query, top_k=1000)

    The `build` step performs all index construction (kmeans, residual
    quantization, posting lists). `search` only invokes the FastPLAID search
    kernel and converts integer plaid IDs to string doc IDs from the doc
    store the adapter was built on.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        params: FastPlaidParams | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.device = device or _runtime.resolve_device()
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"FastPLAID retriever device must be cpu or cuda, got {self.device!r}")
        self.params = params or FastPlaidParams()

        # Index directory: caller-provided (persistent) or a temp dir we own.
        if index_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="late_interaction_fastplaid_")
            self._index_dir = Path(self._tempdir.name)
        else:
            self._tempdir = None
            self._index_dir = Path(index_dir)
            self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index: Any = None
        self._doc_ids: list[str] | None = None
        # Diagnostics surface used by the evaluator (mirror ExactMaxSimRetriever shape).
        self.last_diagnostics: dict[str, Any] = {}

    # -- Protocol --------------------------------------------------------

    def build(self, docs: TokenEmbeddingStore) -> None:
        """Build the FastPLAID index from `docs`. Times under `build_time_ms`."""
        from fast_plaid import search as fps  # type: ignore

        # FastPLAID expects per-document torch tensors of shape (n_tokens, dim).
        # The cache stores fp16; FastPLAID accepts fp16 or fp32. We pass fp16
        # to avoid an unnecessary 2x copy.
        doc_tensors: list[torch.Tensor] = [
            torch.from_numpy(np.asarray(docs.get(i))) for i in range(len(docs))
        ]

        # Wipe the index dir if anything's there from a prior build (FastPLAID
        # appends, and we want fresh kmeans every build).
        for child in self._index_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass

        index_path = self._index_dir / "fast_plaid_index"
        index_path.mkdir(parents=True, exist_ok=True)
        self._index = fps.FastPlaid(index=str(index_path), device=self.device)
        self._index.create(
            documents_embeddings=doc_tensors,
            kmeans_niters=self.params.kmeans_niters,
            nbits=self.params.nbits,
        )
        self._doc_ids = list(docs.ids)

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Score each query and return ranked `(doc_id, score)` lists.

        Per-query call shape: one query at a time, matching the harness.
        """
        if self._index is None or self._doc_ids is None:
            raise RuntimeError("build(docs) must be called before search()")

        rankings: dict[str, list[tuple[str, float]]] = {}
        diagnostics: dict[str, Any] = {}

        for q_idx, query_id in enumerate(queries.ids):
            q_tokens = np.asarray(queries.get(q_idx))  # (n_qtok, dim) fp16
            # Shape (1, n_qtok, dim) — single-query batch, fp16.
            q_tensor = torch.from_numpy(q_tokens).unsqueeze(0)

            results = self._index.search(
                queries_embeddings=q_tensor,
                top_k=top_k,
                batch_size=self.params.batch_size,
                n_full_scores=self.params.n_full_scores,
                n_ivf_probe=self.params.n_ivf_probe,
                show_progress=False,
                n_processes=1,  # fairness invariant 4
            )
            # `results` is list-of-list; one entry per query, each a list of
            # `(plaid_id, score)` tuples. Map plaid_id -> doc_id.
            ranked = [
                (self._doc_ids[plaid_id], float(score))
                for plaid_id, score in results[0]
                if 0 <= plaid_id < len(self._doc_ids)
            ]
            rankings[query_id] = ranked
            diagnostics[query_id] = {
                "documents_scored": len(ranked),
                "query_tokens_used": int(q_tokens.shape[0]),
                "n_full_scores": self.params.n_full_scores,
                "n_ivf_probe": self.params.n_ivf_probe,
            }

        self.last_diagnostics = diagnostics
        return rankings

    # -- cleanup ---------------------------------------------------------

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# Back-compat alias for code that still imports FastPlaidRetriever.
FastPlaidRetriever = LateInteractionRetriever

__all__ = ["LateInteractionRetriever", "FastPlaidRetriever", "FastPlaidParams"]
