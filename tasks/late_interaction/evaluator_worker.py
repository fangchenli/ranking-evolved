"""Worker utilities for late-interaction evaluation."""

from __future__ import annotations

# ruff: noqa: I001

# Import _runtime first so BLAS pinning takes effect before numpy/torch import.
from tasks.late_interaction import _runtime  # noqa: F401

import gc
import importlib.util
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks._shared.metrics import (
    alpha_ndcg_at_k,
    aspect_recall_at_k,
    graded_ndcg_at_k,
    ndcg_at_k,
    recall_at_k,
)
from tasks.late_interaction.embedding_cache import TokenEmbeddingStore, load_embedding_cache
from tasks.late_interaction.library import ExactMaxSimRetriever


@dataclass(frozen=True)
class WorkerResult:
    dataset: str
    recall_at_k: float
    ndcg_at_k: float
    recall_k: int
    ndcg_k: int
    qrels_mode: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    build_time_ms: float
    num_queries: int
    warmup_queries: int
    timed_repeats: int
    documents_scored: float
    query_tokens_used: float
    document_tokens_loaded: float
    # `corpus_coverage = documents_scored / corpus_size`, averaged across
    # measured queries. Surfaces "approximations" that touch most of the
    # corpus and so cannot be faster than exact MaxSim. 0.0 when the
    # candidate doesn't report it.
    corpus_size: float = 0.0
    corpus_coverage: float = 0.0
    # Hardware fingerprint captured during the run; embedded so any downstream
    # consumer can refuse to mix CPU and GPU latency numbers silently.
    fingerprint: dict[str, Any] = field(default_factory=dict)
    # Recall at additional k (10, 100, ...) — populated when `extra_recall_ks`
    # is passed to `evaluate_cache_dataset`. Empty by default.
    extra_recall: dict[int, float] = field(default_factory=dict)
    extra_metrics: dict[str, float] = field(default_factory=dict)
    cache_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recall_at_1000(self) -> float:
        return self.recall_at_k

    @property
    def ndcg_at_10(self) -> float:
        return self.ndcg_at_k

    def to_metrics(self) -> dict[str, float]:
        # `query_latency_median_ms` is the framework-level key the
        # latency-aware objective reads; alias it to p50 here so the
        # late-interaction worker plays the same role bm25's worker plays.
        out: dict[str, float] = {
            f"{self.dataset}_recall_at_{self.recall_k}": self.recall_at_k,
            f"{self.dataset}_ndcg_at_{self.ndcg_k}": self.ndcg_at_k,
            f"{self.dataset}_latency_p50_ms": self.latency_p50_ms,
            f"{self.dataset}_latency_p95_ms": self.latency_p95_ms,
            f"{self.dataset}_latency_mean_ms": self.latency_mean_ms,
            f"{self.dataset}_query_latency_median_ms": self.latency_p50_ms,
            f"{self.dataset}_build_time_ms": self.build_time_ms,
            f"{self.dataset}_num_queries": float(self.num_queries),
            f"{self.dataset}_warmup_queries": float(self.warmup_queries),
            f"{self.dataset}_timed_repeats": float(self.timed_repeats),
            f"{self.dataset}_documents_scored": self.documents_scored,
            f"{self.dataset}_query_tokens_used": self.query_tokens_used,
            f"{self.dataset}_document_tokens_loaded": self.document_tokens_loaded,
            f"{self.dataset}_corpus_size": self.corpus_size,
            f"{self.dataset}_corpus_coverage": self.corpus_coverage,
        }
        for k, v in self.extra_recall.items():
            out[f"{self.dataset}_recall_at_{k}"] = float(v)
        for key, value in self.extra_metrics.items():
            out[f"{self.dataset}_{key}"] = float(value)
        return out


def evaluate_cache_dataset(
    *,
    cache_dir: str | Path,
    program_path: str | Path | None,
    sample_queries: int,
    recall_k: int,
    ndcg_k: int,
    warmup_queries: int,
    timed_repeats: int | None = None,
    retriever_factory: Callable[[], Any] | None = None,
    extra_recall_ks: tuple[int, ...] = (),
    qrels_mode: str = "gold",
    aspect_alpha: float = 0.5,
    progress: bool = False,
) -> WorkerResult:
    """Evaluate one cached dataset with exact/supplied late-interaction retrieval.

    Fairness-relevant arguments:
      timed_repeats:     run `search` N times per measured query and take the
                         median (default: env `EVAL_TIMED_REPEATS` else 3).
                         Reduces single-call variance without inflating the
                         mean when a query has occasional outliers.
      retriever_factory: optional callable returning a pre-built retriever
                         instance (used by `compare_baselines` to inject the
                         FastPLAID adapter through the same harness). When
                         provided, `program_path` is ignored.
      extra_recall_ks:   compute recall@k for each k in this tuple in addition
                         to recall@`recall_k`. Useful for k-mismatch sanity in
                         baseline comparisons.
    """

    if timed_repeats is None:
        timed_repeats = int(os.environ.get("EVAL_TIMED_REPEATS", "3"))
    if timed_repeats < 1:
        raise ValueError(f"timed_repeats must be >= 1, got {timed_repeats}")

    cache = load_embedding_cache(cache_dir, qrels_mode=qrels_mode)
    total_available = len(cache.queries)

    # Unified warmup rule: warmup queries are always a prefix of the measured
    # set. We run them once untimed (to absorb page faults, lazy CUDA init,
    # k-means deferred initialization, etc.) then run the WHOLE measured set
    # timed — including the warmup queries themselves on the second pass.
    #
    # This is the same rule for sample mode (`sample_queries < total`) and
    # full-eval mode (`sample_queries >= total`); the only difference is how
    # many queries land in the measured set.
    measured_query_count = min(sample_queries, total_available)
    warmup_count = min(warmup_queries, measured_query_count)

    selected_queries = slice_query_store(cache.queries, 0, measured_query_count)
    measured_query_ids = selected_queries.ids

    if retriever_factory is not None:
        retriever = retriever_factory()
    else:
        retriever = load_retriever(program_path)

    start_build = time.perf_counter()
    retriever.build(cache.docs)
    build_time_ms = (time.perf_counter() - start_build) * 1000.0

    top_k = max(recall_k, ndcg_k)
    max_excluded = max(
        (len(cache.excluded_ids.get(query_id, [])) for query_id in measured_query_ids),
        default=0,
    )
    search_top_k = min(len(cache.docs), top_k + max_excluded)
    latencies_ms: list[float] = []
    rankings: dict[str, list[tuple[str, float]]] = {}

    # Fairness invariants 8 & 10: drain pending GPU work before timing, and
    # disable Python GC inside the timed region to remove bimodal tails.
    _maybe_cuda_sync()
    gc.collect()
    gc.disable()
    try:
        # Pass 1: warmup, untimed. Same first `warmup_count` queries; results
        # discarded. Absorbs page faults, lazy CUDA init, k-means deferred
        # initialization.
        for warmup_idx in range(warmup_count):
            wq = slice_query_store(selected_queries, warmup_idx, warmup_idx + 1)
            _maybe_cuda_sync()
            retriever.search(wq, top_k=search_top_k)
            _maybe_cuda_sync()

        # Pass 2: timed pass over the WHOLE measured set (including the
        # warmup-prefix queries on this second pass). Each query is called
        # `timed_repeats` times; per-query latency is the median of those.
        timed_iter: Iterable[int] = range(measured_query_count)
        if progress:
            timed_iter = _maybe_tqdm(
                timed_iter,
                total=measured_query_count,
                desc=f"{cache.metadata.dataset_name} timed",
                unit="q",
            )
        for query_index in timed_iter:
            one_query = slice_query_store(selected_queries, query_index, query_index + 1)
            per_call_ms: list[float] = []
            last_ranking: dict[str, list[tuple[str, float]]] | None = None
            for _ in range(timed_repeats):
                _maybe_cuda_sync()
                start = time.perf_counter()
                last_ranking = retriever.search(one_query, top_k=search_top_k)
                _maybe_cuda_sync()
                per_call_ms.append((time.perf_counter() - start) * 1000.0)
            latencies_ms.append(float(statistics.median(per_call_ms)))
            assert last_ranking is not None
            query_id = selected_queries.ids[query_index]
            rankings[query_id] = _filter_excluded(
                last_ranking[query_id],
                excluded_ids=set(cache.excluded_ids.get(query_id, [])),
                top_k=top_k,
            )
            # Live p50 update on the bar (every query is cheap).
            if progress and hasattr(timed_iter, "set_postfix"):
                timed_iter.set_postfix(  # type: ignore[attr-defined]
                    p50=f"{statistics.median(latencies_ms):.1f}ms",
                    refresh=False,
                )
    finally:
        gc.enable()

    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    graded_ndcg_scores: list[float] = []
    alpha_ndcg_scores: list[float] = []
    aspect_recall_scores: list[float] = []
    extra_recall_scores: dict[int, list[float]] = {k: [] for k in extra_recall_ks}
    for query_id in measured_query_ids:
        qrel = cache.qrels.get(query_id, {})
        relevant = relevant_indices(qrel, cache.docs)
        retrieved = retrieved_indices(rankings[query_id], cache.docs)
        retrieved_docs = [doc_id for doc_id, _score in rankings[query_id]]
        recall_scores.append(recall_at_k(relevant, retrieved, recall_k))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, ndcg_k))
        graded_ndcg_scores.append(graded_ndcg_at_k(qrel, retrieved_docs, ndcg_k))
        aspect = _aspect_for_query(cache.aspect_annotations, query_id)
        if aspect is not None:
            doc_to_aspect, aspect_weights = aspect
            alpha_ndcg_scores.append(
                alpha_ndcg_at_k(
                    retrieved_docs,
                    doc_to_aspect,
                    aspect_weights,
                    ndcg_k,
                    alpha=aspect_alpha,
                )
            )
            aspect_recall_scores.append(
                aspect_recall_at_k(retrieved_docs, doc_to_aspect, aspect_weights, recall_k)
            )
        for k in extra_recall_ks:
            extra_recall_scores[k].append(recall_at_k(relevant, retrieved, k))

    diagnostics = getattr(retriever, "last_diagnostics", {})
    measured_diagnostics = [diagnostics.get(query_id) for query_id in measured_query_ids]
    measured_diagnostics = [diag for diag in measured_diagnostics if diag is not None]

    extra_metrics: dict[str, float] = {
        f"graded_ndcg_at_{ndcg_k}": float(np.mean(graded_ndcg_scores)) if graded_ndcg_scores else 0.0,
    }
    if alpha_ndcg_scores:
        extra_metrics[f"alpha_ndcg_at_{ndcg_k}"] = float(np.mean(alpha_ndcg_scores))
    if aspect_recall_scores:
        extra_metrics[f"aspect_recall_at_{recall_k}"] = float(np.mean(aspect_recall_scores))

    return WorkerResult(
        dataset=cache.metadata.dataset_name,
        recall_at_k=float(np.mean(recall_scores)) if recall_scores else 0.0,
        ndcg_at_k=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        qrels_mode=cache.qrels_mode,
        latency_p50_ms=float(statistics.median(latencies_ms)) if latencies_ms else 0.0,
        latency_p95_ms=percentile(latencies_ms, 95.0),
        latency_mean_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        build_time_ms=float(build_time_ms),
        num_queries=len(measured_query_ids),
        warmup_queries=warmup_count,
        timed_repeats=timed_repeats,
        documents_scored=_diagnostic_mean(measured_diagnostics, "documents_scored"),
        query_tokens_used=_diagnostic_mean(measured_diagnostics, "query_tokens_used"),
        document_tokens_loaded=_diagnostic_mean(measured_diagnostics, "document_tokens_loaded"),
        corpus_size=_diagnostic_mean(measured_diagnostics, "corpus_size"),
        corpus_coverage=_diagnostic_mean(measured_diagnostics, "corpus_coverage"),
        fingerprint=_runtime.runtime_fingerprint(),
        extra_recall={k: (float(np.mean(v)) if v else 0.0) for k, v in extra_recall_scores.items()},
        extra_metrics=extra_metrics,
        cache_metadata=dict(cache.metadata.__dict__),
    )


def _maybe_cuda_sync() -> None:
    """Block until pending CUDA kernels complete; no-op on CPU (invariant 8/9)."""
    if _runtime.resolve_device() != "cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _maybe_tqdm(iterable: Iterable, **kwargs: Any) -> Iterable:
    """Wrap `iterable` in a tqdm progress bar when stderr is a TTY.

    Uses `tqdm.auto` so notebooks get the rich widget and terminals get the
    text bar; auto-disables when stderr is redirected so logs stay clean.
    Falls back to the bare iterable when tqdm is not installed.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, disable=not sys.stderr.isatty(), leave=False, **kwargs)


def _diagnostic_mean(items: list[Any], attr: str) -> float:
    """Mean over diagnostics that may be dataclass objects or plain dicts."""
    values: list[float] = []
    for item in items:
        if hasattr(item, attr):
            values.append(float(getattr(item, attr)))
        elif isinstance(item, dict) and attr in item:
            values.append(float(item[attr]))
    return float(np.mean(values)) if values else 0.0


def load_retriever(program_path: str | Path | None) -> Any:
    if program_path is None or str(program_path) == "":
        return ExactMaxSimRetriever()

    path = Path(program_path)
    spec = importlib.util.spec_from_file_location("candidate_late_interaction", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: required for `@dataclass`
    # in the loaded file. dataclass.__post_init__ does `sys.modules.get(
    # cls.__module__).__dict__` and crashes with AttributeError on None when
    # the module isn't registered. (Same fix as tasks/bm25/evaluator.py:230.)
    sys.modules["candidate_late_interaction"] = module
    spec.loader.exec_module(module)

    retriever_cls = getattr(module, "LateInteractionRetriever", None)
    if retriever_cls is None:
        raise AttributeError("candidate must define LateInteractionRetriever")
    return retriever_cls()


def slice_query_store(store: TokenEmbeddingStore, start: int, end: int) -> TokenEmbeddingStore:
    ids = store.ids[start:end]
    arrays = [store.get(index) for index in range(start, end)]
    lengths = np.asarray([arr.shape[0] for arr in arrays], dtype=np.int64)
    offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    if arrays:
        embeddings = np.concatenate(arrays, axis=0).astype(store.embeddings.dtype, copy=False)
    else:
        embeddings = np.empty((0, store.embedding_dim), dtype=store.embeddings.dtype)
    return TokenEmbeddingStore(ids=ids, embeddings=embeddings, lengths=lengths, offsets=offsets)


def relevant_indices(qrels: dict[str, int], docs: TokenEmbeddingStore) -> np.ndarray:
    indices = [
        docs.index_of(doc_id) for doc_id, rel in qrels.items() if rel > 0 and doc_id in docs.ids
    ]
    return np.asarray(indices, dtype=np.int64)


def retrieved_indices(ranking: list[tuple[str, float]], docs: TokenEmbeddingStore) -> np.ndarray:
    return np.asarray([docs.index_of(doc_id) for doc_id, _score in ranking], dtype=np.int64)


def _filter_excluded(
    ranking: list[tuple[str, float]],
    *,
    excluded_ids: set[str],
    top_k: int,
) -> list[tuple[str, float]]:
    if not excluded_ids:
        return ranking[:top_k]
    return [(doc_id, score) for doc_id, score in ranking if doc_id not in excluded_ids][:top_k]


def _aspect_for_query(
    annotations: dict[str, Any] | None,
    query_id: str,
) -> tuple[dict[str, str], dict[str, float]] | None:
    if not annotations:
        return None
    doc_to_aspect = (annotations.get("query_doc_to_aspect") or {}).get(query_id, {})
    aspect_weights = (annotations.get("query_aspect_weights") or {}).get(query_id, {})
    if not isinstance(doc_to_aspect, dict) or not isinstance(aspect_weights, dict):
        return None
    if not doc_to_aspect or not aspect_weights:
        return None
    return (
        {str(doc_id): str(aspect_id) for doc_id, aspect_id in doc_to_aspect.items()},
        {str(aspect_id): float(weight) for aspect_id, weight in aspect_weights.items()},
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))
