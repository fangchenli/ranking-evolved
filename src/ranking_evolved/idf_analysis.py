"""Utilities for computing and comparing arbitrary IDF variants."""

from collections import Counter
from typing import Callable, Iterable

import numpy as np

from ranking_evolved.bm25_biology import Corpus


def compute_idf(
    corpus: Corpus, idf_func: Callable[[np.ndarray, int], np.ndarray]
) -> dict[str, float]:
    """
    Apply an arbitrary IDF function to a corpus.

    Args:
        corpus: Corpus with document_frequency.
        idf_func: Callable that accepts (corpus, df_array, N_docs) and returns a NumPy array.

    Returns:
        Dict mapping term -> idf value.
    """
    df_counter: Counter[str] = corpus.document_frequency
    terms: list[str] = list(df_counter.keys())
    df_array = np.array([df_counter[t] for t in terms], dtype=np.float32)
    idf_array = idf_func(df_array, len(corpus))
    return {t: float(v) for t, v in zip(terms, idf_array)}


def compare_idf(
    corpus: Corpus,
    idf_func_a: Callable[[np.ndarray, int], np.ndarray],
    idf_func_b: Callable[[np.ndarray, int], np.ndarray],
    top_k: int = 10,
) -> dict:
    """
    Compare two IDF variants.

    Returns:
        {
          "per_term": {term: {"a": ..., "b": ..., "delta": ...}},
          "top_increases": [(term, delta, a, b)],
          "top_decreases": [(term, delta, a, b)],
          "stats": {"mean": ..., "std": ..., "min": ..., "max": ...},
        }
    """
    idf_a = compute_idf(corpus, idf_func_a)
    idf_b = compute_idf(corpus, idf_func_b)

    per_term = {}
    deltas = []
    for term in idf_a:
        a = idf_a[term]
        b = idf_b.get(term, 0.0)
        delta = b - a
        per_term[term] = {"a": a, "b": b, "delta": delta}
        deltas.append((term, delta, a, b))

    deltas_sorted = sorted(deltas, key=lambda x: x[1], reverse=True)
    increases = deltas_sorted[:top_k]
    decreases = list(reversed(deltas_sorted[-top_k:]))

    delta_vals = np.array([d[1] for d in deltas], dtype=np.float32)
    stats = {
        "mean": float(delta_vals.mean()) if delta_vals.size else 0.0,
        "std": float(delta_vals.std()) if delta_vals.size else 0.0,
        "min": float(delta_vals.min()) if delta_vals.size else 0.0,
        "max": float(delta_vals.max()) if delta_vals.size else 0.0,
    }

    return {
        "per_term": per_term,
        "top_increases": increases,
        "top_decreases": decreases,
        "stats": stats,
    }


def idf_histogram_data(idf_map: dict[str, float]) -> dict:
    """
    Prepare histogram-friendly data from an IDF map (no plotting).
    """
    values = sorted(float(v) for v in idf_map.values())
    arr = np.array(values, dtype=np.float32)
    return {
        "values": values,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def idf_profile_for_query(
    query_terms: Iterable[str],
    idf_a: dict[str, float],
    idf_b: dict[str, float],
) -> list[dict[str, float | str]]:
    """
    Per-term IDF shift for a query between two IDF maps.
    """
    profile = []
    for term in query_terms:
        a = idf_a.get(term, 0.0)
        b = idf_b.get(term, 0.0)
        profile.append({"term": term, "idf_a": a, "idf_b": b, "delta": b - a})
    return profile


def idf_sensitivity_report(
    corpus: Corpus,
    idf_func_a: Callable[[np.ndarray, int], np.ndarray],
    idf_func_b: Callable[[np.ndarray, int], np.ndarray],
    delta_threshold: float = 0.1,
    rare_thresh: float = 0.01,
    common_thresh: float = 0.2,
) -> dict:
    """
    High-level summary of how IDF changes across the corpus.
    """
    cmp = compare_idf(corpus, idf_func_a, idf_func_b, top_k=10)

    df_counter: Counter[str] = corpus.document_frequency
    N = len(corpus) or 1
    deltas = []
    rare_shift = []
    mid_shift = []
    common_shift = []
    clip_counts = {"min_a": 0, "max_a": 0, "min_b": 0, "max_b": 0}

    idf_a = compute_idf(corpus, idf_func_a)
    idf_b = compute_idf(corpus, idf_func_b)
    vals_a = np.array(list(idf_a.values()), dtype=np.float32)
    vals_b = np.array(list(idf_b.values()), dtype=np.float32)
    min_a, max_a = float(vals_a.min()), float(vals_a.max())
    min_b, max_b = float(vals_b.min()), float(vals_b.max())

    for term, df in df_counter.items():
        a = idf_a.get(term, 0.0)
        b = idf_b.get(term, 0.0)
        delta = b - a
        deltas.append(delta)

        df_ratio = df / N
        if df_ratio < rare_thresh:
            rare_shift.append(delta)
        elif df_ratio > common_thresh:
            common_shift.append(delta)
        else:
            mid_shift.append(delta)

        if a == min_a:
            clip_counts["min_a"] += 1
        if a == max_a:
            clip_counts["max_a"] += 1
        if b == min_b:
            clip_counts["min_b"] += 1
        if b == max_b:
            clip_counts["max_b"] += 1

    delta_arr = np.array(deltas, dtype=np.float32)
    stats = {
        "mean": float(delta_arr.mean()) if delta_arr.size else 0.0,
        "std": float(delta_arr.std()) if delta_arr.size else 0.0,
        "min": float(delta_arr.min()) if delta_arr.size else 0.0,
        "max": float(delta_arr.max()) if delta_arr.size else 0.0,
        "pct_gt_threshold": float((np.abs(delta_arr) > delta_threshold).mean())
        if delta_arr.size
        else 0.0,
    }

    def _agg(arr: list[float]) -> float:
        return float(np.mean(arr)) if arr else 0.0

    return {
        "rare_terms_shift": _agg(rare_shift),
        "mid_terms_shift": _agg(mid_shift),
        "common_terms_shift": _agg(common_shift),
        "clip_counts": clip_counts,
        "global_stats": stats,
        "top_increases": cmp["top_increases"],
        "top_decreases": cmp["top_decreases"],
    }
