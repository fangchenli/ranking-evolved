"""Tests for the BM25 retrieval objective math.

The evolution-algorithm invariants (sampling, admission, migration) are
tested separately in tests/search/test_evolution_algo_invariants.py and
do not depend on this objective.
"""
from __future__ import annotations

import math

import pytest

from ranking_evolved.config.objective import (
    LatencyConfig,
    ObjectiveConfig,
    ObjectiveWeights,
)
from ranking_evolved.evaluation.objective_math import (
    compute_objective,
    is_hard_slowdown,
    latency_score_from_ratio,
)


def test_latency_transform_ratio_one():
    """Equal latency to the baseline → score 0.5."""
    assert latency_score_from_ratio(1.0) == pytest.approx(0.5)


def test_latency_transform_ratio_half():
    """Twice as fast as baseline → score 1 / 1.5."""
    assert latency_score_from_ratio(0.5) == pytest.approx(1.0 / 1.5)


def test_latency_transform_ratio_four():
    """Four times slower than baseline → score 0.2."""
    assert latency_score_from_ratio(4.0) == pytest.approx(0.2)


def test_latency_transform_rejects_negative():
    with pytest.raises(ValueError):
        latency_score_from_ratio(-0.1)


def test_hard_slowdown_triggered_at_5x():
    """501 ms vs 100 ms baseline at threshold 5.0 → triggered."""
    assert is_hard_slowdown(501.0, 100.0, 5.0) is True


def test_hard_slowdown_not_triggered_below_threshold():
    """500 ms vs 100 ms baseline at threshold 5.0 → not triggered (5.0 == 5.0)."""
    assert is_hard_slowdown(500.0, 100.0, 5.0) is False


def test_hard_slowdown_zero_baseline_never_triggers():
    """Zero baseline (no usable comparison) must not trip the penalty."""
    assert is_hard_slowdown(1000.0, 0.0, 5.0) is False


def test_combined_score_known_values():
    """0.45*0.6 + 0.20*0.2 + 0.35*0.5 = 0.485 (the exact spec example)."""
    cfg = _latency_aware_cfg(weights=ObjectiveWeights(recall=0.45, ndcg=0.20, latency=0.35))
    per_dataset = {
        "ds_a": {"recall@1000": 0.6, "ndcg@10": 0.2, "query_latency_median_ms": 100.0},
        "ds_b": {"recall@1000": 0.6, "ndcg@10": 0.2, "query_latency_median_ms": 100.0},
    }
    baseline = {"ds_a": 100.0, "ds_b": 100.0}  # ratio=1 → latency_score=0.5

    outcome = compute_objective(per_dataset, baseline, cfg)

    assert outcome.avg_recall == pytest.approx(0.6)
    assert outcome.avg_ndcg == pytest.approx(0.2)
    assert outcome.avg_latency_score == pytest.approx(0.5)
    assert outcome.combined_score == pytest.approx(0.485)
    assert outcome.objective_recall_component == pytest.approx(0.45 * 0.6)
    assert outcome.objective_ndcg_component == pytest.approx(0.20 * 0.2)
    assert outcome.objective_latency_component == pytest.approx(0.35 * 0.5)
    assert outcome.latency_penalty_triggered == 0.0


def test_hard_slowdown_zeroes_score_for_offending_dataset():
    """Triggered: latency_score_d == 0.0, latency_penalty_triggered_d == 1.0."""
    cfg = _latency_aware_cfg()
    per_dataset = {
        # ratio = 1000 / 100 = 10 > 5.0 → trigger
        "slow_ds": {"recall@1000": 0.6, "ndcg@10": 0.2, "query_latency_median_ms": 1000.0},
        # ratio = 100 / 100 = 1 → score 0.5
        "fast_ds": {"recall@1000": 0.6, "ndcg@10": 0.2, "query_latency_median_ms": 100.0},
    }
    baseline = {"slow_ds": 100.0, "fast_ds": 100.0}

    outcome = compute_objective(per_dataset, baseline, cfg)

    assert outcome.per_dataset["slow_ds"]["latency_score"] == 0.0
    assert outcome.per_dataset["slow_ds"]["latency_penalty_triggered"] == 1.0
    assert outcome.per_dataset["fast_ds"]["latency_score"] == pytest.approx(0.5)
    assert outcome.per_dataset["fast_ds"]["latency_penalty_triggered"] == 0.0
    assert outcome.latency_penalty_triggered == 1.0
    # Aggregate: (0.0 + 0.5) / 2 = 0.25
    assert outcome.avg_latency_score == pytest.approx(0.25)


def test_legacy_objective_unchanged():
    """name=recall100_ndcg10 with weights 0.8/0.2/0.0: combined = 0.8r + 0.2n.

    No latency lookup is required even if baseline is None.
    """
    cfg = ObjectiveConfig(
        name="recall100_ndcg10",
        recall_k=100,
        ndcg_k=10,
        weights=ObjectiveWeights(recall=0.8, ndcg=0.2, latency=0.0),
        latency=LatencyConfig(enabled=False),
    )
    per_dataset = {
        "ds_a": {"recall@100": 0.5, "ndcg@10": 0.4},
        "ds_b": {"recall@100": 0.7, "ndcg@10": 0.6},
    }

    outcome = compute_objective(per_dataset, baseline_latency_by_dataset=None, cfg=cfg)

    expected = 0.8 * 0.6 + 0.2 * 0.5  # avg_recall=0.6, avg_ndcg=0.5
    assert outcome.combined_score == pytest.approx(expected)
    assert outcome.objective_latency_component == 0.0
    assert outcome.latency_penalty_triggered == 0.0


def _latency_aware_cfg(weights: ObjectiveWeights | None = None) -> ObjectiveConfig:
    return ObjectiveConfig(
        name="recall1000_ndcg10_latency",
        recall_k=1000,
        ndcg_k=10,
        weights=weights or ObjectiveWeights(recall=0.45, ndcg=0.20, latency=0.35),
        latency=LatencyConfig(
            enabled=True,
            warmup_queries=0,
            hard_slowdown_threshold=5.0,
        ),
    )
