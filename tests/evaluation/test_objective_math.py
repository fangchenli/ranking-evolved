"""Tests for the BM25 retrieval objective math.

The evolution-algorithm invariants (sampling, admission, migration) are
tested separately in tests/search/test_evolution_algo_invariants.py and
do not depend on this objective.
"""

from __future__ import annotations

import math

import pytest

from ranking_evolved.config.objective import (
    AggregationConfig,
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


def test_combined_score_accepts_evaluator_metric_names():
    """Late-interaction evaluator emits recall_at_k / ndcg_at_k names."""
    cfg = _latency_aware_cfg(weights=ObjectiveWeights(recall=0.40, ndcg=0.20, latency=0.40))
    per_dataset = {
        "ds_a": {
            "recall_at_1000": 0.5,
            "ndcg_at_10": 0.25,
            "query_latency_median_ms": 100.0,
        },
        "ds_b": {
            "recall_at_1000": 0.7,
            "ndcg_at_10": 0.35,
            "query_latency_median_ms": 100.0,
        },
    }

    outcome = compute_objective(per_dataset, {"ds_a": 100.0, "ds_b": 100.0}, cfg)

    assert outcome.avg_recall == pytest.approx(0.6)
    assert outcome.avg_ndcg == pytest.approx(0.3)
    assert outcome.objective_recall_component == pytest.approx(0.40 * 0.6)
    assert outcome.objective_ndcg_component == pytest.approx(0.20 * 0.3)


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


def _latency_aware_cfg(
    weights: ObjectiveWeights | None = None,
    *,
    aggregation: AggregationConfig | None = None,
    min_recall: float = 0.0,
) -> ObjectiveConfig:
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
        aggregation=aggregation or AggregationConfig(),
        min_recall=min_recall,
    )


# ----------------------------------------------------------------------------
# Geometric-mean aggregation
# ----------------------------------------------------------------------------


def test_geometric_mean_matches_arithmetic_when_values_are_equal():
    """If every dataset reports the same metric, geometric == arithmetic."""
    cfg = _latency_aware_cfg(
        weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3),
        aggregation=AggregationConfig(mode="geometric", eps=1e-3),
    )
    per_dataset = {
        f"ds_{i}": {
            "recall@1000": 0.5,
            "ndcg@10": 0.3,
            "query_latency_median_ms": 100.0,
        }
        for i in range(3)
    }
    baseline = {f"ds_{i}": 100.0 for i in range(3)}

    outcome = compute_objective(per_dataset, baseline, cfg)

    assert outcome.aggregation_mode == "geometric"
    assert outcome.avg_recall == pytest.approx(0.5)
    assert outcome.avg_ndcg == pytest.approx(0.3)
    assert outcome.avg_latency_score == pytest.approx(0.5)


def test_geometric_mean_punishes_one_dataset_collapse():
    """Geometric mean of [0.7, 0.7, 0.001] is dominated by the small value."""
    cfg = _latency_aware_cfg(
        weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3),
        aggregation=AggregationConfig(mode="geometric", eps=1e-3),
    )
    per_dataset = {
        "good_a": {"recall@1000": 0.7, "ndcg@10": 0.5, "query_latency_median_ms": 100.0},
        "good_b": {"recall@1000": 0.7, "ndcg@10": 0.5, "query_latency_median_ms": 100.0},
        "bad": {"recall@1000": 0.001, "ndcg@10": 0.5, "query_latency_median_ms": 100.0},
    }
    baseline = {"good_a": 100.0, "good_b": 100.0, "bad": 100.0}

    outcome = compute_objective(per_dataset, baseline, cfg)

    # arithmetic recall would be 0.467; geometric is (0.7 * 0.7 * 0.001)^(1/3) = 0.0772
    arithmetic_recall = (0.7 + 0.7 + 0.001) / 3
    assert outcome.avg_recall < arithmetic_recall * 0.25
    assert outcome.avg_recall == pytest.approx(0.0772, rel=0.05)


def test_geometric_mean_eps_floor_prevents_log_zero():
    """A literal zero must not blow up the geometric mean — it floors to eps."""
    cfg = _latency_aware_cfg(
        aggregation=AggregationConfig(mode="geometric", eps=1e-3),
    )
    per_dataset = {
        "ok": {"recall@1000": 0.5, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
        "zero": {"recall@1000": 0.0, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
    }
    baseline = {"ok": 100.0, "zero": 100.0}

    outcome = compute_objective(per_dataset, baseline, cfg)

    # Geometric of [0.5, eps=1e-3] = sqrt(0.5 * 1e-3) ≈ 0.0224
    assert outcome.avg_recall == pytest.approx((0.5 * 1e-3) ** 0.5, rel=1e-6)


def test_geometric_aggregation_rejects_invalid_mode():
    cfg = _latency_aware_cfg(aggregation=AggregationConfig(mode="harmonic"))
    per_dataset = {
        "ds": {"recall@1000": 0.5, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
    }
    with pytest.raises(ValueError, match="aggregation.mode"):
        compute_objective(per_dataset, {"ds": 100.0}, cfg)


# ----------------------------------------------------------------------------
# Recall floor (whole-candidate gate)
# ----------------------------------------------------------------------------


def test_recall_floor_below_threshold_zeroes_combined_and_sets_flag():
    """Any per-dataset recall < min_recall → combined_score = 0, flag set."""
    cfg = _latency_aware_cfg(
        weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3),
        min_recall=0.10,
    )
    per_dataset = {
        "good": {"recall@1000": 0.5, "ndcg@10": 0.3, "query_latency_median_ms": 50.0},
        "bad": {"recall@1000": 0.05, "ndcg@10": 0.4, "query_latency_median_ms": 50.0},
    }
    baseline = {"good": 100.0, "bad": 100.0}

    outcome = compute_objective(per_dataset, baseline, cfg)

    assert outcome.recall_floor_triggered == 1.0
    assert outcome.combined_score == 0.0
    # Component values are still surfaced for transparency.
    assert outcome.objective_recall_component > 0.0
    assert outcome.objective_latency_component > 0.0


def test_recall_floor_at_threshold_is_inclusive_pass():
    """Recall exactly at the floor passes — only strictly below trips it."""
    cfg = _latency_aware_cfg(
        weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3),
        min_recall=0.10,
    )
    per_dataset = {
        "exact": {"recall@1000": 0.10, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
        "high": {"recall@1000": 0.50, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
    }
    baseline = {"exact": 100.0, "high": 100.0}

    outcome = compute_objective(per_dataset, baseline, cfg)

    assert outcome.recall_floor_triggered == 0.0
    assert outcome.combined_score > 0.0


def test_recall_floor_zero_means_disabled():
    """Default min_recall=0 must never trigger, even on a zero-recall dataset."""
    cfg = _latency_aware_cfg(weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3))
    per_dataset = {
        "zero": {"recall@1000": 0.0, "ndcg@10": 0.3, "query_latency_median_ms": 100.0},
    }
    outcome = compute_objective(per_dataset, {"zero": 100.0}, cfg)
    assert outcome.recall_floor_triggered == 0.0


def test_recall_floor_combines_with_geometric_mean():
    """Floor + geometric mean: floor wins (combined_score=0) regardless of agg."""
    cfg = _latency_aware_cfg(
        weights=ObjectiveWeights(recall=0.4, ndcg=0.3, latency=0.3),
        aggregation=AggregationConfig(mode="geometric"),
        min_recall=0.10,
    )
    per_dataset = {
        "good": {"recall@1000": 0.7, "ndcg@10": 0.4, "query_latency_median_ms": 50.0},
        "bad": {"recall@1000": 0.02, "ndcg@10": 0.4, "query_latency_median_ms": 50.0},
    }
    outcome = compute_objective(per_dataset, {"good": 100.0, "bad": 100.0}, cfg)
    assert outcome.combined_score == 0.0
    assert outcome.recall_floor_triggered == 1.0
    assert outcome.aggregation_mode == "geometric"
