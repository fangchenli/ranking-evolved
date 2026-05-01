"""Optimization-objective configuration.

This is the BM25 retrieval optimization objective. It controls how
`combined_score` is computed from raw evaluator metrics. The evolution
algorithm's invariants (sampling, admission, migration, MAP-Elites) are
independent of this and are tested in
`tests/search/test_evolution_algo_invariants.py`.

Two presets are useful out of the box:

  - `recall100_ndcg10` (legacy): combined_score = 0.8*recall@100 + 0.2*ndcg@10.
    `latency.enabled = False`. This matches what `evaluator_parallel.py`
    has historically computed.

  - `recall1000_ndcg10_latency` (latency-aware): combined_score =
    0.45*recall@1000 + 0.20*ndcg@10 + 0.35*avg_latency_score, where
    `latency_score_d = 1 / (1 + ratio_d)` per dataset and
    `ratio_d = candidate_query_latency_median_ms / baseline_query_latency_median_ms`.
    Hard slowdown (`ratio_d > hard_slowdown_threshold`) zeroes that
    dataset's latency_score and trips a `latency_penalty_triggered` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectiveWeights:
    """Linear weights on the three objective components."""

    recall: float = 0.8
    ndcg: float = 0.2
    latency: float = 0.0


@dataclass
class LatencyConfig:
    """Per-query latency measurement + scoring policy.

    `enabled=False` means the framework does not run a baseline seed
    evaluation, the evaluator does no warmup, and `combined_score`
    ignores latency entirely (legacy behavior).
    """

    enabled: bool = False
    # Where the baseline median per-query latency comes from.
    # "seed" — measure the seed program once before the evolution loop.
    # "fixed" — read from a static JSON file (future; not yet wired).
    baseline_source: str = "seed"
    relative_to: str = "seed"
    # Number of queries to run untimed before the timed phase, per dataset.
    warmup_queries: int = 20
    # Reserved for future use (repeating timed phase to lower variance).
    timed_repeats: int = 1
    aggregation: str = "median_per_query"
    # `inverse_one_plus_ratio`: latency_score = 1 / (1 + ratio_d).
    ratio_transform: str = "inverse_one_plus_ratio"
    # Any dataset with ratio above this threshold is considered "too slow":
    # its latency_score becomes 0.0 and `latency_penalty_triggered` is set.
    hard_slowdown_threshold: float = 5.0
    # "zero_latency_score" — non-destructive (zero score for slow datasets).
    # "reject" — reserved for future use; would set combined_score = -1e9.
    penalty_mode: str = "zero_latency_score"


@dataclass
class ObjectiveConfig:
    """Top-level objective config.

    `recall_k` is also the `top_k` passed to `bm25.rank(...)` during
    evaluation, so that recall@k and the timed retrieval phase use the
    same retrieval depth.
    """

    name: str = "recall100_ndcg10"
    recall_k: int = 100
    ndcg_k: int = 10
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
