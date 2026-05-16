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
    #   "seed"     — measure the seed program once before the evolution loop.
    #   "external" — load from a static JSON file written by an external
    #                baseline (e.g. tasks/late_interaction/compare_baselines.py
    #                emits one per device). The file must carry a
    #                `_fingerprint.device` field; the controller asserts it
    #                matches the runtime device before using the baseline.
    #   "fixed"    — reserved.
    baseline_source: str = "seed"
    # Path to the baseline JSON when `baseline_source == "external"`.
    # `${EVAL_DEVICE}` is interpolated to the resolved device ("cpu" / "cuda")
    # so a single config can target either host (e.g.
    # "tasks/late_interaction/baselines/fastplaid_baseline.${EVAL_DEVICE}.json").
    baseline_path: str = ""
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
class AggregationConfig:
    """How per-dataset metrics are combined into a single scalar.

    `mode="arithmetic"` (default, legacy) — arithmetic mean across datasets.
    A catastrophic regression on one dataset is diluted by the others.

    `mode="geometric"` — geometric mean across datasets, computed as
    `exp(mean(log(max(x, eps))))`. A near-zero on any one dataset drives the
    aggregate near zero. Use this when you want a candidate that flops on
    one dataset to be uncompetitive regardless of how good it is on the rest.
    `eps` is the floor inside the log to avoid log(0); a small positive
    value (default 1e-3) leaves "good" scores essentially unchanged but
    keeps zeros from collapsing the geometric mean to exactly 0.
    """

    mode: str = "arithmetic"
    eps: float = 1e-3


@dataclass
class ObjectiveConfig:
    """Top-level objective config.

    `recall_k` is also the `top_k` passed to `bm25.rank(...)` during
    evaluation, so that recall@k and the timed retrieval phase use the
    same retrieval depth.

    `min_recall` (default 0.0 = off): if any dataset's recall@recall_k falls
    below this floor, the candidate's combined_score becomes 0 and
    `recall_floor_triggered` is set to 1.0. The motivation is that a
    candidate that collapses on the largest corpus is not a usable solution
    regardless of how fast or accurate it is on the smaller ones; we don't
    want it occupying archive cells or seeding offspring.
    """

    name: str = "recall100_ndcg10"
    recall_k: int = 100
    ndcg_k: int = 10
    # Optional evaluator-emitted metric names to optimize instead of the
    # canonical recall@k / ndcg@k aliases. Useful for aspect-aware datasets
    # such as BRIGHT-Pro (`aspect_recall_at_25`, `alpha_ndcg_at_25`).
    recall_metric_key: str = ""
    ndcg_metric_key: str = ""
    # If an explicit key is absent for a dataset, fall back to the canonical
    # recall/nDCG key. Set false for strict single-benchmark configs.
    metric_key_fallback: bool = True
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    min_recall: float = 0.0
