"""Pure objective math: latency transform, hard-slowdown, combined score.

This module is the retrieval objective shared by every task. It is
intentionally side-effect-free so the formulas can be unit-tested without
spinning up the evaluator. The evolution algorithm is independent and is
tested in `tests/search/test_evolution_algo_invariants.py`.

Per-dataset signals (when `latency.enabled=True`):

  ratio_d            = candidate_query_latency_median_ms_d
                       / baseline_query_latency_median_ms_d
  if ratio_d > hard_slowdown_threshold:
      latency_score_d                = 0.0
      latency_penalty_triggered_d    = 1.0
  else:
      latency_score_d                = 1.0 / (1.0 + ratio_d)
      latency_penalty_triggered_d    = 0.0

Aggregation across datasets:

  avg_recall, avg_ndcg, avg_latency_score = aggregate(per-dataset values)

  - aggregation.mode="arithmetic" (default, legacy): plain mean.
  - aggregation.mode="geometric": exp(mean(log(max(x, eps)))). A near-zero
    on any dataset drops the aggregate near zero — useful when you don't
    want one-dataset wins to mask catastrophic regressions on others.

Combined score:

  effectiveness_score = w.recall * avg_recall + w.ndcg * avg_ndcg
  latency_component   = w.latency * avg_latency_score
  combined_score      = effectiveness_score + latency_component

Recall floor:

  If `min_recall > 0` and any dataset's recall@recall_k falls below
  `min_recall`, combined_score is set to 0 and `recall_floor_triggered`
  is 1.0. The candidate is treated as unusable and will lose every
  cell-replacement contest in MAP-Elites; aggregate fields are still
  reported for transparency.

When `latency.enabled=False` the latency component is 0 regardless of
weights and no per-dataset latency lookup is required.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..config.objective import AggregationConfig, ObjectiveConfig


def latency_score_from_ratio(ratio: float) -> float:
    """`inverse_one_plus_ratio` transform. ratio>=0 expected."""
    if ratio < 0.0:
        raise ValueError(f"latency ratio must be non-negative, got {ratio}")
    return 1.0 / (1.0 + ratio)


def is_hard_slowdown(
    candidate_latency_ms: float,
    baseline_latency_ms: float,
    threshold: float,
) -> bool:
    """True when candidate is more than `threshold` times slower than baseline.

    `baseline_latency_ms <= 0` is treated as "no usable baseline" and never
    trips the penalty (we cannot fairly compare against a zero baseline).
    """
    if baseline_latency_ms <= 0.0:
        return False
    return (candidate_latency_ms / baseline_latency_ms) > threshold


@dataclass(frozen=True)
class ObjectiveOutcome:
    """Resolved objective values for a single candidate."""

    combined_score: float
    effectiveness_score: float
    objective_recall_component: float
    objective_ndcg_component: float
    objective_latency_component: float
    avg_recall: float
    avg_ndcg: float
    avg_latency_score: float
    avg_latency_ratio: float
    avg_query_latency_median_ms: float
    avg_baseline_query_latency_median_ms: float
    latency_penalty_triggered: float
    recall_floor_triggered: float
    aggregation_mode: str
    per_dataset: dict[str, dict[str, float]]


def _recall_key(recall_k: int) -> str:
    return f"recall@{recall_k}"


def _ndcg_key(ndcg_k: int) -> str:
    return f"ndcg@{ndcg_k}"


def _metric_value(metrics: Mapping[str, float], canonical: str, alias: str) -> float | None:
    if canonical in metrics:
        return float(metrics[canonical])
    if alias in metrics:
        return float(metrics[alias])
    return None


def _objective_metric_value(
    *,
    dataset: str,
    metrics: Mapping[str, float],
    explicit_key: str,
    canonical: str,
    alias: str,
    fallback: bool,
) -> tuple[float | None, str | None]:
    if explicit_key:
        if explicit_key in metrics:
            return float(metrics[explicit_key]), explicit_key
        if not fallback:
            raise ValueError(
                f"dataset {dataset!r} is missing objective metric {explicit_key!r}; "
                "enable objective.metric_key_fallback or choose a metric key emitted by every dataset"
            )
    value = _metric_value(metrics, canonical, alias)
    return value, (canonical if value is not None else None)


def compute_objective(
    per_dataset_metrics: Mapping[str, Mapping[str, float]],
    baseline_latency_by_dataset: Mapping[str, float] | None,
    cfg: ObjectiveConfig,
) -> ObjectiveOutcome:
    """Resolve the configured objective from raw per-dataset metrics.

    `per_dataset_metrics[dataset]` must contain at least `recall@<recall_k>`
    / `recall_at_<recall_k>` and `ndcg@<ndcg_k>` / `ndcg_at_<ndcg_k>`.
    When `cfg.latency.enabled` is True, the same dict must also contain
    `query_latency_median_ms`.

    `baseline_latency_by_dataset` is required when `cfg.latency.enabled`.
    Datasets missing from the baseline map are skipped from the latency
    average (their effectiveness contribution still counts).
    """
    rk = _recall_key(cfg.recall_k)
    nk = _ndcg_key(cfg.ndcg_k)
    recall_alias = f"recall_at_{cfg.recall_k}"
    ndcg_alias = f"ndcg_at_{cfg.ndcg_k}"

    recalls: list[float] = []
    ndcgs: list[float] = []
    latency_scores: list[float] = []
    latency_ratios: list[float] = []
    cand_latencies: list[float] = []
    base_latencies: list[float] = []
    any_penalty = 0.0
    out_per_dataset: dict[str, dict[str, float]] = {}

    threshold = cfg.latency.hard_slowdown_threshold

    for dataset, metrics in per_dataset_metrics.items():
        ds_out: dict[str, float] = {}
        recall_value, _ = _objective_metric_value(
            dataset=dataset,
            metrics=metrics,
            explicit_key=cfg.recall_metric_key,
            canonical=rk,
            alias=recall_alias,
            fallback=cfg.metric_key_fallback,
        )
        if recall_value is not None:
            recalls.append(recall_value)
            ds_out[rk] = recall_value
        ndcg_value, _ = _objective_metric_value(
            dataset=dataset,
            metrics=metrics,
            explicit_key=cfg.ndcg_metric_key,
            canonical=nk,
            alias=ndcg_alias,
            fallback=cfg.metric_key_fallback,
        )
        if ndcg_value is not None:
            ndcgs.append(ndcg_value)
            ds_out[nk] = ndcg_value

        if cfg.latency.enabled:
            cand_ms = float(metrics.get("query_latency_median_ms", 0.0))
            base_ms = (
                float(baseline_latency_by_dataset.get(dataset, 0.0))
                if baseline_latency_by_dataset is not None
                else 0.0
            )
            ds_out["query_latency_median_ms"] = cand_ms
            ds_out["baseline_query_latency_median_ms"] = base_ms

            if base_ms > 0.0:
                ratio = cand_ms / base_ms
                if ratio > threshold:
                    score = 0.0
                    triggered = 1.0
                    any_penalty = 1.0
                else:
                    score = latency_score_from_ratio(ratio)
                    triggered = 0.0
                latency_scores.append(score)
                latency_ratios.append(ratio)
                cand_latencies.append(cand_ms)
                base_latencies.append(base_ms)
                ds_out["latency_ratio"] = ratio
                ds_out["latency_score"] = score
                ds_out["latency_penalty_triggered"] = triggered
            else:
                # No usable baseline — neutral neutral-score, do not pollute
                # the average with a fabricated value.
                ds_out["latency_ratio"] = 0.0
                ds_out["latency_score"] = 0.0
                ds_out["latency_penalty_triggered"] = 0.0

        out_per_dataset[dataset] = ds_out

    aggregator = _make_aggregator(cfg.aggregation)
    # Effectiveness signals (recall, ndcg) and latency_score are aggregated
    # under the configured mode. Latency raw averages (ratios, ms) stay on
    # arithmetic mean — they're informational, not part of the score.
    avg_recall = aggregator(recalls)
    avg_ndcg = aggregator(ndcgs)
    avg_latency_score = aggregator(latency_scores) if latency_scores else 0.0
    avg_ratio = _mean(latency_ratios) if latency_ratios else 0.0
    avg_cand = _mean(cand_latencies) if cand_latencies else 0.0
    avg_base = _mean(base_latencies) if base_latencies else 0.0

    # Recall floor: any dataset below `min_recall` zeroes the entire
    # candidate. We still report aggregates for transparency but combined
    # score collapses so the candidate cannot win cell contests in MAP-Elites.
    recall_floor_triggered = 0.0
    if cfg.min_recall > 0.0 and recalls:
        if any(r < cfg.min_recall for r in recalls):
            recall_floor_triggered = 1.0

    w = cfg.weights
    recall_component = w.recall * avg_recall
    ndcg_component = w.ndcg * avg_ndcg
    effectiveness = recall_component + ndcg_component
    latency_component = w.latency * avg_latency_score if cfg.latency.enabled else 0.0
    combined = effectiveness + latency_component
    if recall_floor_triggered:
        combined = 0.0

    return ObjectiveOutcome(
        combined_score=combined,
        effectiveness_score=effectiveness,
        objective_recall_component=recall_component,
        objective_ndcg_component=ndcg_component,
        objective_latency_component=latency_component,
        avg_recall=avg_recall,
        avg_ndcg=avg_ndcg,
        avg_latency_score=avg_latency_score,
        avg_latency_ratio=avg_ratio,
        avg_query_latency_median_ms=avg_cand,
        avg_baseline_query_latency_median_ms=avg_base,
        latency_penalty_triggered=any_penalty,
        recall_floor_triggered=recall_floor_triggered,
        aggregation_mode=cfg.aggregation.mode,
        per_dataset=out_per_dataset,
    )


def _make_aggregator(cfg: AggregationConfig):
    mode = (cfg.mode or "arithmetic").lower()
    if mode == "arithmetic":
        return _mean
    if mode == "geometric":
        eps = float(cfg.eps)
        if eps <= 0.0:
            raise ValueError(f"aggregation.eps must be > 0, got {cfg.eps!r}")

        def _geom(values: list[float]) -> float:
            if not values:
                return 0.0
            log_sum = 0.0
            for v in values:
                log_sum += math.log(max(float(v), eps))
            return float(math.exp(log_sum / len(values)))

        return _geom
    raise ValueError(
        f"unsupported aggregation.mode={cfg.mode!r}; expected 'arithmetic' or 'geometric'"
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
