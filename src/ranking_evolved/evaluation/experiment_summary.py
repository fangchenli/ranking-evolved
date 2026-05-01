"""Run-level experiment summary and per-program metrics writers.

Each run directory ends up with three files relevant to objective-formulation
experiments:

  * `baseline_latency.json`    — written before the evolution loop, captures
                                  per-dataset seed median latencies.
  * `program_metrics.jsonl`    — append-only; one row per evaluated program.
  * `experiment_summary.json`  — written at run end with the resolved
                                  objective config and best-program details.

These files are independent of the SQLite run store and the replay JSON; they
exist specifically to make cross-run comparison easy without grepping replay
dashboards.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config.objective import ObjectiveConfig


PROGRAM_METRICS_FILE = "program_metrics.jsonl"
EXPERIMENT_SUMMARY_FILE = "experiment_summary.json"
BASELINE_LATENCY_FILE = "baseline_latency.json"


def write_baseline_latency(
    run_dir: Path,
    *,
    objective: ObjectiveConfig,
    baseline_latency_by_dataset: Mapping[str, float],
) -> Path:
    path = Path(run_dir) / BASELINE_LATENCY_FILE
    payload = {
        "objective": objective.name,
        "recall_k": objective.recall_k,
        "ndcg_k": objective.ndcg_k,
        "warmup_queries": objective.latency.warmup_queries,
        "baseline_latency_by_dataset": dict(baseline_latency_by_dataset),
        "captured_at": _iso_now(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def append_program_metrics(
    run_dir: Path,
    row: Mapping[str, Any],
) -> Path:
    """Append one JSON line to `<run_dir>/program_metrics.jsonl`.

    `row` should already contain `program_id`, `parent_id`, `island`,
    `generation`, `iteration_found`, `combined_score`, the per-objective
    components, and a `per_dataset` block. Caller controls schema.
    """
    path = Path(run_dir) / PROGRAM_METRICS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), default=str))
        f.write("\n")
    return path


def write_experiment_summary(
    run_dir: Path,
    *,
    run_id: str,
    config_path: Path,
    objective: ObjectiveConfig,
    seed_program_id: str,
    best_program_id: str,
    best_combined_score: float,
    best_metrics: Mapping[str, float],
    baseline_latency_by_dataset: Mapping[str, float] | None,
    datasets: list[str],
    timestamp_start: str | None,
    timestamp_end: str | None,
) -> Path:
    path = Path(run_dir) / EXPERIMENT_SUMMARY_FILE
    payload = {
        "run_id": run_id,
        "config_path": str(config_path),
        "objective_name": objective.name,
        "objective_config": _jsonable(asdict(objective)),
        "seed_program_id": seed_program_id,
        "best_program_id": best_program_id,
        "best_combined_score": float(best_combined_score),
        "best_metrics": _jsonable(dict(best_metrics)),
        "baseline_latency_by_dataset": (
            dict(baseline_latency_by_dataset) if baseline_latency_by_dataset else None
        ),
        "datasets": list(datasets),
        "top_k_used_for_recall": objective.recall_k,
        "ndcg_k": objective.ndcg_k,
        "latency_policy": _jsonable(asdict(objective.latency)),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
