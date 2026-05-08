"""Late-interaction evaluator — single source of truth for metrics + latency.

Two entry points, **same code path** so pre-evolution sanity comparison and
the evolution loop see identical numbers:

  1. The framework calls `evaluate(program_path)` once per candidate. It
     returns `dict[str, float]` (per-dataset metrics + a `combined_score`
     placeholder; the controller recomputes `combined_score` from the
     latency-aware objective).

  2. The user runs `python -m tasks.late_interaction.evaluator --program ... \\
     --datasets ...` to produce a self-contained JSON in
     `tasks/late_interaction/baselines/`. Same `evaluate_cache_dataset` per
     dataset, same fingerprint, same warmup / repeats / GC controls. The JSON
     can then be:
       - read by the controller (`baseline_source: external`),
       - read by the recall-floor wrapper,
       - diffed against another program's JSON to verify equivalence.

Both flows respect:
  EVAL_DEVICE              cpu | cuda  (see `_runtime`)
  EVAL_CACHE_DIR           default cache/late_interaction/lightonai__LateOn
  EVAL_DATASETS            comma-separated dataset ids
  EVAL_SAMPLE_QUERIES      default 50
  EVAL_WARMUP_QUERIES      default 10
  EVAL_TIMED_REPEATS       default 3 (per-query, take median)
  EVAL_PROGRESS            1/true/yes/on to print dataset/query progress
  EVAL_RECALL_K            default 1000
  EVAL_NDCG_K              default 10
"""

from __future__ import annotations

# Import _runtime first so BLAS pinning takes effect before numpy/torch import.
from tasks.late_interaction import _runtime  # noqa: F401

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tasks.late_interaction.evaluator_worker import WorkerResult, evaluate_cache_dataset

DEFAULT_CACHE_ROOT = "cache/late_interaction/lightonai__LateOn"
DEFAULT_DATASETS = ["beir_scifact"]
DEFAULT_BASELINE_DIR = "tasks/late_interaction/baselines"
DEFAULT_RECALL_KS_TO_REPORT = (10, 100)


@dataclass(frozen=True)
class EvaluationOutput:
    """Structured output of one full evaluator run (one program, N datasets).

    The CLI serializes this to JSON; `evaluate(program_path)` flattens it for
    the controller's `dict[str, float]` expectation.
    """
    program_path: str
    datasets: list[str]
    per_dataset: dict[str, WorkerResult]
    fingerprint: dict


def run_evaluation(
    *,
    program_path: str | Path | None,
    datasets: Iterable[str],
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    sample_queries: int = 50,
    warmup_queries: int = 10,
    timed_repeats: int = 3,
    recall_k: int = 1000,
    ndcg_k: int = 10,
    extra_recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS_TO_REPORT,
    progress: bool = False,
    output_path: Path | None = None,
    resume: bool = True,
) -> EvaluationOutput:
    """Run one program over the listed datasets; return per-dataset results.

    When `output_path` is given, the result JSON is written **atomically after
    each dataset completes** (tempfile + os.replace). On a re-run, datasets
    already present in the existing JSON are skipped — provided the loaded
    `_fingerprint.device` and `_run_config` match the current invocation.
    Mismatch hard-errors so apples and oranges can't silently mix. Pass
    `resume=False` to ignore an existing file and re-run everything.
    """
    cache_root = Path(cache_root)
    dataset_list = list(datasets)
    fingerprint = _runtime.runtime_fingerprint()
    run_config = {
        "sample_queries": sample_queries,
        "warmup_queries": warmup_queries,
        "timed_repeats": timed_repeats,
        "recall_k": recall_k,
        "ndcg_k": ndcg_k,
        "extra_recall_ks": list(extra_recall_ks),
    }

    # Load existing payload (resume) or start fresh.
    existing_dataset_payloads, completed_resumed = _load_existing_for_resume(
        output_path=output_path,
        resume=resume,
        fingerprint=fingerprint,
        run_config=run_config,
        program_path=program_path,
    )
    # `dataset_payloads` accumulates JSON-shaped per-dataset blocks (loaded
    # ones from disk + newly computed ones). This is the canonical state
    # written to the JSON file after every dataset completes.
    dataset_payloads: dict[str, dict[str, float | int]] = dict(existing_dataset_payloads)
    fresh_results: dict[str, WorkerResult] = {}

    for dataset_idx, dataset_name in enumerate(dataset_list, start=1):
        cache_dir = cache_root / dataset_name
        if not cache_dir.exists():
            if progress:
                print(f"[eval] SKIP {dataset_name}: cache missing at {cache_dir}", flush=True)
            continue
        if dataset_name in dataset_payloads:
            if progress:
                print(
                    f"[eval] === {dataset_name} ({dataset_idx}/{len(dataset_list)}) — RESUMED, skipping ===",
                    flush=True,
                )
            continue
        if progress:
            print(
                f"[eval] === {dataset_name} ({dataset_idx}/{len(dataset_list)}) ===",
                flush=True,
            )
        result = evaluate_cache_dataset(
            cache_dir=cache_dir,
            program_path=program_path,
            sample_queries=sample_queries,
            recall_k=recall_k,
            ndcg_k=ndcg_k,
            warmup_queries=warmup_queries,
            timed_repeats=timed_repeats,
            extra_recall_ks=extra_recall_ks,
            progress=progress,
        )
        fresh_results[dataset_name] = result
        dataset_payloads[dataset_name] = _serialize_dataset_result(result)
        if progress:
            _print_result_line(dataset_name, result)
        # Atomic incremental write — survives Ctrl-C / kill -9 / power loss.
        if output_path is not None:
            _write_payload_atomic(
                output_path,
                fingerprint=fingerprint,
                run_config=run_config,
                program_path=program_path,
                ordered_datasets=dataset_list,
                dataset_payloads=dataset_payloads,
            )
    return EvaluationOutput(
        program_path=str(program_path) if program_path else "<exact_maxsim default>",
        datasets=list(dataset_payloads.keys()),
        per_dataset=fresh_results,
        fingerprint=fingerprint,
    )


def evaluate(program_path: str = "") -> dict[str, float]:
    """Framework-side entry point. Called once per candidate by the controller.

    Returns a flat `dict[str, float]` containing per-dataset metrics and an
    initial placeholder `combined_score` (the controller overwrites this from
    the latency-aware objective when enabled).

    All knobs come from environment variables — the controller does not pass
    them. Defaults are tuned for the late-interaction task; see the module
    docstring.
    """
    cache_root = Path(os.environ.get("EVAL_CACHE_DIR", DEFAULT_CACHE_ROOT))
    dataset_names = _parse_csv_env("EVAL_DATASETS", DEFAULT_DATASETS)
    sample_queries = int(os.environ.get("EVAL_SAMPLE_QUERIES", "50"))
    recall_k = int(os.environ.get("EVAL_RECALL_K", "1000"))
    ndcg_k = int(os.environ.get("EVAL_NDCG_K", "10"))
    warmup_queries = int(os.environ.get("EVAL_WARMUP_QUERIES", "10"))
    timed_repeats = int(os.environ.get("EVAL_TIMED_REPEATS", "3"))
    progress = _parse_bool_env("EVAL_PROGRESS", default=False)

    output = run_evaluation(
        program_path=program_path or None,
        datasets=dataset_names,
        cache_root=cache_root,
        sample_queries=sample_queries,
        warmup_queries=warmup_queries,
        timed_repeats=timed_repeats,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        progress=progress,
    )
    return _flatten_metrics(output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a late-interaction retrieval program over one or more cached "
            "datasets. Writes a single JSON with per-dataset metrics, an "
            "average block, and the runtime hardware fingerprint."
        )
    )
    parser.add_argument(
        "--program",
        required=True,
        help=(
            "Path to a Python file exposing `LateInteractionRetriever` "
            "(e.g. tasks/late_interaction/programs/exact_maxsim.py, "
            "tasks/late_interaction/programs/fastplaid.py, "
            "tasks/late_interaction/seeds/freeform.py)."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=lambda s: [item.strip() for item in s.split(",") if item.strip()],
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset names (default: beir_scifact).",
    )
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=None,
        help=(
            "Number of measured queries per dataset. "
            "Default: ALL queries (full eval). "
            "Pass a positive integer to sample that many instead."
        ),
    )
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument("--recall-k", type=int, default=1000)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the result JSON. Default: "
            "tasks/late_interaction/baselines/<program-stem>.<device>.json"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any existing output JSON and re-run all datasets. "
            "Default behavior: resume — skip datasets already present in the "
            "output JSON (with matching device + run config)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    program_path = Path(args.program).resolve()
    device = _runtime.resolve_device()

    if not program_path.exists():
        print(f"[eval] program not found: {program_path}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(DEFAULT_BASELINE_DIR) / f"{program_path.stem}.{device}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve --sample-queries default: None means "all queries". The worker
    # clamps to min(sample_queries, len(cache.queries)), so passing a very
    # large int is the simple way to request the full set without per-dataset
    # bookkeeping in this layer.
    sample_queries_arg = args.sample_queries
    sample_queries_label = "all" if sample_queries_arg is None else str(sample_queries_arg)
    sample_queries_value = 10**9 if sample_queries_arg is None else sample_queries_arg

    print(
        f"[eval] program={program_path.relative_to(Path.cwd()) if program_path.is_absolute() and program_path.is_relative_to(Path.cwd()) else program_path}",
        flush=True,
    )
    print(
        f"[eval] device={device}, datasets={','.join(args.datasets)}, "
        f"sample_queries={sample_queries_label}, warmup_queries={args.warmup_queries}, "
        f"timed_repeats={args.timed_repeats}",
        flush=True,
    )

    run_evaluation(
        program_path=program_path,
        datasets=args.datasets,
        cache_root=args.cache_root,
        sample_queries=sample_queries_value,
        warmup_queries=args.warmup_queries,
        timed_repeats=args.timed_repeats,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        progress=True,
        output_path=output_path,
        resume=not args.no_resume,
    )
    # The JSON was written incrementally after each dataset; nothing more to
    # do here. Confirm to the operator where it landed.
    print(f"\n[eval] wrote {output_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _flatten_metrics(output: EvaluationOutput) -> dict[str, float]:
    """Flatten an `EvaluationOutput` into the controller's `dict[str, float]`."""
    results = list(output.per_dataset.values())
    recall = float(np.mean([r.recall_at_1000 for r in results])) if results else 0.0
    ndcg = float(np.mean([r.ndcg_at_10 for r in results])) if results else 0.0
    latency_p50 = float(np.mean([r.latency_p50_ms for r in results])) if results else 0.0
    latency_p95 = float(np.mean([r.latency_p95_ms for r in results])) if results else 0.0
    latency_mean = float(np.mean([r.latency_mean_ms for r in results])) if results else 0.0

    metrics: dict[str, float] = {
        "recall_at_1000": recall,
        "ndcg_at_10": ndcg,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "latency_mean_ms": latency_mean,
        "combined_score": 0.8 * recall + 0.2 * ndcg,  # overwritten by the latency-aware controller
        "num_datasets": float(len(results)),
    }
    for r in results:
        metrics.update(r.to_metrics())
    return metrics


def _build_payload_from_dicts(
    *,
    fingerprint: dict,
    run_config: dict,
    program_path: str | Path | None,
    ordered_datasets: list[str],
    dataset_payloads: dict[str, dict[str, float | int]],
) -> dict:
    """Assemble the on-disk JSON from already-serialized per-dataset dicts.

    Keys starting with `_` are skipped by the controller's external-baseline
    loader, so the per-dataset blocks remain the unambiguous lookup target.
    """
    completed = [d for d in ordered_datasets if d in dataset_payloads]
    payload: dict = {
        "_fingerprint": fingerprint,
        "_program": str(program_path) if program_path else "",
        "_datasets": ordered_datasets,
        "_completed": completed,
        "_run_config": run_config,
    }
    for dataset_name, block in dataset_payloads.items():
        payload[dataset_name] = block

    if dataset_payloads:
        recall_ks: set[int] = set()
        for block in dataset_payloads.values():
            for key in block:
                if key.startswith("recall_at_"):
                    try:
                        recall_ks.add(int(key.split("_")[-1]))
                    except ValueError:
                        pass

        def _avg(field: str) -> float:
            return float(np.mean([
                float(block[field]) for block in dataset_payloads.values() if field in block
            ]))

        average: dict[str, float | int] = {
            "median_query_latency_ms": _avg("median_query_latency_ms"),
            "p95_query_latency_ms": _avg("p95_query_latency_ms"),
            "mean_query_latency_ms": _avg("mean_query_latency_ms"),
            "build_time_ms": _avg("build_time_ms"),
            "ndcg_at_10": _avg("ndcg_at_10"),
        }
        for k in sorted(recall_ks):
            average[f"recall_at_{k}"] = _avg(f"recall_at_{k}")
        payload["_average"] = average
    return payload


def _write_payload_atomic(
    output_path: Path,
    *,
    fingerprint: dict,
    run_config: dict,
    program_path: str | Path | None,
    ordered_datasets: list[str],
    dataset_payloads: dict[str, dict[str, float | int]],
) -> None:
    """Write the JSON via tempfile + os.replace so a crash mid-write can't
    leave the file half-written. Posix rename is atomic on the same fs."""
    payload = _build_payload_from_dicts(
        fingerprint=fingerprint,
        run_config=run_config,
        program_path=program_path,
        ordered_datasets=ordered_datasets,
        dataset_payloads=dataset_payloads,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, output_path)


def _load_existing_for_resume(
    *,
    output_path: Path | None,
    resume: bool,
    fingerprint: dict,
    run_config: dict,
    program_path: str | Path | None,
) -> tuple[dict[str, dict[str, float | int]], list[str]]:
    """Read the existing output JSON for resume; validate it's compatible.

    Returns `(dataset_payloads, completed_dataset_names)`. Empty when there's
    no file to resume from. Hard-errors on device or run-config mismatch.
    """
    if output_path is None or not resume or not output_path.exists():
        return {}, []
    try:
        existing = json.loads(output_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"existing output {output_path} is not valid JSON: {exc}. "
            "Pass --no-resume to ignore and overwrite."
        ) from exc
    existing_fp = existing.get("_fingerprint", {}) or {}
    if existing_fp.get("device") != fingerprint.get("device"):
        raise RuntimeError(
            f"resume aborted — existing {output_path} was recorded on device "
            f"{existing_fp.get('device')!r} but this run is on "
            f"{fingerprint.get('device')!r}. Re-run with EVAL_DEVICE matching, "
            "or pass --no-resume to overwrite."
        )
    existing_cfg = existing.get("_run_config", {}) or {}
    if existing_cfg and existing_cfg != run_config:
        diff = {k: (existing_cfg.get(k), run_config.get(k))
                for k in set(existing_cfg) | set(run_config)
                if existing_cfg.get(k) != run_config.get(k)}
        raise RuntimeError(
            f"resume aborted — existing {output_path} used a different run "
            f"config: {diff}. Pass --no-resume to overwrite, or change flags "
            "to match the existing run."
        )
    existing_program = existing.get("_program", "") or ""
    if program_path and existing_program and str(program_path) != existing_program:
        raise RuntimeError(
            f"resume aborted — existing {output_path} was produced by program "
            f"{existing_program!r} but this run uses {str(program_path)!r}. "
            "Pass --no-resume to overwrite, or use the matching --program."
        )
    payloads: dict[str, dict[str, float | int]] = {}
    for key, value in existing.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and "median_query_latency_ms" in value:
            payloads[key] = value
    return payloads, list(payloads.keys())


def _serialize_dataset_result(result: WorkerResult) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "median_query_latency_ms": result.latency_p50_ms,
        "p95_query_latency_ms": result.latency_p95_ms,
        "mean_query_latency_ms": result.latency_mean_ms,
        "build_time_ms": result.build_time_ms,
        "recall_at_1000": result.recall_at_1000,
        "ndcg_at_10": result.ndcg_at_10,
        "num_queries": result.num_queries,
        "warmup_queries": result.warmup_queries,
        "timed_repeats": result.timed_repeats,
    }
    for k, v in result.extra_recall.items():
        out[f"recall_at_{k}"] = float(v)
    return out


def _print_result_line(dataset_name: str, result: WorkerResult) -> None:
    extras = " ".join(f"recall@{k}={v:.4f}" for k, v in sorted(result.extra_recall.items()))
    print(
        f"  {dataset_name:24s} | recall@1000={result.recall_at_1000:.4f} "
        f"ndcg@10={result.ndcg_at_10:.4f} {extras} "
        f"| p50={result.latency_p50_ms:7.2f}ms p95={result.latency_p95_ms:7.2f}ms "
        f"mean={result.latency_mean_ms:7.2f}ms build={result.build_time_ms:7.1f}ms "
        f"| n={result.num_queries} warmup={result.warmup_queries} repeats={result.timed_repeats}",
        flush=True,
    )


# Back-compat: legacy callers used `parse_csv_env` (no underscore).
parse_csv_env = _parse_csv_env


if __name__ == "__main__":
    raise SystemExit(main())
