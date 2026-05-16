"""Run exhaustive exact-MaxSim baselines for the new benchmark suites.

Example:
    uv run python -m tasks.late_interaction.run_new_benchmark_baseline --suite curated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tasks.late_interaction import _runtime
from tasks.late_interaction.encode_embeddings import (
    NEW_BENCHMARKS_ALL_SUITE,
    NEW_BENCHMARKS_CURATED_SUITE,
    normalize_dataset_spec,
    safe_model_name,
)
from tasks.late_interaction.evaluator import DEFAULT_CACHE_MODEL, run_evaluation

SUITES = {
    "curated": NEW_BENCHMARKS_CURATED_SUITE,
    "all": NEW_BENCHMARKS_ALL_SUITE,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="curated")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset id/spec to evaluate. Overrides --suite when provided.",
    )
    parser.add_argument("--program", default="tasks/late_interaction/programs/exact_maxsim.py")
    parser.add_argument("--cache-root", default="cache/late_interaction/lightonai__LateOn")
    parser.add_argument("--output", default=None)
    parser.add_argument("--qrels-mode", choices=["gold", "pooled"], default="gold")
    parser.add_argument("--recall-k", type=int, default=25)
    parser.add_argument("--ndcg-k", type=int, default=25)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument("--cache-model", default=DEFAULT_CACHE_MODEL)
    parser.add_argument("--cache-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--beir-data-dir", default="datasets/beir")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    dataset_specs = args.dataset or SUITES[args.suite]
    dataset_ids = [_dataset_id(spec) for spec in dataset_specs]
    device = _runtime.resolve_device()
    output = Path(args.output) if args.output else _default_output_path(
        suite=args.suite if not args.dataset else "custom",
        qrels_mode=args.qrels_mode,
        device=device,
        model_name=args.cache_model,
    )

    print(
        f"[new-benchmark-baseline] suite={args.suite} datasets={','.join(dataset_ids)} "
        f"device={device} output={output}",
        flush=True,
    )
    run_evaluation(
        program_path=Path(args.program),
        datasets=dataset_ids,
        cache_root=args.cache_root,
        sample_queries=10**9,
        warmup_queries=args.warmup_queries,
        timed_repeats=args.timed_repeats,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        qrels_mode=args.qrels_mode,
        auto_cache=True,
        cache_model=args.cache_model,
        cache_dtype=args.cache_dtype,
        cache_batch_size=args.cache_batch_size,
        beir_data_dir=args.beir_data_dir,
        progress=True,
        output_path=output,
        resume=not args.no_resume,
    )
    print(f"[new-benchmark-baseline] wrote {output}", flush=True)
    return 0


def _dataset_id(spec: str) -> str:
    benchmark, name = normalize_dataset_spec(spec).split(":", maxsplit=1)
    if benchmark == "bright_pro":
        return f"bright_pro_{name}"
    return f"{benchmark}_{name}"


def _default_output_path(*, suite: str, qrels_mode: str, device: str, model_name: str) -> Path:
    model_slug = safe_model_name(model_name).replace("__", "_").lower()
    return (
        Path("tasks/late_interaction/baselines")
        / f"exact_maxsim.{model_slug}.new_benchmarks_{suite}.{qrels_mode}.{device}.json"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
