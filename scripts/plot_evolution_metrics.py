#!/usr/bin/env python3
"""
Plot per-dataset nDCG@10 and Recall@100 across evolution steps.

Reads evolution_trace.jsonl from an OpenEvolve run and plots:
- X-axis: step (0 = seed, 1..N = iterations)
- Y-axis: metric value
- One line per dataset; "best-so-far" at each step (by combined_score).

Usage:
  uv run python scripts/plot_evolution_metrics.py OUTPUT_DIR [--save FIGURE_PATH]
  e.g. uv run python scripts/plot_evolution_metrics.py output/openevolve_output_freeform_fast/20260201_215150 --save evolution_metrics.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_trace(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def best_so_far_series(records: list[dict]) -> tuple[list[int], list[dict]]:
    """
    For each step t, return the metrics of the best program so far (by combined_score).
    Step 0 = seed (first record's parent_metrics); steps 1..N = best among seed and all children.
    Returns (steps, list of metric dicts per step).
    """
    if not records:
        return [], []

    steps = [0]
    seed_metrics = records[0]["parent_metrics"]
    seed_score = seed_metrics.get("combined_score", 0.0)
    best_scores = [seed_score]
    best_metrics_per_step = [seed_metrics]

    for r in records:
        child_metrics = r.get("child_metrics", {})
        child_score = child_metrics.get("combined_score", 0.0)
        steps.append(r["iteration"])
        if child_score >= best_scores[-1]:
            best_scores.append(child_score)
            best_metrics_per_step.append(child_metrics)
        else:
            best_scores.append(best_scores[-1])
            best_metrics_per_step.append(best_metrics_per_step[-1])

    return steps, best_metrics_per_step


def dataset_names_from_metrics(metrics: dict) -> list[str]:
    """Extract dataset names from keys like beir_nfcorpus_ndcg@10, bright_pony_recall@100."""
    seen: set[str] = set()
    for k in metrics:
        if k.endswith("_ndcg@10") and k != "avg_ndcg@10":
            seen.add(k.replace("_ndcg@10", ""))
        if k.endswith("_recall@100") and k != "avg_recall@100":
            seen.add(k.replace("_recall@100", ""))
    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot nDCG@10 and Recall@100 per dataset across steps")
    parser.add_argument("output_dir", type=Path, help="OpenEvolve run dir containing evolution_trace.jsonl")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not show interactive plot (only save)")
    args = parser.parse_args()

    trace_path = args.output_dir / "evolution_trace.jsonl"
    if not trace_path.exists():
        raise SystemExit(f"Trace not found: {trace_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise SystemExit("matplotlib is required. Install with: uv add --dev matplotlib")

    records = load_trace(trace_path)
    if not records:
        raise SystemExit("No records in trace.")

    steps, best_metrics_per_step = best_so_far_series(records)
    datasets = dataset_names_from_metrics(best_metrics_per_step[0])
    if not datasets:
        raise SystemExit("No per-dataset metrics found in trace.")

    ndcg_key = lambda d: f"{d}_ndcg@10"
    recall_key = lambda d: f"{d}_recall@100"

    fig, (ax_ndcg, ax_recall) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ds in datasets:
        nk = ndcg_key(ds)
        rk = recall_key(ds)
        ndcg_vals = [m.get(nk, 0.0) for m in best_metrics_per_step]
        recall_vals = [m.get(rk, 0.0) for m in best_metrics_per_step]
        ax_ndcg.plot(steps, ndcg_vals, label=ds, alpha=0.8)
        ax_recall.plot(steps, recall_vals, label=ds, alpha=0.8)

    ax_ndcg.set_ylabel("nDCG@10")
    ax_ndcg.set_title("Best-so-far nDCG@10 per dataset")
    ax_ndcg.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    ax_ndcg.grid(True, alpha=0.3)

    ax_recall.set_ylabel("Recall@100")
    ax_recall.set_xlabel("Step (0 = seed)")
    ax_recall.set_title("Best-so-far Recall@100 per dataset")
    ax_recall.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    ax_recall.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
