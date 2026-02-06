#!/usr/bin/env python3
"""
Plot average Recall@100 and nDCG@10 across evolution steps (best-so-far).

Creates TWO side-by-side graphs:
- Left: Avg Recall@100
- Right: Avg nDCG@10

Each graph includes:
- Solid line = evolved ranker (best-so-far average)
- Red dotted line = classic BM25 (seed)

Clean, paper-friendly layout.
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
            if line:
                records.append(json.loads(line))
    return records


def best_so_far_series(records: list[dict]) -> tuple[list[int], list[dict]]:
    steps = [0]
    seed_metrics = records[0]["parent_metrics"]
    best_score = float(seed_metrics.get("combined_score", 0.0))
    best_metrics = seed_metrics
    best_metrics_per_step = [best_metrics]

    for r in records:
        steps.append(int(r["iteration"]))
        child_metrics = r.get("child_metrics", {}) or {}
        child_score = float(child_metrics.get("combined_score", 0.0))

        if child_score >= best_score:
            best_score = child_score
            best_metrics = child_metrics

        best_metrics_per_step.append(best_metrics)

    return steps, best_metrics_per_step


def dataset_names_from_metrics(metrics: dict) -> list[str]:
    seen = set()
    for k in metrics:
        if k.endswith("_ndcg@10") and k != "avg_ndcg@10":
            seen.add(k.replace("_ndcg@10", ""))
        if k.endswith("_recall@100") and k != "avg_recall@100":
            seen.add(k.replace("_recall@100", ""))
    return sorted(seen)


def avg_metric(m: dict, datasets: list[str], suffix: str) -> float:
    vals = [float(m.get(f"{ds}_{suffix}", 0.0)) for ds in datasets]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--figsize", type=str, default="7.0,3.0")
    args = parser.parse_args()

    trace_path = args.output_dir / "evolution_trace.jsonl"
    if not trace_path.exists():
        raise SystemExit(f"Trace not found: {trace_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise SystemExit("matplotlib is required.")

    records = load_trace(trace_path)
    if not records:
        raise SystemExit("No records in trace.")

    steps, best_metrics_per_step = best_so_far_series(records)
    datasets = dataset_names_from_metrics(best_metrics_per_step[0])

    w, h = (float(x.strip()) for x in args.figsize.split(","))
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 2.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(w, h), sharex=True)

    # averages
    avg_recall = [avg_metric(m, datasets, "recall@100") for m in best_metrics_per_step]
    avg_ndcg = [avg_metric(m, datasets, "ndcg@10") for m in best_metrics_per_step]

    seed = best_metrics_per_step[0]
    seed_recall = avg_metric(seed, datasets, "recall@100")
    seed_ndcg = avg_metric(seed, datasets, "ndcg@10")

    # Recall plot
    ax_r.plot(steps, avg_recall, label="Avg Recall@100")
    ax_r.plot(
        steps,
        [seed_recall] * len(steps),
        linestyle=":",
        color="red",
        label="Classic BM25",
    )

    ax_r.set_title("Average Recall@100")
    ax_r.set_xlabel("Step")
    ax_r.set_ylabel("Recall@100")
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(frameon=False)

    # nDCG plot
    ax_n.plot(steps, avg_ndcg, label="Avg nDCG@10")
    ax_n.plot(
        steps,
        [seed_ndcg] * len(steps),
        linestyle=":",
        color="red",
        label="Classic BM25",
    )

    ax_n.set_title("Average nDCG@10")
    ax_n.set_xlabel("Step")
    ax_n.set_ylabel("nDCG@10")
    ax_n.grid(True, alpha=0.25)
    ax_n.legend(frameon=False)

    fig.tight_layout(pad=0.5, w_pad=1.0)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=250)
        print(f"Saved: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
