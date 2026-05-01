"""Run-level optimization plots.

The controller calls this after a run finishes. The functions are also usable
from scripts/plot_run_metrics.py for regenerating plots after older runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment_summary import PROGRAM_METRICS_FILE


PLOTS_DIR = "plots"
OPTIMIZATION_CURVES = "optimization_curves.pdf"
OBJECTIVE_COMPONENTS = "objective_components.pdf"
LATENCY_TRADEOFF = "latency_tradeoff.pdf"


def generate_run_plots(run_dir: str | Path) -> list[Path]:
    """Generate/overwrite standard plots for one run directory.

    Returns paths written. Raises FileNotFoundError if program_metrics.jsonl is
    absent; lets callers decide whether plotting is required or best-effort.
    """
    run_dir = Path(run_dir)
    rows = load_program_metrics(run_dir / PROGRAM_METRICS_FILE)
    if not rows:
        return []

    plot_dir = run_dir / PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("matplotlib is required to generate run plots") from exc

    _remove_legacy_pngs(plot_dir)
    _set_style(plt, fm)

    written = [
        _plot_optimization_curves(plt, rows, plot_dir / OPTIMIZATION_CURVES),
        _plot_objective_components(plt, rows, plot_dir / OBJECTIVE_COMPONENTS),
        _plot_latency_tradeoff(plt, rows, plot_dir / LATENCY_TRADEOFF),
    ]
    return written


def load_program_metrics(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def best_so_far_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, Any] | None = None
    out: list[dict[str, Any]] = []
    for row in rows:
        if best is None or _float(row.get("combined_score")) >= _float(best.get("combined_score")):
            best = row
        out.append(best)
    return out


def _plot_optimization_curves(plt, rows: list[dict[str, Any]], out: Path) -> Path:  # type: ignore[no-untyped-def]
    steps = [_iteration(row) for row in rows]
    scores = [_float(row.get("combined_score")) for row in rows]
    best_scores = [_float(row.get("combined_score")) for row in best_so_far_rows(rows)]
    seed_score = scores[0] if scores else 0.0

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(steps, scores, color="#9ecae1", linewidth=1.4, alpha=0.75, label="Candidate")
    ax.plot(steps, best_scores, color="#1f77b4", linewidth=2.4, label="Best so far")
    ax.axhline(seed_score, color="#d62728", linestyle=":", linewidth=1.6, label="Seed")
    ax.set_title("Optimization Target", fontsize=15.5, fontweight="bold", pad=6)
    ax.set_xlabel("Evolution Step", fontsize=17.5)
    ax.set_ylabel("Combined Score", fontsize=17.5)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.86)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_objective_components(plt, rows: list[dict[str, Any]], out: Path) -> Path:  # type: ignore[no-untyped-def]
    steps = [_iteration(row) for row in rows]
    best_rows = best_so_far_rows(rows)

    recall_key = _first_key(rows, "avg_recall@")
    ndcg_key = _first_key(rows, "avg_ndcg@")
    series = [
        ("Combined score", "combined_score"),
        (recall_key.replace("avg_", "Avg ") if recall_key else "Avg recall", recall_key),
        (ndcg_key.replace("avg_", "Avg ") if ndcg_key else "Avg nDCG", ndcg_key),
        ("Latency score", "avg_latency_score"),
        ("Query latency median (ms)", "avg_query_latency_median_ms"),
        ("Recall component", "objective_recall_component"),
        ("nDCG component", "objective_ndcg_component"),
        ("Latency component", "objective_latency_component"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0), sharex=True)
    axes_flat = list(axes.ravel())
    for ax, (title, key) in zip(axes_flat, series):
        if key is None or not any(key in row for row in rows):
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.15)
            continue

        values = [_float(row.get(key)) for row in rows]
        best_values = [_float(row.get(key)) for row in best_rows]
        seed = values[0] if values else 0.0
        ax.plot(steps, values, color="#bdbdbd", linewidth=1.2, alpha=0.85, label="Candidate")
        ax.plot(steps, best_values, color="#1f77b4", linewidth=2.2, label="Best so far")
        ax.axhline(seed, color="#d62728", linestyle=":", linewidth=1.4)
        ax.set_title(title, fontsize=15.5, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", which="both", direction="out")

    for ax in axes[-1]:
        ax.set_xlabel("Evolution Step", fontsize=17.5)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.11, top=0.86, wspace=0.35, hspace=0.42)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_latency_tradeoff(plt, rows: list[dict[str, Any]], out: Path) -> Path:  # type: ignore[no-untyped-def]
    recall_key = _first_key(rows, "avg_recall@")
    if recall_key is None or not any("avg_query_latency_median_ms" in row for row in rows):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "latency/recall metrics not available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return out

    xs = [_float(row.get("avg_query_latency_median_ms")) for row in rows]
    ys = [_float(row.get(recall_key)) for row in rows]
    scores = [_float(row.get("combined_score")) for row in rows]
    best = max(rows, key=lambda row: _float(row.get("combined_score")))
    seed = rows[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    scatter = ax.scatter(xs, ys, c=scores, cmap="viridis", s=38, alpha=0.85)
    ax.scatter(
        [_float(seed.get("avg_query_latency_median_ms"))],
        [_float(seed.get(recall_key))],
        marker="x",
        color="#d62728",
        s=90,
        linewidths=2.0,
        label="Seed",
    )
    ax.scatter(
        [_float(best.get("avg_query_latency_median_ms"))],
        [_float(best.get(recall_key))],
        marker="*",
        color="#ffbf00",
        edgecolor="black",
        s=180,
        linewidths=0.7,
        label="Best",
    )
    ax.set_title("Recall / Latency Tradeoff", fontsize=15.5, fontweight="bold", pad=6)
    ax.set_xlabel("Avg Median Query Latency (ms)", fontsize=17.5)
    ax.set_ylabel(recall_key.replace("avg_", "Avg "), fontsize=17.5)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Combined Score")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.86)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _set_style(plt, fm) -> None:  # type: ignore[no-untyped-def]
    _register_palatino_bold(fm)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua", "URW Palladio L"],
            "font.size": 15.5,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 12.0,
            "lines.linewidth": 2.4,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _register_palatino_bold(fm) -> None:  # type: ignore[no-untyped-def]
    palatino = Path("/System/Library/Fonts/Palatino.ttc")
    if not palatino.exists():
        return
    try:
        import tempfile
        from fontTools.ttLib import TTCollection

        ttc = TTCollection(str(palatino))
        for ttfont in ttc.fonts:
            subfamily = ttfont["name"].getDebugName(2)
            if subfamily and "Bold" in subfamily:
                tmp = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
                ttfont.save(tmp.name)
                tmp.close()
                fm.fontManager.addfont(tmp.name)
    except Exception:
        pass


def _remove_legacy_pngs(plot_dir: Path) -> None:
    for name in ("optimization_curves.png", "objective_components.png", "latency_tradeoff.png"):
        path = plot_dir / name
        if path.exists():
            path.unlink()


def _first_key(rows: list[dict[str, Any]], prefix: str) -> str | None:
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                return key
    return None


def _iteration(row: dict[str, Any]) -> int:
    return int(row.get("iteration_found", row.get("iteration", 0)))


def _float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
