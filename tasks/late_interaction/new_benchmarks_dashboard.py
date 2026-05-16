"""Render the OBLIQ-Bench / BRIGHT-Pro implementation review dashboard."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any

CURATED_DATASETS = [
    "bright_pro_economics",
    "bright_pro_stackoverflow",
    "bright_pro_earth_science",
    "obliq_twitter",
    "obliq_congress",
]


def render_dashboard(
    *,
    repo_root: Path = Path("."),
    out_path: Path = Path("reports/late_interaction_new_benchmarks_dashboard.html"),
) -> Path:
    repo_root = repo_root.resolve()
    out_path = (repo_root / out_path).resolve() if not out_path.is_absolute() else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(repo_root), encoding="utf-8")
    return out_path


def _render(repo_root: Path) -> str:
    baseline_dir = repo_root / "tasks" / "late_interaction" / "baselines"
    cache_root = repo_root / "cache" / "late_interaction" / "lightonai__LateOn"
    baselines = sorted(baseline_dir.glob("exact_maxsim.*new_benchmarks*.json"))
    return _PAGE.format(
        generated_at=html.escape(time.strftime("%Y-%m-%d %H:%M:%S")),
        cache_rows="\n".join(_cache_row(cache_root, name) for name in CURATED_DATASETS),
        baseline_rows="\n".join(_baseline_row(path) for path in baselines)
        or '<tr><td colspan="7" class="muted">No new-benchmark baseline JSON found yet.</td></tr>',
    )


def _cache_row(cache_root: Path, dataset: str) -> str:
    cache_dir = cache_root / dataset
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return (
            f"<tr><td><code>{html.escape(dataset)}</code></td>"
            '<td class="bad">missing</td><td colspan="5" class="muted">will auto-encode on first run</td></tr>'
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        f"<tr><td><code>{html.escape(dataset)}</code></td><td class='ok'>ready</td>"
        f"<td>{metadata.get('num_docs', '')}</td><td>{metadata.get('num_queries', '')}</td>"
        f"<td>{metadata.get('embedding_dim', '')}</td><td>{html.escape(str(metadata.get('qrels_modes', ['gold'])))}</td>"
        f"<td>{'yes' if metadata.get('has_aspect_annotations') else 'no'}</td></tr>"
    )


def _baseline_row(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"<tr><td><code>{html.escape(path.name)}</code></td><td colspan='6' class='bad'>invalid JSON</td></tr>"
    average = payload.get("_average") or {}
    fp = payload.get("_fingerprint") or {}
    return (
        f"<tr><td><code>{html.escape(path.name)}</code></td>"
        f"<td>{html.escape(str(fp.get('device', '')))}</td>"
        f"<td>{_fmt(average.get('recall_at_25'))}</td>"
        f"<td>{_fmt(average.get('ndcg_at_25'))}</td>"
        f"<td>{_fmt(average.get('alpha_ndcg_at_25'))}</td>"
        f"<td>{_fmt(average.get('aspect_recall_at_25'))}</td>"
        f"<td>{_fmt(average.get('median_query_latency_ms'))} ms</td></tr>"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/late_interaction_new_benchmarks_dashboard.html"))
    args = parser.parse_args(argv)
    rendered = render_dashboard(out_path=args.out)
    print(f"[new-benchmarks-dashboard] wrote {rendered}")
    return 0


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RankEvolve New Benchmarks Review</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #202124; }}
h1 {{ margin: 0 0 4px; }}
h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
.meta, .muted {{ color: #666; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; }}
th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: #f4f6f8; }}
code {{ font-size: 0.9em; }}
.ok {{ color: #0b6b2b; font-weight: 700; }}
.bad {{ color: #9a1c1c; font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
.panel {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; background: #fbfbfb; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>RankEvolve New Benchmarks Review</h1>
<div class="meta">Generated {generated_at}</div>

<h2>Benchmark Readout</h2>
<div class="grid">
  <div class="panel">
    <h3>OBLIQ-Bench</h3>
    <p>Five oblique retrieval tasks over math, writing, Twitter, WildChat, and congressional-hearing corpora. Relevance is latent: a document can be easy for a reasoning verifier to recognize after it is surfaced, while first-stage retrieval remains hard.</p>
    <p>Evaluation uses gold qrels by default and pooled qrels when requested. Math and writing include source-document exclusions that are filtered before scoring.</p>
    <p>Source: <a href="https://arxiv.org/abs/2605.06235">paper</a>, <a href="https://huggingface.co/datasets/dianetc/OBLIQ-Bench">dataset</a>.</p>
  </div>
  <div class="panel">
    <h3>BRIGHT-Pro</h3>
    <p>Seven StackExchange domains with 739 queries, 2,763 reasoning aspects, and 5,272 gold passages. Each gold passage maps to exactly one aspect, and aspect weights are normalized per query.</p>
    <p>Static evaluation defaults to α-nDCG@25 with α=0.5 and weighted A-Recall@25. Standard recall/nDCG are retained as diagnostics.</p>
    <p>Source: <a href="https://arxiv.org/abs/2605.04018">paper</a>, <a href="https://huggingface.co/datasets/yale-nlp/Bright-Pro">dataset</a>.</p>
  </div>
</div>

<h2>Curated Evolution Suite</h2>
<table>
<thead><tr><th>Dataset</th><th>Status</th><th>Docs</th><th>Queries</th><th>Dim</th><th>Qrels modes</th><th>Aspects</th></tr></thead>
<tbody>
{cache_rows}
</tbody>
</table>

<h2>Exact MaxSim Baselines</h2>
<p class="muted">Run: <code>uv run python -m tasks.late_interaction.run_new_benchmark_baseline --suite curated</code></p>
<table>
<thead><tr><th>File</th><th>Device</th><th>Recall@25</th><th>nDCG@25</th><th>α-nDCG@25</th><th>A-Recall@25</th><th>p50 latency</th></tr></thead>
<tbody>
{baseline_rows}
</tbody>
</table>

<h2>Paper Comparison Policy</h2>
<p>OBLIQ paper numbers are used as a setup sanity check, not as an exact equality target. The paper reports LateOn with PyLate/PLAID-style retrieval, while this baseline is exhaustive raw MaxSim over the same fixed LateOn token embeddings. Large unexplained metric gaps should block rollout; expected gaps should be attributed to qrels mode, cutoff, model revision, or exact-vs-indexed retrieval.</p>

<h2>Proof Checklist</h2>
<table>
<thead><tr><th>Capability</th><th>Proof source</th></tr></thead>
<tbody>
<tr><td>Dataset loaders</td><td><code>tests/tasks/late_interaction/test_new_benchmark_loaders.py</code></td></tr>
<tr><td>Aspect metrics</td><td><code>tests/test_metrics.py</code></td></tr>
<tr><td>Evaluator sidecars and dynamic metric keys</td><td><code>tests/tasks/late_interaction/test_smoke_evaluator.py</code></td></tr>
<tr><td>Objective metric aliases</td><td><code>tests/evaluation/test_objective_math.py</code></td></tr>
<tr><td>Config parse</td><td><code>tests/tasks/late_interaction/test_configs_load.py</code></td></tr>
</tbody>
</table>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
