"""Replay dashboard renderer.

Reads `<run_dir>/replay/step_*.json` and (optionally)
`<run_dir>/replay/reference/step_*.json`, produces a single-page HTML
report at `reports/replay_dashboard.html` (or a custom path) with:

  * an index table of all steps (iteration, parent, child, scores, Δ,
    diff status, migration flag, plus a "ref?" column),
  * per-step `<details>` blocks expanding to show every section of the
    ReplayStep with the reference (if present) rendered side-by-side.

Mismatches in scalar fields are highlighted in red. Code/prompt/raw_response
fields are not diffed — the user is meant to scan visually for those.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_dashboard(
    run_dir: Path,
    *,
    out_path: Path,
    title: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    steps = sorted((run_dir / "replay").glob("step_*.json"))
    if not steps:
        raise FileNotFoundError(f"no replay/step_*.json under {run_dir}")
    ref_dir = run_dir / "replay" / "reference"
    refs = {p.name: p for p in ref_dir.glob("step_*.json")} if ref_dir.exists() else {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render(steps=steps, refs=refs, title=title or f"Replay Dashboard — {run_dir.name}")
    )
    return out_path


def _render(*, steps: list[Path], refs: dict[str, Path], title: str) -> str:
    rows_html: list[str] = []
    details_html: list[str] = []
    for step_path in steps:
        step = json.loads(step_path.read_text())
        ref_path = refs.get(step_path.name)
        ref = json.loads(ref_path.read_text()) if ref_path else None
        rows_html.append(_index_row(step, ref))
        details_html.append(_step_section(step, ref))

    return _PAGE.format(
        title=html.escape(title),
        rows="\n".join(rows_html),
        details="\n".join(details_html),
    )


def _index_row(step: dict, ref: dict | None) -> str:
    it = step["iteration"]
    parent_id = _short_id(step.get("parent", {}).get("id"))
    child_eval = step.get("child_eval") or {}
    child_metrics = child_eval.get("metrics") or {}
    child_score = _score_str(child_metrics.get("combined_score"))
    parent_score = _score_str((step.get("parent") or {}).get("metrics", {}).get("combined_score"))
    delta = _delta_str(parent_score, child_score)
    island = (step.get("admission") or {}).get("target_island")
    migrated = (step.get("admission") or {}).get("migration_fired")
    diff = step.get("diff") or {}
    diff_status = _diff_status(diff)
    has_ref = "✓" if ref else ""
    return (
        f'<tr>'
        f'<td><a href="#step-{it:04d}">{it}</a></td>'
        f'<td>{parent_id}</td>'
        f'<td>{parent_score}</td>'
        f'<td>{child_score}</td>'
        f'<td>{delta}</td>'
        f'<td>{island}</td>'
        f'<td>{"✓" if migrated else ""}</td>'
        f'<td>{diff_status}</td>'
        f'<td>{has_ref}</td>'
        f'</tr>'
    )


def _step_section(step: dict, ref: dict | None) -> str:
    it = step["iteration"]
    body_parts: list[str] = []

    # Sampling
    body_parts.append(_pair_section("Sampling", step.get("sampling"), ref and ref.get("sampling")))

    # Parent
    body_parts.append(_pair_section("Parent program", step.get("parent"), ref and ref.get("parent")))

    # Inspirations / Top / Previous
    for label, key in [("Inspirations", "inspirations"), ("Top programs", "top_programs"),
                       ("Previous programs", "previous_programs")]:
        body_parts.append(_pair_section(label, step.get(key), ref and ref.get(key)))

    # Prompt
    body_parts.append(_prompt_section(step.get("prompt"), ref and ref.get("prompt")))

    # LLM
    body_parts.append(_llm_section(step.get("llm"), ref and ref.get("llm")))

    # Diff
    body_parts.append(_diff_section(step.get("diff"), ref and ref.get("diff")))

    # Child code
    body_parts.append(_codepair_section("Child code", step.get("child_code"), ref and ref.get("child_code")))

    # Child eval
    body_parts.append(_pair_section("Child evaluation", step.get("child_eval"), ref and ref.get("child_eval")))

    # DB snapshots
    body_parts.append(_pair_section("Population (before)", step.get("db_before"), ref and ref.get("db_before")))
    body_parts.append(_pair_section("Population (after)", step.get("db_after"), ref and ref.get("db_after")))

    # Admission
    body_parts.append(_pair_section("Admission", step.get("admission"), ref and ref.get("admission")))

    return (
        f'<details id="step-{it:04d}">'
        f'<summary>Iteration {it}</summary>'
        f'<div class="step">{"".join(body_parts)}</div>'
        f'</details>'
    )


def _pair_section(label: str, ours: Any, theirs: Any) -> str:
    if theirs is None:
        return _single_section(label, ours)
    return (
        f'<section class="pair">'
        f'<h3>{html.escape(label)}</h3>'
        f'<div class="cols">'
        f'<div class="col"><h4>ours</h4>{_pretty(ours)}</div>'
        f'<div class="col ref"><h4>reference</h4>{_pretty(theirs)}</div>'
        f'</div>'
        f'{_compare_scalars(ours, theirs)}'
        f'</section>'
    )


def _single_section(label: str, ours: Any) -> str:
    return (
        f'<section class="single"><h3>{html.escape(label)}</h3>{_pretty(ours)}</section>'
    )


def _prompt_section(ours: Any, theirs: Any) -> str:
    return _pair_section("Prompt", ours, theirs)


def _llm_section(ours: Any, theirs: Any) -> str:
    return _pair_section("LLM", ours, theirs)


def _diff_section(ours: Any, theirs: Any) -> str:
    return _pair_section("Diff application", ours, theirs)


def _codepair_section(label: str, ours: Any, theirs: Any) -> str:
    if not ours and not theirs:
        return ""
    if theirs is None:
        return (
            f'<section class="single"><h3>{html.escape(label)}</h3>'
            f'<pre class="code">{html.escape(ours or "")}</pre></section>'
        )
    return (
        f'<section class="pair">'
        f'<h3>{html.escape(label)}</h3>'
        f'<div class="cols">'
        f'<div class="col"><h4>ours</h4><pre class="code">{html.escape(ours or "")}</pre></div>'
        f'<div class="col ref"><h4>reference</h4><pre class="code">{html.escape(theirs or "")}</pre></div>'
        f'</div>'
        f'</section>'
    )


def _pretty(obj: Any) -> str:
    try:
        return f'<pre>{html.escape(json.dumps(obj, indent=2, ensure_ascii=False, default=str))}</pre>'
    except Exception:
        return f'<pre>{html.escape(repr(obj))}</pre>'


def _compare_scalars(ours: Any, theirs: Any) -> str:
    """Find scalar field mismatches; return a small <ul> of them, or empty."""
    if not isinstance(ours, dict) or not isinstance(theirs, dict):
        return ""
    diffs: list[str] = []
    for k, v in ours.items():
        if k not in theirs:
            continue
        ref_v = theirs[k]
        if isinstance(v, (str, int, float, bool)) and isinstance(ref_v, (str, int, float, bool)):
            if v != ref_v:
                diffs.append(
                    f'<li><strong>{html.escape(k)}</strong>: '
                    f'<code>{html.escape(str(v))}</code> ≠ '
                    f'<code class="ref">{html.escape(str(ref_v))}</code></li>'
                )
    return f'<ul class="mismatches">{"".join(diffs)}</ul>' if diffs else ""


def _short_id(s: str | None) -> str:
    if not s:
        return ""
    return s if len(s) <= 12 else s[:12] + "…"


def _score_str(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _delta_str(parent: str, child: str) -> str:
    try:
        d = float(child) - float(parent)
        return f"{d:+.4f}"
    except Exception:
        return ""


def _diff_status(diff: dict) -> str:
    if diff.get("fatal_error"):
        return f'<span class="err">{html.escape(diff["fatal_error"])}</span>'
    n_e, n_a = diff.get("n_extracted"), diff.get("n_applied")
    if n_e is None:
        return "—"
    return f"{n_a}/{n_e}"


_PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin: 0 0 16px; }}
  table.index {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  table.index th, table.index td {{
    border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 13px;
  }}
  table.index th {{ background: #f5f5f5; }}
  details {{ margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; padding: 6px 10px; }}
  details summary {{ font-weight: 600; cursor: pointer; }}
  section {{ margin: 12px 0; padding: 8px; border-left: 3px solid #eee; }}
  section h3 {{ margin: 0 0 6px; font-size: 14px; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .col h4 {{ margin: 4px 0; font-size: 12px; color: #666; }}
  .col.ref {{ background: #f9f9ff; }}
  pre {{ background: #f7f7f7; padding: 8px; overflow-x: auto; font-size: 12px;
         max-height: 320px; white-space: pre-wrap; word-break: break-word; }}
  pre.code {{ font-family: ui-monospace, monospace; }}
  ul.mismatches li {{ color: #b00; }}
  ul.mismatches code.ref {{ color: #06b; }}
  span.err {{ color: #b00; font-weight: 600; }}
</style>
</head>
<body>
<h1>{title}</h1>
<table class="index">
  <thead>
    <tr>
      <th>Iter</th><th>Parent</th><th>Parent score</th><th>Child score</th>
      <th>Δ</th><th>Island</th><th>Migrated</th><th>Diff</th><th>Ref</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

{details}

</body>
</html>
"""
