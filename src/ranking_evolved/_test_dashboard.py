"""Shared writer for the test dashboard (used by tests/conftest.py and cli.py).

Single source of truth for the JSON + HTML output, so the CLI can produce a valid
empty dashboard without invoking pytest at all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def write_dashboard(
    *,
    repo_root: Path,
    records: list[dict[str, Any]],
    exit_status: int,
) -> tuple[Path, Path]:
    """Write dashboard JSON + HTML under <repo_root>/reports/. Returns (json_path, html_path)."""
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "test_dashboard.json"
    html_path = reports_dir / "test_dashboard.html"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_status": int(exit_status),
        "total": len(records),
        "passed": sum(1 for r in records if r["status"] == "passed"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return json_path, html_path


def _render_html(payload: dict[str, Any]) -> str:
    if payload["records"]:
        rows = "\n".join(_render_row(r) for r in payload["records"])
    else:
        rows = (
            '<tr><td colspan="6" class="empty">'
            "No <code>record_io</code> entries yet &mdash; dashboard is empty."
            "</td></tr>"
        )
    return _HTML_TEMPLATE.format(
        generated_at=_html_escape(payload["generated_at"]),
        exit_status=payload["exit_status"],
        total=payload["total"],
        passed=payload["passed"],
        failed=payload["failed"],
        rows=rows,
    )


def _render_row(record: dict[str, Any]) -> str:
    icon = "PASS" if record["status"] == "passed" else "FAIL"
    status_class = f"status-{record['status']}"
    output_field = (
        record["output"] if record["status"] == "passed" else (record.get("error") or "")
    )
    return (
        "<tr>"
        f"<td><code>{_html_escape(record['module'])}</code></td>"
        f"<td><code>{_html_escape(record['function'])}</code></td>"
        f"<td><pre>{_html_escape(record['input'])}</pre></td>"
        f"<td><pre>{_html_escape(output_field)}</pre></td>"
        f"<td class='{status_class}'>{icon}</td>"
        f"<td class='duration'>{record['duration_ms']} ms</td>"
        "</tr>"
    )


def _html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ranking-evolved test dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 2rem; color: #1a1a1a; }}
h1 {{ margin-bottom: 0.25rem; }}
.meta {{ color: #666; margin-bottom: 1rem; }}
.summary span {{ display: inline-block; padding: 0.4rem 0.9rem; border-radius: 6px;
                margin-right: 0.5rem; font-weight: 600; font-size: 0.95rem; }}
.summary .total {{ background: #e6e6e6; color: #333; }}
.summary .pass  {{ background: #d4f4dd; color: #1b6e34; }}
.summary .fail  {{ background: #f8d4d4; color: #8a1c1c; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.55rem 0.75rem; text-align: left;
         vertical-align: top; }}
th {{ background: #f3f3f3; font-size: 0.92rem; }}
tr:nth-child(even) td {{ background: #fafafa; }}
td.status-passed {{ color: #1b6e34; font-weight: 700; }}
td.status-failed {{ color: #8a1c1c; font-weight: 700; }}
td.empty {{ text-align: center; color: #888; font-style: italic; padding: 2.5rem; }}
pre {{ margin: 0; font-size: 0.82rem; max-height: 240px; overflow: auto;
      white-space: pre-wrap; word-break: break-word; background: #f6f8fa;
      padding: 0.5rem; border-radius: 4px; }}
.duration {{ color: #555; font-variant-numeric: tabular-nums; white-space: nowrap; }}
code {{ font-size: 0.88rem; }}
</style>
</head>
<body>
<h1>ranking-evolved test dashboard</h1>
<div class="meta">Generated {generated_at} &middot; pytest exit {exit_status}</div>
<div class="summary">
  <span class="total">Total: {total}</span>
  <span class="pass">Passed: {passed}</span>
  <span class="fail">Failed: {failed}</span>
</div>
<table>
  <thead><tr>
    <th>Module</th><th>Function</th><th>Input</th><th>Output</th>
    <th>Status</th><th>Time</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
</body>
</html>
"""
