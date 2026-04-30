"""
Test dashboard infrastructure.

Provides the `record_io` fixture: every call records {module, function, input, output,
status, duration} into a session-scoped accumulator. After the test session ends, results
are written to:

- reports/test_dashboard.json (machine-readable)
- reports/test_dashboard.html (rendered table, single-page, self-contained)

Usage in a test:

    def test_apply_diff(record_io):
        out = record_io(
            module="src/ranking_evolved/prompts/diff.py",
            function="apply_search_replace",
            input={"parent": parent, "diff": diff},
            run=lambda: apply_search_replace(parent, diff),
        )
        assert out == expected
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from ranking_evolved._test_dashboard import write_dashboard

REPO_ROOT = Path(__file__).resolve().parents[1]

_SESSION_RECORDS: list[dict[str, Any]] = []
_MAX_FIELD_CHARS = 4000


@pytest.fixture
def record_io(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Run a callable, record its exact input/output, and return the output."""

    def _record(
        *,
        module: str,
        function: str,
        input: Any,
        run: Callable[[], Any],
    ) -> Any:
        start = time.perf_counter()
        status = "passed"
        error: str | None = None
        output: Any = None
        try:
            output = run()
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            _SESSION_RECORDS.append(
                {
                    "test_id": request.node.nodeid,
                    "module": module,
                    "function": function,
                    "input": _safe_display(input),
                    "output": _safe_display(output) if status == "passed" else None,
                    "status": status,
                    "error": error,
                    "duration_ms": duration_ms,
                }
            )
        return output

    return _record


def _safe_display(value: Any) -> str:
    """Convert any value to a bounded display string."""
    try:
        text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > _MAX_FIELD_CHARS:
        text = text[:_MAX_FIELD_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the dashboard at session end (only if record_io was used).

    Skipping the write when no records were captured prevents legacy / unrelated
    test runs (`pytest tests/test_bm25.py`) from clobbering a real dashboard.
    """
    if not _SESSION_RECORDS:
        return
    write_dashboard(
        repo_root=REPO_ROOT,
        records=_SESSION_RECORDS,
        exit_status=int(exitstatus),
    )
