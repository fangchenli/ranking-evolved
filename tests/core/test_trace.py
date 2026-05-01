"""Tests for core.trace: append-only JSONL projection."""
from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.core.trace import TraceWriter


def test_trace_appends_line_per_iteration(tmp_path: Path, record_io):
    path = tmp_path / "trace.jsonl"

    def run() -> list[dict]:
        with TraceWriter(path) as t:
            for i in range(3):
                t.append(
                    iteration=i,
                    parent_id=f"p{i-1}" if i else None,
                    child_id=f"p{i}",
                    parent_metrics={"combined_score": 0.5} if i else None,
                    child_metrics={"combined_score": 0.5 + i * 0.1},
                    improvement_delta=0.1 if i else None,
                    prompt={"system": "s", "user": "u"},
                    llm_response="resp",
                    diff_summary={"n_blocks": 1},
                    island=i % 2,
                    eval_duration_s=0.05,
                )
        return [json.loads(line) for line in path.read_text().splitlines()]

    out = record_io(
        module="src/ranking_evolved/core/trace.py",
        function="TraceWriter.append",
        input={"n_iters": 3},
        run=run,
    )
    assert len(out) == 3
    assert [e["iteration"] for e in out] == [0, 1, 2]
    assert all(e["schema_version"] == TraceWriter.SCHEMA_VERSION for e in out)
    assert out[0]["parent_id"] is None
    assert out[2]["child_metrics"]["combined_score"] == 0.7
