"""Streaming JSONL projection of the run.

`trace.jsonl` is a derived artifact — the controller writes one line per
iteration, capturing the same fields persisted to the `iterations` table
in `run.db` plus a few convenience fields. Downstream tools (existing
plotting scripts, live dashboards) tail this file. The DB remains the
source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceWriter:
    """Append-only JSONL writer for per-iteration events."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def append(
        self,
        *,
        iteration: int,
        parent_id: str | None,
        child_id: str | None,
        parent_metrics: dict[str, float] | None,
        child_metrics: dict[str, float] | None,
        improvement_delta: float | None,
        prompt: dict[str, str] | None,
        llm_response: str | None,
        diff_summary: dict[str, Any] | None,
        island: int | None,
        eval_duration_s: float | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "schema_version": self.SCHEMA_VERSION,
            "iteration": iteration,
            "parent_id": parent_id,
            "child_id": child_id,
            "parent_metrics": parent_metrics,
            "child_metrics": child_metrics,
            "improvement_delta": improvement_delta,
            "prompt": prompt,
            "llm_response": llm_response,
            "diff_summary": diff_summary,
            "island": island,
            "eval_duration_s": eval_duration_s,
        }
        if extra:
            event.update(extra)
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
