"""Tests for core.manifest: writes valid run identification, updates on close."""
from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.core.manifest import (
    Manifest,
    build_manifest,
    make_run_id,
    update_manifest,
    write_manifest,
)


def test_manifest_write_and_update(tmp_path: Path, record_io):
    run_id = make_run_id("bm25")

    def run() -> dict:
        m = build_manifest(
            run_id=run_id, task="bm25", config_path="tasks/bm25/configs/freeform.yaml",
            repo_root=tmp_path,  # not a git repo; git fields will be None
        )
        write_manifest(tmp_path, m)
        update_manifest(tmp_path, ended_at="2026-04-30T00:00:00Z", exit_status="ok")
        data = json.loads((tmp_path / "manifest.json").read_text())
        return {
            "task": data["task"],
            "ended_at": data["ended_at"],
            "exit_status": data["exit_status"],
            "git_sha": data["git_sha"],
        }

    out = record_io(
        module="src/ranking_evolved/core/manifest.py",
        function="write_manifest+update_manifest",
        input={"task": "bm25", "config_path": "tasks/bm25/configs/freeform.yaml"},
        run=run,
    )
    assert out == {
        "task": "bm25",
        "ended_at": "2026-04-30T00:00:00Z",
        "exit_status": "ok",
        "git_sha": None,
    }
