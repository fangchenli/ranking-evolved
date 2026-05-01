"""Tests for core.replay_dashboard: HTML render with optional reference column."""
from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.core.replay_dashboard import render_dashboard


def _write_step(path: Path, iteration: int, *, parent_score: float, child_score: float) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "iteration": iteration,
        "sampling": {"parent_id": "p", "parent_island": 0},
        "parent": {
            "id": "p", "parent_id": None, "island": 0,
            "generation": 0, "iteration_found": 0,
            "metrics": {"combined_score": parent_score}, "feature_coords": {},
            "code_sha256": "", "code_preview": "",
        },
        "inspirations": [],
        "top_programs": [],
        "previous_programs": [],
        "parent_artifacts": None,
        "prompt": {"system": "s", "user": "u", "template_key": "diff_user"},
        "llm": {"proposer": "fake", "model": "m", "raw_response": "r",
                "tokens_in": None, "tokens_out": None, "latency_ms": 1.0},
        "diff": {"pattern": "p", "blocks": [], "n_extracted": 1, "n_applied": 1, "fatal_error": None},
        "child_code": "child",
        "child_eval": {"metrics": {"combined_score": child_score},
                       "per_dataset": {}, "artifacts": {}, "duration_s": 0.1, "error": None},
        "db_before": {"n_programs": 1, "islands": [], "island_generations": [],
                      "current_island": 0, "last_migration_generation": 0,
                      "archive_cells": {}, "archive_size": 0,
                      "best_program_id": None, "island_best_programs": []},
        "db_after": {"n_programs": 2, "islands": [], "island_generations": [],
                     "current_island": 0, "last_migration_generation": 0,
                     "archive_cells": {}, "archive_size": 0,
                     "best_program_id": None, "island_best_programs": []},
        "admission": {"target_island": 0, "feature_coords": {}, "cell_key": "0",
                      "evicted_program_id": None, "migration_fired": False,
                      "migration_details": None},
    }))


def test_dashboard_renders_index_and_step_details(tmp_path: Path, record_io):
    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    _write_step(run_dir / "replay" / "step_0001.json", 1, parent_score=0.4, child_score=0.5)
    _write_step(run_dir / "replay" / "step_0002.json", 2, parent_score=0.5, child_score=0.55)

    def run() -> dict:
        out = run_dir / "replay_dashboard.html"
        rendered = render_dashboard(run_dir, out_path=out)
        text = rendered.read_text()
        return {
            "exists": rendered.exists(),
            "has_index_table": '<table class="index">' in text,
            "rows_for_iters": all(f">{i}<" in text or f'href="#step-000{i}"' in text for i in (1, 2)),
            "has_step_anchors": all(f'id="step-{i:04d}"' in text for i in (1, 2)),
            "shows_diff_status": '1/1' in text,
            "no_reference_marker_when_absent": "Right contains" not in text,  # sanity
        }

    out = record_io(
        module="src/ranking_evolved/core/replay_dashboard.py",
        function="render_dashboard (no reference)",
        input={"n_steps": 2},
        run=run,
    )
    assert out["exists"] is True
    assert out["has_index_table"] is True
    assert out["rows_for_iters"] is True
    assert out["has_step_anchors"] is True
    assert out["shows_diff_status"] is True


def test_dashboard_renders_reference_side_by_side(tmp_path: Path, record_io):
    """Reference column rendered, and mismatches at the scalar level surface in <ul class=mismatches>."""
    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "reference").mkdir(parents=True)
    _write_step(run_dir / "replay" / "step_0001.json", 1, parent_score=0.4, child_score=0.5)
    # Build a reference whose `db_after` has a scalar field (n_programs) that
    # differs from ours — this triggers _compare_scalars to render a mismatch
    # entry. (Score lives deep inside child_eval.metrics; mismatch detection
    # only sees direct scalar children of a section dict.)
    _write_step(run_dir / "replay" / "reference" / "step_0001.json",
                1, parent_score=0.4, child_score=0.5)
    ref_path = run_dir / "replay" / "reference" / "step_0001.json"
    ref = json.loads(ref_path.read_text())
    ref["db_after"]["n_programs"] = 999  # differ from our 2
    ref["db_after"]["current_island"] = 7  # differ from our 0
    ref_path.write_text(json.dumps(ref))

    def run() -> dict:
        out = run_dir / "replay_dashboard.html"
        rendered = render_dashboard(run_dir, out_path=out)
        text = rendered.read_text()
        return {
            "has_ref_col": "<h4>reference</h4>" in text,
            "shows_mismatch": "ul class=\"mismatches\"" in text,
            "ref_marker_in_index": ">✓<" in text,
        }

    out = record_io(
        module="src/ranking_evolved/core/replay_dashboard.py",
        function="render_dashboard (with reference)",
        input={"n_steps": 1, "reference_score_diff": 0.01},
        run=run,
    )
    assert out["has_ref_col"] is True
    assert out["shows_mismatch"] is True
    assert out["ref_marker_in_index"] is True
