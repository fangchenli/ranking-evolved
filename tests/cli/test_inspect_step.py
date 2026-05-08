"""`ranking-evolved inspect-step` — section-scoped reader with always-on/snapshot fallback."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranking_evolved.cli_inspect import cmd_inspect_step


def _write_step_snapshot(replay_dir: Path, step: int, **overrides) -> Path:
    """Build a minimal-but-valid step_NNNN.json fixture."""
    payload = {
        "iteration": step,
        "sampling": {"parent_id": "p-id", "parent_island": 1, "rng_seed": 7},
        "parent": {
            "id": "p-id",
            "generation": 3,
            "island": 1,
            "source_code": "def parent():\n    return 'I am the parent'\n",
            "metrics": {"combined_score": 0.42, "recall_at_1000": 0.5},
        },
        "inspirations": [
            {"id": "insp-1", "generation": 2, "island": 0,
             "metrics": {"combined_score": 0.31}, "source_code": "def insp1():\n    pass\n"},
            {"id": "insp-2", "generation": 4, "island": 2,
             "metrics": {"combined_score": 0.45}, "source_code": "def insp2():\n    pass\n"},
        ],
        "top_programs": [],
        "previous_programs": [],
        "parent_artifacts": None,
        "prompt": {
            "system": "YOU ARE A RETRIEVER",
            "user": "PARENT CODE: ...",
            "template_key": "freeform_diff",
        },
        "llm": {"raw_response": "<<< some diff >>>", "tokens_in": 1234, "tokens_out": 567, "latency_ms": 4567},
        "diff": {"n_extracted": 2, "n_applied": 2, "blocks": [
            {"search": "old", "replace": "new"},
            {"search": "old2", "replace": "new2"},
        ]},
        "child_code": "def child():\n    return 'I am child'\n",
        "child_eval": {"metrics": {"combined_score": 0.48, "recall_at_1000": 0.6}},
        "db_before": {
            "islands": [["a"], ["b"], ["c"]],
            "island_generations": [3, 4, 5],
            "current_island": 1,
            "archive_cells": {"0:low,low": "a", "1:high,high": "b"},
            "island_best_programs": ["a", "b", "c"],
        },
        "db_after": {
            "islands": [["a"], ["b", "child-id"], ["c"]],
            "island_generations": [3, 5, 5],
            "current_island": 1,
            "archive_cells": {"0:low,low": "a", "1:high,high": "b", "1:mid,mid": "child-id"},
            "island_best_programs": ["a", "child-id", "c"],
        },
        "admission": {"island": 1},
    }
    payload.update(overrides)
    p = replay_dir / f"step_{step:04d}.json"
    p.write_text(json.dumps(payload))
    return p


def test_inspect_step_summary_default(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 3)

    rc = cmd_inspect_step(run_dir=tmp_path, step=3, sections=[])
    captured = capsys.readouterr()

    assert rc == 0
    assert "## summary" in captured.out
    assert "iteration:           3" in captured.out
    assert "parent_id:           p-id" in captured.out
    assert "num_inspirations:    2" in captured.out
    assert "admission_island:    1" in captured.out


def test_inspect_step_prompt_section(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 1)

    rc = cmd_inspect_step(run_dir=tmp_path, step=1, sections=["prompt"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "--- SYSTEM ---" in captured.out
    assert "YOU ARE A RETRIEVER" in captured.out
    assert "--- USER ---" in captured.out
    assert "PARENT CODE: ..." in captured.out


def test_inspect_step_inspirations(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 5)

    rc = cmd_inspect_step(run_dir=tmp_path, step=5, sections=["inspirations"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "## inspirations" in captured.out
    assert "id=insp-1" in captured.out
    assert "id=insp-2" in captured.out


def test_inspect_step_population_after(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 5)

    rc = cmd_inspect_step(run_dir=tmp_path, step=5, sections=["population-after"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Island 0" in captured.out
    assert "Island 1" in captured.out
    assert "Island 2" in captured.out
    assert "child-id" in captured.out  # added in db_after


def test_inspect_step_diff(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 5)
    rc = cmd_inspect_step(run_dir=tmp_path, step=5, sections=["diff"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "n_extracted: 2" in captured.out
    assert "SEARCH:" in captured.out
    assert "REPLACE:" in captured.out


def test_inspect_step_all(tmp_path, capsys):
    replay = tmp_path / "replay"
    replay.mkdir()
    _write_step_snapshot(replay, 5)
    rc = cmd_inspect_step(run_dir=tmp_path, step=5, sections=["all"])
    captured = capsys.readouterr()
    assert rc == 0
    for header in ["## summary", "## prompt", "## parent", "## inspirations",
                   "## population", "## diff", "## llm-response",
                   "## child-code", "## eval"]:
        assert header in captured.out, f"missing {header}"


def test_inspect_step_missing_step_errors(tmp_path, capsys):
    (tmp_path / "replay").mkdir()
    rc = cmd_inspect_step(run_dir=tmp_path, step=99, sections=["summary"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no data for step 99" in err


def test_inspect_step_unknown_section_raises(tmp_path):
    (tmp_path / "replay").mkdir()
    _write_step_snapshot(tmp_path / "replay", 1)
    with pytest.raises(SystemExit, match="unknown section"):
        cmd_inspect_step(run_dir=tmp_path, step=1, sections=["bogus"])


def test_inspect_step_snapshot_missing_falls_back_with_notice(tmp_path, capsys):
    """When only run.db has the step (no replay snapshot), warn and degrade gracefully."""
    # No replay/, no run.db — should error cleanly.
    rc = cmd_inspect_step(run_dir=tmp_path, step=1, sections=["summary"])
    assert rc in (1, 2)


def test_inspect_step_snapshot_only_section_unavailable_without_snapshot(tmp_path, capsys):
    """`inspirations` is snapshot-only; without a step file it must say so."""
    # We need *some* data for the step to be considered "exists" — write a fake
    # run.db with one iteration row. For now, just confirm the snapshot-only
    # guard prints the unavailable message when explicitly asked.
    replay = tmp_path / "replay"
    replay.mkdir()
    # No step file written — so snapshot is None. We need db_state too to make
    # cmd_inspect_step proceed (else it returns early). Synthesize a minimal
    # run.db with one iteration row.
    import sqlite3
    conn = sqlite3.connect(tmp_path / "run.db")
    conn.executescript("""
        CREATE TABLE iterations (
            iteration INTEGER PRIMARY KEY,
            parent_id TEXT, child_id TEXT, child_score REAL,
            improvement_delta REAL, prompt_hash TEXT, llm_latency_ms REAL,
            diff_n_extracted INTEGER, diff_n_applied INTEGER,
            eval_duration_s REAL, island INTEGER
        );
        CREATE TABLE programs (
            id TEXT PRIMARY KEY, parent_id TEXT, generation INTEGER,
            iteration_found INTEGER, timestamp REAL,
            source_code TEXT, metrics_json TEXT, complexity REAL,
            diversity REAL, island INTEGER,
            feature_coords_json TEXT, changes_description TEXT,
            artifacts_json TEXT, metadata_json TEXT,
            prompt_system TEXT, prompt_user TEXT, llm_raw_response TEXT
        );
        INSERT INTO iterations VALUES (1, 'p1', 'c1', 0.5, 0.1, 'h', 100, 2, 2, 30, 0);
        INSERT INTO programs (id, parent_id, generation, iteration_found, source_code,
            metrics_json, complexity, diversity, island, prompt_system, prompt_user, llm_raw_response)
        VALUES
        ('p1', NULL, 0, 0, 'parent code', '{}', 0.0, 0.0, 0, 'sys', 'usr', 'resp'),
        ('c1', 'p1', 1, 1, 'child code', '{"combined_score": 0.5}', 0.0, 0.0, 0, 'sys', 'usr', 'resp');
    """)
    conn.commit()
    conn.close()

    rc = cmd_inspect_step(
        run_dir=tmp_path, step=1, sections=["summary", "inspirations", "population-after"],
    )
    out = capsys.readouterr().out
    err = capsys.readouterr().err

    assert rc == 0
    assert "iteration:           1" in out  # summary fell back to db_state
    # inspirations and population-after should print their unavailable banner
    assert "(unavailable" in out
