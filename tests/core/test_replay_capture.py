"""Tests for core.replay_capture: convert OE trace + checkpoint to ReplayStep schema."""
from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.core.replay_capture import capture_reference


def _write_oe_run(root: Path, *, n_iters: int = 2) -> Path:
    """Synthesize a minimal OpenEvolve-shaped output dir."""
    root.mkdir(parents=True, exist_ok=True)
    trace = root / "evolution_trace.jsonl"
    lines = []
    for i in range(1, n_iters + 1):
        event = {
            "iteration": i,
            "timestamp": 1.0 * i,
            "parent_id": f"P{i}",
            "child_id": f"C{i}",
            "parent_metrics": json.dumps({"combined_score": 0.4}),
            "child_metrics": json.dumps({"combined_score": 0.5 + 0.1 * i}),
            "parent_code": f"# parent {i}\n",
            "child_code": f"# child {i}\n",
            "prompt": json.dumps({"system": "sys", "user": f"user-{i}"}),
            "llm_response": f"<<< SEARCH >>>...{i}",
            "improvement_delta": json.dumps({"combined_score": 0.1}),
            "island_id": (i - 1) % 2,
            "generation": i,
            "metadata": json.dumps({"iteration_time": 0.123, "model": "gpt-x"}),
        }
        lines.append(json.dumps(event))
    trace.write_text("\n".join(lines))

    ckpt = root / "checkpoints" / "checkpoint_2"
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text(json.dumps({
        "island_feature_maps": [{"1-1": "P1"}, {"2-2": "P2"}],
        "islands": [["P1", "C1"], ["P2", "C2"]],
        "archive": ["P1", "P2"],
        "best_program_id": "C2",
        "island_best_programs": ["P1", "C2"],
        "current_island": 0,
        "island_generations": [2, 1],
        "last_migration_generation": 0,
    }))
    return root


def test_capture_reference_writes_one_file_per_iter(tmp_path: Path, record_io):
    oe_root = _write_oe_run(tmp_path / "oe", n_iters=2)
    out = tmp_path / "ref"

    def run() -> dict:
        written = capture_reference(openevolve_output=oe_root, out_dir=out)
        # Inspect step_0001.json
        step1 = json.loads((out / "step_0001.json").read_text())
        return {
            "n_files": len(written),
            "filenames": sorted(p.name for p in written),
            "iter": step1["iteration"],
            "parent_id": step1["parent"]["id"],
            "child_metrics_score": step1["child_eval"]["metrics"].get("combined_score"),
            "prompt_user_starts_with": step1["prompt"]["user"][:6],
            "source": step1["_source"],
            "partial": step1["_partial"],
            "db_after_n_programs": step1["db_after"]["n_programs"],
        }

    out_d = record_io(
        module="src/ranking_evolved/core/replay_capture.py",
        function="capture_reference",
        input={"n_iters": 2},
        run=run,
    )
    assert out_d["n_files"] == 2
    assert out_d["filenames"] == ["step_0001.json", "step_0002.json"]
    assert out_d["iter"] == 1
    assert out_d["parent_id"] == "P1"
    assert out_d["child_metrics_score"] == 0.6  # 0.5 + 0.1*1
    assert out_d["prompt_user_starts_with"] == "user-1"
    assert out_d["source"] == "openevolve"
    assert out_d["partial"] is True
    # db_after pulled from checkpoint_2, which has 4 programs across 2 islands.
    assert out_d["db_after_n_programs"] == 4


def test_capture_reference_handles_missing_checkpoint(tmp_path: Path, record_io):
    """When no checkpoint covers the iter, we still emit a step with empty pop snapshots."""
    root = tmp_path / "oe"
    root.mkdir()
    (root / "evolution_trace.jsonl").write_text(json.dumps({
        "iteration": 1, "timestamp": 1.0, "parent_id": "P", "child_id": "C",
        "parent_metrics": "{'combined_score': 0.1}",
        "child_metrics": "{'combined_score': 0.2}",
        "parent_code": "# p\n", "child_code": "# c\n",
    }))
    out = tmp_path / "ref"

    def run() -> dict:
        written = capture_reference(openevolve_output=root, out_dir=out)
        step = json.loads(written[0].read_text())
        return {
            "n_files": len(written),
            "db_before_n": step["db_before"]["n_programs"],
            "db_after_n": step["db_after"]["n_programs"],
            "child_score": step["child_eval"]["metrics"]["combined_score"],
        }

    out_d = record_io(
        module="src/ranking_evolved/core/replay_capture.py",
        function="capture_reference (no checkpoints)",
        input={"trace_only": True},
        run=run,
    )
    assert out_d["n_files"] == 1
    assert out_d["db_before_n"] == 0
    assert out_d["db_after_n"] == 0
    assert out_d["child_score"] == 0.2  # parsed from the python-repr string
