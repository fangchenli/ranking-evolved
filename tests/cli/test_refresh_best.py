from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.cli import main
from ranking_evolved.core.run_store import RunStore
from ranking_evolved.core.types import Program


def _program(program_id: str, *, score: float, iteration: int) -> Program:
    return Program(
        id=program_id,
        source_code=f"# {program_id}\n",
        parent_id="parent" if iteration > 0 else None,
        generation=iteration,
        iteration_found=iteration,
        timestamp=0.0,
        metrics={"combined_score": score},
        complexity=1.0,
        diversity=0.0,
        island=0,
        feature_coords={},
        changes_description="test",
        artifacts={},
        metadata={},
    )


def test_refresh_best_writes_step_provenance(tmp_path: Path, record_io):
    run_dir = tmp_path / "run"
    store = RunStore(run_dir / "run.db")
    try:
        store.add_program(_program("seed", score=0.1, iteration=0))
        store.add_program(_program("winner", score=0.9, iteration=7))
    finally:
        store.close()

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="refresh-best",
        input={"run": str(run_dir)},
        run=lambda: main(["refresh-best", "--run", str(run_dir)]),
    )

    assert out == 0
    best_dir = run_dir / "best"
    metadata = json.loads((best_dir / "metadata.json").read_text())
    assert metadata["program_id"] == "winner"
    assert metadata["iteration_found"] == 7
    assert metadata["replay_path"] == "../replay/step_0007.json"
    assert (best_dir / "created_at_step_0007.txt").exists()
    assert (best_dir / "README.md").exists()
