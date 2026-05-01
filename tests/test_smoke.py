"""End-to-end smoke test: 5-iter loop with FakeProposer + stub evaluator.

This is the Phase-1 gate that proves the loop wires everything together:
  * core/types ↔ core/run_store ↔ core/trace ↔ core/replay
  * search/map_elites_islands.MapElitesIslandsStrategy
  * prompts/sampler + prompts/diff
  * proposers/fake
  * evaluation/runner (inline mode, callable evaluator)
  * core/controller orchestrating the above

We don't use the CLI here — we drive the Controller directly so the test is
hermetic and fast (< 1s).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ranking_evolved.config.base import (
    Config,
    EvaluationConfig,
    EvolutionConfig,
    LoggingConfig,
    ProposerConfig,
    RunStoreConfig,
    TaskConfig,
    TraceConfig,
)
from ranking_evolved.core.controller import Controller
from ranking_evolved.evaluation.runner import EvaluatorRunner
from ranking_evolved.prompts.sampler import PromptConfig
from ranking_evolved.proposers.fake import FakeProposer
from ranking_evolved.search.map_elites_islands import MapElitesIslandsConfig


# A trivial seed program. The evaluator scores it by literal "x = N" magnitude.
SEED_SOURCE = """\
# seed program
x = 1
"""


def _make_evaluator(tmp_path: Path) -> Path:
    """Write a tiny evaluator that scores a program by reading `x = <int>`."""
    src = '''\
import re
from pathlib import Path

def evaluate(program_path: str) -> dict:
    text = Path(program_path).read_text()
    m = re.search(r"^x\\s*=\\s*(\\d+)", text, flags=re.MULTILINE)
    n = int(m.group(1)) if m else 0
    return {
        "metrics": {"combined_score": float(n) / 10.0},
        "per_dataset": {},
        "artifacts": {"x_value": str(n)},
    }
'''
    p = tmp_path / "evaluator.py"
    p.write_text(src)
    return p


def _make_seed(tmp_path: Path) -> Path:
    p = tmp_path / "seed.py"
    p.write_text(SEED_SOURCE)
    return p


def _diff(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE"


def _make_config(seed: Path, evaluator: Path, *, capture_replay: bool) -> Config:
    return Config(
        task=TaskConfig(seed=str(seed), evaluator=str(evaluator)),
        evolution=EvolutionConfig(max_iterations=5, random_seed=42, capture_replay=capture_replay),
        search=MapElitesIslandsConfig(
            population_size=20, archive_size=4, num_islands=2,
            migration_interval=10, migration_rate=0.5,
            feature_dimensions=["complexity"], feature_bins=4,
            num_inspirations=2, random_seed=42,
        ),
        proposer=ProposerConfig(kind="fake"),
        prompt=PromptConfig(diff_based=True, num_top_programs=2, num_diverse_programs=2),
        evaluation=EvaluationConfig(timeout=10.0, isolation="inline"),
        trace=TraceConfig(enabled=True, include_prompts=True),
        logging=LoggingConfig(),
        run_store=RunStoreConfig(vacuum_on_close=False),
    )


def test_smoke_5_iterations_produces_complete_run_dir(tmp_path: Path, record_io):
    seed = _make_seed(tmp_path)
    evaluator = _make_evaluator(tmp_path)
    run_dir = tmp_path / "run"

    # The strategy may sample any island's parent. Use a callback that reads
    # the parent's current `x = N` value from the prompt and proposes a diff
    # that increments it. This way every iteration applies cleanly regardless
    # of which island the parent came from.
    import re
    def _callback(prompt, iteration):
        m = re.search(r"^x\s*=\s*(\d+)", prompt.user, flags=re.MULTILINE)
        n = int(m.group(1)) if m else 0
        return _diff(f"x = {n}\n", f"x = {n + 1}\n")

    proposer = FakeProposer(callback=_callback)
    config = _make_config(seed, evaluator, capture_replay=True)
    runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
    controller = Controller(config=config, run_dir=run_dir, proposer=proposer, runner=runner)

    async def go_inner() -> dict:
        best = await controller.run(seed_path=seed)
        controller.close()
        # Inspect the run directory.
        files = sorted(p.name for p in run_dir.iterdir())
        trace_lines = (run_dir / "trace.jsonl").read_text().splitlines()
        replay_files = sorted(p.name for p in (run_dir / "replay").iterdir())
        # SQLite content.
        conn = sqlite3.connect(run_dir / "run.db")
        n_programs = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        n_iterations = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        conn.close()
        plot_files = sorted(p.name for p in (run_dir / "plots").iterdir())
        return {
            "files": files,
            "n_trace_lines": len(trace_lines),
            "replay_files": replay_files,
            "plot_files": plot_files,
            "n_programs": n_programs,
            "n_iterations": n_iterations,
            "best_score": best.metrics["combined_score"],
            "best_program_file_exists": (run_dir / "best" / "program.py").exists(),
        }

    out = record_io(
        module="src/ranking_evolved/core/controller.py",
        function="Controller.run (5-iter smoke)",
        input={
            "max_iterations": 5,
            "proposer": "fake",
            "transcript": "x bumps from 1 to 6",
        },
        run=lambda: __import__("asyncio").run(go_inner()),
    )

    # The run directory has all the expected artifacts.
    assert "run.db" in out["files"]
    assert "trace.jsonl" in out["files"]
    assert "manifest.json" not in out["files"]  # manifest is CLI-level, not Controller-level
    assert "best" in out["files"]
    assert "replay" in out["files"]
    assert "plots" in out["files"]
    assert out["plot_files"] == [
        "latency_tradeoff.pdf",
        "objective_components.pdf",
        "optimization_curves.pdf",
    ]

    # Trace has 6 entries: seed (iter 0) + 5 iterations.
    assert out["n_trace_lines"] == 6

    # Replay has 5 step files (one per iteration; seed is not a "step").
    assert out["replay_files"] == [f"step_{i:04d}.json" for i in range(1, 6)]

    # Programs persisted: at least seed + 1 per iter (plus copies for islands).
    assert out["n_programs"] >= 6

    # 5 iteration rows.
    assert out["n_iterations"] == 5

    # Best score improved over the seed (seed has x=1 -> 0.1). The exact
    # final value depends on which islands were sampled across the 5 iters
    # (the strategy picks parents from any island, not always the latest
    # child), but the best should be at least 0.2 (one successful increment).
    assert out["best_score"] > 0.1
    assert out["best_program_file_exists"] is True


def test_controller_resume_continues_existing_run_without_duplicate_seed(tmp_path: Path, record_io):
    seed = _make_seed(tmp_path)
    evaluator = _make_evaluator(tmp_path)
    run_dir = tmp_path / "run"

    import re
    def _callback(prompt, iteration):
        m = re.search(r"^x\s*=\s*(\d+)", prompt.user, flags=re.MULTILINE)
        n = int(m.group(1)) if m else 0
        return _diff(f"x = {n}\n", f"x = {n + 1}\n")

    async def go_inner() -> dict:
        first_config = _make_config(seed, evaluator, capture_replay=True)
        first_config.evolution.max_iterations = 2
        first_runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
        first = Controller(
            config=first_config,
            run_dir=run_dir,
            proposer=FakeProposer(callback=_callback),
            runner=first_runner,
        )
        try:
            await first.run(seed_path=seed)
        finally:
            first.close()

        second_config = _make_config(seed, evaluator, capture_replay=True)
        second_config.evolution.max_iterations = 4
        second_runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
        second = Controller(
            config=second_config,
            run_dir=run_dir,
            proposer=FakeProposer(callback=_callback),
            runner=second_runner,
        )
        try:
            best = await second.run(seed_path=seed, resume=True)
        finally:
            second.close()

        metrics_rows = [
            json.loads(line)
            for line in (run_dir / "program_metrics.jsonl").read_text().splitlines()
        ]
        trace_rows = (run_dir / "trace.jsonl").read_text().splitlines()
        conn = sqlite3.connect(run_dir / "run.db")
        last_iter = conn.execute("SELECT value FROM run_meta WHERE key = 'last_iter'").fetchone()[0]
        n_seed_rows = conn.execute("SELECT COUNT(*) FROM programs WHERE id = 'seed'").fetchone()[0]
        n_iterations = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        has_strategy_state = conn.execute(
            "SELECT COUNT(*) FROM run_meta WHERE key = 'strategy_state'"
        ).fetchone()[0]
        conn.close()
        return {
            "metric_iterations": [row["iteration_found"] for row in metrics_rows],
            "seed_metric_rows": sum(1 for row in metrics_rows if row["program_id"] == "seed"),
            "n_trace_lines": len(trace_rows),
            "last_iter_meta": json.loads(last_iter),
            "n_seed_db_rows": n_seed_rows,
            "n_iterations": n_iterations,
            "has_strategy_state": has_strategy_state,
            "best_score": best.metrics["combined_score"],
        }

    out = record_io(
        module="src/ranking_evolved/core/controller.py",
        function="Controller.run (resume)",
        input={"first_max_iterations": 2, "second_total_max_iterations": 4},
        run=lambda: __import__("asyncio").run(go_inner()),
    )
    assert out["metric_iterations"] == [0, 1, 2, 3, 4]
    assert out["seed_metric_rows"] == 1
    assert out["n_trace_lines"] == 5
    assert out["last_iter_meta"] == 4
    assert out["n_seed_db_rows"] == 1
    assert out["n_iterations"] == 4
    assert out["has_strategy_state"] == 1
    assert out["best_score"] > 0.1


def test_smoke_replay_step_has_expected_shape(tmp_path: Path, record_io):
    """A replay step JSON file contains every section the dashboard renders."""
    seed = _make_seed(tmp_path)
    evaluator = _make_evaluator(tmp_path)
    run_dir = tmp_path / "run"

    proposer = FakeProposer(responses=[(_diff("x = 1\n", "x = 2\n"), "fake-1")])
    config = _make_config(seed, evaluator, capture_replay=True)
    config.evolution.max_iterations = 1
    runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
    controller = Controller(config=config, run_dir=run_dir, proposer=proposer, runner=runner)

    async def go_inner() -> dict:
        await controller.run(seed_path=seed)
        controller.close()
        return json.loads((run_dir / "replay" / "step_0001.json").read_text())

    step = record_io(
        module="src/ranking_evolved/core/controller.py",
        function="Controller._step (replay capture shape)",
        input={"iteration": 1},
        run=lambda: __import__("asyncio").run(go_inner()),
    )

    expected_keys = {
        "schema_version", "iteration", "sampling", "parent", "inspirations",
        "top_programs", "previous_programs", "parent_artifacts", "prompt",
        "llm", "diff", "child_code", "child_eval", "db_before", "db_after",
        "admission",
    }
    assert expected_keys.issubset(set(step.keys()))
    assert step["iteration"] == 1
    assert step["llm"]["proposer"] == "fake"
    assert step["diff"]["n_extracted"] == 1
    assert step["diff"]["n_applied"] == 1
    assert step["child_eval"]["metrics"]["combined_score"] == pytest.approx(0.2, abs=1e-6)
    # db_before should have fewer programs than db_after.
    assert step["db_after"]["n_programs"] >= step["db_before"]["n_programs"]
