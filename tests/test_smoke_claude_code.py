"""Smoke test: 3-iter run driven by ClaudeCodeProposer with a mocked subprocess runner.

Same shape as the openai_chat smoke; here the runner reads the prompt's
`x = N` from stdin and prints a SEARCH/REPLACE that increments it.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from ranking_evolved.config.base import (
    Config, EvaluationConfig, EvolutionConfig, LoggingConfig,
    ProposerConfig, RunStoreConfig, TaskConfig, TraceConfig,
)
from ranking_evolved.core.controller import Controller
from ranking_evolved.evaluation.runner import EvaluatorRunner
from ranking_evolved.prompts.sampler import PromptConfig
from ranking_evolved.proposers.claude_code import ClaudeCodeProposer
from ranking_evolved.search.map_elites_islands import MapElitesIslandsConfig


SEED_SOURCE = "x = 1\n"


class _StubRunner:
    """Pretends to be the `claude` CLI: read x value from stdin, emit a diff."""
    def __init__(self):
        self.calls = 0

    def __call__(self, *, args, input, timeout):
        self.calls += 1
        text = input.decode()
        m = re.search(r"^x\s*=\s*(\d+)", text, flags=re.MULTILINE)
        n = int(m.group(1)) if m else 0
        diff = f"<<<<<<< SEARCH\nx = {n}\n=======\nx = {n + 1}\n>>>>>>> REPLACE"
        return (diff, "", 0)


def _write_evaluator(tmp: Path) -> Path:
    src = '''\
import re
from pathlib import Path

def evaluate(program_path: str) -> dict:
    text = Path(program_path).read_text()
    m = re.search(r"^x\\s*=\\s*(\\d+)", text, flags=re.MULTILINE)
    n = int(m.group(1)) if m else 0
    return {"metrics": {"combined_score": float(n) / 10.0}}
'''
    p = tmp / "evaluator.py"
    p.write_text(src)
    return p


def test_smoke_claude_code_3_iter(tmp_path: Path, record_io):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED_SOURCE)
    evaluator = _write_evaluator(tmp_path)
    run_dir = tmp_path / "run"

    runner_stub = _StubRunner()
    proposer = ClaudeCodeProposer(
        binary="claude", model="claude-opus-4-7", runner=runner_stub,
    )

    config = Config(
        task=TaskConfig(seed=str(seed), evaluator=str(evaluator)),
        evolution=EvolutionConfig(max_iterations=3, random_seed=21, capture_replay=True),
        search=MapElitesIslandsConfig(
            population_size=10, archive_size=3, num_islands=2,
            migration_interval=10, migration_rate=0.5,
            feature_dimensions=["complexity"], feature_bins=4,
            num_inspirations=1, random_seed=21,
        ),
        proposer=ProposerConfig(kind="claude_code"),
        prompt=PromptConfig(diff_based=True, num_top_programs=1, num_diverse_programs=1),
        evaluation=EvaluationConfig(timeout=10.0, isolation="inline"),
        trace=TraceConfig(enabled=True, include_prompts=True),
        logging=LoggingConfig(),
        run_store=RunStoreConfig(vacuum_on_close=False),
    )
    eval_runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
    controller = Controller(config=config, run_dir=run_dir, proposer=proposer, runner=eval_runner)

    async def go():
        best = await controller.run(seed_path=seed)
        controller.close()
        conn = sqlite3.connect(run_dir / "run.db")
        n_progs = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        n_iters = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        conn.close()
        return {
            "subprocess_calls": runner_stub.calls,
            "best_score": best.metrics["combined_score"],
            "n_programs": n_progs,
            "n_iters": n_iters,
            "trace_lines": len((run_dir / "trace.jsonl").read_text().splitlines()),
            "replay_files": sorted(p.name for p in (run_dir / "replay").iterdir()),
        }

    out = record_io(
        module="src/ranking_evolved/proposers/claude_code.py",
        function="ClaudeCodeProposer + Controller (3-iter smoke)",
        input={"max_iterations": 3, "proposer": "claude_code"},
        run=lambda: __import__("asyncio").run(go()),
    )
    assert out["subprocess_calls"] == 3
    assert out["n_iters"] == 3
    assert out["best_score"] > 0.1
    assert out["trace_lines"] == 4
    assert out["replay_files"] == [f"step_{i:04d}.json" for i in (1, 2, 3)]
