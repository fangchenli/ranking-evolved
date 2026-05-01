"""Smoke test: 3-iter run driven by OpenAIResponsesProposer with a mocked HTTP client.

This proves the controller composes correctly with a real (non-Fake) proposer:
  prompt -> proposer.propose -> diff blocks -> diff applies -> eval -> admit.

The mocked client reads the parent's `x = N` value from the Responses API
`input` field and returns `output_text` containing a SEARCH/REPLACE that
increments it, so the diff applies on every iteration regardless of which
island parent the strategy sampled.
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
from ranking_evolved.proposers.openai_chat import OpenAIResponsesProposer
from ranking_evolved.search.map_elites_islands import MapElitesIslandsConfig


SEED_SOURCE = "x = 1\n"


class _StubResponse:
    def __init__(self, content: str):
        self._data = {
            "model": "gpt-test",
            "output_text": content,
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }
        self.status_code = 200
        self.text = ""

    def json(self) -> dict:
        return self._data


class _StubClient:
    """httpx.AsyncClient look-alike: derives the diff from the prompt's x value."""

    def __init__(self):
        self.calls = 0

    async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
        self.calls += 1
        # Responses API body: input is a list of {role, content} dicts.
        user_msg = json["input"][0]["content"]
        m = re.search(r"^x\s*=\s*(\d+)", user_msg, flags=re.MULTILINE)
        n = int(m.group(1)) if m else 0
        diff = (
            f"<<<<<<< SEARCH\nx = {n}\n=======\nx = {n + 1}\n>>>>>>> REPLACE"
        )
        return _StubResponse(diff)

    async def aclose(self) -> None:
        return None


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


def test_smoke_openai_responses_3_iter(tmp_path: Path, record_io):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED_SOURCE)
    evaluator = _write_evaluator(tmp_path)
    run_dir = tmp_path / "run"

    client = _StubClient()
    proposer = OpenAIResponsesProposer(
        api_key="sk-test", model="gpt-test", retries=1, client=client,
    )

    config = Config(
        task=TaskConfig(seed=str(seed), evaluator=str(evaluator)),
        evolution=EvolutionConfig(max_iterations=3, random_seed=11, capture_replay=True),
        search=MapElitesIslandsConfig(
            population_size=10, archive_size=3, num_islands=2,
            migration_interval=10, migration_rate=0.5,
            feature_dimensions=["complexity"], feature_bins=4,
            num_inspirations=1, random_seed=11,
        ),
        proposer=ProposerConfig(kind="openai_responses", api_key="sk-test"),
        prompt=PromptConfig(diff_based=True, num_top_programs=1, num_diverse_programs=1),
        evaluation=EvaluationConfig(timeout=10.0, isolation="inline"),
        trace=TraceConfig(enabled=True, include_prompts=True),
        logging=LoggingConfig(),
        run_store=RunStoreConfig(vacuum_on_close=False),
    )
    runner = EvaluatorRunner(evaluator, timeout_s=10.0, isolation="inline")
    controller = Controller(config=config, run_dir=run_dir, proposer=proposer, runner=runner)

    async def go():
        best = await controller.run(seed_path=seed)
        controller.close()
        conn = sqlite3.connect(run_dir / "run.db")
        n_progs = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        n_iters = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        conn.close()
        return {
            "http_calls": client.calls,
            "best_score": best.metrics["combined_score"],
            "n_programs": n_progs,
            "n_iters": n_iters,
            "trace_lines": len((run_dir / "trace.jsonl").read_text().splitlines()),
            "replay_files": sorted(p.name for p in (run_dir / "replay").iterdir()),
            "has_best_export": (run_dir / "best" / "program.py").exists(),
        }

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer + Controller (3-iter smoke)",
        input={"max_iterations": 3, "proposer": "openai_responses"},
        run=lambda: __import__("asyncio").run(go()),
    )
    assert out["http_calls"] == 3                    # one HTTP call per iteration
    assert out["n_iters"] == 3                       # 3 iteration rows
    assert out["best_score"] > 0.1                   # improved from seed (x=1 -> 0.1)
    assert out["trace_lines"] == 4                   # seed + 3 iters
    assert out["replay_files"] == [f"step_{i:04d}.json" for i in (1, 2, 3)]
    assert out["has_best_export"] is True
