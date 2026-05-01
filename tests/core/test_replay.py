"""Tests for core.replay: ReplayStep capture is a faithful, JSON-serializable snapshot."""
from __future__ import annotations

import json
from pathlib import Path

from ranking_evolved.core.replay import ReplayWriter, build_step
from ranking_evolved.core.types import (
    AdmissionDecisions,
    DiffApplication,
    DiffBlock,
    EvaluationResult,
    PopulationSnapshot,
    Program,
    Prompt,
    SamplingDecisions,
)


def _mk_program(id: str, parent_id: str | None = None, iter_: int = 0, island: int = 0) -> Program:
    return Program(
        id=id, source_code=f"# {id}\n", parent_id=parent_id,
        generation=iter_, iteration_found=iter_, timestamp=0.0,
        metrics={"combined_score": 0.5}, complexity=10.0, diversity=0.1,
        island=island, feature_coords={"complexity": 0.5, "diversity": 0.5},
    )


def _mk_pop(programs: list[Program]) -> PopulationSnapshot:
    return PopulationSnapshot(
        n_programs=len(programs),
        islands=[[p.id for p in programs]],
        island_generations=[0],
        current_island=0,
        last_migration_generation=0,
        archive_cells={"5,5": programs[0].id} if programs else {},
        archive_size=len(programs),
        best_program_id=programs[0].id if programs else None,
        island_best_programs=[programs[0].id if programs else None],
    )


def test_replay_writer_dumps_full_step(tmp_path: Path, record_io):
    parent = _mk_program("seed")
    child = _mk_program("c1", parent_id="seed", iter_=1)
    prompt = Prompt(
        system="s", user="u", template_key="diff_user",
        iteration=1, parent_id="seed", inspiration_ids=(),
    )
    diff = DiffApplication(
        pattern=r"<<<<<<< SEARCH...",
        blocks=(DiffBlock(search="x", replace="y", matched_at_line=1),),
        n_extracted=1, n_applied=1, fatal_error=None,
    )
    eval_ = EvaluationResult(
        metrics={"combined_score": 0.6}, per_dataset={"scifact": {"ndcg@10": 0.6}},
        artifacts={}, duration_s=0.7, error=None,
    )

    def run() -> dict:
        step = build_step(
            iteration=1,
            sampling=SamplingDecisions(
                rng_seed_hash="abc", parent_id="seed", parent_island=0,
                inspiration_ids=[], inspiration_strategy="random",
                top_program_ids=["seed"], previous_program_ids=["seed"],
            ),
            parent=parent, inspirations=[], top_programs=[parent], previous_programs=[parent],
            parent_artifacts=None, prompt=prompt,
            llm_proposer="fake", llm_model="fake-1", llm_raw="resp",
            llm_tokens_in=10, llm_tokens_out=20, llm_latency_ms=5.0,
            diff=diff, child_code=child.source_code, child_eval=eval_,
            db_before=_mk_pop([parent]), db_after=_mk_pop([parent, child]),
            admission=AdmissionDecisions(
                target_island=0, feature_coords={"complexity": 0.5, "diversity": 0.5},
                cell_key="5,5", evicted_program_id=None,
                migration_fired=False, migration_details=None,
            ),
        )
        writer = ReplayWriter(tmp_path)
        out = writer.write(step)
        return {"path": out.name, "json_keys": sorted(json.loads(out.read_text()).keys())}

    out = record_io(
        module="src/ranking_evolved/core/replay.py",
        function="ReplayWriter.write",
        input={"iteration": 1, "parent": "seed", "child": "c1"},
        run=run,
    )
    assert out["path"] == "step_0001.json"
    assert "sampling" in out["json_keys"]
    assert "db_before" in out["json_keys"]
    assert "db_after" in out["json_keys"]
    assert "diff" in out["json_keys"]
