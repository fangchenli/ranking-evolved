"""Tests for core.run_store: insert programs, query lineage, transactional commits."""
from __future__ import annotations

from pathlib import Path

from ranking_evolved.core.run_store import RunStore
from ranking_evolved.core.types import Program


def _mk_program(
    *,
    id: str,
    parent_id: str | None = None,
    iteration: int = 0,
    score: float = 0.5,
    island: int = 0,
) -> Program:
    return Program(
        id=id,
        source_code=f"# {id}\nx = {iteration}\n",
        parent_id=parent_id,
        generation=iteration,
        iteration_found=iteration,
        timestamp=1234567890.0 + iteration,
        metrics={"combined_score": score},
        complexity=10.0 * (iteration + 1),
        diversity=0.1 * (iteration + 1),
        island=island,
        feature_coords={"complexity": 0.1, "diversity": 0.2},
    )


def test_run_store_roundtrip(tmp_path: Path, record_io):
    db = tmp_path / "run.db"

    def run() -> dict:
        store = RunStore(db)
        try:
            seed = _mk_program(id="seed", iteration=0, score=0.5)
            store.add_program(seed, prompt_system="sys", prompt_user="usr")
            child = _mk_program(id="c1", parent_id="seed", iteration=1, score=0.6)
            store.add_program(child, llm_raw_response="<<< SEARCH >>>")
            store.add_iteration(
                iteration=1, parent_id="seed", child_id="c1",
                prompt_hash="abc", llm_latency_ms=12.5,
                diff_n_extracted=1, diff_n_applied=1,
                eval_duration_s=0.7, child_score=0.6,
                improvement_delta=0.1,
            )
            return {
                "n": store.count_programs(),
                "seed": store.get_program("seed").id,
                "child_parent": store.get_program("c1").parent_id,
                "ids_in_order": [p.id for p in store.iter_programs()],
                "last_iter": store.last_iteration(),
            }
        finally:
            store.close()

    out = record_io(
        module="src/ranking_evolved/core/run_store.py",
        function="RunStore.add_program/get_program/iter_programs",
        input={"programs": ["seed", "c1"], "iterations": [1]},
        run=run,
    )
    assert out == {
        "n": 2,
        "seed": "seed",
        "child_parent": "seed",
        "ids_in_order": ["seed", "c1"],
        "last_iter": 1,
    }


def test_run_store_archive_cells(tmp_path: Path, record_io):
    db = tmp_path / "run.db"

    def run() -> dict:
        store = RunStore(db)
        try:
            store.upsert_archive_cell(0, "1,2", "p1")
            store.upsert_archive_cell(0, "3,4", "p2")
            store.upsert_archive_cell(0, "1,2", "p3")  # eviction
            store.upsert_archive_cell(1, "1,2", "p4")  # different island
            return store.archive_cells()
        finally:
            store.close()

    out = record_io(
        module="src/ranking_evolved/core/run_store.py",
        function="RunStore.upsert_archive_cell",
        input={"writes": [(0, "1,2", "p1"), (0, "3,4", "p2"), (0, "1,2", "p3"), (1, "1,2", "p4")]},
        run=run,
    )
    assert out == {(0, "1,2"): "p3", (0, "3,4"): "p2", (1, "1,2"): "p4"}


def test_run_store_replace_archive_cells_removes_stale_rows(tmp_path: Path, record_io):
    db = tmp_path / "run.db"

    def run() -> dict:
        store = RunStore(db)
        try:
            store.upsert_archive_cell(0, "old", "p_old")
            store.upsert_archive_cell(1, "also_old", "p_other")
            store.replace_archive_cells([(0, "new", "p_new")])
            return store.archive_cells()
        finally:
            store.close()

    out = record_io(
        module="src/ranking_evolved/core/run_store.py",
        function="RunStore.replace_archive_cells",
        input={"cells": [(0, "new", "p_new")]},
        run=run,
    )
    assert out == {(0, "new"): "p_new"}


def test_run_store_transaction_rollback(tmp_path: Path, record_io):
    db = tmp_path / "run.db"

    def run() -> int:
        store = RunStore(db)
        try:
            seed = _mk_program(id="seed", iteration=0)
            store.add_program(seed)
            try:
                with store.transaction() as conn:
                    store.add_program(_mk_program(id="c1", iteration=1), conn=conn)
                    store.add_program(_mk_program(id="c1", iteration=2), conn=conn)  # PK conflict
            except Exception:
                pass
            return store.count_programs()
        finally:
            store.close()

    out = record_io(
        module="src/ranking_evolved/core/run_store.py",
        function="RunStore.transaction",
        input={"scenario": "duplicate primary key inside transaction"},
        run=run,
    )
    assert out == 1  # rollback dropped the c1 row, only seed remains
