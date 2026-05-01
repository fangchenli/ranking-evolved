"""SQLite-backed run store: programs, lineage, metrics, prompts, iterations.

Single source of truth for run state. `core.trace` writes a derived JSONL
projection within the same transaction; `best/program.py` is exported on
improvement; checkpoints are logical (`WHERE iteration_found <= N`) — there
are no checkpoint directories.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .types import Program


_SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  generation INTEGER NOT NULL,
  iteration_found INTEGER NOT NULL,
  timestamp REAL NOT NULL,
  source_code TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  per_dataset_metrics_json TEXT,
  complexity REAL NOT NULL,
  diversity REAL NOT NULL,
  island INTEGER NOT NULL,
  feature_coords_json TEXT NOT NULL,
  changes_description TEXT,
  prompt_system TEXT,
  prompt_user TEXT,
  llm_raw_response TEXT,
  artifacts_json TEXT,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_programs_iter   ON programs(iteration_found);
CREATE INDEX IF NOT EXISTS idx_programs_island ON programs(island);

CREATE TABLE IF NOT EXISTS archive_cells (
  island INTEGER NOT NULL,
  cell_key TEXT NOT NULL,
  program_id TEXT NOT NULL,
  PRIMARY KEY (island, cell_key)
);

CREATE TABLE IF NOT EXISTS migrations (
  iteration INTEGER NOT NULL,
  src_island INTEGER NOT NULL,
  dst_island INTEGER NOT NULL,
  program_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iterations (
  iteration INTEGER PRIMARY KEY,
  parent_id TEXT,
  child_id TEXT,
  prompt_hash TEXT,
  llm_latency_ms REAL,
  diff_n_extracted INTEGER,
  diff_n_applied INTEGER,
  eval_duration_s REAL,
  child_score REAL,
  improvement_delta REAL
);

CREATE TABLE IF NOT EXISTS run_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class RunStore:
    """Thin wrapper over sqlite3 for the evolution loop.

    Use the `transaction()` context manager to group a program insert + its
    iteration row + archive/migration updates atomically; this is what makes
    `trace.jsonl` and `run.db` agree at every iteration boundary.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self, *, vacuum: bool = False) -> None:
        if vacuum:
            self._conn.execute("VACUUM")
        self._conn.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # programs
    # ------------------------------------------------------------------

    def add_program(
        self,
        program: Program,
        *,
        prompt_system: str | None = None,
        prompt_user: str | None = None,
        llm_raw_response: str | None = None,
        per_dataset_metrics: dict[str, dict[str, float]] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        c = conn or self._conn
        c.execute(
            """
            INSERT INTO programs (
              id, parent_id, generation, iteration_found, timestamp,
              source_code, metrics_json, per_dataset_metrics_json,
              complexity, diversity, island, feature_coords_json,
              changes_description, prompt_system, prompt_user,
              llm_raw_response, artifacts_json, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                program.id,
                program.parent_id,
                program.generation,
                program.iteration_found,
                program.timestamp,
                program.source_code,
                json.dumps(program.metrics),
                json.dumps(per_dataset_metrics) if per_dataset_metrics else None,
                program.complexity,
                program.diversity,
                program.island,
                json.dumps(program.feature_coords),
                program.changes_description,
                prompt_system,
                prompt_user,
                llm_raw_response,
                json.dumps(_jsonable(program.artifacts)) if program.artifacts else None,
                json.dumps(_jsonable(program.metadata)) if program.metadata else None,
            ),
        )

    def get_program(self, program_id: str) -> Program | None:
        row = self._conn.execute(
            "SELECT id, parent_id, generation, iteration_found, timestamp, "
            "source_code, metrics_json, complexity, diversity, island, "
            "feature_coords_json, changes_description, artifacts_json, metadata_json "
            "FROM programs WHERE id = ?",
            (program_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_program(row)

    def iter_programs(self, *, max_iteration: int | None = None) -> Iterator[Program]:
        sql = (
            "SELECT id, parent_id, generation, iteration_found, timestamp, "
            "source_code, metrics_json, complexity, diversity, island, "
            "feature_coords_json, changes_description, artifacts_json, metadata_json "
            "FROM programs"
        )
        params: tuple[Any, ...] = ()
        if max_iteration is not None:
            sql += " WHERE iteration_found <= ?"
            params = (max_iteration,)
        sql += " ORDER BY iteration_found ASC"
        for row in self._conn.execute(sql, params):
            yield _row_to_program(row)

    def count_programs(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]

    # ------------------------------------------------------------------
    # archive cells
    # ------------------------------------------------------------------

    def upsert_archive_cell(
        self, island: int, cell_key: str, program_id: str,
        *, conn: sqlite3.Connection | None = None,
    ) -> None:
        c = conn or self._conn
        c.execute(
            "INSERT INTO archive_cells(island, cell_key, program_id) VALUES (?,?,?) "
            "ON CONFLICT(island, cell_key) DO UPDATE SET program_id=excluded.program_id",
            (island, cell_key, program_id),
        )

    def archive_cells(self) -> dict[tuple[int, str], str]:
        rows = self._conn.execute(
            "SELECT island, cell_key, program_id FROM archive_cells"
        ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    # ------------------------------------------------------------------
    # migrations
    # ------------------------------------------------------------------

    def record_migration(
        self, iteration: int, src: int, dst: int, program_id: str,
        *, conn: sqlite3.Connection | None = None,
    ) -> None:
        c = conn or self._conn
        c.execute(
            "INSERT INTO migrations(iteration, src_island, dst_island, program_id) VALUES (?,?,?,?)",
            (iteration, src, dst, program_id),
        )

    # ------------------------------------------------------------------
    # iterations
    # ------------------------------------------------------------------

    def add_iteration(
        self,
        *,
        iteration: int,
        parent_id: str | None,
        child_id: str | None,
        prompt_hash: str | None,
        llm_latency_ms: float | None,
        diff_n_extracted: int | None,
        diff_n_applied: int | None,
        eval_duration_s: float | None,
        child_score: float | None,
        improvement_delta: float | None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        c = conn or self._conn
        c.execute(
            "INSERT OR REPLACE INTO iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                iteration, parent_id, child_id, prompt_hash, llm_latency_ms,
                diff_n_extracted, diff_n_applied, eval_duration_s,
                child_score, improvement_delta,
            ),
        )

    def last_iteration(self) -> int | None:
        row = self._conn.execute("SELECT MAX(iteration) FROM iterations").fetchone()
        return row[0] if row and row[0] is not None else None

    # ------------------------------------------------------------------
    # run meta
    # ------------------------------------------------------------------

    def set_meta(
        self, key: str, value: Any,
        *, conn: sqlite3.Connection | None = None,
    ) -> None:
        c = conn or self._conn
        c.execute(
            "INSERT OR REPLACE INTO run_meta(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default


def _row_to_program(row: tuple[Any, ...]) -> Program:
    (
        id_, parent_id, generation, iteration_found, timestamp,
        source_code, metrics_json, complexity, diversity, island,
        feature_coords_json, changes_description, artifacts_json, metadata_json,
    ) = row
    return Program(
        id=id_,
        parent_id=parent_id,
        generation=generation,
        iteration_found=iteration_found,
        timestamp=timestamp,
        source_code=source_code,
        metrics=json.loads(metrics_json),
        complexity=complexity,
        diversity=diversity,
        island=island,
        feature_coords=json.loads(feature_coords_json),
        changes_description=changes_description or "",
        artifacts=json.loads(artifacts_json) if artifacts_json else {},
        metadata=json.loads(metadata_json) if metadata_json else {},
    )


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Drop bytes values (artifacts can carry them) so json.dumps doesn't choke.

    Bytes values are summarized as "<bytes:N>"; large in-DB blobs aren't part
    of v1's design.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            out[k] = f"<bytes:{len(v)}>"
        else:
            out[k] = v
    return out
