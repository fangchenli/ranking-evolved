"""Pinned-down data contracts shared by every backend.

Every Proposer, SearchStrategy, and Evaluator implementation in this codebase
exchanges these types — they are the framework's public seams. Keep them
narrow, frozen, and JSON-serializable so `core.replay` can faithfully snapshot
every loop iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ----------------------------------------------------------------------------
# Program & population
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Program:
    id: str
    source_code: str
    parent_id: str | None
    generation: int
    iteration_found: int
    timestamp: float
    metrics: dict[str, float]
    complexity: float
    diversity: float
    island: int
    feature_coords: dict[str, float]
    changes_description: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PopulationSnapshot:
    """Snapshot of the search-strategy state at a single iteration boundary."""
    n_programs: int
    islands: list[list[str]]
    island_generations: list[int]
    current_island: int
    last_migration_generation: int
    archive_cells: dict[str, str]
    archive_size: int
    best_program_id: str | None
    island_best_programs: list[str | None]


@dataclass(frozen=True)
class SamplingDecisions:
    rng_seed_hash: str
    parent_id: str
    parent_island: int
    inspiration_ids: list[str]
    inspiration_strategy: str
    top_program_ids: list[str]
    previous_program_ids: list[str]


@dataclass(frozen=True)
class AdmissionDecisions:
    target_island: int
    feature_coords: dict[str, float]
    cell_key: str
    evicted_program_id: str | None
    migration_fired: bool
    migration_details: dict[str, Any] | None


# ----------------------------------------------------------------------------
# Prompt & proposer
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    template_key: str
    iteration: int
    parent_id: str
    inspiration_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiffBlock:
    search: str
    replace: str
    matched_at_line: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class DiffApplication:
    pattern: str
    blocks: tuple[DiffBlock, ...]
    n_extracted: int
    n_applied: int
    fatal_error: str | None = None


@dataclass(frozen=True)
class ProposedCandidate:
    raw_response: str
    diff_blocks: tuple[DiffBlock, ...]
    full_rewrite: str | None
    proposer: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    per_dataset: dict[str, dict[str, float]]
    artifacts: dict[str, Any]
    duration_s: float
    error: str | None = None


# ----------------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------------

@runtime_checkable
class Proposer(Protocol):
    name: str
    async def propose(self, prompt: Prompt) -> ProposedCandidate: ...


@runtime_checkable
class SearchStrategy(Protocol):
    def initialize(self, seed: Program) -> None: ...
    def sample(self, iteration: int) -> tuple[Program, list[Program]]: ...
    def admit(self, child: Program, iteration: int) -> AdmissionDecisions: ...
    def best(self) -> Program: ...
    def snapshot(self) -> PopulationSnapshot: ...


@runtime_checkable
class Evaluator(Protocol):
    async def evaluate(self, program_path: str) -> EvaluationResult: ...
