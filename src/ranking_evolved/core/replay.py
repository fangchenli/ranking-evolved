"""Per-iteration replay capture: every intermediate state, written to disk.

This is the framework's correctness gate. When `evolution.capture_replay`
is on (or `--replay` is passed), every loop iteration writes a
`replay/step_<NNNN>.json` containing the full sampling decisions, prompt,
LLM call, diff application, child evaluation, and pre/post population
snapshots. The replay dashboard renders these for manual review and
side-by-side comparison against a captured OpenEvolve reference.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .types import (
    AdmissionDecisions,
    DiffApplication,
    EvaluationResult,
    PopulationSnapshot,
    Program,
    Prompt,
    SamplingDecisions,
)


@dataclass(frozen=True)
class ProgramSnapshot:
    """Compact subset of Program for use inside replay records."""
    id: str
    parent_id: str | None
    island: int
    generation: int
    iteration_found: int
    metrics: dict[str, float]
    feature_coords: dict[str, float]
    code_sha256: str
    code_preview: str

    @classmethod
    def of(cls, p: Program) -> ProgramSnapshot:
        sha = hashlib.sha256(p.source_code.encode()).hexdigest()
        return cls(
            id=p.id,
            parent_id=p.parent_id,
            island=p.island,
            generation=p.generation,
            iteration_found=p.iteration_found,
            metrics=dict(p.metrics),
            feature_coords=dict(p.feature_coords),
            code_sha256=sha,
            code_preview=p.source_code[:200],
        )


@dataclass(frozen=True)
class PromptRecord:
    system: str
    user: str
    template_key: str

    @classmethod
    def of(cls, p: Prompt) -> PromptRecord:
        return cls(system=p.system, user=p.user, template_key=p.template_key)


@dataclass(frozen=True)
class LLMCall:
    proposer: str
    model: str
    raw_response: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float


@dataclass(frozen=True)
class ReplayStep:
    schema_version: int
    iteration: int
    sampling: SamplingDecisions
    parent: ProgramSnapshot
    inspirations: list[ProgramSnapshot]
    top_programs: list[ProgramSnapshot]
    previous_programs: list[ProgramSnapshot]
    parent_artifacts: dict[str, Any] | None
    prompt: PromptRecord
    llm: LLMCall
    diff: DiffApplication
    child_code: str
    child_eval: EvaluationResult
    db_before: PopulationSnapshot
    db_after: PopulationSnapshot
    admission: AdmissionDecisions


class ReplayWriter:
    """Writes one `step_<NNNN>.json` per iteration into `<run_dir>/replay/`."""

    SCHEMA_VERSION = 1

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "replay"
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, step: ReplayStep) -> Path:
        out = self.dir / f"step_{step.iteration:04d}.json"
        out.write_text(json.dumps(_jsonable(asdict(step)), indent=2, ensure_ascii=False))
        return out


def build_step(
    *,
    iteration: int,
    sampling: SamplingDecisions,
    parent: Program,
    inspirations: list[Program],
    top_programs: list[Program],
    previous_programs: list[Program],
    parent_artifacts: dict[str, Any] | None,
    prompt: Prompt,
    llm_proposer: str,
    llm_model: str,
    llm_raw: str,
    llm_tokens_in: int | None,
    llm_tokens_out: int | None,
    llm_latency_ms: float,
    diff: DiffApplication,
    child_code: str,
    child_eval: EvaluationResult,
    db_before: PopulationSnapshot,
    db_after: PopulationSnapshot,
    admission: AdmissionDecisions,
) -> ReplayStep:
    return ReplayStep(
        schema_version=ReplayWriter.SCHEMA_VERSION,
        iteration=iteration,
        sampling=sampling,
        parent=ProgramSnapshot.of(parent),
        inspirations=[ProgramSnapshot.of(p) for p in inspirations],
        top_programs=[ProgramSnapshot.of(p) for p in top_programs],
        previous_programs=[ProgramSnapshot.of(p) for p in previous_programs],
        parent_artifacts=parent_artifacts,
        prompt=PromptRecord.of(prompt),
        llm=LLMCall(
            proposer=llm_proposer,
            model=llm_model,
            raw_response=llm_raw,
            tokens_in=llm_tokens_in,
            tokens_out=llm_tokens_out,
            latency_ms=llm_latency_ms,
        ),
        diff=diff,
        child_code=child_code,
        child_eval=child_eval,
        db_before=db_before,
        db_after=db_after,
        admission=admission,
    )


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclass instances and bytes for JSON dumping."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if dataclasses.is_dataclass(obj):
        return _jsonable(asdict(obj))
    return obj
