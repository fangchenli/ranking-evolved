"""Capture an OpenEvolve run as `ReplayStep`-shaped reference snapshots.

Reads an OE output directory's `evolution_trace.jsonl` (per-iter parent/
child + prompt + LLM response + metrics) and `checkpoints/checkpoint_*/`
(population snapshots — programs + island_feature_maps + archive +
island_generations + last_migration_generation), then emits one
`reference/step_<NNNN>.json` per iteration in the SAME schema we use
for native runs (with `_partial=True` flag on fields we can't recover).

The replay dashboard places these side-by-side with our run's steps so
the user can manually verify field-level equivalence.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .replay import ReplayStep


def capture_reference(
    *,
    openevolve_output: Path,
    out_dir: Path,
    max_steps: int | None = None,
) -> list[Path]:
    """Walk `openevolve_output/`, write reference step files to `out_dir`.

    Returns the list of files written.
    """
    openevolve_output = Path(openevolve_output)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = openevolve_output / "evolution_trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"no evolution_trace.jsonl under {openevolve_output}")

    # Find all checkpoints, sorted by iteration.
    ckpt_dir = openevolve_output / "checkpoints"
    checkpoints: dict[int, Path] = {}
    if ckpt_dir.exists():
        for d in ckpt_dir.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint_"):
                try:
                    checkpoints[int(d.name.split("_", 1)[1])] = d
                except ValueError:
                    continue

    written: list[Path] = []
    with trace_path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_steps and i >= max_steps:
                break
            event = json.loads(line)
            iteration = int(event["iteration"])
            ckpt_iter = _nearest_geq(iteration, sorted(checkpoints.keys()))
            ckpt = checkpoints.get(ckpt_iter) if ckpt_iter is not None else None
            ref = _convert(event, ckpt)
            out = out_dir / f"step_{iteration:04d}.json"
            out.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
            written.append(out)
    return written


def _nearest_geq(iteration: int, sorted_iters: list[int]) -> int | None:
    for it in sorted_iters:
        if it >= iteration:
            return it
    return None


def _convert(event: dict, ckpt: Path | None) -> dict:
    """Map an OE trace event (+ optional checkpoint) into our schema.

    Fields we can fully recover go in directly. Fields the trace doesn't
    expose (sampling RNG state, exact diff_blocks, db_before snapshot) are
    marked `null` with `_partial=True` so the dashboard can render them
    as "no reference" rather than spurious mismatches.
    """
    parent_metrics = _maybe_parse_dict(event.get("parent_metrics"))
    child_metrics = _maybe_parse_dict(event.get("child_metrics"))
    prompt = _maybe_parse_dict(event.get("prompt"))
    metadata = _maybe_parse_dict(event.get("metadata"))

    parent_code = event.get("parent_code")
    child_code = event.get("child_code")

    # Compose a ReplayStep-shaped record (matching our `core.replay`):
    record = {
        "schema_version": 1,
        "iteration": int(event["iteration"]),
        "_source": "openevolve",
        "_partial": True,
        "sampling": {
            "rng_seed_hash": None,
            "parent_id": event.get("parent_id"),
            "parent_island": event.get("island_id"),
            "inspiration_ids": None,
            "inspiration_strategy": None,
            "top_program_ids": None,
            "previous_program_ids": None,
        },
        "parent": _program_snapshot_from_trace(
            program_id=event.get("parent_id"),
            code=parent_code,
            metrics=parent_metrics,
            island=event.get("island_id"),
            generation=(event.get("generation") or 0) - 1,
        ),
        "inspirations": [],
        "top_programs": [],
        "previous_programs": [],
        "parent_artifacts": None,
        "prompt": {
            "system": (prompt or {}).get("system", ""),
            "user": (prompt or {}).get("user", ""),
            "template_key": (prompt or {}).get("template_key") or "diff_user",
        },
        "llm": {
            "proposer": "openevolve",
            "model": (metadata or {}).get("model", "unknown"),
            "raw_response": event.get("llm_response", ""),
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": float((metadata or {}).get("iteration_time", 0.0)) * 1000.0,
        },
        "diff": {
            "pattern": "<<<<<<< SEARCH(.*?)=======(.*?)>>>>>>> REPLACE",
            "blocks": [],     # OE doesn't expose block-level matches in trace
            "n_extracted": None,
            "n_applied": None,
            "fatal_error": None,
        },
        "child_code": child_code,
        "child_eval": {
            "metrics": child_metrics or {},
            "per_dataset": {},
            "artifacts": {},
            "duration_s": None,
            "error": None,
        },
        "db_before": _checkpoint_snapshot(ckpt, before=True),
        "db_after": _checkpoint_snapshot(ckpt, before=False),
        "admission": {
            "target_island": event.get("island_id"),
            "feature_coords": {},
            "cell_key": None,
            "evicted_program_id": None,
            "migration_fired": None,
            "migration_details": None,
        },
    }
    return record


def _maybe_parse_dict(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        # OE traces often serialize dicts as Python repr (e.g. "{'a': 1}")
        # and sometimes as JSON. Try both.
        try:
            return json.loads(value)
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
    return None


def _program_snapshot_from_trace(
    *,
    program_id: str | None,
    code: str | None,
    metrics: dict | None,
    island: int | None,
    generation: int,
) -> dict:
    if program_id is None:
        return {
            "id": "?", "parent_id": None, "island": island or 0,
            "generation": max(0, generation), "iteration_found": 0,
            "metrics": {}, "feature_coords": {},
            "code_sha256": "", "code_preview": "",
        }
    import hashlib
    code = code or ""
    return {
        "id": program_id,
        "parent_id": None,
        "island": island if island is not None else 0,
        "generation": max(0, generation),
        "iteration_found": 0,
        "metrics": metrics or {},
        "feature_coords": {},
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "code_preview": code[:200],
    }


def _checkpoint_snapshot(ckpt: Path | None, *, before: bool) -> dict:
    """Read OE checkpoint metadata.json and translate to PopulationSnapshot shape.

    OE only stores a single snapshot per checkpoint; we use the same view
    for both `db_before` and `db_after` (the dashboard will render them
    side-by-side and the user can ignore the redundancy). Native runs do
    distinguish — the side-by-side will show that distinction.
    """
    if ckpt is None or not (ckpt / "metadata.json").exists():
        return _empty_snapshot()
    md = json.loads((ckpt / "metadata.json").read_text())

    islands_lists: list[list[str]] = []
    for island in md.get("islands", []):
        if isinstance(island, list):
            islands_lists.append(list(island))
        else:
            islands_lists.append([])

    flat_cells: dict[str, str] = {}
    for i, m in enumerate(md.get("island_feature_maps", []) or []):
        if isinstance(m, dict):
            for cell, pid in m.items():
                flat_cells[f"{i}:{cell}"] = pid

    return {
        "n_programs": sum(len(i) for i in islands_lists),
        "islands": islands_lists,
        "island_generations": md.get("island_generations", []) or [],
        "current_island": md.get("current_island", 0),
        "last_migration_generation": md.get("last_migration_generation", 0),
        "archive_cells": flat_cells,
        "archive_size": len(md.get("archive", []) or []),
        "best_program_id": md.get("best_program_id"),
        "island_best_programs": md.get("island_best_programs", []) or [],
    }


def _empty_snapshot() -> dict:
    return {
        "n_programs": 0, "islands": [], "island_generations": [],
        "current_island": 0, "last_migration_generation": 0,
        "archive_cells": {}, "archive_size": 0,
        "best_program_id": None, "island_best_programs": [],
    }
