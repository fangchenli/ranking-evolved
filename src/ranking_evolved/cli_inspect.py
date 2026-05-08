"""`ranking-evolved inspect-step` — read what happened at a single iteration.

Two data layers exist on disk per run, and this command reads both:

  * **always-on** — `run.db` (programs + iterations + archive_cells tables) and
    `trace.jsonl`. Every iteration is here regardless of `capture_replay`,
    with full prompt and LLM response stored per program.

  * **per-step snapshot** — `<run>/replay/step_NNNN.json` when capture_replay
    was enabled. Adds the inspirations list, before/after population
    snapshots, and the diff text. Sampled by `evolution.capture_replay_every`.

`inspect-step` prefers the snapshot when present and falls back to the
always-on layer for sections that don't require snapshot-only data
(`inspirations`, `population-before`, `population-after` are snapshot-only;
everything else has a fallback).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

KNOWN_SECTIONS = (
    "summary", "prompt", "parent", "inspirations",
    "population", "population-before", "population-after",
    "diff", "llm-response", "child-code", "eval",
)
SNAPSHOT_ONLY_SECTIONS = {"inspirations", "population", "population-before", "population-after"}


def cmd_inspect_step(*, run_dir: Path, step: int, sections: list[str]) -> int:
    """Render scoped views of a single iteration. Returns process exit code."""
    if not run_dir.exists():
        print(f"[inspect-step] run dir not found: {run_dir}", file=sys.stderr)
        return 2

    requested = _resolve_sections(sections)
    snapshot = _load_snapshot(run_dir, step)
    db_state = _load_from_run_db(run_dir, step)

    if snapshot is None and db_state is None:
        print(
            f"[inspect-step] no data for step {step}: neither replay/step_{step:04d}.json "
            f"nor run.db has a record. Range is 1..{_last_iteration(run_dir) or '?'}.",
            file=sys.stderr,
        )
        return 1

    if snapshot is None:
        print(
            f"note: replay/step_{step:04d}.json not captured "
            f"(capture_replay disabled or sampled out by capture_replay_every). "
            "Falling back to run.db / trace.jsonl — inspirations and population "
            "snapshots are NOT available for this step.\n",
            file=sys.stderr,
        )

    for section in requested:
        if section in SNAPSHOT_ONLY_SECTIONS and snapshot is None:
            _print_section_header(section)
            print("(unavailable — replay snapshot not captured for this step)")
            continue
        renderer = _RENDERERS[section]
        renderer(snapshot=snapshot, db_state=db_state, step=step, run_dir=run_dir)

    return 0


# ---------------------------------------------------------------------------
# section selection
# ---------------------------------------------------------------------------


def _resolve_sections(sections: list[str]) -> list[str]:
    if not sections:
        return ["summary"]
    if "all" in sections:
        return list(KNOWN_SECTIONS)
    out: list[str] = []
    seen: set[str] = set()
    for s in sections:
        if s not in KNOWN_SECTIONS and s != "all":
            raise SystemExit(
                f"unknown section: {s!r}. valid: {', '.join(['all', *KNOWN_SECTIONS])}"
            )
        # Normalize "population" → "population-after" alias is kept distinct;
        # callers using "population" get the after-snapshot.
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------


def _load_snapshot(run_dir: Path, step: int) -> dict[str, Any] | None:
    p = run_dir / "replay" / f"step_{step:04d}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        print(f"[inspect-step] WARN: {p} is not valid JSON: {exc}", file=sys.stderr)
        return None


def _load_from_run_db(run_dir: Path, step: int) -> dict[str, Any] | None:
    """Read iteration row + child program row + parent program row from run.db.

    Returns None if run.db doesn't exist OR has no iteration row for `step`.
    """
    db_path = run_dir / "run.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # The iterations table primary key may differ; SELECT by iteration_id
        # column (see core/run_store.py schema).
        iter_row = conn.execute(
            "SELECT iteration, parent_id, child_id, child_score, "
            "improvement_delta, prompt_hash, llm_latency_ms, "
            "diff_n_extracted, diff_n_applied, eval_duration_s, island "
            "FROM iterations WHERE iteration = ?",
            (step,),
        ).fetchone()
        if iter_row is None:
            return None
        keys = ["iteration", "parent_id", "child_id", "child_score",
                "improvement_delta", "prompt_hash", "llm_latency_ms",
                "diff_n_extracted", "diff_n_applied", "eval_duration_s", "island"]
        iter_data = dict(zip(keys, iter_row, strict=True))

        child = _read_program(conn, iter_data["child_id"]) if iter_data.get("child_id") else None
        parent = _read_program(conn, iter_data["parent_id"]) if iter_data.get("parent_id") else None

        return {"iteration": iter_data, "child": child, "parent": parent}
    except sqlite3.OperationalError as exc:
        print(
            f"[inspect-step] WARN: run.db schema mismatch ({exc}); "
            "always-on fallback partially unavailable.",
            file=sys.stderr,
        )
        return None
    finally:
        conn.close()


def _read_program(conn: sqlite3.Connection, program_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, parent_id, generation, iteration_found, timestamp, "
        "source_code, metrics_json, complexity, diversity, island, "
        "prompt_system, prompt_user, llm_raw_response "
        "FROM programs WHERE id = ?",
        (program_id,),
    ).fetchone()
    if row is None:
        return None
    keys = ["id", "parent_id", "generation", "iteration_found", "timestamp",
            "source_code", "metrics_json", "complexity", "diversity", "island",
            "prompt_system", "prompt_user", "llm_raw_response"]
    rec = dict(zip(keys, row, strict=True))
    if rec.get("metrics_json"):
        try:
            rec["metrics"] = json.loads(rec["metrics_json"])
        except json.JSONDecodeError:
            rec["metrics"] = {}
    else:
        rec["metrics"] = {}
    return rec


def _last_iteration(run_dir: Path) -> int | None:
    db_path = run_dir / "run.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT MAX(iteration) FROM iterations").fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# section renderers
# ---------------------------------------------------------------------------


def _print_section_header(section: str) -> None:
    print(f"\n{'='*78}\n## {section}\n{'='*78}")


def _render_summary(*, snapshot, db_state, step, run_dir, **_) -> None:
    _print_section_header("summary")
    if snapshot is not None:
        s = snapshot
        sampling = s.get("sampling") or {}
        adm = s.get("admission") or {}
        print(f"iteration:           {s.get('iteration', step)}")
        print(f"parent_id:           {(s.get('parent') or {}).get('id', '<unknown>')}")
        print(f"parent_island:       {sampling.get('parent_island', '<unknown>')}")
        print(f"num_inspirations:    {len(s.get('inspirations') or [])}")
        print(f"child_id:            {(s.get('child_eval') or {}).get('program_id', '<see snapshot>')}")
        print(f"admission_island:    {adm.get('island', '<unknown>')}")
        print(f"diff_extracted:      {(s.get('diff') or {}).get('n_extracted', '?')}")
        print(f"diff_applied:        {(s.get('diff') or {}).get('n_applied', '?')}")
        print(f"prompt_template_key: {(s.get('prompt') or {}).get('template_key', '<unknown>')}")
        return
    if db_state is None:
        print("(no data)")
        return
    it = db_state["iteration"]
    print(f"iteration:           {it['iteration']}")
    print(f"parent_id:           {it['parent_id']}")
    print(f"child_id:            {it['child_id']}")
    print(f"child_score:         {it['child_score']}")
    print(f"improvement_delta:   {it['improvement_delta']}")
    print(f"island:              {it['island']}")
    print(f"prompt_hash:         {it['prompt_hash']}")
    print(f"llm_latency_ms:      {it['llm_latency_ms']}")
    print(f"eval_duration_s:     {it['eval_duration_s']}")
    print(f"diff_extracted:      {it['diff_n_extracted']}")
    print(f"diff_applied:        {it['diff_n_applied']}")


def _render_prompt(*, snapshot, db_state, **_) -> None:
    _print_section_header("prompt")
    system, user = "", ""
    if snapshot is not None:
        prompt = snapshot.get("prompt") or {}
        system = prompt.get("system") or ""
        user = prompt.get("user") or ""
    elif db_state is not None and db_state.get("child"):
        system = db_state["child"].get("prompt_system") or ""
        user = db_state["child"].get("prompt_user") or ""
    if not system and not user:
        print("(no prompt recorded)")
        return
    print("--- SYSTEM ---")
    print(system)
    print("\n--- USER ---")
    print(user)


def _render_parent(*, snapshot, db_state, **_) -> None:
    _print_section_header("parent")
    parent: dict[str, Any] | None = None
    if snapshot is not None:
        parent = snapshot.get("parent")
    elif db_state is not None:
        parent = db_state.get("parent")
    if not parent:
        print("(no parent recorded — likely the seed)")
        return
    print(f"id:               {parent.get('id', '<unknown>')}")
    print(f"generation:       {parent.get('generation', '?')}")
    print(f"island:           {parent.get('island', '?')}")
    metrics = parent.get("metrics") or {}
    if metrics:
        keys = ["combined_score", "recall_at_1000", "ndcg_at_10", "latency_p50_ms"]
        for k in keys:
            if k in metrics:
                print(f"{k:<18}{metrics[k]}")
    print("\n--- source_code ---")
    print(parent.get("source_code") or "(no source recorded)")


def _render_inspirations(*, snapshot, **_) -> None:
    _print_section_header("inspirations")
    if snapshot is None:
        return  # already handled by snapshot-only guard
    inspirations = snapshot.get("inspirations") or []
    if not inspirations:
        print("(no inspirations sampled this step)")
        return
    for i, insp in enumerate(inspirations, start=1):
        metrics = insp.get("metrics") or {}
        head_score = metrics.get("combined_score", "?")
        print(f"\n[{i}] id={insp.get('id', '?')}  gen={insp.get('generation', '?')}  "
              f"island={insp.get('island', '?')}  combined_score={head_score}")
        src = insp.get("source_code") or ""
        head = "\n".join(src.splitlines()[:30])
        print(head)
        if src.count("\n") > 30:
            print(f"... ({src.count(chr(10)) - 30} more lines)")


def _render_population(*, snapshot, which: str, **_) -> None:
    _print_section_header(f"population-{which}")
    if snapshot is None:
        return
    db_block = snapshot.get(f"db_{which}") or {}
    islands = db_block.get("islands") or []
    archive_cells = db_block.get("archive_cells") or {}
    island_best = db_block.get("island_best_programs") or []
    island_gens = db_block.get("island_generations") or []
    if not islands and not archive_cells:
        print("(no population snapshot recorded)")
        return
    # Group archive cells by island.
    cells_by_island: dict[int, list[tuple[str, str]]] = {}
    for cell_key, program_id in archive_cells.items():
        # cell_key format is "island:rest"
        if ":" in cell_key:
            isl_str, rest = cell_key.split(":", 1)
            try:
                isl = int(isl_str)
            except ValueError:
                isl, rest = -1, cell_key
        else:
            isl, rest = -1, cell_key
        cells_by_island.setdefault(isl, []).append((rest, program_id))
    for isl, members in enumerate(islands):
        gen = island_gens[isl] if isl < len(island_gens) else "?"
        best = island_best[isl] if isl < len(island_best) else "?"
        cells = cells_by_island.get(isl, [])
        print(f"\nIsland {isl}  (gen={gen}, members={len(members)}, archive_cells={len(cells)}, best={best})")
        for cell_key, program_id in sorted(cells):
            print(f"  cell {cell_key}:  {program_id}")


def _render_population_before(**kwargs) -> None:
    _render_population(which="before", **kwargs)


def _render_population_after(**kwargs) -> None:
    _render_population(which="after", **kwargs)


def _render_diff(*, snapshot, db_state, **_) -> None:
    _print_section_header("diff")
    if snapshot is not None:
        diff = snapshot.get("diff") or {}
        print(f"n_extracted: {diff.get('n_extracted', '?')}")
        print(f"n_applied:   {diff.get('n_applied', '?')}")
        if diff.get("fatal_error"):
            print(f"fatal_error: {diff['fatal_error']}")
        blocks = diff.get("blocks") or []
        for i, blk in enumerate(blocks, start=1):
            print(f"\n--- block {i} ---")
            if isinstance(blk, dict):
                print("SEARCH:")
                print(blk.get("search") or "")
                print("REPLACE:")
                print(blk.get("replace") or "")
            else:
                print(blk)
        return
    if db_state is not None:
        it = db_state["iteration"]
        print(f"n_extracted: {it['diff_n_extracted']}")
        print(f"n_applied:   {it['diff_n_applied']}")
        print("(full SEARCH/REPLACE blocks not available — replay snapshot was not captured)")


def _render_llm_response(*, snapshot, db_state, **_) -> None:
    _print_section_header("llm-response")
    text = ""
    if snapshot is not None:
        text = (snapshot.get("llm") or {}).get("raw_response") or ""
    elif db_state is not None and db_state.get("child"):
        text = db_state["child"].get("llm_raw_response") or ""
    if not text:
        print("(no LLM response recorded)")
        return
    print(text)


def _render_child_code(*, snapshot, db_state, **_) -> None:
    _print_section_header("child-code")
    code = ""
    if snapshot is not None:
        code = snapshot.get("child_code") or ""
    elif db_state is not None and db_state.get("child"):
        code = db_state["child"].get("source_code") or ""
    if not code:
        print("(no child code recorded)")
        return
    print(code)


def _render_eval(*, snapshot, db_state, **_) -> None:
    _print_section_header("eval")
    metrics: dict[str, Any] = {}
    if snapshot is not None:
        ce = snapshot.get("child_eval") or {}
        metrics = ce.get("metrics") or {}
    if not metrics and db_state is not None and db_state.get("child"):
        metrics = db_state["child"].get("metrics") or {}
    if not metrics:
        print("(no eval metrics recorded)")
        return
    print(json.dumps(metrics, indent=2, sort_keys=True))


_RENDERERS: dict[str, Any] = {
    "summary": _render_summary,
    "prompt": _render_prompt,
    "parent": _render_parent,
    "inspirations": _render_inspirations,
    "population": _render_population_after,
    "population-before": _render_population_before,
    "population-after": _render_population_after,
    "diff": _render_diff,
    "llm-response": _render_llm_response,
    "child-code": _render_child_code,
    "eval": _render_eval,
}
