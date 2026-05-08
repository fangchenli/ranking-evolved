from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tasks/evolution_algo_test/configs/evolution_algo_test.yaml"


def _run_evolution_algo_test() -> Path:
    cmd = [
        sys.executable,
        "-m",
        "ranking_evolved.cli",
        "run",
        "--config",
        str(CONFIG_PATH),
        "--replay",
        "--max-iterations",
        "24",
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"evolution algo test run failed\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    run_dir_line = None
    for line in result.stdout.splitlines():
        if line.startswith("[ranking-evolved] run dir:"):
            run_dir_line = line
            break

    assert run_dir_line is not None, (
        "Could not find run directory in CLI output.\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    run_dir = Path(run_dir_line.split(":", 1)[1].strip())
    assert run_dir.exists(), f"Run directory does not exist: {run_dir}"
    return run_dir


def _load_replay_steps(run_dir: Path) -> list[dict]:
    replay_dir = run_dir / "replay"
    assert replay_dir.exists(), f"Replay directory does not exist: {replay_dir}"

    paths = sorted(replay_dir.glob("step_*.json"))
    assert len(paths) == 24, f"Expected 24 replay steps, found {len(paths)}"

    return [json.loads(p.read_text()) for p in paths]


def test_evolution_algo_core_invariants() -> None:
    run_dir = _run_evolution_algo_test()
    steps = _load_replay_steps(run_dir)

    errors: list[tuple] = []
    migration_steps: list[int] = []

    for step in steps:
        i = step["iteration"]

        parent = step["parent"]
        sampling = step["sampling"]
        admission = step["admission"]
        diff = step["diff"]

        parent_score = parent["metrics"]["combined_score"]
        child_score = step["child_eval"]["metrics"]["combined_score"]

        if diff["n_extracted"] != 1:
            errors.append((i, "bad_n_extracted", diff["n_extracted"]))

        if diff["n_applied"] != 1:
            errors.append((i, "bad_n_applied", diff["n_applied"]))

        if diff["fatal_error"] is not None:
            errors.append((i, "diff_fatal_error", diff["fatal_error"]))

        if parent["id"] != sampling["parent_id"]:
            errors.append(
                (
                    i,
                    "parent_id_mismatch",
                    parent["id"],
                    sampling["parent_id"],
                )
            )

        if parent["island"] != sampling["parent_island"]:
            errors.append(
                (
                    i,
                    "parent_island_mismatch",
                    parent["island"],
                    sampling["parent_island"],
                )
            )

        # MAP-Elites cell-eviction invariant: a child can only evict the
        # program currently occupying its (complexity, diversity) cell when
        # the child's fitness is STRICTLY GREATER than that program's. The
        # comparison is against the EVICTED program's score, not the parent's
        # — under a true 2-D grid the child often lands in a cell occupied
        # by some non-parent program. (Earlier versions of this test compared
        # to the parent because diversity was hard-coded to 0.0, which
        # collapsed the grid to 1-D and made child and parent always share
        # a cell. Once diversity actually varies that incidental property
        # disappears.)
        evicted = admission.get("evicted_program_id")
        if evicted is not None:
            db_before = step.get("db_before") or {}
            evicted_score = None
            for entry in db_before.get("programs", []) or []:
                if entry.get("id") == evicted:
                    evicted_score = (entry.get("metrics") or {}).get("combined_score")
                    break
            if evicted_score is not None and child_score <= evicted_score:
                errors.append(
                    (
                        i,
                        "worse_child_evicted_program",
                        "evicted_score",
                        evicted_score,
                        "child_score",
                        child_score,
                        "evicted",
                        evicted,
                    )
                )

        if admission.get("migration_fired"):
            migration_steps.append(i)

    assert errors == []
    # Migration must fire at least once across 24 iterations (with
    # migration_interval=4 and 3 islands the algorithm crosses the
    # threshold multiple times). The exact iterations are not asserted
    # because they depend on UUID-keyed set iteration in `_migrate`,
    # which is non-deterministic across processes — what matters here
    # is that the migration mechanism IS firing at all and the other
    # invariants (parent/inspiration sampling, admission, no
    # worse-child evictions) hold.
    assert migration_steps, "expected at least one migration_fired event"

    final_step = steps[-1]
    final_best_id = final_step["db_after"]["best_program_id"]

    # Find the final best program score from visible replay state.
    # The scripted proposal sequence contains a unique best score of 0.50.
    best_score_seen = max(
        step["child_eval"]["metrics"]["combined_score"]
        for step in steps
        if step["child_eval"]["metrics"].get("error", 0.0) == 0.0
    )

    assert best_score_seen == 0.50
    assert final_best_id is not None
