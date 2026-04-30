# Architecture

This document describes the modular framework layered over the original RankEvolve work.
Phase-by-phase content is filled in as each migration phase lands; for the full plan see
`C:\Users\Jinming\.claude\plans\examine-ranking-evolved-i-need-fancy-eclipse.md`.

## Dependency management — uv only

All dependency changes go through **uv**. Do not use `pip`, `poetry`, or `pip-tools`.

| Action               | Command                          |
| -------------------- | -------------------------------- |
| Add a runtime dep    | `uv add <pkg>`                   |
| Add a dev dep        | `uv add --dev <pkg>`             |
| Remove a dep         | `uv remove <pkg>`                |
| Sync local env       | `uv sync`                        |
| Verify lock is fresh | `uv lock --check`                |
| CI install           | `uv sync --frozen`               |

The `uv.lock` file is committed. Any PR that changes dependencies must regenerate the
lockfile in the same commit.

## Run state & log management

(Defined in the plan; first implementation lands in Phase 1.)

Each `ranking-evolved run` produces one self-contained directory under
`output/<task>/<timestamp>_<short-hash>/` containing:

- `config.resolved.yaml` — fully-resolved config
- `run.db` — SQLite source of truth (programs, lineage, metrics, prompts, artifacts)
- `trace.jsonl` — streaming append-only event log
- `checkpoints/` — logical iteration cutoffs (DB views, not file copies)
- `best/` — current best program + metrics
- `logs/` — rotated, size-capped logs (`run.log`, `debug.log`, `optimizer.log`,
  `evaluator.log`); rotated optimizer logs are gzipped
- `manifest.json` — run id, git sha, uv lock hash, host, timing, exit status

`output/` is gitignored. `results/` (curated paper artifacts) is committed.

## Test dashboard

Every module ships with at least one `record_io`-style test (see `tests/conftest.py`).
The dashboard makes correctness auditable: each row pins the exact input fed to the
module under test and the exact output observed.

| Path                            | Purpose                                  |
| ------------------------------- | ---------------------------------------- |
| `tests/conftest.py`             | `record_io` fixture + dashboard renderer |
| `reports/test_dashboard.json`   | machine-readable record of every test    |
| `reports/test_dashboard.html`   | rendered table; one row per recorded I/O |

Regenerate with `uv run ranking-evolved test-dashboard`. CI runs the same command
and uploads the HTML as a build artifact on every PR.
