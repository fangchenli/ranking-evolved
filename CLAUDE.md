# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

`ranking-evolved` is a modular framework for evolving retrieval algorithms with
LLM-driven program synthesis. Each task (BM25, QL, etc.) lives in its own
folder under `tasks/` with seeds + evaluator + configs. The framework engine
is in `src/ranking_evolved/`.

The OpenEvolve dependency was removed during the restructuring; the engine is
now in-house. Pre-restructuring code lives in `legacy/` and **must not be
imported** from `src/` or `tasks/`.

## Layout

- `src/ranking_evolved/` — framework only: `core/`, `search/`, `proposers/`,
  `prompts/`, `evaluation/`, `config/`, `cli.py`.
- `tasks/bm25/` — active BM25 task: `library.py` (reference BM25),
  `evaluator.py` + `evaluator_worker.py`, `seeds/{freeform,constrained,composable}.py`,
  `interface.py` (BM25Candidate Protocol), `configs/*.yaml`.
- `tasks/_shared/` — `datasets.py` (BRIGHT/BEIR/TREC-DL loaders), `metrics.py`.
- `tasks/ql/` — structurally relocated only; every QL file carries a
  "STATUS: NOT YET MIGRATED" header. Do not run via `ranking-evolved`.
- `tasks/evolution_algo_test/` — smoke fixture for `tests/test_smoke.py`.
- `tests/` — `core/`, `search/`, `proposers/`, `prompts/`, `evaluation/`,
  `config/`, plus library tests (`test_bm25.py`, `test_metrics.py`,
  `test_lucene_tokenizer.py`, `test_bm25_regression.py`) and the framework
  smoke test. The `record_io` fixture in `tests/conftest.py` feeds
  `reports/test_dashboard.{html,json}`.
- `legacy/` — frozen archive of pre-restructuring code (benchmarks,
  evolved_programs, old configs/scripts/docs, unused evaluators, paper variants).
- `docs/architecture.md` — framework overview.

## Common commands

```bash
# Install / sync.
uv sync

# Run the dashboard tests (Phase-1 modules + library + smoke).
uv run ranking-evolved test-dashboard

# Drive an evolution loop on a config.
uv run ranking-evolved run --config tasks/bm25/configs/freeform_latency_aware.yaml --replay --max-iterations 50

# Resume an in-progress run.
uv run ranking-evolved run --resume output/bm25_freeform_latency_aware/<run_id> --max-iterations 100

# Render the per-step replay dashboard for a run.
uv run ranking-evolved replay-dashboard --run output/bm25_freeform_latency_aware/<run_id>

# Lint / type-check.
uv run ruff format
uv run mypy src/

# Run targeted tests.
uv run pytest tests/test_bm25_regression.py
uv run pytest tests/proposers/
```

## Run directory layout

`output/<task>/<timestamp>_<short-hash>/` is self-contained:
`config.resolved.yaml`, `manifest.json`, `run.db` (SQLite — programs,
metrics, lineage, prompts), `trace.jsonl` (one event per iteration),
`replay/step_NNNN.json` (when `--replay`), `baseline_latency.json`,
`experiment_summary.json`, `plots/*.pdf`, `best/program.py +
metrics.json`, `logs/run.log`.

## Architecture pointers

- The four extension points are `Optimizer` (proposers), `SearchStrategy`
  (search), `PromptBuilder` (prompts/sampler.py), and the user-supplied
  `evaluate(program_path)` function. See `docs/architecture.md` for the
  Protocol shapes.
- The controller (`src/ranking_evolved/core/controller.py`) is the main
  loop: parent select → inspiration select → prompt build → propose → diff
  apply → evaluate → admit → trace.
- `src/ranking_evolved/evaluation/runner.py` loads the evaluator by file
  path with `importlib.util.spec_from_file_location` and isolates
  optionally as a subprocess; env vars from `ObjectiveConfig` are scoped
  per call.
- The latency-aware objective (`src/ranking_evolved/evaluation/objective_math.py`)
  reads `baseline_latency.json` written during seed evaluation and
  recomputes `combined_score = w_recall*recall + w_ndcg*ndcg + w_lat*lat_score`.

## Important conventions

- **uv only.** All dep changes go through `uv add` / `uv remove` /
  `uv sync`. The lockfile is committed. No `pip install` in docs or scripts.
- **No imports from `legacy/`.** The framework and `tasks/` must not depend
  on anything under `legacy/`.
- **QL files are quarantined.** Until they are migrated, do not import them
  from active code, do not run them through the framework.
- **Tests must use `record_io`.** New modules in `src/ranking_evolved/`
  ship with at least one dashboard-instrumented test.

## Useful environment variables

```bash
EVAL_EXCLUDE_DATASETS=...       # comma-separated dataset ids to skip
EVAL_BRIGHT_DOMAINS=...         # restrict BRIGHT to listed domains
EVAL_BEIR_DATASETS=...          # restrict BEIR to listed datasets
EVAL_SAMPLE_QUERIES=20          # downsample queries per dataset
EVAL_TOKENIZER=lucene           # lucene | simple
EVAL_MAX_WORKERS=8              # parallel dataset workers (0 = auto)
EVAL_THREADS_PER_WORKER=8       # threads per worker
EVAL_OBJECTIVE_NAME=...         # set by the framework from ObjectiveConfig
EVAL_RECALL_K / EVAL_NDCG_K     # set by the framework from ObjectiveConfig
EVAL_WARMUP_QUERIES             # set by the framework when latency.enabled
```
