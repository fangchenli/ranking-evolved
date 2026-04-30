# Plan: Restructure `ranking-evolved` into a Modular Retrieval-Evolution Framework

## Context

`ranking-evolved` is the codebase behind the RankEvolve paper (Nian et al., 2026) which uses LLM-driven evolution to discover retrieval algorithms. Today it's tightly coupled to a single workflow: BM25/QL seeds + 9 partially-redundant evaluators + 5 OpenEvolve YAML configs + the OpenEvolve git dependency. There is **no `import openevolve` anywhere** — coupling is purely via `python -m openevolve.cli` invocations and a fragile convention (the seed must expose `BM25, Corpus, tokenize, LuceneTokenizer`).

The owner wants the repo to become a **modular research framework for many retrieval-evolution problems** (BM25 accuracy, late-interaction efficiency, LLM reranking, etc.) used by many researchers, with:

1. Core evolution loop reimplemented in-house (no runtime OpenEvolve dep), preserving its good ideas.
2. Pluggable **optimizer backends** (OpenAI/Anthropic API, Claude Code subprocess, Codex hook, local model, manual).
3. Pluggable **search algorithms** (MAP-Elites + islands today; easy to swap for Pareto, hill-climb, GEPA, etc.).
4. Adding a new **task** (= seed + evaluator + benchmark) should be one folder, not surgery on shared files.
5. Reproducibility, clean structure, lean CLI.

The exploration phase confirmed:
- The **OpenEvolve coupling surface is small** (CLI invocation, evaluator returning a `dict[str, float]` with `combined_score`, SEARCH/REPLACE diff format, JSON-per-program checkpoints) — feasible to reimplement.
- **OpenEvolve's good ideas worth keeping**: MAP-Elites + island population, rich prompt context (top + diverse + recent + artifacts), artifact side-channel, dataclass+YAML config, async evaluator.
- **OpenEvolve's pain points to fix**: no evaluator subprocess isolation, JSON-per-program checkpoints (slow at scale), fragile diff parsing, basic ensemble.
- **skydiscover patterns worth borrowing**: single CLI entry, dataclass configs with env interpolation + CLI overrides, algorithm registry pattern (each algorithm = its own module), `FakeLLMPool` for deterministic tests, evaluator-as-Python-file simplicity.

## Goals & Non-Goals

**Goals**
- A core engine that reproduces OpenEvolve's loop *in spirit* (same population dynamics, same prompt richness, same checkpoint semantics).
- Clean Protocol-based seams at the four extension points: **Optimizer**, **SearchStrategy**, **PromptBuilder**, **Evaluator**.
- A `tasks/` layer where each task is a self-contained folder; the core never imports task code.
- One CLI (`ranking-evolved`), one config schema, one evaluator contract.
- Deterministic CI via a `FakeOptimizer` mock — no real LLM calls in tests.

**Non-Goals**
- Bit-for-bit numerical reproduction of past OpenEvolve runs. We'll match seeds/feature dimensions/selection logic but not promise byte-identical RNG sequences. Past evolved programs in `evolved_programs/` are preserved as artifacts; we don't replay them. *(Confirmed: behavioral equivalence target.)*
- A general-purpose "evolve any code" framework. This stays focused on retrieval research; abstractions are Protocol-based but the docs and tasks are retrieval-shaped.
- Live OpenEvolve compatibility. The OpenEvolve dep is removed, not kept as a fallback.

**Resolved decisions (from user)**
- Migration: **incremental in-place** — new core lives alongside old files; old files deleted phase-by-phase after verification. Same git repo, no `v2/` subtree.
- Tasks layout: **top-level `tasks/`** at repo root, sibling to `src/` and `docs/`.
- Parity target: **behavioral equivalence** — same dynamics, no RNG-identity promise.
- v1 optimizer scope: **all five** ship in v1 — `openai_chat`, `fake`, `anthropic`, `claude_code`, `codex`. (Claude Code and Codex are subprocess hooks; Anthropic uses the native SDK with prompt caching.)
- Dependency manager: **uv only**. All dependency changes go through `uv add` / `uv remove` / `uv sync`; the lockfile (`uv.lock`) is committed. No `pip install` in docs or scripts.
- Run state + logs: **first-class concern**, not an afterthought. Each run gets one self-contained directory with bounded, rotated logs and a single source-of-truth state store (see "Run State & Log Management" below).
- Test discipline: **every module ships with tests**. No module is "done" until a per-module test exists with explicit input/output captured. A test dashboard is generated from these tests as proof of correctness (see "Test Discipline & Dashboard" below).

## Target Repo Structure

```
ranking-evolved/
├── pyproject.toml            # drop openevolve dep; add anthropic, click/typer
├── README.md                 # rewritten: framework overview + add-a-task quickstart
├── LICENSE
├── docs/
│   ├── architecture.md       # core engine + extension points
│   ├── adding_a_task.md      # walk-through: seed + evaluator + config
│   ├── adding_an_optimizer.md
│   ├── adding_a_search_algorithm.md
│   ├── reproducibility.md    # seeding, checkpoints, trace format
│   └── results/              # paper figures, tables (kept)
│
├── src/ranking_evolved/      # KEEP package name (paper citation continuity)
│   ├── __init__.py           # public API: run_evolution(), Config
│   ├── cli.py                # `ranking-evolved {run,resume,evaluate}`
│   │
│   ├── config/
│   │   ├── base.py           # Config (top-level), EvolutionConfig
│   │   ├── llm.py            # OptimizerConfig (formerly LLMConfig)
│   │   ├── search.py         # SearchConfig + algorithm-specific subtypes
│   │   ├── evaluator.py
│   │   ├── prompt.py
│   │   └── loader.py         # YAML + ${ENV} interpolation + CLI overrides
│   │
│   ├── core/                 # the engine (replaces openevolve dep)
│   │   ├── controller.py     # main loop: orchestrates search + optimizer + evaluator
│   │   ├── program.py        # Program dataclass (id, source, metrics, parent_id, ...)
│   │   ├── population.py     # Population container
│   │   ├── run_store.py      # SQLite-backed run state (programs, lineage, metrics, prompts)
│   │   ├── checkpoint.py     # thin wrapper: snapshot/restore via run_store
│   │   ├── trace.py          # streaming JSONL trace (one event per iteration)
│   │   └── logs.py           # rotated, level-bucketed run logger (size-capped)
│   │
│   ├── search/               # pluggable search strategies (registry pattern)
│   │   ├── __init__.py       # REGISTRY = {"map_elites_islands": ..., "hillclimb": ...}
│   │   ├── base.py           # SearchStrategy Protocol
│   │   ├── map_elites_islands.py   # default; matches OpenEvolve behaviour
│   │   ├── hillclimb.py      # simple beam/greedy
│   │   └── pareto.py         # multi-objective
│   │
│   ├── optimizers/           # pluggable LLM proposers (registry pattern)
│   │   ├── __init__.py       # REGISTRY = {"openai_chat": ..., "claude_code": ...}
│   │   ├── base.py           # Optimizer Protocol
│   │   ├── openai_chat.py    # OpenAI-compatible (covers OpenAI, OpenRouter, local)
│   │   ├── anthropic.py      # native Anthropic SDK with prompt caching
│   │   ├── claude_code.py    # subprocess hook to `claude` CLI
│   │   ├── codex.py          # subprocess hook to Codex CLI
│   │   ├── ensemble.py       # weighted multi-optimizer (port from OpenEvolve)
│   │   ├── manual.py         # human-in-the-loop write-to-file mode
│   │   └── fake.py           # deterministic mock for tests/CI
│   │
│   ├── prompts/
│   │   ├── sampler.py        # builds prompt from parent + inspiration + artifacts
│   │   ├── diff.py           # SEARCH/REPLACE parse+apply (with whitespace tolerance)
│   │   └── templates/        # jinja-style: system, user (diff/full), feedback
│   │
│   ├── evaluation/
│   │   ├── runner.py         # invokes user evaluator with timeout + isolation
│   │   ├── result.py         # EvaluationResult dataclass
│   │   ├── isolation.py      # subprocess-based isolation (improvement over OpenEvolve)
│   │   └── cascade.py        # cheap → expensive staged eval
│   │
│   └── utils/
│       ├── seeding.py
│       └── io.py
│
├── tasks/                    # ONE FOLDER PER PROBLEM (top-level, discoverable)
│   ├── README.md             # how to add a task
│   │
│   ├── _shared/              # retrieval-side utilities reused across IR tasks
│   │   ├── ir_evaluator.py   # collapses 9 evaluator files → 1 generic IR runner
│   │   ├── datasets.py       # BRIGHT/BEIR/TREC-DL loaders (centralised)
│   │   ├── metrics.py        # nDCG, recall, MRR, MAP
│   │   └── tokenizers.py     # Lucene + simple
│   │
│   ├── bm25/                 # the work the paper covers
│   │   ├── README.md         # what evolves, what stays fixed, how to run
│   │   ├── library.py        # reference BM25 (current bm25.py, slimmed)
│   │   ├── seeds/
│   │   │   ├── constrained.py
│   │   │   ├── composable.py
│   │   │   └── freeform.py
│   │   ├── evaluator.py      # 30-line shim over tasks/_shared/ir_evaluator.py
│   │   ├── interface.py      # formal Protocol for what a BM25 candidate must expose
│   │   └── configs/
│   │       ├── constrained.yaml
│   │       ├── composable.yaml
│   │       └── freeform.yaml
│   │
│   ├── ql/                   # Query Likelihood
│   │   ├── library.py
│   │   ├── seeds/freeform.py
│   │   ├── evaluator.py      # ALSO a shim over the same _shared/ir_evaluator.py
│   │   ├── interface.py
│   │   └── configs/freeform.yaml
│   │
│   ├── late_interaction/     # placeholder for future ColBERT-style efficiency work
│   │   └── README.md         # stub: what this task should evolve
│   │
│   └── llm_reranker/         # placeholder for future LLM-reranking work
│       └── README.md
│
├── benchmarks/               # validation harnesses (kept, lightly cleaned)
│   ├── pyserini_baseline.py  # consolidates compare_pyserini.py + scripts/
│   └── ...
│
├── scripts/                  # plotting + SLURM submission (kept, deduped)
│   ├── plot_evolution.py
│   ├── slurm/
│   └── visualizer.sh
│
├── tests/
│   ├── test_smoke.py         # E2E with FakeOptimizer (2 iter, no real LLM)
│   ├── core/                 # controller, run_store, checkpoint, trace, logs
│   ├── search/               # MAP-Elites correctness; island migration
│   ├── prompts/              # SEARCH/REPLACE parser; whitespace tolerance
│   ├── optimizers/           # one test per optimizer w/ mocked transport
│   ├── evaluation/           # subprocess timeout; isolation; cascade
│   ├── config/               # YAML+ENV+override loading
│   ├── tasks/bm25/           # regression: existing seeds parse, eval returns dict
│   └── conftest.py           # captures every test's input + output → reports/test_dashboard.json
│
├── reports/
│   ├── test_dashboard.json   # machine-readable: per-test {module, function, input, output, status}
│   └── test_dashboard.html   # rendered table (one row per test) — proof-of-correctness view
│
├── results/                  # paper-result JSONs + figures (kept)
└── output/                   # gitignored run artifacts (per-run dirs, see Run State below)
```

## Core Abstractions

Four Protocols at [src/ranking_evolved/](src/ranking_evolved/):

```python
# core/program.py
@dataclass
class Program:
    id: str
    source_code: str
    metrics: dict[str, float]
    parent_id: str | None
    generation: int
    iteration_found: int
    artifacts: dict[str, Any] = field(default_factory=dict)
    feature_coords: dict[str, float] | None = None  # for MAP-Elites

# search/base.py
class SearchStrategy(Protocol):
    def initialize(self, seed: Program) -> Population: ...
    def select_parent(self, pop: Population, rng: Random) -> Program: ...
    def select_inspiration(self, pop: Population, parent: Program, rng: Random) -> list[Program]: ...
    def admit(self, pop: Population, child: Program) -> None: ...
    def best(self, pop: Population) -> Program: ...

# optimizers/base.py
class Optimizer(Protocol):
    async def propose(self, prompt: Prompt) -> ProposedCandidate: ...
    # ProposedCandidate = {source_code, changes_description, raw_response}

# evaluation contract (user side, no inheritance — just a function)
def evaluate(program_path: str) -> EvaluationResult: ...
# EvaluationResult = {metrics: dict, artifacts: dict, error: str | None}
```

The controller is a ~150-line loop:
```
load(config)
seed = parse_seed(config.task.seed)
seed.metrics = await runner.run(seed, evaluator)
pop = strategy.initialize(seed)
for it in range(config.evolution.max_iterations):
    parent = strategy.select_parent(pop, rng)
    inspiration = strategy.select_inspiration(pop, parent, rng)
    prompt = prompt_builder.build(parent, inspiration, artifacts=last_eval_artifacts)
    candidate = await optimizer.propose(prompt)
    child = apply_diff(parent, candidate)  # or full-rewrite
    child.metrics = await runner.run(child, evaluator)
    strategy.admit(pop, child)
    trace.write(it, parent, child, prompt, candidate)
    if it % checkpoint_interval == 0: checkpoint.save(pop, it)
return strategy.best(pop)
```

## Run State & Log Management

Each invocation of `ranking-evolved run` produces **one self-contained run directory** under `output/<task>/<timestamp>_<short-hash>/`. Everything about the run lives there; nothing is scattered. Layout:

```
output/bm25/20260429_143022_a3f1/
├── config.resolved.yaml      # the fully-resolved config (after ENV + CLI overrides)
├── run.db                    # SQLite: programs, metrics, lineage, prompts, artifacts
├── trace.jsonl               # streaming append-only event log (one line per iteration)
├── checkpoints/
│   ├── checkpoint_010/       # symlink/dirref into run.db at iter 10
│   └── checkpoint_020/
├── best/
│   ├── program.py            # always points to current best
│   └── metrics.json
├── logs/
│   ├── run.log               # INFO+ rotated (10 MB × 5 files = 50 MB hard cap)
│   ├── debug.log             # DEBUG rotated (50 MB × 3 files = 150 MB hard cap, off by default)
│   ├── optimizer.log         # raw LLM request/response transcripts (compressed, gzip rotation)
│   └── evaluator.log         # per-iteration evaluator stdout/stderr (rotated)
└── manifest.json             # run id, git sha, uv lock hash, host, start/end time, exit status
```

Design choices that keep this elegant at scale:

- **SQLite is the single source of truth** for run state (`run_store.py`). Programs, metrics, parent links, prompts, artifacts all live in one file. Fixes OpenEvolve's slow "JSON-per-program" pattern; a 10K-iteration run loads in seconds. `checkpoint_N` is a logical view (an iteration cutoff), not a file copy — resume just queries `WHERE iteration <= N`.
- **Logs are rotated and size-capped** via `logging.handlers.RotatingFileHandler` per stream. Hard ceilings are config knobs (`logging.run_log_max_mb`, `logging.optimizer_log_max_mb`, etc.). `debug.log` is OFF by default and opt-in via `--debug` — debug runs do not silently fill disks.
- **Optimizer transcripts are gzipped** on rotation. LLM prompt+response bodies are the bulkiest artifact; gzip cuts ~10× for free.
- **`trace.jsonl` is the streaming public API** — append-only, one event per iteration, schema-stable. Existing `scripts/plot_evolution_metrics.py` keeps working without changes. The DB is for queries; the JSONL is for streaming consumers (live dashboards, downstream tools).
- **Resume is one command**: `ranking-evolved resume --run output/bm25/20260429_143022_a3f1`. The CLI reads `manifest.json`, opens `run.db`, replays state. No more "find the latest checkpoint dir" guesswork.
- **`output/` is gitignored**; `results/` (curated paper artifacts) is committed.
- **GC sub-command**: `ranking-evolved gc --keep-best --older-than 30d` prunes old runs, keeping the manifest + best/ + final metrics for provenance, deleting bulk logs and intermediate prompts.

## Test Discipline & Dashboard

Every module shipped requires a co-located test that **explicitly records input and output** so correctness is auditable.

**Convention.** Tests use a small `record_io` fixture (defined in `tests/conftest.py`) that wraps an assertion and writes a structured record to `reports/test_dashboard.json`. Example:

```python
def test_search_replace_diff_apply(record_io):
    parent = "def f():\n    return 1\n"
    diff = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace",
        input={"parent": parent, "diff": diff},
        run=lambda: apply_search_replace(parent, diff),
    )
    assert out == "def f():\n    return 2\n"
```

`record_io` captures: module, function, exact input, exact output, pass/fail, duration. After `pytest`, `conftest.py` writes `reports/test_dashboard.json` and renders `reports/test_dashboard.html` — a single-page table:

| Module | Function | Input (collapsed) | Output (collapsed) | Status | Time |
| --- | --- | --- | --- | --- | --- |
| `core/run_store.py` | `RunStore.add_program` | `Program(id="p1", ...)` | `True` (row inserted) | ✅ | 3 ms |
| `prompts/diff.py` | `apply_search_replace` | parent + diff (shown) | new source (shown) | ✅ | 1 ms |
| `optimizers/anthropic.py` | `AnthropicOptimizer.propose` | mocked prompt | `ProposedCandidate(...)` | ✅ | 12 ms |
| ... | ... | ... | ... | ... | ... |

CLI: `ranking-evolved test-dashboard` runs pytest and opens the HTML report. CI publishes `reports/test_dashboard.html` as an artifact on every PR.

**Coverage rule per phase.** No phase is "done" until:
- every new module has at least one `record_io` test,
- the smoke test passes,
- `reports/test_dashboard.html` is regenerated and reviewed.

## Configuration Design

One YAML schema, loaded into nested dataclasses (skydiscover-style, not Pydantic):

```yaml
task:
  seed: tasks/bm25/seeds/freeform.py
  evaluator: tasks/bm25/evaluator.py

evolution:
  max_iterations: 200
  checkpoint_interval: 10
  random_seed: 42

search:
  algorithm: map_elites_islands
  population_size: 200
  archive_size: 50
  num_islands: 3
  migration_interval: 20
  migration_rate: 0.1
  feature_dimensions: [complexity, diversity]
  feature_bins: 10

optimizer:
  kind: openai_chat
  models:
    - {name: gpt-5.2, weight: 1.0}
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 8192
  timeout: 180
  retries: 3

prompt:
  diff_based: true
  num_top_programs: 3
  num_diverse_programs: 2
  use_template_stochasticity: true
  include_artifacts: true
  system_message: |
    You are evolving a constrained BM25 ranking function...

evaluation:
  timeout: 1800
  parallel_evaluations: 1
  isolation: subprocess
  cascade: false

trace:
  enabled: true
  format: jsonl
  include_prompts: true

logging:
  level: INFO              # run.log threshold
  debug_log: false         # opt-in DEBUG stream
  run_log_max_mb: 10
  run_log_backups: 5
  optimizer_log_max_mb: 50
  optimizer_log_backups: 3
  optimizer_log_gzip: true
  evaluator_log_max_mb: 25
  evaluator_log_backups: 3

run_store:
  backend: sqlite          # only option in v1
  vacuum_on_close: true
```

CLI overrides via `--set search.population_size=500` (matches skydiscover's `apply_overrides`). The current 5 OpenEvolve configs collapse into 4 task configs (one per seed strategy) under `tasks/{bm25,ql}/configs/`.

## CLI

```bash
ranking-evolved run            --config tasks/bm25/configs/freeform.yaml
ranking-evolved resume         --run output/bm25/20260429_143022_a3f1   # auto-finds latest checkpoint
ranking-evolved resume         --run output/bm25/20260429_143022_a3f1 --checkpoint 50
ranking-evolved eval           --program path.py --evaluator tasks/bm25/evaluator.py
ranking-evolved list-algorithms                       # introspect search/optimizer registries
ranking-evolved gc             --keep-best --older-than 30d    # prune bulk logs, keep manifests
ranking-evolved test-dashboard                        # run pytest, render reports/test_dashboard.html
ranking-evolved inspect        --run output/bm25/<id> # dump summary from run.db
```

All commands are wired through `uv run ranking-evolved ...`; the project script is registered in `[project.scripts]` of `pyproject.toml`. Dependencies are added/removed exclusively via `uv add` / `uv remove`; `uv.lock` is committed.

Replaces all of: [run_evolve.sh](run_evolve.sh), [run_evaluation.sh](run_evaluation.sh), [run_evaluation_parallel.sh](run_evaluation_parallel.sh), [run_evaluation_and_seeds_parallel.sh](run_evaluation_and_seeds_parallel.sh), [resume_evolve.sh](resume_evolve.sh), [resume_evolve_ql.sh](resume_evolve_ql.sh), [commands.txt](commands.txt).

## Migration Phases

**Phase 0 — Repo scaffolding & test-dashboard infrastructure**
- Lock dependency management: confirm `uv.lock` is current; CI runs `uv sync --frozen`; document `uv add` / `uv remove` as the only dep-change path in [docs/architecture.md](docs/architecture.md).
- Add `tests/conftest.py` with the `record_io` fixture and the dashboard renderer (writes `reports/test_dashboard.json`, renders `reports/test_dashboard.html`).
- Add `ranking-evolved test-dashboard` CLI stub.
- Done when: `uv run ranking-evolved test-dashboard` produces an empty-but-valid dashboard HTML.

**Phase 1 — Core engine + all optimizers (no behaviour change to existing workflow yet)**
- Build `src/ranking_evolved/{config,core,search,optimizers,prompts,evaluation}/`.
- Implement `core/run_store.py` (SQLite-backed program/lineage/metric store), `core/checkpoint.py` (logical snapshots into the store), `core/trace.py` (streaming JSONL), `core/logs.py` (rotated, size-capped logger with gzip on rotation). Run-directory layout per "Run State & Log Management".
- Implement `map_elites_islands` matching OpenEvolve's selection + migration logic *behaviorally* (not RNG-identical); cross-check qualitatively on a toy seed.
- Implement SEARCH/REPLACE diff parser with whitespace-tolerant matching (improvement over OpenEvolve's strict matching).
- Implement subprocess-isolated evaluator runner (improvement over OpenEvolve's inline async).
- Ship **all five v1 optimizers**, each behind the same `Optimizer` Protocol:
  - `openai_chat` — OpenAI-compatible HTTP client (covers OpenAI, OpenRouter, OptiLLM, vLLM/Ollama via `api_base`). The default for current workflows.
  - `anthropic` — native Anthropic SDK with **prompt caching** for the system message + inspiration block (saves cost on long runs where these stay stable across iterations).
  - `claude_code` — subprocess hook that pipes the prompt to the `claude` CLI and parses the response back. Useful for agentic proposals.
  - `codex` — subprocess hook to the Codex CLI, same shape as `claude_code`.
  - `fake` — deterministic mock returning canned diffs; required for CI and the smoke test.
- **Tests written alongside each module**, every one using `record_io`:
  - `tests/core/test_run_store.py` — insert program, query lineage, snapshot+restore at iter N (input: 5 fake programs; output: lineage tree + restored state).
  - `tests/core/test_checkpoint.py` — round-trip a 20-program population through checkpoint/restore.
  - `tests/core/test_trace.py` — append events, verify schema, verify resume reads same content.
  - `tests/core/test_logs.py` — write past rotation threshold, verify file count + gzip of rotated.
  - `tests/search/test_map_elites_islands.py` — admit programs, verify cell occupancy + island migration on schedule.
  - `tests/prompts/test_diff.py` — SEARCH/REPLACE happy path + whitespace-mismatch tolerance + multi-block.
  - `tests/optimizers/test_{openai_chat,anthropic,claude_code,codex,fake}.py` — each with mocked transport (httpx mock for HTTP, subprocess mock for CLIs); records the exact prompt sent and the parsed candidate returned.
  - `tests/evaluation/test_runner.py` — subprocess timeout, error capture, result roundtrip.
  - `tests/config/test_loader.py` — YAML + `${ENV}` interpolation + `--set` overrides.
  - `tests/test_smoke.py` — 2-iteration E2E with `fake` optimizer on a stub task; verifies a complete run directory is produced (run.db non-empty, trace.jsonl has 2 lines, logs/run.log exists, best/program.py present).
- Done when: `reports/test_dashboard.html` shows every Phase-1 module with at least one ✅ row, and the smoke test passes.
- Old `evaluator_parallel.py`-style files left untouched; old OpenEvolve workflow still runs in parallel during transition.

**Phase 2 — Migrate BM25/QL tasks**
- Collapse 9 evaluator files → `tasks/_shared/ir_evaluator.py` parameterised by:
  - candidate-class names (`BM25` vs `QL`),
  - score formula (`0.8*recall@100 + 0.2*ndcg@10` becomes a config field),
  - dataset filter set.
- `tasks/bm25/evaluator.py` and `tasks/ql/evaluator.py` are 30-line shims.
- Move seeds: `src/ranking_evolved/bm25_*_fast.py` → `tasks/bm25/seeds/`; ditto QL.
- Move `bm25.py` → `tasks/bm25/library.py`; `datasets.py`, `metrics.py` → `tasks/_shared/`.
- Convert 5 OpenEvolve YAML configs → 4 new-schema configs under `tasks/{bm25,ql}/configs/`.
- Add `tasks/bm25/interface.py` with formal Protocol (kills the magic-string contract).
- **Tests added** (all using `record_io`):
  - `tests/tasks/_shared/test_ir_evaluator.py` — input: a tiny 10-doc / 3-query corpus + a known-good BM25 seed; output: nDCG@10, recall@100 matching hand-computed values.
  - `tests/tasks/_shared/test_datasets.py` — load BRIGHT/biology subset, verify shape; mock HF dataset for hermetic CI.
  - `tests/tasks/_shared/test_metrics.py` — nDCG/recall/MRR with a 4-item ranking; explicit input/output table.
  - `tests/tasks/bm25/test_seeds_parse.py` — each seed file imports cleanly and exposes the required Protocol.
  - `tests/tasks/bm25/test_eval_regression.py` — running each seed through the new evaluator on a fixed sample reproduces metrics from the old `evaluator.py` within 1%.
- Done when: dashboard rows for every new module are ✅ and the regression test confirms parity with the old evaluator.

**Phase 3 — Drop OpenEvolve dep**
- `uv remove openevolve` (lockfile updates automatically); confirm `uv sync` is clean.
- Verify: a 20-iteration `freeform` run via `ranking-evolved run` produces a metric trajectory whose final `combined_score` is within ±2% of the prior OpenEvolve run (behavioral, not bit-identical).
- Confirm the run directory contains everything per "Run State & Log Management" (run.db populated, trace.jsonl streamed, logs rotated as expected, best/ updated, manifest.json complete).
- Update [README.md](README.md): new quickstart + add-a-task walkthrough; all install/run commands use `uv`.

**Phase 4 — Future-task scaffolding**
- `tasks/late_interaction/README.md`: stub describing what the task evolves (compute/memory tradeoffs in ColBERT-style retrieval), what the candidate Protocol looks like, what benchmarks measure.
- `tasks/llm_reranker/README.md`: stub for LLM-reranking efficiency/quality.
- `docs/adding_a_task.md`: full walkthrough using `bm25/` as the example.
- `docs/adding_an_optimizer.md`: walkthrough using `claude_code` as the example of a subprocess-style optimizer.
- `docs/adding_a_search_algorithm.md`: walkthrough using `hillclimb.py` as a minimal example.

**Phase 5 — Cleanup (delete files)**
- [evaluator.py](evaluator.py), [evaluator_beir.py](evaluator_beir.py), [evaluator_bright.py](evaluator_bright.py), [evaluator_parallel.py](evaluator_parallel.py), [evaluator_parallel_wave.py](evaluator_parallel_wave.py), [evaluator_parallel_wave_worker.py](evaluator_parallel_wave_worker.py), [evaluator_parallel_worker.py](evaluator_parallel_worker.py), [evaluator_ql_parallel.py](evaluator_ql_parallel.py), [evaluator_ql_parallel_worker.py](evaluator_ql_parallel_worker.py), [eval_unified.py](eval_unified.py)
- [compare_pyserini.py](compare_pyserini.py), [verify_all_implementations.py](verify_all_implementations.py), [hyperparam_search.py](hyperparam_search.py), [optuna_search.py](optuna_search.py)
- [comparison_report.md](comparison_report.md), [comparison_results.json](comparison_results.json), [hyperparam_results.json](hyperparam_results.json), [commands.txt](commands.txt)
- All [run_*.sh](run_evolve.sh) / [resume_*.sh](resume_evolve.sh) shell scripts (replaced by CLI).

## Reuse Map (existing files → new location/role)

| Existing | New role |
| --- | --- |
| [evaluator_parallel.py](evaluator_parallel.py) (1583 LOC) | logic moves to [tasks/_shared/ir_evaluator.py](tasks/_shared/ir_evaluator.py), parameterised |
| [src/ranking_evolved/bm25.py](src/ranking_evolved/bm25.py) (71 KB) | → [tasks/bm25/library.py](tasks/bm25/library.py), slimmed |
| [src/ranking_evolved/datasets.py](src/ranking_evolved/datasets.py) | → [tasks/_shared/datasets.py](tasks/_shared/datasets.py) |
| [src/ranking_evolved/metrics.py](src/ranking_evolved/metrics.py) | → [tasks/_shared/metrics.py](tasks/_shared/metrics.py) |
| [src/ranking_evolved/bm25_*_fast.py](src/ranking_evolved/) | → [tasks/bm25/seeds/](tasks/bm25/seeds/) |
| [src/ranking_evolved/ql_*.py](src/ranking_evolved/) | → [tasks/ql/](tasks/ql/) |
| [openevolve_config_*.yaml](openevolve_config_constrained.yaml) | → [tasks/{bm25,ql}/configs/*.yaml](tasks/) (new schema) |
| [scripts/plot_evolution_metrics.py](scripts/) | unchanged; reads same JSONL trace format |
| [evolved_programs/](evolved_programs/) | unchanged; preserved as paper artifacts |
| [tests/](tests/) | core/regression tests preserved; new core tests added |

## Verification

End-to-end checks before declaring each phase done:

- **After Phase 0**: `uv run ranking-evolved test-dashboard` produces a valid (empty) `reports/test_dashboard.html`; CI runs `uv sync --frozen` cleanly.
- **After Phase 1**: `pytest tests/test_smoke.py` passes — 2-iteration run with `FakeOptimizer` on a stub task — and `reports/test_dashboard.html` shows a ✅ row for every Phase-1 module (run_store, checkpoint, trace, logs, map_elites_islands, diff parser, each of the 5 optimizers, evaluator runner, config loader). Each row records the exact input fed to the function and the exact output observed.
- **After Phase 2**: `ranking-evolved eval --program tasks/bm25/seeds/constrained.py --evaluator tasks/bm25/evaluator.py --datasets bright:biology` produces metrics within 1% of `evaluator.py src/ranking_evolved/bm25_constrained_fast.py --bright biology`. Dashboard adds rows for `ir_evaluator`, `datasets`, `metrics`, and the BM25/QL regression tests.
- **After Phase 3**: 20-iteration `ranking-evolved run --config tasks/bm25/configs/freeform.yaml` produces a metric trajectory whose final `combined_score` is within ±2% of an equivalent OpenEvolve run from before. The resulting run directory passes a `tests/test_run_directory_layout.py` check that asserts every expected file exists, log files are within size caps, and `run.db` has the expected number of programs. `uv tree` shows no `openevolve` node.
- **Phase 1 per-optimizer**: each of the 5 optimizers (`openai_chat`, `anthropic`, `claude_code`, `codex`, `fake`) passes its unit test (mocked transport — `record_io` captures the exact prompt sent and parsed candidate returned) and a 2-iteration smoke run on a stub task.
- **Continuous**: every PR regenerates `reports/test_dashboard.html` and uploads it as a CI artifact. Adding a module without adding a `record_io` test fails CI via a `tests/test_dashboard_coverage.py` rule that scans `src/ranking_evolved/` for modules with no corresponding test entry in the dashboard.
