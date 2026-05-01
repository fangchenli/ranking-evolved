"""ranking-evolved CLI.

Subcommands:
  run             — run an evolution loop end-to-end against a config YAML.
  test-dashboard  — run dashboard-instrumented tests, render reports/test_dashboard.html.

Future phases will add: resume, eval, replay-capture-reference, replay-dashboard,
list-algorithms, gc, inspect.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from ranking_evolved._test_dashboard import write_dashboard

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
DASHBOARD_HTML = REPORTS_DIR / "test_dashboard.html"
DASHBOARD_JSON = REPORTS_DIR / "test_dashboard.json"

# Tests that participate in the dashboard live in these locations. They use the
# `record_io` fixture from tests/conftest.py. The legacy IR-benchmark tests at the
# top level of tests/ download HuggingFace datasets and take minutes; they are
# NOT part of the dashboard and are excluded by default.
DASHBOARD_TEST_PATHS = [
    "tests/core",
    "tests/search",
    "tests/proposers",
    "tests/prompts",
    "tests/evaluation",
    "tests/config",
    "tests/test_smoke.py",
]

_PYTEST_NO_TESTS_COLLECTED = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ranking-evolved")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_capref = sub.add_parser(
        "replay-capture-reference",
        help="Convert an OpenEvolve run dir into reference replay step files.",
    )
    p_capref.add_argument("--openevolve-output", required=True, type=Path,
                          help="Path to an OE run dir (containing evolution_trace.jsonl).")
    p_capref.add_argument("--out", required=True, type=Path,
                          help="Destination dir for step_NNNN.json files (e.g. <run>/replay/reference).")
    p_capref.add_argument("--max-steps", type=int, default=None,
                          help="Convert at most N steps (default: all).")

    p_dash = sub.add_parser(
        "replay-dashboard",
        help="Render <run>/replay/*.json (and optionally reference/) into HTML.",
    )
    p_dash.add_argument("--run", required=True, type=Path, help="Path to a run directory.")
    p_dash.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: <run>/replay_dashboard.html).")

    p_run = sub.add_parser("run", help="Run an evolution loop against a config YAML.")
    p_run.add_argument("--config", required=True, type=Path, help="Path to config YAML.")
    p_run.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override a config value: --set search.population_size=100",
    )
    p_run.add_argument(
        "--replay", action="store_true",
        help="Capture per-iteration replay snapshots (overrides config).",
    )
    p_run.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override evolution.max_iterations from the CLI.",
    )
    p_run.add_argument(
        "--resume", type=Path, default=None,
        help="Resume an existing run directory; --max-iterations is the total target iteration.",
    )

    p_test_dash = sub.add_parser(
        "test-dashboard",
        help="Run dashboard-instrumented tests and render reports/test_dashboard.html",
    )
    p_test_dash.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also run legacy IR-benchmark tests (slow; downloads HuggingFace data).",
    )
    p_test_dash.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (after `--`).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(
            config_path=args.config,
            overrides=args.overrides,
            replay=args.replay,
            max_iterations=args.max_iterations,
            resume=args.resume,
        )
    if args.cmd == "replay-capture-reference":
        return _cmd_replay_capture(
            openevolve_output=args.openevolve_output, out=args.out, max_steps=args.max_steps,
        )
    if args.cmd == "replay-dashboard":
        return _cmd_replay_dashboard(run=args.run, out=args.out)
    if args.cmd == "test-dashboard":
        return _cmd_test_dashboard(args.pytest_args, include_legacy=args.include_legacy)

    parser.print_help()
    return 2


def _cmd_run(
    *,
    config_path: Path,
    overrides: list[str],
    replay: bool,
    max_iterations: int | None,
    resume: Path | None = None,
) -> int:
    from ranking_evolved.config.loader import load_config
    from ranking_evolved.core.controller import Controller
    from ranking_evolved.core.logs import make_run_logger
    from ranking_evolved.core.manifest import (
        build_manifest, make_run_id, update_manifest, write_manifest,
    )
    from ranking_evolved.evaluation.runner import EvaluatorRunner
    # Importing the proposers package side-effect-registers every proposer.
    import ranking_evolved.proposers  # noqa: F401
    from ranking_evolved.proposers.base import REGISTRY as PROPOSERS

    config = load_config(config_path, overrides=overrides)
    if replay:
        config.evolution.capture_replay = True
    if max_iterations is not None:
        config.evolution.max_iterations = max_iterations

    task_label = _derive_task_label(config_path)
    if resume is not None:
        run_dir = resume.resolve()
        if not run_dir.exists():
            print(f"[ranking-evolved] resume run dir does not exist: {run_dir}", file=sys.stderr)
            return 2
        if not (run_dir / "run.db").exists():
            print(f"[ranking-evolved] resume run dir has no run.db: {run_dir}", file=sys.stderr)
            return 2
        run_id = run_dir.name
    else:
        run_id = make_run_id(task_label)
        run_dir = REPO_ROOT / "output" / task_label / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the resolved config + manifest.
        (run_dir / "config.resolved.yaml").write_text(_dump_yaml_safely(config))
        manifest = build_manifest(
            run_id=run_id, task=task_label, config_path=config_path, repo_root=REPO_ROOT,
        )
        write_manifest(run_dir, manifest)

    proposer_kind = config.proposer.kind
    if proposer_kind not in PROPOSERS:
        print(
            f"[ranking-evolved] unknown proposer {proposer_kind!r}; "
            f"available: {sorted(PROPOSERS.keys())}",
            file=sys.stderr,
        )
        return 2
    proposer = _build_proposer(proposer_kind, config.proposer)

    runner = EvaluatorRunner(
        config.task.evaluator,
        timeout_s=config.evaluation.timeout,
        isolation=config.evaluation.isolation,
        extra_env=_objective_env(config.objective),
    )

    logger = make_run_logger(
        "ranking_evolved.controller",
        run_dir / "run.log",
        level=config.logging.level,
        max_mb=config.logging.run_log_max_mb,
        backups=config.logging.run_log_backups,
    )
    _ensure_console_logging(logger, level=config.logging.level)

    controller = Controller(
        config=config, run_dir=run_dir, proposer=proposer, runner=runner,
        logger=logger, config_path=config_path, run_id=run_id,
    )
    try:
        best = asyncio.run(controller.run(seed_path=Path(config.task.seed), resume=resume is not None))
    finally:
        controller.close()
    update_manifest(run_dir, ended_at=_iso_now(), exit_status="ok")

    print(f"[ranking-evolved] run dir: {run_dir}")
    print(f"[ranking-evolved] best: {best.id}")
    print(f"[ranking-evolved] best metrics: {json.dumps(best.metrics, indent=2)}")
    summary_path = run_dir / "experiment_summary.json"
    metrics_path = run_dir / "program_metrics.jsonl"
    baseline_path = run_dir / "baseline_latency.json"
    if summary_path.exists():
        print(f"[ranking-evolved] summary:        {summary_path}")
    if metrics_path.exists():
        print(f"[ranking-evolved] per-program:    {metrics_path}")
    if baseline_path.exists():
        print(f"[ranking-evolved] baseline:       {baseline_path}")
    print(
        f"[ranking-evolved] replay HTML:    "
        f"uv run ranking-evolved replay-dashboard --run {run_dir} --out report.html"
    )
    return 0


def _ensure_console_logging(logger: logging.Logger, *, level: str) -> None:
    """Attach one concise stdout handler for live long-run progress."""
    has_console = any(getattr(handler, "_ranking_evolved_console", False) for handler in logger.handlers)
    if has_console:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler._ranking_evolved_console = True  # type: ignore[attr-defined]
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def _objective_env(objective) -> dict[str, str]:
    """Translate ObjectiveConfig into env vars the evaluator can read.

    Only emit env vars when the objective differs from the legacy defaults
    so that runs using the historical objective don't gain new vars in
    `os.environ`. The evaluator (evaluator_parallel.py) reads these at
    module import time, so they must be set before the EvaluatorRunner
    loads it.
    """
    env: dict[str, str] = {}
    if objective.name != "recall100_ndcg10":
        env["EVAL_OBJECTIVE_NAME"] = str(objective.name)
    if objective.recall_k != 100:
        env["EVAL_RECALL_K"] = str(objective.recall_k)
    if objective.ndcg_k != 10:
        env["EVAL_NDCG_K"] = str(objective.ndcg_k)
    if objective.latency.enabled:
        env["EVAL_WARMUP_QUERIES"] = str(objective.latency.warmup_queries)
    return env


def _cmd_replay_capture(*, openevolve_output: Path, out: Path, max_steps: int | None) -> int:
    from ranking_evolved.core.replay_capture import capture_reference
    written = capture_reference(openevolve_output=openevolve_output, out_dir=out, max_steps=max_steps)
    print(f"[ranking-evolved] wrote {len(written)} reference step file(s) to {out}")
    if written:
        print(f"[ranking-evolved]   first: {written[0].name}")
        print(f"[ranking-evolved]   last:  {written[-1].name}")
    return 0


def _cmd_replay_dashboard(*, run: Path, out: Path | None) -> int:
    from ranking_evolved.core.replay_dashboard import render_dashboard
    out_path = out or (run / "replay_dashboard.html")
    rendered = render_dashboard(run, out_path=out_path)
    print(f"[ranking-evolved] dashboard: {rendered}")
    return 0


def _derive_task_label(config_path: Path) -> str:
    """Derive `<task>_<config_stem>` from the canonical layout.

    `tasks/<task>/configs/<stem>.yaml`  -> `<task>_<stem>` (e.g. `bm25_freeform`).
    Anything else                       -> `<stem>` (the config file's stem).

    The label is the second segment of the run directory:
    `output/<task_label>/<run_id>/`.
    """
    p = Path(config_path).resolve()
    parts = p.parts
    if (
        len(parts) >= 4
        and p.parent.name == "configs"
        and p.parent.parent.parent.name == "tasks"
    ):
        return f"{p.parent.parent.name}_{p.stem}"
    return p.stem or "run"


def _build_proposer(kind: str, cfg) -> object:
    """Instantiate a proposer from config.

    Proposer modules self-register via `@register_proposer`; importing the
    `proposers` package (done at the top of `_cmd_run`) populates REGISTRY.
    """
    from ranking_evolved.proposers.base import REGISTRY

    cls = REGISTRY[kind]
    primary_model = (cfg.models[0]["name"] if cfg.models else None) or "default"

    if kind == "fake":
        if cfg.transcript_path is None:
            raise ValueError("proposer.transcript_path required for kind=fake")
        responses: list[tuple[str, str]] = []
        for line in Path(cfg.transcript_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            responses.append((d["raw_response"], d.get("model", "fake-1")))
        return cls(responses=responses)
    if kind == "openai_responses":
        return cls(
            api_base=cfg.api_base or "https://api.openai.com/v1",
            api_key=cfg.api_key,
            model=primary_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            reasoning_effort=cfg.reasoning_effort,  # None → omit from body
            timeout=cfg.timeout,
            retries=cfg.retries,
        )
    if kind == "anthropic":
        return cls(
            api_key=cfg.api_key,
            model=primary_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if kind == "claude_code":
        return cls(model=primary_model, timeout=float(cfg.timeout))
    if kind == "codex":
        return cls(timeout=float(cfg.timeout))
    if kind == "ensemble":
        # Each member is a proposer config dict in cfg.models[*]; for v1 we
        # only support same-kind ensembles via the YAML schema. CLI users who
        # need richer ensembles construct in code.
        raise NotImplementedError(
            "ensemble proposer requires programmatic construction; not yet "
            "wired through YAML."
        )
    if kind == "scripted":
        if cfg.proposals_jsonl is None:
            raise ValueError("proposer.proposals_jsonl required for kind=scripted")
        return cls(
            proposals_jsonl=cfg.proposals_jsonl,
            timeout=cfg.timeout,
            retries=cfg.retries,
        )
    raise NotImplementedError(f"Proposer {kind!r} not yet wired in `_build_proposer`.")


def _dump_yaml_safely(obj) -> str:
    """Serialize a Config dataclass tree to YAML for `config.resolved.yaml`."""
    import dataclasses
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return json.dumps(_to_dict(obj), indent=2, default=str)
    return yaml.safe_dump(_to_dict(obj), sort_keys=False)


def _to_dict(obj):  # type: ignore[no-untyped-def]
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _iso_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _cmd_test_dashboard(extra_pytest_args: list[str], *, include_legacy: bool) -> int:
    extra = list(extra_pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if include_legacy:
        targets = ["tests"]
    else:
        targets = [p for p in DASHBOARD_TEST_PATHS if (REPO_ROOT / p).exists()]

    if not targets:
        # Nothing to run yet (Phase 0 state). Write a valid empty dashboard directly.
        json_path, html_path = write_dashboard(
            repo_root=REPO_ROOT, records=[], exit_status=0
        )
        print(
            "[ranking-evolved] no dashboard-instrumented tests yet "
            "(skipping pytest); wrote empty dashboard."
        )
        print(f"[ranking-evolved] dashboard: {html_path}")
        print(f"[ranking-evolved] json:      {json_path}")
        return 0

    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q", *targets, *extra]
    print(f"[ranking-evolved] running: {' '.join(cmd)}")
    print(f"[ranking-evolved] cwd: {REPO_ROOT}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if not DASHBOARD_HTML.exists():
        # No record_io entries were captured (e.g., the targets contained only
        # tests that don't use the fixture). Write an explicit empty dashboard.
        write_dashboard(repo_root=REPO_ROOT, records=[], exit_status=int(result.returncode))

    print()
    print(f"[ranking-evolved] dashboard: {DASHBOARD_HTML}")
    print(f"[ranking-evolved] json:      {DASHBOARD_JSON}")

    if result.returncode == _PYTEST_NO_TESTS_COLLECTED:
        return 0
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
