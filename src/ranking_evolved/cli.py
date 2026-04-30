"""ranking-evolved CLI.

Phase 0 ships the `test-dashboard` subcommand. Future phases will add
`run`, `resume`, `eval`, `list-algorithms`, `gc`, and `inspect`.
"""
from __future__ import annotations

import argparse
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
    "tests/optimizers",
    "tests/prompts",
    "tests/evaluation",
    "tests/config",
    "tests/tasks",
    "tests/test_smoke.py",
]

_PYTEST_NO_TESTS_COLLECTED = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ranking-evolved")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser(
        "test-dashboard",
        help="Run dashboard-instrumented tests and render reports/test_dashboard.html",
    )
    p_dash.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also run legacy IR-benchmark tests (slow; downloads HuggingFace data).",
    )
    p_dash.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (after `--`).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "test-dashboard":
        return _cmd_test_dashboard(args.pytest_args, include_legacy=args.include_legacy)

    parser.print_help()
    return 2


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
