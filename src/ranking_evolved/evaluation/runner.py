"""Evaluator runner.

Loads a user-provided evaluator from a Python file (must expose
`evaluate(program_path: str) -> dict | EvaluationResult`) and invokes it
with a timeout. Inline mode runs in-process; subprocess mode forks a
worker. Inline is the default — subprocess is opt-in via config because
it doubles the fork cost for already-process-pool-isolated evaluators
(see plan section "Evaluation parallelism").
"""
from __future__ import annotations

import asyncio
import importlib.util
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ..core.types import EvaluationResult


class EvaluatorRunner:
    """Wraps a user `evaluate()` function with timeout + result normalization.

    `extra_env` is applied to `os.environ` for the duration of each
    `evaluate(...)` call (and restored after). The controller uses this to
    pass per-run objective settings (`EVAL_RECALL_K`, `EVAL_WARMUP_QUERIES`,
    etc.) through to evaluators that read them at module-load or per-call
    time.
    """

    def __init__(
        self,
        evaluator_path: str | Path,
        *,
        timeout_s: float = 1800,
        isolation: str = "inline",
        extra_env: dict[str, str] | None = None,
    ):
        self.evaluator_path = Path(evaluator_path)
        self.timeout_s = timeout_s
        if isolation not in ("inline", "subprocess"):
            raise ValueError(f"Unknown isolation mode: {isolation!r}")
        self.isolation = isolation
        self.extra_env = dict(extra_env or {})
        # Apply env vars BEFORE loading the evaluator: many evaluators
        # (including evaluator_parallel.py) read env at import time.
        with _scoped_env(self.extra_env):
            self._evaluate_fn = _load_evaluate(self.evaluator_path)

    async def evaluate(self, program_path: str | Path) -> EvaluationResult:
        program_path = str(program_path)
        start = time.perf_counter()
        try:
            if self.isolation == "subprocess":
                raw = await asyncio.wait_for(
                    asyncio.to_thread(
                        _run_in_subprocess,
                        self.evaluator_path,
                        program_path,
                        self.extra_env,
                    ),
                    timeout=self.timeout_s,
                )
            else:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(_call_with_env, self._evaluate_fn, program_path, self.extra_env),
                    timeout=self.timeout_s,
                )
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start
            return EvaluationResult(
                metrics={}, per_dataset={}, artifacts={},
                duration_s=duration, error=f"timeout after {self.timeout_s}s",
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            return EvaluationResult(
                metrics={}, per_dataset={}, artifacts={},
                duration_s=duration,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

        duration = time.perf_counter() - start
        return _normalize(raw, duration)


class _scoped_env:
    """Context manager that overlays env vars and restores on exit."""

    def __init__(self, env: dict[str, str]):
        self.env = env
        self._previous: dict[str, str | None] = {}

    def __enter__(self) -> "_scoped_env":
        for key, value in self.env.items():
            self._previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, prev in self._previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _call_with_env(fn: Callable[[str], Any], program_path: str, env: dict[str, str]) -> Any:
    with _scoped_env(env):
        return fn(program_path)


def _load_evaluate(path: Path) -> Callable[[str], Any]:
    module_name = f"_user_evaluator_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator at {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Python 3.11's @dataclass looks up cls.__module__
    # in sys.modules during class construction. If the module isn't there,
    # sys.modules.get(...) returns None and `.__dict__` raises AttributeError.
    sys.modules[module_name] = module
    # Put the evaluator file's directory on sys.path so its sibling imports
    # (e.g. `from evaluator_parallel_worker import ...`) resolve. Stays on
    # sys.path for the lifetime of the process — same scope evaluator workers
    # need anyway when they're spawned by the evaluator at eval time.
    parent = str(path.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    fn = getattr(module, "evaluate", None)
    if not callable(fn):
        raise AttributeError(f"Evaluator module {path} missing `evaluate(program_path)`")
    return fn


def _normalize(raw: Any, duration: float) -> EvaluationResult:
    """Accept either an EvaluationResult, a dict, or a (metrics, artifacts) tuple."""
    if isinstance(raw, EvaluationResult):
        return raw
    if isinstance(raw, dict):
        # Allow user to return either {"metrics": {...}, "artifacts": {...}, "per_dataset": {...}}
        # or just a flat metrics dict.
        if "metrics" in raw and isinstance(raw["metrics"], dict):
            return EvaluationResult(
                metrics=dict(raw["metrics"]),
                per_dataset=dict(raw.get("per_dataset", {})),
                artifacts=dict(raw.get("artifacts", {})),
                duration_s=duration,
                error=raw.get("error"),
            )
        return EvaluationResult(
            metrics={k: float(v) for k, v in raw.items() if isinstance(v, (int, float))},
            per_dataset={}, artifacts={}, duration_s=duration, error=None,
        )
    raise TypeError(
        f"evaluator returned {type(raw).__name__}; expected EvaluationResult or dict"
    )


def _run_in_subprocess(
    evaluator_path: Path,
    program_path: str,
    extra_env: dict[str, str] | None = None,
) -> Any:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_subprocess_entry,
        args=(str(evaluator_path), program_path, child_conn, dict(extra_env or {})),
    )
    proc.start()
    try:
        if not parent_conn.poll(timeout=None):  # block until child sends or exits
            pass
        result = parent_conn.recv()
    finally:
        proc.join()
    if isinstance(result, dict) and result.get("__error__"):
        raise RuntimeError(result["__error__"])
    return result


def _subprocess_entry(  # type: ignore[no-untyped-def]
    evaluator_path: str, program_path: str, conn, extra_env: dict[str, str],
) -> None:
    try:
        for key, value in (extra_env or {}).items():
            os.environ[key] = str(value)
        fn = _load_evaluate(Path(evaluator_path))
        out = fn(program_path)
        conn.send(out)
    except Exception as exc:
        conn.send({"__error__": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        conn.close()
