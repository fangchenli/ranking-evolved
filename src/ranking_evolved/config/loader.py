"""YAML loader with `${ENV}` interpolation + `--set k.v=value` overrides.

Also reads a `.env` file (if present) at the repo root and seeds env-vars
from it BEFORE interpolation, so secrets like `OPENAI_API_KEY` can live
in `.env` rather than the shell session.

Keeps the surface tiny: PyYAML if available, otherwise a fallback that
asks the user to install. Strict-mode: any unknown top-level key raises.
"""

from __future__ import annotations

import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .base import (
    Config,
    EvaluationConfig,
    EvolutionConfig,
    LoggingConfig,
    ProposerConfig,
    RunStoreConfig,
    TaskConfig,
    TraceConfig,
)
from .objective import (
    AggregationConfig,
    LatencyConfig,
    ObjectiveConfig,
    ObjectiveWeights,
)
from ..prompts.sampler import PromptConfig
from ..search.map_elites_islands import MapElitesIslandsConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    env_file: str | Path | None = None,
) -> Config:
    """Load YAML, interpolate ${ENV}, apply --set overrides, build Config.

    `env_file` is loaded into the process env (without overwriting existing
    vars) before interpolation. If `None`, looks for `.env` next to the
    config file, and then in the repo root (cwd-walk up 3 levels).
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install via `uv add pyyaml`.") from exc

    config_path = Path(path)
    _load_env_file(_resolve_env_file(config_path, env_file))

    raw_text = config_path.read_text()
    # Pre-pass: parse uninterpolated YAML once to extract `evaluation.env`
    # and seed it into os.environ (without overwriting shell vars). This lets
    # `${EVAL_DEVICE}` inside the YAML (e.g. in objective.latency.baseline_path)
    # resolve from the same `evaluation.env` block that the runner will pass
    # to the evaluator at call time — so the YAML alone fully describes the
    # run, no shell prerequisites required.
    try:
        prepass = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        # Will raise again on the real parse below — keep behavior consistent.
        prepass = {}
    _seed_env_from_evaluation_block(prepass)

    # Refuse literal API keys in the YAML BEFORE env interpolation —
    # interpolated `${OPENAI_API_KEY}` values are fine even if they expand to
    # a real-shaped key, but a hard-coded `api_key: sk-...` line in the YAML
    # is not.
    _enforce_no_literal_secrets_in_raw(raw_text, config_path)
    raw_text = _interpolate_env(raw_text)
    data = yaml.safe_load(raw_text) or {}
    if overrides:
        for ov in overrides:
            _apply_override(data, ov)
    return _build_config(data)


# Match a YAML line of the form `<some_key>: <maybe-quoted><secret>` where
# `<secret>` looks like an API key. Catches `api_key:`, `openai_api_key:`,
# `auth_token:`, etc. — anything ending in `_key`/`_token`/`_secret` plus
# the bare names `api_key`/`password`/etc.
_LITERAL_SECRET_LINE = re.compile(
    r"""(?xm)
    ^[ \t]*
    (?:[A-Za-z_][A-Za-z0-9_]*_(?:key|token|secret|password)|api_key|password|secret|auth_token)
    [ \t]*:[ \t]*
    ["']?
    (sk-[A-Za-z0-9_\-]{16,})       # OpenAI / Anthropic / project key shape
    ["']?
    [ \t]*$
    """
)


def _enforce_no_literal_secrets_in_raw(raw_yaml: str, config_path: Path) -> None:
    """Refuse to start when a literal API key is hard-coded in the YAML.

    Runs against the uninterpolated YAML text so that `${OPENAI_API_KEY}` is
    still allowed (it doesn't match the `sk-...` shape until after
    interpolation, which we don't do here).
    """
    m = _LITERAL_SECRET_LINE.search(raw_yaml)
    if m:
        raise ValueError(
            f"{config_path} contains a literal API-key-shaped value "
            f"(matched: {m.group(1)[:8]}...). Move it to the environment "
            "(e.g. `export OPENAI_API_KEY=...` or a `.env` file) and "
            'reference it as `api_key: "${OPENAI_API_KEY}"`.'
        )


def _seed_env_from_evaluation_block(data: dict) -> None:
    """Push values from the YAML's `evaluation.env` into os.environ.

    Existing env vars are NOT overwritten (shell wins). This runs BEFORE
    `_interpolate_env` so any `${VAR}` inside the YAML can be resolved from
    the YAML-declared env. Values are coerced to str (YAML parses "10" as
    int by default).
    """
    evaluation = data.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return
    env_block = evaluation.get("env") or {}
    if not isinstance(env_block, dict):
        return
    for key, value in env_block.items():
        if value is None:
            continue
        key_s = str(key)
        if key_s and key_s not in os.environ:
            os.environ[key_s] = str(value)


def _resolve_env_file(config_path: Path, explicit: str | Path | None) -> Path | None:
    """Find the .env file to load. Search order:
    1) `explicit` (if given) — must exist or we raise.
    2) `<config_dir>/.env` — for task-scoped overrides.
    3) walk up from cwd; first `.env` found wins.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"env_file does not exist: {p}")
        return p
    candidate = config_path.resolve().parent / ".env"
    if candidate.exists():
        return candidate
    cur = Path.cwd().resolve()
    for _ in range(8):
        c = cur / ".env"
        if c.exists():
            return c
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _load_env_file(path: Path | None) -> None:
    """Hand-parsed, dotenv-compatible: KEY=VALUE per line, # comments,
    optional surrounding quotes. Existing env-vars are NOT overwritten so
    a shell-set var still wins."""
    if path is None:
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _interpolate_env(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))

    return _ENV_PATTERN.sub(_sub, text)


def _apply_override(data: dict, override: str) -> None:
    """Apply `key.subkey=value` (value parsed as YAML scalar)."""
    import yaml  # noqa: PLC0415

    if "=" not in override:
        raise ValueError(f"--set expects key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    parts = key.split(".")
    cursor = data
    for i, part in enumerate(parts[:-1]):
        next_part = parts[i + 1]
        if isinstance(cursor, dict):
            if part not in cursor or cursor[part] is None:
                cursor[part] = [] if next_part.isdigit() else {}
            cursor = cursor[part]
        elif isinstance(cursor, list):
            if not part.isdigit():
                raise ValueError(f"--set conflict: {part} is not a list index")
            idx = int(part)
            if idx >= len(cursor):
                raise ValueError(f"--set list index out of range: {part}")
            cursor = cursor[idx]
        else:
            raise ValueError(f"--set conflict: {part} is not a mapping or list")

    final = parts[-1]
    value = yaml.safe_load(raw_value)
    if isinstance(cursor, dict):
        cursor[final] = value
    elif isinstance(cursor, list):
        if not final.isdigit():
            raise ValueError(f"--set conflict: {final} is not a list index")
        idx = int(final)
        if idx >= len(cursor):
            raise ValueError(f"--set list index out of range: {final}")
        cursor[idx] = value
    else:
        raise ValueError(f"--set conflict: {final} is not a mapping or list")


def _build_config(data: dict) -> Config:
    if "task" not in data:
        raise ValueError("config missing required `task:` section")
    task = TaskConfig(**_extract(data["task"], TaskConfig))

    # `search.algorithm` selects which strategy class to instantiate; strip
    # it before building the strategy-specific config dataclass.
    search_data = dict(data.get("search", {}) or {})
    algorithm = search_data.pop("algorithm", "map_elites_islands")
    if algorithm != "map_elites_islands":
        raise ValueError(
            f"Phase-1 supports only search.algorithm=map_elites_islands, got {algorithm!r}"
        )

    return Config(
        task=task,
        evolution=_section(data, "evolution", EvolutionConfig),
        search=MapElitesIslandsConfig(**_extract(search_data, MapElitesIslandsConfig)),
        proposer=_section(data, "proposer", ProposerConfig),
        prompt=_section(data, "prompt", PromptConfig),
        evaluation=_section(data, "evaluation", EvaluationConfig),
        objective=_objective_section(data),
        trace=_section(data, "trace", TraceConfig),
        logging=_section(data, "logging", LoggingConfig),
        run_store=_section(data, "run_store", RunStoreConfig),
    )


def _objective_section(data: dict) -> ObjectiveConfig:
    """Build ObjectiveConfig from the YAML `objective:` block.

    Three nested sub-dataclasses (`weights:`, `latency:`, `aggregation:`)
    need their own unknown-key validation, so we don't go through the
    generic `_section()` helper.
    """
    sub = data.get("objective", {}) or {}
    weights_raw = sub.get("weights", {}) or {}
    latency_raw = sub.get("latency", {}) or {}
    aggregation_raw = sub.get("aggregation", {}) or {}
    nested = ("weights", "latency", "aggregation")
    top = {k: v for k, v in sub.items() if k not in nested}
    unknown = set(top) - {f.name for f in fields(ObjectiveConfig)} - set(nested)
    if unknown:
        raise ValueError(f"unknown ObjectiveConfig keys: {sorted(unknown)}")
    return ObjectiveConfig(
        **_extract(top, ObjectiveConfig),
        weights=ObjectiveWeights(**_extract(weights_raw, ObjectiveWeights)),
        latency=LatencyConfig(**_extract(latency_raw, LatencyConfig)),
        aggregation=AggregationConfig(**_extract(aggregation_raw, AggregationConfig)),
    )


def _section(data: dict, key: str, cls: type) -> Any:
    sub = data.get(key, {}) or {}
    return cls(**_extract(sub, cls))


def _extract(d: dict, cls: type) -> dict:
    if not is_dataclass(cls):
        return dict(d)
    valid = {f.name for f in fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in valid}
