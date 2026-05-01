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
from .objective import LatencyConfig, ObjectiveConfig, ObjectiveWeights
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
        raise ImportError(
            "PyYAML is required. Install via `uv add pyyaml`."
        ) from exc

    config_path = Path(path)
    _load_env_file(_resolve_env_file(config_path, env_file))

    raw_text = config_path.read_text()
    raw_text = _interpolate_env(raw_text)
    data = yaml.safe_load(raw_text) or {}
    if overrides:
        for ov in overrides:
            _apply_override(data, ov)
    return _build_config(data)


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
            line = line[len("export "):].lstrip()
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
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"--set conflict: {part} is not a mapping")
    cursor[parts[-1]] = yaml.safe_load(raw_value)


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

    Two nested sub-dataclasses (`weights:`, `latency:`) need their own
    unknown-key validation, so we don't go through the generic
    `_section()` helper.
    """
    sub = data.get("objective", {}) or {}
    weights_raw = sub.get("weights", {}) or {}
    latency_raw = sub.get("latency", {}) or {}
    top = {k: v for k, v in sub.items() if k not in ("weights", "latency")}
    unknown = set(top) - {f.name for f in fields(ObjectiveConfig)} - {"weights", "latency"}
    if unknown:
        raise ValueError(f"unknown ObjectiveConfig keys: {sorted(unknown)}")
    return ObjectiveConfig(
        **_extract(top, ObjectiveConfig),
        weights=ObjectiveWeights(**_extract(weights_raw, ObjectiveWeights)),
        latency=LatencyConfig(**_extract(latency_raw, LatencyConfig)),
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
