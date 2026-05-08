"""Load a static external baseline JSON for the latency-aware objective.

When `objective.latency.baseline_source == "external"` the controller skips
the seed-eval baseline capture and reads from a file produced by an external
baseline run (e.g. FastPLAID via `tasks/late_interaction/compare_baselines.py`).

Schema of the baseline file:

    {
      "_fingerprint": {
        "device": "cpu" | "cuda",
        "device_name": "...",
        "...": "..."
      },
      "<dataset_name>": {
        "median_query_latency_ms": float,
        "recall_at_1000": float,         # optional
        "ndcg_at_10": float,             # optional
        "build_time_ms": float           # optional
      },
      ...
    }

The loader returns a `dict[str, float]` mapping dataset name to baseline
median latency in ms. Optional fields are ignored by the controller; they're
present in the file for downstream consumers (e.g. the recall-floor wrapper
in the late-interaction evaluator).

Hard-error behavior:
  - Missing file or unparsable JSON.
  - Missing `_fingerprint.device` field.
  - `_fingerprint.device` does not match the active runtime device. This is
    the load-bearing fairness check — a baseline measured on CPU must not be
    silently compared to a GPU run.
  - No dataset entries (file would produce an empty baseline map).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_FINGERPRINT_KEY = "_fingerprint"


class ExternalBaselineError(RuntimeError):
    """Raised on any malformed or device-mismatched external baseline."""


def resolve_baseline_path(template: str, *, device: str) -> Path:
    """Substitute ${EVAL_DEVICE} in `template` with the active device.

    Allows a single YAML config to target either host:
        baseline_path: tasks/late_interaction/baselines/fastplaid_baseline.${EVAL_DEVICE}.json
    """
    if not template:
        raise ExternalBaselineError(
            "baseline_source=external requires objective.latency.baseline_path."
        )
    if device not in ("cpu", "cuda"):
        raise ExternalBaselineError(
            f"device must be 'cpu' or 'cuda', got {device!r}"
        )
    # Support $EVAL_DEVICE and ${EVAL_DEVICE}.
    expanded = template.replace("${EVAL_DEVICE}", device).replace("$EVAL_DEVICE", device)
    return Path(expanded)


def load_external_baseline(
    path: Path,
    *,
    runtime_device: str,
) -> dict[str, float]:
    """Read `path`, assert fingerprint matches `runtime_device`, return latency map.

    Returns `{dataset_name: median_query_latency_ms}` ready to feed into
    `compute_objective(...)` as the `baseline_latency_by_dataset` argument.
    """
    if not path.exists():
        raise ExternalBaselineError(
            f"external baseline not found: {path}. "
            f"Run `tasks/late_interaction/compare_baselines.py` on this host first."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ExternalBaselineError(
            f"external baseline {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ExternalBaselineError(
            f"external baseline {path} must be a JSON object, got {type(data).__name__}"
        )

    fingerprint = data.get(_FINGERPRINT_KEY)
    if not isinstance(fingerprint, dict):
        raise ExternalBaselineError(
            f"external baseline {path} missing required '_fingerprint' object."
        )
    loaded_device = str(fingerprint.get("device", "")).lower()
    if loaded_device not in ("cpu", "cuda"):
        raise ExternalBaselineError(
            f"external baseline {path}: '_fingerprint.device' must be "
            f"'cpu' or 'cuda', got {fingerprint.get('device')!r}"
        )
    if loaded_device != runtime_device:
        raise ExternalBaselineError(
            f"external baseline {path}: device mismatch — recorded on "
            f"{loaded_device!r} ({fingerprint.get('device_name')}), "
            f"this run is on {runtime_device!r}. "
            "Re-run compare_baselines on this host first, "
            "or switch EVAL_DEVICE to match."
        )

    baseline: dict[str, float] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        latency_ms = value.get("median_query_latency_ms")
        if latency_ms is None:
            raise ExternalBaselineError(
                f"external baseline {path}: dataset {key!r} missing "
                "'median_query_latency_ms'."
            )
        baseline[str(key)] = float(latency_ms)

    if not baseline:
        raise ExternalBaselineError(
            f"external baseline {path} contains no dataset entries."
        )
    return baseline


def load_external_baseline_recall_at_k(
    path: Path,
    *,
    k: int,
    runtime_device: str,
) -> dict[str, float]:
    """Return `{dataset_name: baseline_recall_at_k}` from the same baseline file.

    Used by the late-interaction evaluator's recall-floor wrapper. Same
    fingerprint check as `load_external_baseline`. Datasets without
    `recall_at_<k>` are silently omitted from the returned map (the floor is
    only applied to datasets where a baseline recall exists).
    """
    if not path.exists():
        raise ExternalBaselineError(f"external baseline not found: {path}")
    data = json.loads(path.read_text())
    fingerprint = data.get(_FINGERPRINT_KEY) or {}
    loaded_device = str(fingerprint.get("device", "")).lower()
    if loaded_device != runtime_device:
        raise ExternalBaselineError(
            f"external baseline {path}: device mismatch (loaded={loaded_device!r}, "
            f"runtime={runtime_device!r})"
        )
    field = f"recall_at_{k}"
    out: dict[str, float] = {}
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        if field in value:
            out[str(key)] = float(value[field])
    return out


def detect_runtime_device() -> str:
    """Best-effort device resolution for callers outside `tasks.late_interaction`.

    Mirrors the resolution rule in `tasks.late_interaction._runtime` without
    importing it (so this module stays framework-side and dependency-light).
    """
    raw = os.environ.get("EVAL_DEVICE", "").strip().lower()
    if raw in ("cpu", "cuda"):
        return raw
    if raw:
        raise ExternalBaselineError(
            f"EVAL_DEVICE must be 'cpu' or 'cuda', got {raw!r}"
        )
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def expand_path(template: str | Any) -> str:
    """Pass-through `${EVAL_DEVICE}` interpolation for non-Path consumers."""
    template = str(template)
    return resolve_baseline_path(template, device=detect_runtime_device()).as_posix()
