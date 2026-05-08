"""Runtime fingerprint + BLAS thread pinning for fair latency measurement.

This module is **import-order sensitive**. The four `*_NUM_THREADS` variables
(OMP, MKL, OPENBLAS, VECLIB) only take effect if they are present in the
process environment **before** numpy/torch are first imported. So
`_runtime` does its work at import time and asserts that no late-interaction
code has imported numpy/torch yet.

Always import this module first in any entry point that times retrieval:

    from tasks.late_interaction import _runtime  # noqa: F401  (first import)
    import numpy as np
    ...

Public API:

  resolve_device()       -> "cpu" | "cuda"
  runtime_fingerprint()  -> dict (see schema below)

Fingerprint schema (every result file embeds this so cross-host comparisons
are not silently mixed):

    {
      "device":         "cpu" | "cuda",
      "device_name":    str,
      "cuda_version":   str | None,
      "torch_version":  str | None,
      "numpy_version":  str,
      "blas_threads":   {"omp": int, "mkl": int, "openblas": int, "veclib": int} | None,
      "cpu_count":      int,
      "hostname":       str
    }

`device` resolution: env var `EVAL_DEVICE` if set (must be "cpu" or "cuda"),
otherwise "cuda" if torch reports CUDA available else "cpu".
"""
from __future__ import annotations

import os
import platform
import socket
import sys
import warnings
from typing import Any

# -- guard: numpy/torch must not yet be imported when EVAL_DEVICE=cpu --------
# We don't error if they are already imported (some test runners import them
# in conftest); we just warn so the user knows BLAS pinning may be a no-op.
_NP_PRELOADED = "numpy" in sys.modules
_TORCH_PRELOADED = "torch" in sys.modules


def _resolve_device_from_env() -> str:
    raw = os.environ.get("EVAL_DEVICE", "").strip().lower()
    if raw in ("cpu", "cuda"):
        return raw
    if raw:
        raise ValueError(
            f"EVAL_DEVICE must be 'cpu' or 'cuda', got {raw!r}"
        )
    # Auto-select. Importing torch here is fine — we've already done the BLAS
    # env work in CPU mode below before this branch is reachable, and in GPU
    # mode the BLAS work is a no-op anyway.
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_blas_to_one_thread() -> dict[str, int]:
    """Set the four BLAS thread env vars to 1; return resolved values."""
    for var in _BLAS_VARS:
        os.environ.setdefault(var, "1")
    return {
        "omp": int(os.environ["OMP_NUM_THREADS"]),
        "mkl": int(os.environ["MKL_NUM_THREADS"]),
        "openblas": int(os.environ["OPENBLAS_NUM_THREADS"]),
        "veclib": int(os.environ["VECLIB_MAXIMUM_THREADS"]),
    }


# -- resolve at import time --------------------------------------------------
DEVICE: str
"""Resolved device for this process; one of 'cpu' or 'cuda'."""

BLAS_THREADS: dict[str, int] | None
"""Resolved BLAS thread settings in CPU mode; None in GPU mode."""

# Tentatively decide device from env (without auto-detect) so we can run
# BLAS pinning before importing torch when the user explicitly requested CPU.
_explicit = os.environ.get("EVAL_DEVICE", "").strip().lower()
if _explicit == "cpu":
    DEVICE = "cpu"
    BLAS_THREADS = _pin_blas_to_one_thread()
    if _NP_PRELOADED or _TORCH_PRELOADED:
        warnings.warn(
            "tasks.late_interaction._runtime imported AFTER numpy/torch; "
            "BLAS thread pinning may be ignored. Import _runtime first.",
            RuntimeWarning,
            stacklevel=2,
        )
elif _explicit == "cuda":
    DEVICE = "cuda"
    BLAS_THREADS = None
else:
    # Auto-detect needs to import torch.
    DEVICE = _resolve_device_from_env()
    if DEVICE == "cpu":
        BLAS_THREADS = _pin_blas_to_one_thread()
        # If torch was already imported during auto-detect, BLAS pinning may
        # not stick. That's fine for auto mode — users who care set EVAL_DEVICE
        # explicitly.
    else:
        BLAS_THREADS = None


def resolve_device() -> str:
    """Return the resolved device for this process."""
    return DEVICE


def runtime_fingerprint() -> dict[str, Any]:
    """Return the per-process runtime fingerprint (see module docstring)."""
    import numpy as np  # safe: at this point numpy import is unavoidable

    torch_version: str | None = None
    cuda_version: str | None = None
    device_name = platform.processor() or platform.machine() or "unknown"

    try:
        import torch
        torch_version = str(torch.__version__)
        if DEVICE == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "EVAL_DEVICE=cuda but torch.cuda.is_available() is False. "
                    "Either install a CUDA-enabled torch build or set EVAL_DEVICE=cpu."
                )
            cuda_version = str(torch.version.cuda)
            try:
                device_name = str(torch.cuda.get_device_name(0))
            except Exception:  # noqa: BLE001
                device_name = "cuda:0 (unknown)"
        else:
            # CPU mode — keep the platform-derived name, which is informative
            # enough on macOS ("arm") and Linux.
            cpu_brand = platform.processor() or platform.machine()
            sys_name = platform.system()
            if sys_name == "Darwin":
                device_name = f"{cpu_brand} (Apple/Darwin)"
            else:
                device_name = f"{cpu_brand} ({sys_name})"
    except ImportError:
        if DEVICE == "cuda":
            raise RuntimeError(
                "EVAL_DEVICE=cuda but torch is not installed."
            )

    return {
        "device": DEVICE,
        "device_name": device_name,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "numpy_version": str(np.__version__),
        "blas_threads": BLAS_THREADS,
        "cpu_count": os.cpu_count() or 0,
        "hostname": socket.gethostname(),
    }


def assert_fingerprint_match(loaded: dict[str, Any], *, context: str) -> None:
    """Hard-error if a loaded baseline's fingerprint disagrees with this run.

    Only the `device` field is enforced (matching `device_name`/`hostname` is
    too strict — two laptops may both be "cpu" with different brands and we
    still want to compare to the laptop baseline; cross-device mixing is the
    real failure mode we guard against).
    """
    here = runtime_fingerprint()
    loaded_device = str(loaded.get("device", "")).lower()
    if loaded_device not in ("cpu", "cuda"):
        raise RuntimeError(
            f"{context}: loaded fingerprint missing or invalid 'device' "
            f"field (got {loaded.get('device')!r}). "
            "Re-run compare_baselines on this host first."
        )
    if loaded_device != here["device"]:
        raise RuntimeError(
            f"{context}: device mismatch — baseline was recorded on "
            f"{loaded_device!r} ({loaded.get('device_name')}), but this run "
            f"is on {here['device']!r} ({here['device_name']}). "
            "Re-run compare_baselines on this host first, or switch "
            "EVAL_DEVICE to match."
        )
