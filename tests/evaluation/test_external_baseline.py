"""External baseline loader: fingerprint enforcement + path interpolation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ranking_evolved.evaluation.external_baseline import (
    ExternalBaselineError,
    detect_runtime_device,
    load_external_baseline,
    load_external_baseline_recall_at_k,
    resolve_baseline_path,
)


def _write_baseline(
    path: Path,
    *,
    device: str,
    datasets: dict[str, dict[str, float]],
    device_name: str = "Test Device",
) -> None:
    payload: dict = {
        "_fingerprint": {"device": device, "device_name": device_name},
    }
    payload.update(datasets)
    path.write_text(json.dumps(payload))


def test_resolve_baseline_path_interpolates_eval_device():
    p = resolve_baseline_path(
        "tasks/late_interaction/baselines/fastplaid_baseline.${EVAL_DEVICE}.json",
        device="cuda",
    )
    assert p.as_posix().endswith("fastplaid_baseline.cuda.json")

    p_dollar = resolve_baseline_path(
        "fastplaid_baseline.$EVAL_DEVICE.json", device="cpu",
    )
    assert p_dollar.as_posix() == "fastplaid_baseline.cpu.json"


def test_resolve_baseline_path_rejects_empty():
    with pytest.raises(ExternalBaselineError, match="requires"):
        resolve_baseline_path("", device="cpu")


def test_resolve_baseline_path_rejects_unknown_device():
    with pytest.raises(ExternalBaselineError, match="cpu' or 'cuda'"):
        resolve_baseline_path("x.json", device="tpu")


def test_load_external_baseline_returns_per_dataset_latency(tmp_path):
    p = tmp_path / "fp.cpu.json"
    _write_baseline(
        p,
        device="cpu",
        datasets={
            "beir_scifact": {"median_query_latency_ms": 12.5, "recall_at_1000": 0.95},
            "beir_arguana": {"median_query_latency_ms": 30.0, "recall_at_1000": 0.80},
        },
    )
    out = load_external_baseline(p, runtime_device="cpu")
    assert out == {"beir_scifact": 12.5, "beir_arguana": 30.0}


def test_load_external_baseline_rejects_device_mismatch(tmp_path):
    """Invariant 14: a CPU baseline must not be silently used in a GPU run."""
    p = tmp_path / "fp.cpu.json"
    _write_baseline(p, device="cpu", datasets={"beir_scifact": {"median_query_latency_ms": 1.0}})
    with pytest.raises(ExternalBaselineError, match="device mismatch"):
        load_external_baseline(p, runtime_device="cuda")


def test_load_external_baseline_requires_fingerprint(tmp_path):
    p = tmp_path / "fp.json"
    p.write_text(json.dumps({"beir_scifact": {"median_query_latency_ms": 1.0}}))
    with pytest.raises(ExternalBaselineError, match="missing required '_fingerprint'"):
        load_external_baseline(p, runtime_device="cpu")


def test_load_external_baseline_rejects_invalid_fingerprint_device(tmp_path):
    p = tmp_path / "fp.json"
    p.write_text(json.dumps({"_fingerprint": {"device": "tpu"}, "beir_scifact": {"median_query_latency_ms": 1.0}}))
    with pytest.raises(ExternalBaselineError, match="must be 'cpu' or 'cuda'"):
        load_external_baseline(p, runtime_device="cpu")


def test_load_external_baseline_requires_dataset_latency(tmp_path):
    p = tmp_path / "fp.json"
    _write_baseline(p, device="cpu", datasets={"beir_scifact": {"recall_at_1000": 0.9}})
    with pytest.raises(ExternalBaselineError, match="missing 'median_query_latency_ms'"):
        load_external_baseline(p, runtime_device="cpu")


def test_load_external_baseline_rejects_empty(tmp_path):
    p = tmp_path / "fp.json"
    _write_baseline(p, device="cpu", datasets={})
    with pytest.raises(ExternalBaselineError, match="no dataset entries"):
        load_external_baseline(p, runtime_device="cpu")


def test_load_external_baseline_recall_at_k(tmp_path):
    p = tmp_path / "fp.json"
    _write_baseline(
        p,
        device="cpu",
        datasets={
            "beir_scifact": {"median_query_latency_ms": 1.0, "recall_at_1000": 0.93},
            "beir_arguana": {"median_query_latency_ms": 1.0},  # no recall_at_1000
        },
    )
    out = load_external_baseline_recall_at_k(p, k=1000, runtime_device="cpu")
    # Datasets without recall_at_1000 are silently omitted.
    assert out == {"beir_scifact": 0.93}


def test_load_external_baseline_recall_rejects_device_mismatch(tmp_path):
    p = tmp_path / "fp.json"
    _write_baseline(p, device="cpu", datasets={"beir_scifact": {"median_query_latency_ms": 1.0, "recall_at_1000": 0.5}})
    with pytest.raises(ExternalBaselineError, match="device mismatch"):
        load_external_baseline_recall_at_k(p, k=1000, runtime_device="cuda")


def test_load_external_baseline_missing_file(tmp_path):
    with pytest.raises(ExternalBaselineError, match="not found"):
        load_external_baseline(tmp_path / "does-not-exist.json", runtime_device="cpu")


def test_detect_runtime_device_respects_env(monkeypatch):
    monkeypatch.setenv("EVAL_DEVICE", "cpu")
    assert detect_runtime_device() == "cpu"
    monkeypatch.setenv("EVAL_DEVICE", "cuda")
    assert detect_runtime_device() == "cuda"


def test_detect_runtime_device_rejects_unknown(monkeypatch):
    monkeypatch.setenv("EVAL_DEVICE", "tpu")
    with pytest.raises(ExternalBaselineError):
        detect_runtime_device()
