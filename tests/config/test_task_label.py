"""Tests for cli._derive_task_label: the run-dir's task segment.

Canonical layout: `tasks/<task>/configs/<stem>.yaml` -> `<task>_<stem>`.
Anything else falls back to the config file stem.
"""
from __future__ import annotations

from pathlib import Path

from ranking_evolved.cli import _derive_task_label


def test_canonical_layout_yields_task_underscore_stem(tmp_path: Path, record_io):
    cfg = tmp_path / "tasks" / "bm25" / "configs" / "freeform.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("task: {seed: s.py, evaluator: e.py}\n")

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="_derive_task_label",
        input={"config_path": "tasks/bm25/configs/freeform.yaml"},
        run=lambda: _derive_task_label(cfg),
    )
    assert out == "bm25_freeform"


def test_other_canonical_task_layout(tmp_path: Path, record_io):
    cfg = tmp_path / "tasks" / "ql" / "configs" / "JM.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("task: {seed: s.py, evaluator: e.py}\n")

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="_derive_task_label",
        input={"config_path": "tasks/ql/configs/JM.yaml"},
        run=lambda: _derive_task_label(cfg),
    )
    assert out == "ql_JM"


def test_non_canonical_layout_falls_back_to_stem(tmp_path: Path, record_io):
    cfg = tmp_path / "experiments" / "ad_hoc.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("task: {seed: s.py, evaluator: e.py}\n")

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="_derive_task_label (fallback)",
        input={"config_path": "experiments/ad_hoc.yaml"},
        run=lambda: _derive_task_label(cfg),
    )
    assert out == "ad_hoc"
