"""Tests for the optimization-objective config block.

The evolution-algorithm invariants are tested separately in
tests/search/test_evolution_algo_invariants.py and do not depend on this
block.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ranking_evolved.config.loader import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
LATENCY_AWARE_YAML = REPO_ROOT / "tasks/bm25/configs/freeform_latency_aware.yaml"
LEGACY_YAML = REPO_ROOT / "tasks/bm25/configs/freeform.yaml"


def test_latency_aware_config_loads():
    cfg = load_config(LATENCY_AWARE_YAML)

    obj = cfg.objective
    assert obj.name == "recall1000_ndcg10_latency"
    assert obj.recall_k == 1000
    assert obj.ndcg_k == 10
    assert obj.weights.recall == pytest.approx(0.45)
    assert obj.weights.ndcg == pytest.approx(0.20)
    assert obj.weights.latency == pytest.approx(0.35)
    assert obj.latency.enabled is True
    assert obj.latency.warmup_queries == 20
    assert obj.latency.hard_slowdown_threshold == pytest.approx(5.0)
    assert obj.latency.penalty_mode == "zero_latency_score"
    assert obj.latency.ratio_transform == "inverse_one_plus_ratio"


def test_legacy_freeform_defaults_to_legacy_objective():
    """The unmodified freeform.yaml has no `objective:` section → defaults."""
    cfg = load_config(LEGACY_YAML)

    obj = cfg.objective
    assert obj.name == "recall100_ndcg10"
    assert obj.recall_k == 100
    assert obj.ndcg_k == 10
    assert obj.weights.recall == pytest.approx(0.8)
    assert obj.weights.ndcg == pytest.approx(0.2)
    assert obj.weights.latency == pytest.approx(0.0)
    assert obj.latency.enabled is False


def test_unknown_objective_key_raises(tmp_path: Path):
    yaml_text = """\
task:
  seed: seed.py
  evaluator: evaluator.py
search:
  algorithm: map_elites_islands
objective:
  name: custom
  bogus_field: 1
"""
    p = tmp_path / "task.yaml"
    p.write_text(yaml_text)

    with pytest.raises(ValueError) as exc:
        load_config(p)

    assert "ObjectiveConfig" in str(exc.value)
    assert "bogus_field" in str(exc.value)


def test_unknown_latency_key_raises(tmp_path: Path):
    yaml_text = """\
task:
  seed: seed.py
  evaluator: evaluator.py
search:
  algorithm: map_elites_islands
objective:
  latency:
    enabled: true
    not_a_real_field: 1
"""
    p = tmp_path / "task.yaml"
    p.write_text(yaml_text)

    with pytest.raises(ValueError) as exc:
        load_config(p)

    assert "LatencyConfig" in str(exc.value)
    assert "not_a_real_field" in str(exc.value)
