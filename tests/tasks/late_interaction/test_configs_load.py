"""The freeform_latency_aware.yaml must parse through the framework loader.

This catches typos in field names and rejected unknown keys (the loader runs
in strict mode — see `src/ranking_evolved/config/loader.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ranking_evolved.config.loader import load_config


REPO_ROOT = Path(__file__).resolve().parents[3]
LATE_INTERACTION_CONFIG = (
    REPO_ROOT / "tasks" / "late_interaction" / "configs" / "freeform_latency_aware.yaml"
)
NEW_BENCHMARK_CONFIG = (
    REPO_ROOT / "tasks" / "late_interaction" / "configs" / "new_benchmarks_latency_aware.yaml"
)


@pytest.mark.skipif(not LATE_INTERACTION_CONFIG.exists(), reason="config not yet written")
def test_freeform_latency_aware_yaml_parses(monkeypatch):
    # OPENAI_API_KEY interpolation — provide a dummy so the loader doesn't see
    # a leftover "${OPENAI_API_KEY}" substring that confuses downstream code.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-yaml-parse")
    config = load_config(LATE_INTERACTION_CONFIG)

    # Task wiring
    assert config.task.seed.endswith("seeds/freeform.py")
    assert config.task.evaluator.endswith("late_interaction/evaluator.py")

    # Evolution config — capture_replay_every is the new field
    assert config.evolution.max_iterations == 100
    assert config.evolution.capture_replay is True
    assert config.evolution.capture_replay_every == 1

    # Search ratios: 30% exploration, 60% exploitation, 10% random remainder
    assert config.search.exploration_ratio == pytest.approx(0.30)
    assert config.search.exploitation_ratio == pytest.approx(0.60)

    # Evaluation env passthrough — the live YAML's dataset list spans
    # ~8x corpus-size range. We assert presence of at least one entry from
    # each shape rather than the exact list (which gets retuned often).
    env = config.evaluation.env
    assert env.get("EVAL_DEVICE") == "cpu"
    eval_datasets = env.get("EVAL_DATASETS", "")
    assert "beir_fiqa" in eval_datasets
    assert "bright_stackoverflow" in eval_datasets
    assert "bright_theoremqa_questions" in eval_datasets
    assert str(env.get("EVAL_SAMPLE_QUERIES")) == "1000000000"
    assert str(env.get("EVAL_WARMUP_QUERIES")) == "10"
    assert str(env.get("EVAL_TIMED_REPEATS")) == "1"
    assert str(env.get("EVAL_PROGRESS")) == "1"

    # Proposer/prompt context
    assert config.proposer.models[0]["name"] == "gpt-5.2"
    assert config.proposer.reasoning_effort == "medium"
    assert config.prompt.num_recent_programs == 1
    assert config.prompt.num_top_programs == 1
    # 5 diverse picks: now affordable because context programs are
    # rendered as compact metrics + unified diff vs parent (no full source).
    assert config.prompt.num_diverse_programs == 5
    assert config.prompt.num_failed_attempts == 3

    # Search: archive_size = num_islands * 4, AST-based complexity,
    # num_inspirations sized to fill (recent + top + diverse + headroom).
    assert config.search.archive_size == 12
    assert config.search.num_islands == 3
    assert config.search.num_inspirations == 8
    assert config.search.complexity_metric == "ast_nodes"

    # Objective: 0.40 recall / 0.30 ndcg / 0.30 latency (the live YAML).
    assert config.objective.weights.recall == pytest.approx(0.40)
    assert config.objective.weights.ndcg == pytest.approx(0.30)
    assert config.objective.weights.latency == pytest.approx(0.30)
    assert config.objective.latency.enabled is True
    assert config.objective.latency.timed_repeats == 1
    assert config.objective.latency.baseline_source == "external"
    # ${EVAL_DEVICE} should have been interpolated using the value seeded from
    # evaluation.env (cpu) — the loader's pre-pass enables this.
    assert "exact_maxsim.cpu.json" in config.objective.latency.baseline_path


def test_new_benchmarks_latency_aware_yaml_parses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-yaml-parse")
    config = load_config(NEW_BENCHMARK_CONFIG)

    env = config.evaluation.env
    assert "bright_pro_economics" in env["EVAL_DATASETS"]
    assert "obliq_congress" in env["EVAL_DATASETS"]
    assert str(env["EVAL_RECALL_K"]) == "25"
    assert str(env["EVAL_NDCG_K"]) == "25"
    assert env["EVAL_QRELS_MODE"] == "gold"
    assert config.objective.recall_metric_key == "aspect_recall_at_25"
    assert config.objective.ndcg_metric_key == "alpha_ndcg_at_25"
    assert config.objective.metric_key_fallback is True
    assert "new_benchmarks_curated.gold.cpu.json" in config.objective.latency.baseline_path


def test_unknown_evaluation_env_value_does_not_break_loader(tmp_path, monkeypatch):
    """The env block accepts arbitrary string keys; numbers should be coerced."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(
        """
task:
  seed: tasks/late_interaction/seeds/freeform.py
  evaluator: tasks/late_interaction/evaluator.py
evaluation:
  env:
    EVAL_FOO: "abc"
    EVAL_BAR: 42
"""
    )
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    config = load_config(cfg)
    assert config.evaluation.env["EVAL_FOO"] == "abc"
    # YAML parses 42 as int; the merge in cli.py converts to str at use time
    # but the dataclass keeps the raw value.
    assert config.evaluation.env["EVAL_BAR"] in (42, "42")


def test_set_override_supports_list_index(tmp_path, monkeypatch):
    cfg = tmp_path / "test.yaml"
    cfg.write_text(
        """
task:
  seed: tasks/late_interaction/seeds/freeform.py
  evaluator: tasks/late_interaction/evaluator.py
proposer:
  models:
    - {name: "gpt-5.2", weight: 1.0}
"""
    )
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    config = load_config(cfg, overrides=["proposer.models.0.name=gpt-5.4"])
    assert config.proposer.models[0]["name"] == "gpt-5.4"
