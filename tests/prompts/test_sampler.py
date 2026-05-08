"""Tests for prompt sampler context selection and deduplication."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ranking_evolved.core.types import Program
from ranking_evolved.prompts.sampler import PromptConfig, PromptSampler


def _program(id: str, *, iteration: int = 0, score: float = 0.1) -> Program:
    return Program(
        id=id,
        source_code=f"# source for {id}\nvalue = {iteration}\n",
        parent_id=None,
        generation=iteration,
        iteration_found=iteration,
        timestamp=time.time(),
        metrics={"combined_score": score},
        complexity=10.0,
        diversity=0.1,
        island=0,
        feature_coords={},
    )


def test_prompt_context_counts_and_deduplication():
    parent = _program("parent", iteration=10)
    recent = _program("recent", iteration=9)
    top = _program("top", iteration=8)
    inspiration_1 = _program("inspiration-1", iteration=7)
    inspiration_2 = _program("inspiration-2", iteration=6)

    sampler = PromptSampler(
        PromptConfig(
            num_recent_programs=1,
            num_top_programs=1,
            num_diverse_programs=2,
        )
    )
    prompt = sampler.build(
        iteration=30,
        parent=parent,
        # `top` appears here too; it should be skipped because the top section
        # already consumed it.
        inspirations=[top, inspiration_1, inspiration_2],
        # `parent` appears here too; it should be skipped because the current
        # program is already shown at the top of the prompt.
        top_programs=[parent, top],
        previous_programs=[parent, recent, top],
    )

    user = prompt.user
    assert user.count("## Recent programs in this island") == 1
    assert user.count("## Top programs in this island") == 1
    assert user.count("## Inspiration programs") == 1
    assert user.count("### Program recent") == 1
    assert user.count("### Program top") == 1
    assert user.count("### Program inspiration-1") == 1
    assert user.count("### Program inspiration-2") == 1
    assert "### Program parent" not in user
    assert user.count("### Program ") == 4


def test_prompt_omits_empty_context_sections_after_deduplication():
    parent = _program("parent", iteration=10)
    sampler = PromptSampler(
        PromptConfig(
            num_recent_programs=1,
            num_top_programs=1,
            num_diverse_programs=1,
        )
    )
    prompt = sampler.build(
        iteration=1,
        parent=parent,
        inspirations=[parent],
        top_programs=[parent],
        previous_programs=[parent],
    )

    assert "## Recent programs in this island" not in prompt.user
    assert "## Top programs in this island" not in prompt.user
    assert "## Inspiration programs" not in prompt.user


def test_context_programs_are_rendered_as_diff_against_parent():
    """Recent / top / inspiration programs must NOT include the parent's
    full source repeated; they appear as a unified diff vs the parent.
    This is what makes 5 diverse picks affordable."""
    parent = _program("parent", iteration=10)
    other = _program("other", iteration=9, score=0.42)

    sampler = PromptSampler(
        PromptConfig(num_recent_programs=1, num_top_programs=0, num_diverse_programs=0)
    )
    prompt = sampler.build(
        iteration=11,
        parent=parent,
        inspirations=[],
        top_programs=[],
        previous_programs=[other],
    )
    user = prompt.user
    # Parent appears once as a python codeblock at the top of the prompt.
    assert user.count("```python") == 1
    # Context program is rendered as a diff codeblock with --- parent / +++ this.
    assert "```diff" in user
    assert "--- parent" in user
    assert "+++ this" in user
    # The context program's full source is NOT pasted verbatim.
    assert user.count(other.source_code) == 0


def test_summary_metrics_drops_uninteresting_keys():
    """The compact metrics block keeps only configured keys + per-dataset
    suffixes; everything else (per-query token counts, build times, etc.)
    is dropped."""
    parent = Program(
        id="p",
        source_code="x = 1\n",
        parent_id=None,
        generation=0,
        iteration_found=0,
        timestamp=time.time(),
        metrics={
            "combined_score": 0.55,
            "objective_recall_component": 0.20,
            "objective_ndcg_component": 0.10,
            "objective_latency_component": 0.25,
            "beir_fiqa_recall_at_1000": 0.88,
            "beir_fiqa_ndcg_at_10": 0.41,
            "beir_fiqa_query_latency_median_ms": 115.6,
            "beir_fiqa_build_time_ms": 18000.0,
            "beir_fiqa_query_tokens_used": 32.0,
            "beir_fiqa_corpus_size": 57638.0,
        },
        complexity=10.0,
        diversity=0.1,
        island=0,
        feature_coords={},
    )
    sampler = PromptSampler(PromptConfig())
    prompt = sampler.build(
        iteration=1,
        parent=parent,
        inspirations=[],
        top_programs=[],
        previous_programs=[],
    )
    user = prompt.user
    assert "combined_score: 0.5500" in user
    assert "objective_recall_component: 0.2000" in user
    assert "beir_fiqa_recall_at_1000: 0.8800" in user
    assert "beir_fiqa_ndcg_at_10: 0.4100" in user
    assert "beir_fiqa_query_latency_median_ms: 115.6000" in user
    # Dropped: build_time, query_tokens_used, corpus_size.
    assert "beir_fiqa_build_time_ms" not in user
    assert "beir_fiqa_query_tokens_used" not in user
    assert "beir_fiqa_corpus_size" not in user


@dataclass
class _Failure:
    iteration: int
    parent_id: str | None
    parent_island: int
    parent_source_code: str
    child_source_code: str
    error_summary: str


def test_prompt_renders_recent_failures_section():
    """When `recent_failures` is non-empty, the prompt includes a 'do not
    repeat' section with each failure rendered as `(error, diff)`."""
    parent = _program("parent", iteration=10)
    failure = _Failure(
        iteration=8,
        parent_id="parent",
        parent_island=0,
        parent_source_code="x = 1\n",
        child_source_code="x = 1\nimport non_existent_module\n",
        error_summary="ModuleNotFoundError: No module named 'non_existent_module'",
    )
    sampler = PromptSampler(PromptConfig(num_failed_attempts=3))
    prompt = sampler.build(
        iteration=11,
        parent=parent,
        inspirations=[],
        top_programs=[],
        previous_programs=[],
        recent_failures=[failure],
    )
    user = prompt.user
    assert "## Recent failed attempts" in user
    assert "ModuleNotFoundError" in user
    # Failure body includes a diff against the failed parent.
    assert user.count("```diff") == 1


def test_prompt_failure_section_disabled_when_count_zero():
    parent = _program("parent", iteration=10)
    failure = _Failure(
        iteration=8,
        parent_id="parent",
        parent_island=0,
        parent_source_code="x = 1\n",
        child_source_code="x = 2\n",
        error_summary="some error",
    )
    sampler = PromptSampler(PromptConfig(num_failed_attempts=0))
    prompt = sampler.build(
        iteration=11,
        parent=parent,
        inspirations=[],
        top_programs=[],
        previous_programs=[],
        recent_failures=[failure],
    )
    assert "## Recent failed attempts" not in prompt.user
