"""Prompt assembly.

Mirrors the shape of OpenEvolve's `PromptSampler.build_prompt` so a recorded
transcript and ours produce comparable prompt text in the replay dashboard.
The default templates are hardcoded f-strings — there is no jinja layer in
v1 (per plan: dropped speculative scaffolding).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.types import Program, Prompt


DEFAULT_SYSTEM = (
    "You are an expert software engineer evolving a candidate program. "
    "You will be shown the current program, its evaluator metrics, and a "
    "selection of past programs as context. Propose changes using the "
    "SEARCH/REPLACE diff format below."
)

DIFF_INSTRUCTIONS = (
    "Output your changes as one or more SEARCH/REPLACE blocks:\n"
    "<<<<<<< SEARCH\n"
    "<exact lines from the current program>\n"
    "=======\n"
    "<replacement lines>\n"
    ">>>>>>> REPLACE\n"
    "The SEARCH text must match the current program exactly."
)


@dataclass
class PromptConfig:
    diff_based: bool = True
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    include_artifacts: bool = True
    system_message: str = DEFAULT_SYSTEM


class PromptSampler:
    def __init__(self, config: PromptConfig):
        self.config = config

    def build(
        self,
        *,
        iteration: int,
        parent: Program,
        inspirations: list[Program],
        top_programs: list[Program],
        previous_programs: list[Program],
        parent_artifacts: dict[str, Any] | None = None,
    ) -> Prompt:
        sections: list[str] = []
        sections.append(f"# Iteration {iteration}")
        sections.append("\n## Current program\n")
        sections.append(_codeblock(parent.source_code))
        sections.append("\n## Current program metrics\n")
        sections.append(_format_metrics(parent.metrics))

        if previous_programs:
            sections.append("\n## Recent programs in this island\n")
            for p in previous_programs:
                sections.append(_program_summary(p))

        if top_programs:
            sections.append("\n## Top programs in this island\n")
            for p in top_programs[: self.config.num_top_programs]:
                sections.append(_program_summary(p))

        if inspirations:
            sections.append("\n## Inspiration programs\n")
            for p in inspirations[: self.config.num_diverse_programs]:
                sections.append(_program_summary(p))

        if self.config.include_artifacts and parent_artifacts:
            sections.append("\n## Artifacts from the parent's evaluation\n")
            for k, v in parent_artifacts.items():
                sections.append(f"- {k}: {_short(v)}")

        if self.config.diff_based:
            sections.append("\n## Instructions\n")
            sections.append(DIFF_INSTRUCTIONS)
            template_key = "diff_user"
        else:
            sections.append("\n## Instructions\n")
            sections.append(
                "Rewrite the program in full. Output a single fenced ```python block."
            )
            template_key = "full_rewrite_user"

        return Prompt(
            system=self.config.system_message,
            user="\n".join(sections),
            template_key=template_key,
            iteration=iteration,
            parent_id=parent.id,
            inspiration_ids=tuple(p.id for p in inspirations),
        )


def _codeblock(code: str) -> str:
    return f"```python\n{code}\n```"


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "(no metrics)"
    return "\n".join(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in metrics.items())


def _program_summary(p: Program) -> str:
    head = f"### Program {p.id} (gen={p.generation}, island={p.island})"
    metrics = _format_metrics(p.metrics)
    code = _codeblock(p.source_code)
    return f"{head}\n{metrics}\n{code}"


def _short(v: Any, *, maxlen: int = 200) -> str:
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen] + f"... [{len(s)} chars]"
