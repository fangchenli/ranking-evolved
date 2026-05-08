"""Prompt assembly.

Mirrors the shape of OpenEvolve's `PromptSampler.build_prompt` so a recorded
transcript and ours produce comparable prompt text in the replay dashboard.
The default templates are hardcoded f-strings — there is no jinja layer in
v1 (per plan: dropped speculative scaffolding).

Context programs (recent / top / inspiration) are rendered as
`(compact_metrics_summary, unified_diff_against_parent)` rather than as
full source code. Reasons:

  - The parent is already in the prompt verbatim, so showing the full
    source of every other program is mostly redundant token spend on the
    shared interface and boilerplate (class signature, comment headers).
  - SEARCH/REPLACE diffs only need the parent shown verbatim; everyone
    else just needs to communicate "what design choice differs and how
    did it score." A unified diff is exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import unified_diff
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


# Top-level keys we *always* try to render in a context program's metrics
# block. Per-dataset metrics are picked separately by suffix.
_DEFAULT_SUMMARY_METRIC_KEYS = (
    "combined_score",
    "objective_recall_component",
    "objective_ndcg_component",
    "objective_latency_component",
)

# Per-dataset suffixes worth keeping in the summary. The defaults are
# tuned for the latency-aware late-interaction objective (recall@1000,
# nDCG@10, and median query latency); other tasks can override via
# PromptConfig.
_DEFAULT_SUMMARY_PER_DATASET_SUFFIXES = (
    "_recall_at_1000",
    "_ndcg_at_10",
    "_query_latency_median_ms",
)


@dataclass
class PromptConfig:
    diff_based: bool = True
    num_recent_programs: int = 3
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    include_artifacts: bool = True
    system_message: str = DEFAULT_SYSTEM
    # Compact metric block for context programs (and the parent). Only
    # these top-level keys plus per-dataset metrics matching the suffix
    # list below are rendered. Set both to empty to fall back to the
    # full metrics dict (legacy behavior).
    summary_metric_keys: list[str] = field(
        default_factory=lambda: list(_DEFAULT_SUMMARY_METRIC_KEYS)
    )
    summary_per_dataset_suffixes: list[str] = field(
        default_factory=lambda: list(_DEFAULT_SUMMARY_PER_DATASET_SUFFIXES)
    )
    # Maximum unified-diff lines per context program. Long diffs get
    # head/tail-truncated so a single rewrite-everything child doesn't
    # blow the prompt budget.
    max_diff_lines: int = 120
    # How many recent failed (score=0 / crashed) attempts to surface to
    # the LLM under "Recent failed attempts". 0 disables the section.
    num_failed_attempts: int = 3


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
        retry_feedback: list[dict[str, Any]] | None = None,
        recent_failures: list[Any] | None = None,
    ) -> Prompt:
        """Assemble the user prompt.

        Layout:
          1. Current program (parent) — full source + compact metrics.
          2. Recent / Top / Inspiration programs — compact metrics + a
             unified diff against the parent. Identical-to-parent context
             programs are rendered with a "(no code differences)" note.
          3. Recent failed attempts (if `recent_failures` provided).
          4. In-iteration retry feedback (when previous attempts on THIS
             iteration failed).
          5. Diff/rewrite instructions.

        `recent_failures` accepts items with attributes
        `error_summary`, `parent_source_code`, `child_source_code`,
        `iteration`, `parent_id` (a `FailureRecord` or any duck-typed shim).
        """
        sections: list[str] = []
        sections.append(f"# Iteration {iteration}")
        sections.append("\n## Current program\n")
        sections.append(_codeblock(parent.source_code))
        sections.append("\n## Current program metrics\n")
        sections.append(self._summary_metrics(parent.metrics))

        seen_program_ids = {parent.id}

        recent = _dedupe_programs(
            previous_programs,
            limit=self.config.num_recent_programs,
            seen_ids=seen_program_ids,
        )
        if recent:
            sections.append("\n## Recent programs in this island\n")
            sections.append(
                "Each program below is summarized as a compact metrics block "
                "plus a unified diff against the current program (parent)."
            )
            for p in recent:
                sections.append(self._program_diff_summary(p, parent))

        top = _dedupe_programs(
            top_programs,
            limit=self.config.num_top_programs,
            seen_ids=seen_program_ids,
        )
        if top:
            sections.append("\n## Top programs in this island\n")
            for p in top:
                sections.append(self._program_diff_summary(p, parent))

        inspiration_context = _dedupe_programs(
            inspirations,
            limit=self.config.num_diverse_programs,
            seen_ids=seen_program_ids,
        )
        if inspiration_context:
            sections.append("\n## Inspiration programs\n")
            for p in inspiration_context:
                sections.append(self._program_diff_summary(p, parent))

        if self.config.include_artifacts and parent_artifacts:
            sections.append("\n## Artifacts from the parent's evaluation\n")
            for k, v in parent_artifacts.items():
                sections.append(f"- {k}: {_short(v)}")

        failure_section = self._format_failure_section(recent_failures, parent_id=parent.id)
        if failure_section:
            sections.append(failure_section)

        if retry_feedback:
            sections.append("\n## Previous attempts on this iteration FAILED\n")
            sections.append(
                "Earlier attempts at this same iteration failed for the reasons below. "
                "Do NOT repeat the same mistakes — propose a meaningfully different "
                "edit that avoids them.\n"
            )
            for i, fb in enumerate(retry_feedback, start=1):
                kind = str(fb.get("kind") or "unknown").upper()
                detail = _short(fb.get("detail") or "")
                sections.append(f"- Attempt {i} ({kind}): {detail}")

        if self.config.diff_based:
            sections.append("\n## Instructions\n")
            sections.append(DIFF_INSTRUCTIONS)
            template_key = "diff_user"
        else:
            sections.append("\n## Instructions\n")
            sections.append("Rewrite the program in full. Output a single fenced ```python block.")
            template_key = "full_rewrite_user"

        return Prompt(
            system=self.config.system_message,
            user="\n".join(sections),
            template_key=template_key,
            iteration=iteration,
            parent_id=parent.id,
            inspiration_ids=tuple(p.id for p in inspirations),
        )

    # ------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------

    def _summary_metrics(self, metrics: dict[str, float]) -> str:
        """Compact metric block: top-level objective keys + per-dataset
        metrics whose key ends with one of the configured suffixes. Falls
        back to the full dict when both summary lists are empty."""
        if not metrics:
            return "(no metrics)"
        keys = list(self.config.summary_metric_keys or ())
        suffixes = tuple(self.config.summary_per_dataset_suffixes or ())
        if not keys and not suffixes:
            return _format_metrics(metrics)
        lines: list[str] = []
        for k in keys:
            if k in metrics:
                lines.append(_format_metric_line(k, metrics[k]))
        if suffixes:
            per_ds = sorted(
                k for k in metrics if k.endswith(suffixes) and not k.startswith(("avg_", "total_"))
            )
            for k in per_ds:
                lines.append(_format_metric_line(k, metrics[k]))
        if not lines:
            # No configured key matched — fall back so we never silently
            # drop the entire metrics block.
            return _format_metrics(metrics)
        return "\n".join(lines)

    def _program_diff_summary(self, ctx: Program, parent: Program) -> str:
        """One context program rendered as `(metrics, unified_diff_vs_parent)`.

        The interface (class signatures, imports, public surface) is shared
        across all programs; the unified diff naturally omits unchanged
        lines, so the LLM sees only the design delta plus its score.
        """
        head = f"### Program {ctx.id} (gen={ctx.generation}, island={ctx.island})"
        metrics = self._summary_metrics(ctx.metrics)
        diff = _diff_against_parent(
            parent.source_code,
            ctx.source_code,
            max_lines=self.config.max_diff_lines,
        )
        if diff is None:
            body = "(no code differences from the current program)"
        else:
            body = f"```diff\n{diff}\n```"
        return f"{head}\n{metrics}\n{body}"

    def _format_failure_section(
        self,
        recent_failures: list[Any] | None,
        *,
        parent_id: str,
    ) -> str | None:
        """Render the 'Recent failed attempts' section, or None when empty.

        Each failure shows the error summary plus a diff between the
        crashed child and its parent at the time it was tried, so the LLM
        can see precisely what edit broke and avoid re-proposing it.
        """
        n = max(0, int(self.config.num_failed_attempts))
        if not recent_failures or n <= 0:
            return None
        items = list(recent_failures)[:n]
        if not items:
            return None
        out: list[str] = ["\n## Recent failed attempts\n"]
        out.append(
            "These recent candidates crashed or were rejected by the "
            "objective (combined_score = 0). Do NOT repeat their approach. "
            "Each is shown as `error_summary` + diff between the failed "
            "child and its parent at that iteration."
        )
        for f in items:
            iteration = getattr(f, "iteration", "?")
            parent_short = getattr(f, "parent_id", None) or "n/a"
            err = getattr(f, "error_summary", "(no error summary)")
            parent_code = getattr(f, "parent_source_code", "") or ""
            child_code = getattr(f, "child_source_code", "") or ""
            diff = _diff_against_parent(
                parent_code, child_code, max_lines=self.config.max_diff_lines
            )
            head = (
                f"### Failed at iteration {iteration} "
                f"(failed_parent={parent_short[:8]}, "
                f"current_parent_match={'yes' if parent_short == parent_id else 'no'})"
            )
            err_line = f"- error: {err}"
            if diff is None:
                body = "(failed child was identical to its parent)"
            else:
                body = f"```diff\n{diff}\n```"
            out.append(f"{head}\n{err_line}\n{body}")
        return "\n".join(out)


def _codeblock(code: str) -> str:
    return f"```python\n{code}\n```"


def _format_metric_line(k: str, v: Any) -> str:
    if isinstance(v, float):
        return f"- {k}: {v:.4f}"
    if isinstance(v, bool):
        return f"- {k}: {v}"
    if isinstance(v, int):
        return f"- {k}: {v}"
    return f"- {k}: {v}"


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "(no metrics)"
    return "\n".join(_format_metric_line(k, v) for k, v in metrics.items())


def _diff_against_parent(parent_code: str, ctx_code: str, *, max_lines: int = 120) -> str | None:
    """Render a unified diff. Returns None when the two sources are equal.

    The diff is computed line-by-line over `splitlines(keepends=True)` so
    trailing newlines are preserved. Long diffs are head/tail-truncated
    around `max_lines` total lines (an elision marker indicates the cut)
    so a single rewrite-everything child can't blow the prompt budget.
    """
    if parent_code == ctx_code:
        return None
    parent_lines = parent_code.splitlines(keepends=True)
    ctx_lines = ctx_code.splitlines(keepends=True)
    diff = list(
        unified_diff(
            parent_lines,
            ctx_lines,
            fromfile="parent",
            tofile="this",
            n=2,
            lineterm="",
        )
    )
    if not diff:
        return None
    if max_lines > 0 and len(diff) > max_lines:
        half = max_lines // 2
        elided = len(diff) - max_lines
        diff = diff[:half] + [f"... [{elided} diff lines elided]"] + diff[-half:]
    # Strip per-line trailing newlines so join with "\n" doesn't double up.
    return "\n".join(line.rstrip("\n") for line in diff)


def _dedupe_programs(
    programs: list[Program],
    *,
    limit: int,
    seen_ids: set[str],
) -> list[Program]:
    if limit <= 0:
        return []
    out: list[Program] = []
    for p in programs:
        if p.id in seen_ids:
            continue
        seen_ids.add(p.id)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _short(v: Any, *, maxlen: int = 200) -> str:
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen] + f"... [{len(s)} chars]"
