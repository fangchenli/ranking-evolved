"""Exact MaxSim retriever as an evaluator-importable program.

This is a thin alias over `tasks.late_interaction.library.ExactMaxSimRetriever`
so the evaluator's `--program` flag can target it the same way it targets the
freeform seed or an evolved candidate.

    EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \\
        --program tasks/late_interaction/programs/exact_maxsim.py \\
        --datasets beir_scifact

Exposes `LateInteractionRetriever` (the symbol the evaluator's loader looks
for) as an alias for `ExactMaxSimRetriever`.
"""
from __future__ import annotations

# Import _runtime first so BLAS pinning takes effect before numpy is imported
# downstream by `library`.
from tasks.late_interaction import _runtime  # noqa: F401

from tasks.late_interaction.library import ExactMaxSimRetriever as LateInteractionRetriever

__all__ = ["LateInteractionRetriever"]
