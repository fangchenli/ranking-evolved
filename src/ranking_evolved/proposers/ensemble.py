"""Weighted multi-proposer.

Each iteration picks one underlying proposer with probability proportional
to its weight (uses a deterministic `random.Random` seeded from config so
the choice is replayable). The `ProposedCandidate.proposer` field still
identifies which leaf produced the response, so the replay dashboard can
show which proposer was active per step.
"""
from __future__ import annotations

import random

from ..core.types import Prompt, ProposedCandidate
from .base import register_proposer


@register_proposer("ensemble")
class EnsembleProposer:
    name = "ensemble"

    def __init__(
        self,
        *,
        members: list[tuple[object, float]],
        random_seed: int = 0,
    ):
        if not members:
            raise ValueError("EnsembleProposer needs at least one member")
        self._members = members
        self._rng = random.Random(random_seed)

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        weights = [w for _, w in self._members]
        chosen = self._rng.choices(self._members, weights=weights, k=1)[0][0]
        return await chosen.propose(prompt)  # type: ignore[union-attr]
