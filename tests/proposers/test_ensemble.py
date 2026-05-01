"""EnsembleProposer test: weighted dispatch is deterministic & records leaf identity."""
from __future__ import annotations

import asyncio

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.ensemble import EnsembleProposer
from ranking_evolved.proposers.fake import FakeProposer


def _prompt(iter_: int = 1) -> Prompt:
    return Prompt(
        system="s", user="u", template_key="diff_user",
        iteration=iter_, parent_id="p", inspiration_ids=(),
    )


def test_ensemble_dispatches_by_weight_deterministically(record_io):
    a = FakeProposer(responses=[("from-a", "model-a")])
    b = FakeProposer(responses=[("from-b", "model-b")])
    ens = EnsembleProposer(members=[(a, 0.0), (b, 1.0)], random_seed=42)

    def run() -> list[str]:
        return [
            asyncio.run(ens.propose(_prompt(i))).raw_response for i in range(5)
        ]

    out = record_io(
        module="src/ranking_evolved/proposers/ensemble.py",
        function="EnsembleProposer.propose (weighted to b)",
        input={"members": [("a", 0.0), ("b", 1.0)], "n_calls": 5},
        run=run,
    )
    # All weight on b → every call goes to b.
    assert out == ["from-b"] * 5


def test_ensemble_uses_seeded_rng(record_io):
    a = FakeProposer(callback=lambda p, i: "a")
    b = FakeProposer(callback=lambda p, i: "b")

    def run() -> dict:
        ens1 = EnsembleProposer(members=[(a, 1.0), (b, 1.0)], random_seed=99)
        ens2 = EnsembleProposer(members=[(a, 1.0), (b, 1.0)], random_seed=99)
        out1 = [asyncio.run(ens1.propose(_prompt(i))).raw_response for i in range(10)]
        out2 = [asyncio.run(ens2.propose(_prompt(i))).raw_response for i in range(10)]
        return {"out1": out1, "out2": out2, "equal": out1 == out2}

    out = record_io(
        module="src/ranking_evolved/proposers/ensemble.py",
        function="EnsembleProposer.propose (deterministic)",
        input={"random_seed": 99, "n_calls": 10},
        run=run,
    )
    assert out["equal"] is True
