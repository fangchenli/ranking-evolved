"""FakeProposer tests: transcript replay + callback mode."""
from __future__ import annotations

import asyncio

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.fake import FakeProposer


def _prompt(iter_: int = 1) -> Prompt:
    return Prompt(
        system="s", user="u", template_key="diff_user",
        iteration=iter_, parent_id="p", inspiration_ids=(),
    )


def test_fake_transcript_cycles(record_io):
    diff = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
    prop = FakeProposer(responses=[(diff, "m1"), ("ignored", "m2")])

    def run() -> dict:
        c1 = asyncio.run(prop.propose(_prompt(1)))
        c2 = asyncio.run(prop.propose(_prompt(2)))
        c3 = asyncio.run(prop.propose(_prompt(3)))  # cycles back to first
        return {
            "raw_responses": [c1.raw_response, c2.raw_response, c3.raw_response],
            "models": [c1.model, c2.model, c3.model],
            "n_diff_blocks_first": len(c1.diff_blocks),
        }

    out = record_io(
        module="src/ranking_evolved/proposers/fake.py",
        function="FakeProposer.propose (transcript)",
        input={"n_responses": 2, "calls": 3},
        run=run,
    )
    assert out["raw_responses"] == [diff, "ignored", diff]
    assert out["models"] == ["m1", "m2", "m1"]
    assert out["n_diff_blocks_first"] == 1


def test_fake_callback_uses_prompt(record_io):
    def cb(prompt: Prompt, iteration: int) -> str:
        return f"<iter={iteration}, parent={prompt.parent_id}>"

    prop = FakeProposer(callback=cb)

    def run() -> str:
        out = asyncio.run(prop.propose(_prompt(7)))
        return out.raw_response

    out = record_io(
        module="src/ranking_evolved/proposers/fake.py",
        function="FakeProposer.propose (callback)",
        input={"iteration": 7, "parent_id": "p"},
        run=run,
    )
    assert out == "<iter=7, parent=p>"
