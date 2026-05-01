"""CodexProposer test (mocked subprocess runner)."""
from __future__ import annotations

import asyncio

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.codex import CodexProposer


DIFF = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


def test_codex_passes_prompt_via_stdin(record_io):
    captured: dict = {}

    def runner(*, args, input, timeout):
        captured["args"] = list(args)
        captured["stdin"] = input.decode()
        return (DIFF, "", 0)

    prop = CodexProposer(binary="codex", extra_args=["--mode", "diff"], runner=runner)

    def run() -> dict:
        cand = asyncio.run(prop.propose(Prompt(
            system="S", user="U", template_key="diff_user",
            iteration=2, parent_id="p", inspiration_ids=(),
        )))
        return {
            "args": captured["args"],
            "stdin_includes_system": "<system>" in captured["stdin"],
            "n_diff_blocks": len(cand.diff_blocks),
            "proposer": cand.proposer,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/codex.py",
        function="CodexProposer.propose",
        input={"binary": "codex"},
        run=run,
    )
    assert out["args"][0] == "codex"
    assert out["args"][1:] == ["--mode", "diff"]
    assert out["stdin_includes_system"] is True
    assert out["n_diff_blocks"] == 1
    assert out["proposer"] == "codex"
