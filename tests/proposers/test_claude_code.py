"""ClaudeCodeProposer test with a fake subprocess runner."""
from __future__ import annotations

import asyncio

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.claude_code import ClaudeCodeProposer


DIFF = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


def test_claude_code_invokes_with_stdin_pipe(record_io):
    captured: dict = {}

    def fake_runner(*, args, input, timeout):
        captured["args"] = list(args)
        captured["stdin"] = input.decode()
        captured["timeout"] = timeout
        return (DIFF, "", 0)

    prop = ClaudeCodeProposer(
        binary="claude", model="claude-opus-4-7",
        extra_args=["--dangerously-skip-permissions"],
        runner=fake_runner,
    )

    def run() -> dict:
        cand = asyncio.run(prop.propose(Prompt(
            system="SYS", user="USR", template_key="diff_user",
            iteration=1, parent_id="p", inspiration_ids=(),
        )))
        return {
            "args": captured["args"],
            "stdin_has_system": "<system>" in captured["stdin"],
            "stdin_has_user": "<user>" in captured["stdin"],
            "n_diff_blocks": len(cand.diff_blocks),
            "model": cand.model,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/claude_code.py",
        function="ClaudeCodeProposer.propose",
        input={"binary": "claude", "extra_args": ["--dangerously-skip-permissions"]},
        run=run,
    )
    assert out["args"][0] == "claude"
    assert "-p" in out["args"]
    assert "--dangerously-skip-permissions" in out["args"]
    assert out["stdin_has_system"] is True
    assert out["stdin_has_user"] is True
    assert out["n_diff_blocks"] == 1


def test_claude_code_nonzero_exit_raises(record_io):
    def runner(*, args, input, timeout):
        return ("", "boom", 17)

    prop = ClaudeCodeProposer(runner=runner)

    def run() -> str:
        try:
            asyncio.run(prop.propose(Prompt(
                system="s", user="u", template_key="diff_user",
                iteration=1, parent_id="p", inspiration_ids=(),
            )))
            return "no_raise"
        except RuntimeError as exc:
            return str(exc)

    out = record_io(
        module="src/ranking_evolved/proposers/claude_code.py",
        function="ClaudeCodeProposer.propose (nonzero exit)",
        input={"returncode": 17, "stderr": "boom"},
        run=run,
    )
    assert "exited 17" in out
    assert "boom" in out
