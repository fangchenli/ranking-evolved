"""`claude` CLI subprocess proposer (Claude Code).

Pipes the prompt's `system` and `user` content to a `claude` invocation
via stdin and reads the SEARCH/REPLACE blocks from stdout. Used when the
proposer should behave agentically — i.e., have access to file tools —
rather than just returning a single completion.

Tests inject a `runner` callable (`run_subprocess(args, input_bytes,
timeout) -> (stdout, stderr, returncode)`) so we can drive the codepath
without spawning real binaries.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


SubprocessRunner = Callable[..., tuple[str, str, int]]


@register_proposer("claude_code")
class ClaudeCodeProposer:
    name = "claude_code"

    def __init__(
        self,
        *,
        binary: str = "claude",
        model: str = "claude-opus-4-7",
        timeout: float = 300.0,
        extra_args: list[str] | None = None,
        runner: SubprocessRunner | None = None,
    ):
        self._binary = binary
        self._model = model
        self._timeout = timeout
        self._extra_args = list(extra_args or [])
        self._runner = runner

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        args = [self._binary, "-p", "--model", self._model, *self._extra_args]
        # Concatenate system + user as the stdin payload; the `-p` flag tells
        # `claude` to print the response and exit.
        stdin_bytes = (
            f"<system>\n{prompt.system}\n</system>\n\n"
            f"<user>\n{prompt.user}\n</user>\n"
        ).encode()

        start = time.perf_counter()
        if self._runner is not None:
            stdout, stderr, rc = self._runner(args=args, input=stdin_bytes, timeout=self._timeout)
        else:
            stdout, stderr, rc = await _run_real(args, stdin_bytes, self._timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if rc != 0:
            raise RuntimeError(
                f"claude_code: subprocess exited {rc}; stderr={stderr.strip()[:400]}"
            )
        raw = stdout
        pairs = extract_blocks(raw)
        return ProposedCandidate(
            raw_response=raw,
            diff_blocks=tuple(DiffBlock(search=s, replace=r) for s, r in pairs),
            full_rewrite=None,
            proposer="claude_code",
            model=self._model,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
        )


async def _run_real(args: list[str], stdin: bytes, timeout: float) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace"), proc.returncode or 0
