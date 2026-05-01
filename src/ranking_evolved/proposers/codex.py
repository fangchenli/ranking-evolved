"""`codex` CLI subprocess proposer.

Same shape as `claude_code` but invokes the Codex CLI binary. Kept as a
separate module for two reasons: (1) different default flags, (2) makes
the registry self-documenting (`list-algorithms` shows both names).
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


SubprocessRunner = Callable[..., tuple[str, str, int]]


@register_proposer("codex")
class CodexProposer:
    name = "codex"

    def __init__(
        self,
        *,
        binary: str = "codex",
        timeout: float = 300.0,
        extra_args: list[str] | None = None,
        runner: SubprocessRunner | None = None,
    ):
        self._binary = binary
        self._timeout = timeout
        self._extra_args = list(extra_args or [])
        self._runner = runner

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        args = [self._binary, *self._extra_args]
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
            raise RuntimeError(f"codex: subprocess exited {rc}; stderr={stderr.strip()[:400]}")
        pairs = extract_blocks(stdout)
        return ProposedCandidate(
            raw_response=stdout,
            diff_blocks=tuple(DiffBlock(search=s, replace=r) for s, r in pairs),
            full_rewrite=None,
            proposer="codex",
            model=self._binary,
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
