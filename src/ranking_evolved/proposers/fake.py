"""Deterministic, transcript-driven proposer.

Used by the smoke test and the reference-replay sign-off. Construct with
either:

  * a list of `(raw_response, model)` tuples (looped if exhausted), OR
  * a callable `(prompt, iteration) -> raw_response` for tests that want
    to react to prompt content.

Returns a `ProposedCandidate` whose `diff_blocks` are extracted via the
same regex used by `prompts.diff` so the rest of the pipeline sees the
same shape it would see from a real LLM.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


@register_proposer("fake")
class FakeProposer:
    name = "fake"

    def __init__(
        self,
        *,
        responses: Iterable[tuple[str, str]] | None = None,
        callback: Callable[[Prompt, int], str] | None = None,
        model: str = "fake-1",
    ):
        if responses is None and callback is None:
            raise ValueError("FakeProposer needs either `responses` or `callback`")
        self._responses = list(responses) if responses else None
        self._callback = callback
        self._model = model
        self._cursor = 0

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        start = time.perf_counter()
        if self._callback is not None:
            raw = self._callback(prompt, prompt.iteration)
            model = self._model
        else:
            assert self._responses is not None
            raw, model = self._responses[self._cursor % len(self._responses)]
            self._cursor += 1

        pairs = extract_blocks(raw)
        blocks = tuple(DiffBlock(search=s, replace=r) for s, r in pairs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ProposedCandidate(
            raw_response=raw,
            diff_blocks=blocks,
            full_rewrite=None,
            proposer="fake",
            model=model,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
        )
