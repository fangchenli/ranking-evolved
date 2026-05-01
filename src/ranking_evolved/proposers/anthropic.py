"""Anthropic Messages API proposer with prompt caching.

Why caching: long evolution runs reuse the same system message + a slowly
moving block of inspiration programs across iterations. Marking those
blocks `cache_control: ephemeral` cuts cost ~10x on large prompts and
removes a recompute cost the OpenAI-compatible path can't avoid.

Tests inject a fake `client` (any object with an async `messages.create`
method) so the network never touches CI.
"""
from __future__ import annotations

import time

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


@register_proposer("anthropic")
class AnthropicProposer:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        cache_system: bool = True,
        client: object | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache_system = cache_system
        self._client = client

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        client = self._client or _default_client(self._api_key)
        system_blocks = self._build_system(prompt.system)
        start = time.perf_counter()
        message = await client.messages.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt.user}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _to_candidate(message, model=self._model, latency_ms=latency_ms)

    def _build_system(self, system_text: str):
        block: dict = {"type": "text", "text": system_text}
        if self._cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]


def _to_candidate(message, *, model: str, latency_ms: float) -> ProposedCandidate:  # type: ignore[no-untyped-def]
    # message.content is a list of blocks; pull text from the first text block.
    raw = ""
    for block in getattr(message, "content", []) or []:
        b_text = getattr(block, "text", None)
        if b_text is None and isinstance(block, dict):
            b_text = block.get("text")
        if b_text:
            raw = b_text
            break
    usage = getattr(message, "usage", None)
    tokens_in = getattr(usage, "input_tokens", None) if usage else None
    tokens_out = getattr(usage, "output_tokens", None) if usage else None
    pairs = extract_blocks(raw)
    return ProposedCandidate(
        raw_response=raw,
        diff_blocks=tuple(DiffBlock(search=s, replace=r) for s, r in pairs),
        full_rewrite=None,
        proposer="anthropic",
        model=getattr(message, "model", model),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


def _default_client(api_key: str | None):  # type: ignore[no-untyped-def]
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(api_key=api_key)
