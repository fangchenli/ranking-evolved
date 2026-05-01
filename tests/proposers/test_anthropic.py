"""AnthropicProposer test with a fake SDK client."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.anthropic import AnthropicProposer


DIFF = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


class _FakeMessages:
    def __init__(self):
        self.last_call: dict | None = None

    async def create(self, *, model, max_tokens, temperature, system, messages):
        self.last_call = {
            "model": model, "system": system, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        return SimpleNamespace(
            model=model,
            content=[SimpleNamespace(text=DIFF)],
            usage=SimpleNamespace(input_tokens=42, output_tokens=11),
        )


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_anthropic_caches_system(record_io):
    client = _FakeClient()
    prop = AnthropicProposer(
        api_key="anth-x", model="claude-opus-4-7", cache_system=True, client=client,
    )

    def run() -> dict:
        cand = asyncio.run(prop.propose(Prompt(
            system="big static system", user="iter user", template_key="diff_user",
            iteration=1, parent_id="p", inspiration_ids=(),
        )))
        sysblock = client.messages.last_call["system"][0]
        return {
            "system_text": sysblock["text"],
            "system_cache_control": sysblock.get("cache_control"),
            "n_diff_blocks": len(cand.diff_blocks),
            "tokens_in": cand.tokens_in,
            "tokens_out": cand.tokens_out,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/anthropic.py",
        function="AnthropicProposer.propose (with cache)",
        input={"cache_system": True},
        run=run,
    )
    assert out["system_text"] == "big static system"
    assert out["system_cache_control"] == {"type": "ephemeral"}
    assert out["n_diff_blocks"] == 1
    assert out["tokens_in"] == 42
    assert out["tokens_out"] == 11


def test_anthropic_no_cache_when_disabled(record_io):
    client = _FakeClient()
    prop = AnthropicProposer(api_key="x", cache_system=False, client=client)

    def run() -> dict:
        asyncio.run(prop.propose(Prompt(
            system="s", user="u", template_key="diff_user",
            iteration=1, parent_id="p", inspiration_ids=(),
        )))
        sysblock = client.messages.last_call["system"][0]
        return {"has_cache_control": "cache_control" in sysblock}

    out = record_io(
        module="src/ranking_evolved/proposers/anthropic.py",
        function="AnthropicProposer.propose (no cache)",
        input={"cache_system": False},
        run=run,
    )
    assert out == {"has_cache_control": False}
