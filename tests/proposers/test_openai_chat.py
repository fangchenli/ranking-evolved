"""OpenAIResponsesProposer tests with a mocked httpx-like client.

The proposer registers as `openai_responses` and posts to `/responses`.
Body shape: `instructions` + `input` + `max_output_tokens` (+ optional
`reasoning.effort`). Response shape: `output_text` (preferred) or `output`
list with `content[].text` blocks. Usage carries `input_tokens` /
`output_tokens` (NOT `prompt_tokens`/`completion_tokens` like Chat
Completions).
"""
from __future__ import annotations

import asyncio

from ranking_evolved.core.types import Prompt
from ranking_evolved.proposers.openai_chat import OpenAIResponsesProposer


DIFF = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


class _FakeResponse:
    def __init__(self, status_code: int, data: dict | str):
        self.status_code = status_code
        self._data = data
        self.text = data if isinstance(data, str) else "ok"

    def json(self):
        if isinstance(self._data, str):
            raise ValueError("not JSON")
        return self._data


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_call: dict | None = None

    async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
        self.last_call = {"url": url, "json": json, "headers": headers}
        return self._response

    async def aclose(self) -> None:
        return None


def _prompt() -> Prompt:
    return Prompt(
        system="SYSTEM", user="USER", template_key="diff_user",
        iteration=1, parent_id="p", inspiration_ids=(),
    )


def test_responses_happy_path_output_text(record_io):
    response = _FakeResponse(200, {
        "model": "gpt-5.2",
        "output_text": f"intro\n{DIFF}\noutro",
        "usage": {"input_tokens": 100, "output_tokens": 30},
    })
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="sk-test", model="gpt-5.2", client=client)

    def run() -> dict:
        cand = asyncio.run(prop.propose(_prompt()))
        body = client.last_call["json"]
        return {
            "url": client.last_call["url"],
            "auth": client.last_call["headers"]["Authorization"],
            "instructions": body["instructions"],
            "input_role": body["input"][0]["role"],
            "max_output_tokens": body["max_output_tokens"],
            "tokens_in": cand.tokens_in,
            "tokens_out": cand.tokens_out,
            "n_diff_blocks": len(cand.diff_blocks),
            "first_search": cand.diff_blocks[0].search,
            "model": cand.model,
            "proposer": cand.proposer,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (output_text)",
        input={"model": "gpt-5.2"},
        run=run,
    )
    assert out["url"].endswith("/responses")
    assert out["auth"] == "Bearer sk-test"
    assert out["instructions"] == "SYSTEM"
    assert out["input_role"] == "user"
    assert out["max_output_tokens"] == 8192     # default
    assert out["tokens_in"] == 100
    assert out["tokens_out"] == 30
    assert out["n_diff_blocks"] == 1
    assert out["first_search"] == "old\n"
    assert out["model"] == "gpt-5.2"
    assert out["proposer"] == "openai_responses"


def test_responses_falls_back_to_output_list_text(record_io):
    """When `output_text` is absent, extract from the typed `output[].content[].text` chain."""
    response = _FakeResponse(200, {
        "model": "gpt-5.2",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": f"part-A\n{DIFF}\n"},
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 6},
    })
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="x", model="gpt-5.2", client=client)

    def run() -> dict:
        cand = asyncio.run(prop.propose(_prompt()))
        return {"raw_starts_with": cand.raw_response[:6], "n_blocks": len(cand.diff_blocks)}

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (output[] fallback)",
        input={"output_text": "absent"},
        run=run,
    )
    assert out == {"raw_starts_with": "part-A", "n_blocks": 1}


def test_reasoning_effort_omitted_when_none(record_io):
    """Default behavior: `reasoning_effort=None` means do not put the
    `reasoning` block in the request body at all."""
    response = _FakeResponse(200, {"output_text": DIFF, "usage": {}})
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="x", model="gpt-5.2", client=client)
    # default: reasoning_effort=None

    def run() -> dict:
        asyncio.run(prop.propose(_prompt()))
        body = client.last_call["json"]
        return {"has_reasoning": "reasoning" in body, "model": body["model"]}

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (no reasoning_effort)",
        input={"reasoning_effort": None, "model": "gpt-5.2"},
        run=run,
    )
    assert out == {"has_reasoning": False, "model": "gpt-5.2"}


def test_reasoning_effort_attached_when_set(record_io):
    """Setting `reasoning_effort=medium` on a reasoning model puts a
    `reasoning: {effort: medium}` block in the request body."""
    response = _FakeResponse(200, {"output_text": DIFF, "usage": {}})
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(
        api_key="x", model="gpt-5.2", reasoning_effort="medium", client=client,
    )

    def run() -> dict:
        asyncio.run(prop.propose(_prompt()))
        body = client.last_call["json"]
        return {
            "reasoning_block": body.get("reasoning"),
            "no_temperature": "temperature" not in body,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (reasoning_effort=medium)",
        input={"reasoning_effort": "medium"},
        run=run,
    )
    assert out == {"reasoning_block": {"effort": "medium"}, "no_temperature": True}


def test_non_reasoning_model_includes_temperature_no_reasoning(record_io):
    """Classic models (gpt-4o, etc.) should keep `temperature` in the body
    and never get a `reasoning` block, even if reasoning_effort is set."""
    response = _FakeResponse(200, {"output_text": DIFF, "usage": {}})
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(
        api_key="x", model="gpt-4o", temperature=0.7,
        reasoning_effort="medium",  # ignored on non-reasoning model
        client=client,
    )

    def run() -> dict:
        asyncio.run(prop.propose(_prompt()))
        body = client.last_call["json"]
        return {
            "temperature": body.get("temperature"),
            "has_reasoning": "reasoning" in body,
        }

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (non-reasoning)",
        input={"model": "gpt-4o", "reasoning_effort_in_config": "medium"},
        run=run,
    )
    assert out == {"temperature": 0.7, "has_reasoning": False}


def test_responses_4xx_with_error_body_surfaces_message(record_io):
    """4xx with an error-shaped body must raise RuntimeError carrying
    the upstream message — not a KeyError."""
    response = _FakeResponse(401, {
        "error": {"message": "Invalid API key", "type": "invalid_request_error"},
    })
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="bad", model="gpt-5.2", retries=1, client=client)

    def run() -> str:
        try:
            asyncio.run(prop.propose(_prompt()))
            return "no_raise"
        except RuntimeError as exc:
            return str(exc)

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (4xx error body)",
        input={"status": 401},
        run=run,
    )
    assert "Invalid API key" in out


def test_responses_2xx_with_error_body_surfaces_message(record_io):
    """Some proxies return 200 with an error body. We must still raise."""
    response = _FakeResponse(200, {
        "error": {"message": "rate-limited by upstream", "type": "rate_limit"},
    })
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="x", model="gpt-5.2", retries=1, client=client)

    def run() -> str:
        try:
            asyncio.run(prop.propose(_prompt()))
            return "no_raise"
        except RuntimeError as exc:
            return str(exc)

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (200 error body)",
        input={"status": 200},
        run=run,
    )
    assert "rate-limited by upstream" in out


def test_responses_incomplete_max_output_tokens_surfaces_actionable_hint(record_io):
    """Responses can return HTTP 200 with status=incomplete when the output
    budget is exhausted before a visible message is produced."""
    response = _FakeResponse(200, {
        "id": "resp_test",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 1000, "output_tokens": 8192},
    })
    client = _FakeClient(response)
    prop = OpenAIResponsesProposer(api_key="x", model="gpt-5.2", retries=1, client=client)

    def run() -> str:
        try:
            asyncio.run(prop.propose(_prompt()))
            return "no_raise"
        except RuntimeError as exc:
            return str(exc)

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (incomplete max_output_tokens)",
        input={"status": "incomplete", "reason": "max_output_tokens"},
        run=run,
    )
    assert "response incomplete" in out
    assert "max_output_tokens" in out
    assert "Increase proposer.max_tokens or lower proposer.reasoning_effort" in out


def test_responses_5xx_retries_then_succeeds(record_io):
    """5xx is retried; a subsequent 200 succeeds."""
    bad = _FakeResponse(503, {})
    good = _FakeResponse(200, {"output_text": DIFF, "usage": {}})

    class SeqClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
            self.calls += 1
            return bad if self.calls == 1 else good

        async def aclose(self) -> None:
            return None

    client = SeqClient()
    prop = OpenAIResponsesProposer(api_key="sk", model="gpt-5.2", retries=3, client=client)

    def run() -> dict:
        cand = asyncio.run(prop.propose(_prompt()))
        return {"calls": client.calls, "raw_has_search": "SEARCH" in cand.raw_response}

    out = record_io(
        module="src/ranking_evolved/proposers/openai_chat.py",
        function="OpenAIResponsesProposer.propose (retry)",
        input={"sequence": [503, 200]},
        run=run,
    )
    assert out == {"calls": 2, "raw_has_search": True}
