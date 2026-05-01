"""OpenAI Responses API HTTP proposer.

Covers OpenAI Responses API directly. This is not guaranteed to work with
OpenRouter, vLLM, Ollama, or OptiLLM unless those providers implement an
OpenAI-compatible `/responses` endpoint.

Auth: Bearer ${api_key}. The `api_key` is read from the proposer config
through `${ENV}` interpolation in the YAML loader, so secrets stay out of the
resolved-config snapshot.

Reasoning-model handling:
- GPT-5 / o-series models should use the Responses API reasoning object.
- Do not pass temperature/top_p for reasoning models unless the specific model
  and provider explicitly support them.
- Use `reasoning.effort` as the main reasoning-control knob.

Reasoning effort notes:
- "none": fastest, no reasoning. Only supported by some newer models.
- "minimal": very small amount of reasoning, useful for simple tasks.
- "low": good default for cheaper/faster proposal generation.
- "medium": stronger default when proposal quality matters more.
- "high": use for hard code edits or when candidate quality is weak.
- "xhigh": only supported by some newer models; highest cost/latency.
"""
from __future__ import annotations

import time

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


_REASONING_MODEL_PREFIXES = (
    "o1-", "o1",
    "o3-", "o3",
    "o4-",
    "gpt-5-", "gpt-5",
    "gpt-oss-120b", "gpt-oss-20b",
)


def _is_reasoning_model(name: str) -> bool:
    return str(name).lower().startswith(_REASONING_MODEL_PREFIXES)


@register_proposer("openai_responses")
class OpenAIResponsesProposer:
    name = "openai_responses"

    def __init__(
        self,
        *,
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-5.2",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
        timeout: float = 180.0,
        retries: int = 3,
        client: object | None = None,
    ):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._retries = retries
        self._client = client  # tests inject an httpx.AsyncClient mock

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        client = self._client or _default_client(self._timeout)
        owns_client = self._client is None
        try:
            return await self._call(client, prompt)
        finally:
            if owns_client:
                await client.aclose()  # type: ignore[attr-defined]

    async def _call(self, client, prompt: Prompt) -> ProposedCandidate:  # type: ignore[no-untyped-def]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
        }

        # Responses API shape:
        # - `instructions` is the closest replacement for a system message.
        # - `input` is the user/developer/model input.
        # - `max_output_tokens` replaces `max_tokens` / `max_completion_tokens`.
        body: dict = {
            "model": self._model,
            "instructions": prompt.system,
            "input": [
                {"role": "user", "content": prompt.user},
            ],
            "max_output_tokens": self._max_tokens,
        }

        if _is_reasoning_model(self._model):
            # Reasoning effort is the replacement knob for temperature-like
            # control on GPT-5 / o-series reasoning models.
            # Avoid `none` unless you know the model supports it.
            # Avoid `xhigh` unless the model docs say it is supported.
            if self._reasoning_effort is not None:
                body["reasoning"] = {"effort": self._reasoning_effort}
        else:
            # Non-reasoning models may still support temperature in Responses.
            # Keep this only for models/providers where it is accepted.
            body["temperature"] = self._temperature

        url = f"{self._api_base}/responses"

        last_exc: Exception | None = None
        for _ in range(max(1, self._retries)):
            start = time.perf_counter()
            try:
                response = await client.post(url, json=body, headers=headers)
            except Exception as exc:  # network errors
                last_exc = exc
                continue

            latency_ms = (time.perf_counter() - start) * 1000.0
            status = getattr(response, "status_code", 500)

            if status >= 500:
                last_exc = RuntimeError(
                    f"openai_responses: HTTP {status} "
                    f"body={_short(response.text if hasattr(response, 'text') else '')}"
                )
                continue

            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"openai_responses: HTTP {status} returned non-JSON body: "
                    f"{_short(response.text if hasattr(response, 'text') else '')}"
                ) from exc

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"openai_responses: HTTP {status} returned non-dict JSON: "
                    f"body={_short(data)}"
                )

            if data.get("error"):
                raise RuntimeError(
                    f"openai_responses: HTTP {status} error={data.get('error')!r} "
                    f"body={_short(data)}"
                )

            if data.get("status") == "incomplete":
                details = data.get("incomplete_details") or {}
                reason = details.get("reason") if isinstance(details, dict) else None
                hint = (
                    "Increase proposer.max_tokens or lower proposer.reasoning_effort."
                    if reason == "max_output_tokens"
                    else "Inspect the response body and retry settings."
                )
                raise RuntimeError(
                    f"openai_responses: HTTP {status} response incomplete "
                    f"reason={reason!r}. {hint} body={_short(data)}"
                )

            raw = _extract_response_text(data)
            if raw is None:
                raise RuntimeError(
                    f"openai_responses: HTTP {status} response had no text output. "
                    f"body={_short(data)}"
                )

            return _to_candidate(data, raw=raw, model=self._model, latency_ms=latency_ms)

        raise RuntimeError(f"openai_responses: exhausted retries; last_exc={last_exc}")


def _extract_response_text(data: dict) -> str | None:
    """Extract text from a Responses API response.

    The happy path is `output_text`. The fallback handles the typed `output`
    list in case the provider or SDK does not populate `output_text`.
    """
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue

        # Typical item:
        # {
        #   "type": "message",
        #   "content": [
        #     {"type": "output_text", "text": "..."}
        #   ]
        # }
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)

    if chunks:
        return "\n".join(chunks)

    return None


def _to_candidate(
    data: dict,
    *,
    raw: str,
    model: str,
    latency_ms: float,
) -> ProposedCandidate:
    usage = data.get("usage") or {}
    pairs = extract_blocks(raw)

    return ProposedCandidate(
        raw_response=raw,
        diff_blocks=tuple(DiffBlock(search=s, replace=r) for s, r in pairs),
        full_rewrite=None,
        proposer="openai_responses",
        model=data.get("model", model),
        tokens_in=usage.get("input_tokens"),
        tokens_out=usage.get("output_tokens"),
        latency_ms=latency_ms,
    )


def _short(s, *, n: int = 200):  # type: ignore[no-untyped-def]
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def _default_client(timeout: float):  # type: ignore[no-untyped-def]
    import httpx
    return httpx.AsyncClient(timeout=timeout)