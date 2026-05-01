"""Scripted proposer for deterministic evolution algorithm tests."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..core.types import DiffBlock, Prompt, ProposedCandidate
from ..prompts.diff import extract_blocks
from .base import register_proposer


_CONSTANT_BLOCK_RE = re.compile(
    r"SCORE\s*=\s*[-+0-9.eE]+\s*\n"
    r"COMPLEXITY\s*=\s*[-+0-9.eE]+\s*\n"
    r"DIVERSITY\s*=\s*[-+0-9.eE]+",
    re.MULTILINE,
)


@register_proposer("scripted")
class ScriptedProposer:
    name = "scripted"

    def __init__(
        self,
        *,
        proposals_jsonl: str,
        timeout: float = 30.0,
        retries: int = 0,
    ):
        self._proposals_jsonl = Path(proposals_jsonl)
        self._timeout = timeout
        self._retries = retries
        self._targets: list[dict[str, float]] = []
        self._i = 0

        with self._proposals_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self._targets.append(
                    {
                        "score": float(obj["score"]),
                        "complexity": float(obj["complexity"]),
                        "diversity": float(obj["diversity"]),
                    }
                )

        if not self._targets:
            raise ValueError(f"No scripted proposals found in {self._proposals_jsonl}")

    async def propose(self, prompt: Prompt) -> ProposedCandidate:
        start = time.perf_counter()
        raw = self._build_response(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _to_candidate(raw=raw, latency_ms=latency_ms)

    def _build_response(self, prompt: Prompt) -> str:
        current_code = _extract_current_program_code(prompt)
        match = _CONSTANT_BLOCK_RE.search(current_code)

        if match is None:
            raise ValueError(
                "ScriptedProposer could not find this exact block in the current program:\n"
                "SCORE = ...\n"
                "COMPLEXITY = ...\n"
                "DIVERSITY = ..."
            )

        search = match.group(0)

        target = self._targets[self._i % len(self._targets)]
        self._i += 1

        replace = (
            f"SCORE = {target['score']:.2f}\n"
            f"COMPLEXITY = {target['complexity']:.2f}\n"
            f"DIVERSITY = {target['diversity']:.2f}"
        )

        return (
            "<<<<<<< SEARCH\n"
            f"{search}\n"
            "=======\n"
            f"{replace}\n"
            ">>>>>>> REPLACE"
        )


def _extract_current_program_code(prompt: Prompt) -> str:
    text = prompt.user

    marker = "## Current program"
    if marker in text:
        after_marker = text.split(marker, 1)[1]
        blocks = re.findall(r"```python\n(.*?)\n```", after_marker, flags=re.DOTALL)
        if blocks:
            return blocks[0]

    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1]

    return text


def _to_candidate(
    *,
    raw: str,
    latency_ms: float,
) -> ProposedCandidate:
    pairs = extract_blocks(raw)

    return ProposedCandidate(
        raw_response=raw,
        diff_blocks=tuple(DiffBlock(search=s, replace=r) for s, r in pairs),
        full_rewrite=None,
        proposer="scripted",
        model="scripted",
        tokens_in=None,
        tokens_out=None,
        latency_ms=latency_ms,
    )