"""SEARCH/REPLACE diff parser & applier — fails loudly on every ambiguity.

Compatible with OpenEvolve's regex (so recorded LLM transcripts replay
verbatim) but with stricter semantics:

  * Block extracted but search text appears 0 times in parent  -> raise.
  * Block extracted but search text appears >1 times in parent -> raise,
    even with fuzzy=True. Never silently picks one.
  * After processing all blocks, if any errored, raise DiffError with the
    full per-block report. Partial application is not provided.

This matches the user's directive: diffs are where bugs hide, so the
parser must surface every uncertainty.
"""
from __future__ import annotations

import re
from typing import Pattern

from ..core.types import DiffApplication, DiffBlock


DEFAULT_PATTERN: Pattern[str] = re.compile(
    r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)
DEFAULT_PATTERN_STR: str = DEFAULT_PATTERN.pattern


class DiffError(Exception):
    """Raised when a SEARCH/REPLACE block cannot be applied unambiguously."""

    def __init__(self, message: str, application: DiffApplication):
        super().__init__(message)
        self.application = application


def extract_blocks(raw_response: str, *, pattern: Pattern[str] = DEFAULT_PATTERN) -> list[tuple[str, str]]:
    """Pull out every (search, replace) pair. Order preserved."""
    return [(m.group(1), m.group(2)) for m in pattern.finditer(raw_response)]


def apply_search_replace(
    parent: str,
    raw_response: str,
    *,
    fuzzy: bool = False,
    pattern: Pattern[str] = DEFAULT_PATTERN,
) -> tuple[str, DiffApplication]:
    """Apply every SEARCH/REPLACE block in `raw_response` to `parent`.

    Returns the new source plus a `DiffApplication` describing exactly which
    blocks matched where. Raises `DiffError` if any block was ambiguous or
    unmatched (under default settings).
    """
    pairs = extract_blocks(raw_response, pattern=pattern)
    blocks: list[DiffBlock] = []
    current = parent
    n_applied = 0
    fatal: str | None = None

    for search, replace in pairs:
        result = _apply_one(current, search, replace, fuzzy=fuzzy)
        blocks.append(result.block)
        if result.error is not None:
            if fatal is None:
                fatal = result.error
        else:
            current = result.new_text
            n_applied += 1

    application = DiffApplication(
        pattern=pattern.pattern,
        blocks=tuple(blocks),
        n_extracted=len(pairs),
        n_applied=n_applied,
        fatal_error=fatal,
    )

    if fatal is not None:
        report_lines = [f"diff failed: {fatal}"]
        for i, b in enumerate(application.blocks):
            if b.error:
                report_lines.append(f"  block #{i}: {b.error}")
        raise DiffError("\n".join(report_lines), application)

    return current, application


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------

class _Result:
    __slots__ = ("block", "new_text", "error")

    def __init__(self, block: DiffBlock, new_text: str, error: str | None):
        self.block = block
        self.new_text = new_text
        self.error = error


def _apply_one(text: str, search: str, replace: str, *, fuzzy: bool) -> _Result:
    """Apply one SEARCH/REPLACE pair. Exact match preferred; fuzzy is opt-in."""
    n = text.count(search)
    if n == 1:
        idx = text.index(search)
        line = text[:idx].count("\n") + 1
        return _Result(
            DiffBlock(search=search, replace=replace, matched_at_line=line),
            text.replace(search, replace, 1),
            None,
        )
    if n > 1:
        # Always raise on ambiguity, even in fuzzy mode.
        return _Result(
            DiffBlock(
                search=search, replace=replace,
                error=f"ambiguous match ({n} candidates)",
            ),
            text,
            f"ambiguous match ({n} candidates)",
        )

    # n == 0: try fuzzy if enabled
    if fuzzy:
        candidates = _whitespace_normalized_matches(text, search)
        if len(candidates) == 1:
            start, end = candidates[0]
            line = text[:start].count("\n") + 1
            new_text = text[:start] + replace + text[end:]
            return _Result(
                DiffBlock(search=search, replace=replace, matched_at_line=line),
                new_text,
                None,
            )
        if len(candidates) > 1:
            return _Result(
                DiffBlock(
                    search=search, replace=replace,
                    error=f"ambiguous fuzzy match ({len(candidates)} candidates)",
                ),
                text,
                f"ambiguous fuzzy match ({len(candidates)} candidates)",
            )

    return _Result(
        DiffBlock(search=search, replace=replace, error="search text not found"),
        text,
        "search text not found",
    )


def _whitespace_normalized_matches(text: str, search: str) -> list[tuple[int, int]]:
    """Find all spans of `text` whose whitespace-normalized form == search's.

    Whitespace normalization: collapse runs of spaces/tabs into one space;
    strip trailing whitespace on each line; preserve newlines.
    """
    normed_search = _normalize_ws(search)
    matches: list[tuple[int, int]] = []
    # Walk every byte offset and try to match a window. Quadratic but fine for
    # the realistic block sizes (LLM diffs are small) and the input text size
    # is bounded by `max_code_length`.
    text_norm_lines = [(_normalize_ws(line), len(line)) for line in text.split("\n")]
    cursor = 0
    line_starts: list[int] = [0]
    for _, ln in text_norm_lines:
        line_starts.append(line_starts[-1] + ln + 1)  # +1 for the \n we split on

    target = normed_search.rstrip("\n")
    target_lines = target.split("\n")
    n_target = len(target_lines)
    if n_target == 0:
        return []

    norm_lines = [nl for nl, _ in text_norm_lines]
    for i in range(len(norm_lines) - n_target + 1):
        if all(norm_lines[i + k] == target_lines[k] for k in range(n_target)):
            start = line_starts[i]
            end = line_starts[i + n_target] - 1 if i + n_target < len(line_starts) else len(text)
            # include trailing newline if search had one
            if normed_search.endswith("\n") and end < len(text) and text[end] == "\n":
                end += 1
            matches.append((start, end))
    _ = cursor  # silence unused
    return matches


def _normalize_ws(s: str) -> str:
    out_lines = []
    for line in s.split("\n"):
        # collapse runs of internal spaces/tabs to single space; strip trailing
        collapsed = re.sub(r"[ \t]+", " ", line).rstrip()
        out_lines.append(collapsed)
    return "\n".join(out_lines)
