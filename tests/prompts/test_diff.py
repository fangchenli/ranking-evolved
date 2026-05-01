"""Tests for prompts.diff: SEARCH/REPLACE applies, fails loudly on ambiguity."""
from __future__ import annotations

import pytest

from ranking_evolved.prompts.diff import DiffError, apply_search_replace


def _wrap(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE"


def test_apply_single_block(record_io):
    parent = "def f():\n    return 1\n"
    raw = _wrap("    return 1\n", "    return 2\n")

    def run() -> dict:
        new, app = apply_search_replace(parent, raw)
        return {
            "new_text": new,
            "n_extracted": app.n_extracted,
            "n_applied": app.n_applied,
            "matched_at_line": app.blocks[0].matched_at_line,
            "fatal_error": app.fatal_error,
        }

    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace",
        input={"parent": parent, "raw": raw},
        run=run,
    )
    assert out == {
        "new_text": "def f():\n    return 2\n",
        "n_extracted": 1,
        "n_applied": 1,
        "matched_at_line": 2,
        "fatal_error": None,
    }


def test_multi_block_in_order(record_io):
    parent = "a = 1\nb = 2\nc = 3\n"
    raw = _wrap("a = 1\n", "a = 10\n") + "\n" + _wrap("c = 3\n", "c = 30\n")

    def run() -> dict:
        new, app = apply_search_replace(parent, raw)
        return {"new": new, "n_applied": app.n_applied}

    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace",
        input={"parent": parent, "n_blocks": 2},
        run=run,
    )
    assert out == {"new": "a = 10\nb = 2\nc = 30\n", "n_applied": 2}


def test_no_match_raises(record_io):
    parent = "def f():\n    return 1\n"
    raw = _wrap("    return 99\n", "    return 2\n")

    def run() -> dict:
        try:
            apply_search_replace(parent, raw)
            return {"raised": False}
        except DiffError as exc:
            return {
                "raised": True,
                "fatal_error": exc.application.fatal_error,
                "block_error": exc.application.blocks[0].error,
            }

    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace",
        input={"parent": parent, "scenario": "search text not found"},
        run=run,
    )
    assert out == {
        "raised": True,
        "fatal_error": "search text not found",
        "block_error": "search text not found",
    }


def test_ambiguous_match_raises_even_with_fuzzy(record_io):
    parent = "x = 0\nx = 0\n"
    raw = _wrap("x = 0\n", "x = 1\n")

    def run() -> dict:
        results: list[str] = []
        for fuzzy in (False, True):
            try:
                apply_search_replace(parent, raw, fuzzy=fuzzy)
                results.append(f"fuzzy={fuzzy}: did not raise")
            except DiffError as exc:
                results.append(f"fuzzy={fuzzy}: {exc.application.blocks[0].error}")
        return {"results": results}

    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace",
        input={"parent": parent, "scenario": "duplicate search text -> always raise"},
        run=run,
    )
    assert out == {
        "results": [
            "fuzzy=False: ambiguous match (2 candidates)",
            "fuzzy=True: ambiguous match (2 candidates)",
        ],
    }


def test_fuzzy_succeeds_on_unique_whitespace_variant(record_io):
    parent = "def f():\n    return  1\n"  # double space before 1
    raw = _wrap("    return 1\n", "    return 2\n")  # diff has single space

    def run() -> dict:
        # exact match fails -> raises
        with pytest.raises(DiffError):
            apply_search_replace(parent, raw, fuzzy=False)
        # fuzzy normalizes whitespace and finds the match
        new, app = apply_search_replace(parent, raw, fuzzy=True)
        return {"new": new, "n_applied": app.n_applied}

    out = record_io(
        module="src/ranking_evolved/prompts/diff.py",
        function="apply_search_replace[fuzzy=True]",
        input={"parent_has_double_space": True, "diff_has_single_space": True},
        run=run,
    )
    assert out == {"new": "def f():\n    return 2\n", "n_applied": 1}
