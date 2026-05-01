"""Tests for .env loading + ${ENV} interpolation in config/loader.py."""
from __future__ import annotations

import os
from pathlib import Path

from ranking_evolved.config.loader import load_config


_MINIMAL_YAML = """\
task:
  seed: seed.py
  evaluator: evaluator.py

proposer:
  kind: openai_chat
  api_key: ${OPENAI_API_KEY}

search:
  algorithm: map_elites_islands
"""


def test_env_file_loaded_before_interpolation(tmp_path: Path, monkeypatch, record_io):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_MINIMAL_YAML)

    def run() -> dict:
        cfg = load_config(cfg_path)
        return {
            "api_key": cfg.proposer.api_key,
            "env_set": os.environ.get("OPENAI_API_KEY"),
        }

    out = record_io(
        module="src/ranking_evolved/config/loader.py",
        function="load_config (auto-load .env)",
        input={".env": "OPENAI_API_KEY=sk-from-dotenv"},
        run=run,
    )
    assert out["api_key"] == "sk-from-dotenv"
    assert out["env_set"] == "sk-from-dotenv"


def test_shell_env_takes_precedence_over_dotenv(tmp_path: Path, monkeypatch, record_io):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_MINIMAL_YAML)

    def run() -> str:
        cfg = load_config(cfg_path)
        return cfg.proposer.api_key

    out = record_io(
        module="src/ranking_evolved/config/loader.py",
        function="load_config (shell wins)",
        input={"shell": "sk-from-shell", ".env": "sk-from-dotenv"},
        run=run,
    )
    assert out == "sk-from-shell"


def test_dotenv_handles_quotes_comments_and_export(tmp_path: Path, monkeypatch, record_io):
    monkeypatch.delenv("MY_QUOTED", raising=False)
    monkeypatch.delenv("MY_EXPORTED", raising=False)
    monkeypatch.delenv("MY_COMMENTED", raising=False)
    (tmp_path / ".env").write_text(
        '# leading comment\n'
        'MY_QUOTED="hello world"\n'
        "export MY_EXPORTED=exported_value\n"
        "# MY_COMMENTED=ignored\n"
    )
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(
        "task: {seed: s.py, evaluator: e.py}\n"
        "proposer:\n"
        "  kind: openai_chat\n"
        "  api_key: ${MY_QUOTED}\n"
        "  api_base: ${MY_EXPORTED}\n"
        "search:\n  algorithm: map_elites_islands\n"
    )

    def run() -> dict:
        cfg = load_config(cfg_path)
        return {
            "quoted": cfg.proposer.api_key,
            "exported": cfg.proposer.api_base,
            "commented_present": "MY_COMMENTED" in os.environ,
        }

    out = record_io(
        module="src/ranking_evolved/config/loader.py",
        function="_load_env_file (quotes/export/comments)",
        input={".env": "quoted, export, comment"},
        run=run,
    )
    assert out == {
        "quoted": "hello world",
        "exported": "exported_value",
        "commented_present": False,
    }


def test_explicit_env_file_required_to_exist(tmp_path: Path, record_io):
    cfg_path = tmp_path / "task.yaml"
    cfg_path.write_text(_MINIMAL_YAML)

    def run() -> str:
        try:
            load_config(cfg_path, env_file=tmp_path / "missing.env")
            return "no_raise"
        except FileNotFoundError as exc:
            return str(exc)

    out = record_io(
        module="src/ranking_evolved/config/loader.py",
        function="load_config(env_file=...) (missing)",
        input={"env_file": "missing.env"},
        run=run,
    )
    assert "missing.env" in out and "does not exist" in out
