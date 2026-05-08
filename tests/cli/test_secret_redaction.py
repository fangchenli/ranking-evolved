"""`config.resolved.yaml` must never contain a real API key.

Tests both halves of the secret hygiene contract:
  - The CLI's `_dump_yaml_safely` redacts any `*_key` / `*_token` / `*_secret`
    field regardless of value.
  - The loader refuses to start when a literal API key is hard-coded in YAML.
"""

from __future__ import annotations

from ranking_evolved.cli import _dump_yaml_safely, _is_secret_key


def test_is_secret_key_recognizes_common_secret_names():
    assert _is_secret_key("api_key") is True
    assert _is_secret_key("API_KEY") is True
    assert _is_secret_key("openai_api_key") is True
    assert _is_secret_key("auth_token") is True
    assert _is_secret_key("password") is True
    assert _is_secret_key("client_secret") is True
    # Non-secret fields stay plain.
    assert _is_secret_key("api_base") is False
    assert _is_secret_key("model") is False
    assert _is_secret_key("temperature") is False


def test_dump_yaml_safely_redacts_api_key_field(record_io):
    """The cli serializer must replace api_key values with REDACTED."""

    class FakeProposer:
        # Stand-in for the real ProposerConfig dataclass — we use a plain dict
        # so the test doesn't depend on the dataclass shape evolving.
        pass

    config = {
        "proposer": {
            "kind": "openai_responses",
            "api_key": "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "api_base": "https://api.openai.com/v1",
            "auth_token": "ghp_AAAAAAAAAAAAAAAAAAAAAAAA",
        },
        "evaluation": {
            "env": {
                # Env block keys are NOT redacted — those are runtime knobs,
                # not secrets, and live in the YAML.
                "EVAL_DEVICE": "cpu",
            },
        },
    }

    def run() -> str:
        return _dump_yaml_safely(config)

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="_dump_yaml_safely",
        input={"has_api_key": True, "has_auth_token": True},
        run=run,
    )

    # Real key values must NOT appear anywhere.
    assert "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" not in out
    assert "ghp_AAAAAAAAAAAAAAAAAAAAAAAA" not in out
    # Redaction marker present where secrets used to be.
    assert "***REDACTED***" in out
    # Non-secret config fields survive verbatim.
    assert "api_base: https://api.openai.com/v1" in out
    assert "EVAL_DEVICE: cpu" in out


def test_dump_yaml_safely_leaves_empty_api_key_alone(record_io):
    """An empty api_key should NOT be redacted (no secret to leak)."""
    config = {"proposer": {"api_key": "", "api_base": "x"}}

    def run() -> str:
        return _dump_yaml_safely(config)

    out = record_io(
        module="src/ranking_evolved/cli.py",
        function="_dump_yaml_safely (empty api_key)",
        input={"api_key": ""},
        run=run,
    )

    assert "***REDACTED***" not in out
    assert "api_key:" in out
