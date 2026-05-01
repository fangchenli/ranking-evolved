"""Proposer registry."""
from __future__ import annotations

from typing import Any, Callable

REGISTRY: dict[str, Callable[..., Any]] = {}


def register_proposer(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a Proposer factory under `name`."""

    def _wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = factory
        return factory

    return _wrap
