"""Search-strategy registry.

The registry pattern lets users (and tests) instantiate strategies by name
without `core.controller` having to import every concrete strategy.
"""
from __future__ import annotations

from typing import Any, Callable

REGISTRY: dict[str, Callable[..., Any]] = {}


def register_strategy(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a SearchStrategy factory under `name`."""

    def _wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = factory
        return factory

    return _wrap
