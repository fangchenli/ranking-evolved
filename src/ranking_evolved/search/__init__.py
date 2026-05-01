"""Pluggable search strategies."""

from .base import REGISTRY, register_strategy

__all__ = ["REGISTRY", "register_strategy"]
