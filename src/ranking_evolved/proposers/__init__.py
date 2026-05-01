"""Pluggable LLM proposers.

Importing this package eagerly imports every concrete proposer module so
their `@register_proposer` decorators populate `REGISTRY`. Anything that
reads `REGISTRY` (the CLI's membership check, `list-algorithms`, tests)
just needs to import this package; no lazy-import dance.
"""

from .base import REGISTRY, register_proposer

# Side-effect imports — order doesn't matter; each module registers itself.
from . import anthropic  # noqa: F401, E402
from . import claude_code  # noqa: F401, E402
from . import codex  # noqa: F401, E402
from . import ensemble  # noqa: F401, E402
from . import fake  # noqa: F401, E402
from . import openai_chat  # noqa: F401, E402
from . import scripted  # noqa: F401, E402

__all__ = ["REGISTRY", "register_proposer"]