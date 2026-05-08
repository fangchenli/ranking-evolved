"""Structural interfaces for late-interaction candidate programs."""

from __future__ import annotations

from typing import Protocol

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore


class LateInteractionRetriever(Protocol):
    """Evaluator-facing protocol for late-interaction retrievers.

    Candidate programs should keep this surface stable but may freely rewrite
    the internal retrieval algorithm.
    """

    def build(self, docs: TokenEmbeddingStore) -> None:
        """Build any in-memory or on-disk retrieval state from document embeddings."""

    def search(
        self,
        queries: TokenEmbeddingStore,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Return ranked ``(doc_id, score)`` pairs for every query."""


class LateInteractionCandidate(Protocol):
    """Module-level protocol for evolved late-interaction programs."""

    LateInteractionRetriever: type[LateInteractionRetriever]
