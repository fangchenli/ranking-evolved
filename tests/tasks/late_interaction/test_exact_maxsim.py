from __future__ import annotations

import numpy as np

from tasks.late_interaction.embedding_cache import TokenEmbeddingStore
from tasks.late_interaction.library import (
    ExactMaxSimRetriever,
    exact_maxsim_score,
    rank_exact_maxsim,
)


def test_exact_maxsim_matches_manual_score(record_io):
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)

    score = record_io(
        module="tasks/late_interaction/library.py",
        function="exact_maxsim_score",
        input={"query": query.tolist(), "doc": doc.tolist()},
        run=lambda: exact_maxsim_score(query, doc),
    )

    assert score == np.float32(0.8 + 0.9)


def test_rank_exact_maxsim_is_deterministic(record_io):
    docs = TokenEmbeddingStore(
        ids=["d2", "d1", "d3"],
        embeddings=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        lengths=np.array([1, 1, 1], dtype=np.int64),
        offsets=np.array([0, 1, 2], dtype=np.int64),
    )
    queries = TokenEmbeddingStore(
        ids=["q1"],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        lengths=np.array([1], dtype=np.int64),
        offsets=np.array([0], dtype=np.int64),
    )

    ranking = record_io(
        module="tasks/late_interaction/library.py",
        function="rank_exact_maxsim",
        input={"query_ids": queries.ids, "doc_ids": docs.ids, "top_k": 3},
        run=lambda: rank_exact_maxsim(queries, docs, top_k=3),
    )

    assert ranking == {"q1": [("d1", 1.0), ("d2", 1.0), ("d3", 0.0)]}


def test_exact_maxsim_retriever_reports_diagnostics():
    docs = TokenEmbeddingStore(
        ids=["d1"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        lengths=np.array([2], dtype=np.int64),
        offsets=np.array([0], dtype=np.int64),
    )
    queries = TokenEmbeddingStore(
        ids=["q1"],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        lengths=np.array([1], dtype=np.int64),
        offsets=np.array([0], dtype=np.int64),
    )

    retriever = ExactMaxSimRetriever()
    retriever.build(docs)
    ranking = retriever.search(queries, top_k=1)

    assert ranking == {"q1": [("d1", 1.0)]}
    assert retriever.last_diagnostics["q1"].documents_scored == 1
    assert retriever.last_diagnostics["q1"].query_tokens_used == 1
    assert retriever.last_diagnostics["q1"].document_tokens_loaded == 2
