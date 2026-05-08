from __future__ import annotations

import numpy as np
import pytest

from tasks.late_interaction.embedding_cache import (
    build_metadata,
    load_embedding_cache,
    write_embedding_cache,
)


def test_embedding_cache_roundtrip(record_io, tmp_path):
    doc_embeddings = [
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
    ]
    query_embeddings = [np.array([[1.0, 0.0]], dtype=np.float32)]
    metadata = build_metadata(
        dataset_name="toy",
        benchmark="synthetic",
        model_name="toy-model",
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        dtype="float32",
    )

    def run():
        write_embedding_cache(
            tmp_path,
            doc_embeddings=doc_embeddings,
            doc_ids=["d1", "d2"],
            query_embeddings=query_embeddings,
            query_ids=["q1"],
            qrels={"q1": {"d1": 1}},
            metadata=metadata,
        )
        cache = load_embedding_cache(tmp_path)
        return {
            "doc_ids": cache.docs.ids,
            "query_ids": cache.queries.ids,
            "doc0": cache.docs.get_by_id("d1").tolist(),
            "query0": cache.queries.get_by_id("q1").tolist(),
            "qrels": cache.qrels,
            "metadata": cache.metadata.dataset_name,
        }

    result = record_io(
        module="tasks/late_interaction/embedding_cache.py",
        function="write_embedding_cache/load_embedding_cache",
        input={"doc_ids": ["d1", "d2"], "query_ids": ["q1"]},
        run=run,
    )

    assert result["doc_ids"] == ["d1", "d2"]
    assert result["query_ids"] == ["q1"]
    assert result["doc0"] == [[1.0, 0.0], [0.5, 0.5]]
    assert result["query0"] == [[1.0, 0.0]]
    assert result["qrels"] == {"q1": {"d1": 1}}
    assert result["metadata"] == "toy"


def test_embedding_cache_rejects_bad_offsets():
    from tasks.late_interaction.embedding_cache import TokenEmbeddingStore

    with pytest.raises(ValueError, match="offsets"):
        TokenEmbeddingStore(
            ids=["a", "b"],
            embeddings=np.zeros((3, 2), dtype=np.float32),
            lengths=np.array([1, 2], dtype=np.int64),
            offsets=np.array([0, 2], dtype=np.int64),
        )
