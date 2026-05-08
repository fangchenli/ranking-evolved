from __future__ import annotations

import numpy as np

from tasks.late_interaction.embedding_cache import build_metadata, write_embedding_cache
from tasks.late_interaction.evaluator_worker import evaluate_cache_dataset


def test_smoke_worker_returns_recall_ndcg_and_latency(record_io, tmp_path):
    doc_embeddings = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    ]
    query_embeddings = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
    ]
    metadata = build_metadata(
        dataset_name="toy",
        benchmark="synthetic",
        model_name="toy-model",
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        dtype="float32",
    )
    write_embedding_cache(
        tmp_path,
        doc_embeddings=doc_embeddings,
        doc_ids=["d1", "d2", "d3"],
        query_embeddings=query_embeddings,
        query_ids=["q1", "q2"],
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
        metadata=metadata,
    )

    result = record_io(
        module="tasks/late_interaction/evaluator_worker.py",
        function="evaluate_cache_dataset",
        input={"dataset": "toy", "sample_queries": 2, "recall_k": 1000, "ndcg_k": 10},
        run=lambda: evaluate_cache_dataset(
            cache_dir=tmp_path,
            program_path=None,
            sample_queries=2,
            recall_k=1000,
            ndcg_k=10,
            warmup_queries=0,
        ).to_metrics(),
    )

    assert result["toy_recall_at_1000"] == 1.0
    assert result["toy_ndcg_at_10"] == 1.0
    assert result["toy_latency_p50_ms"] >= 0.0
    assert result["toy_num_queries"] == 2.0
