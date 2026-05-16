from __future__ import annotations

import json

import numpy as np

from tasks.late_interaction.embedding_cache import build_metadata, write_embedding_cache
from tasks.late_interaction.evaluator import run_evaluation
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


def test_worker_filters_excluded_ids_and_reports_aspect_metrics(record_io, tmp_path):
    doc_embeddings = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.9, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
    ]
    query_embeddings = [np.array([[1.0, 0.0]], dtype=np.float32)]
    metadata = build_metadata(
        dataset_name="toy_aspects",
        benchmark="synthetic",
        model_name="toy-model",
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        dtype="float32",
        has_excluded_ids=True,
        has_aspect_annotations=True,
    )
    write_embedding_cache(
        tmp_path,
        doc_embeddings=doc_embeddings,
        doc_ids=["source", "relevant", "other"],
        query_embeddings=query_embeddings,
        query_ids=["q1"],
        qrels={"q1": {"relevant": 1}},
        excluded_ids={"q1": ["source"]},
        aspect_annotations={
            "query_aspect_weights": {"q1": {"a1": 1.0}},
            "query_doc_to_aspect": {"q1": {"relevant": "a1"}},
            "query_aspect_content": {"q1": {"a1": "the relevant aspect"}},
        },
        metadata=metadata,
    )

    result = record_io(
        module="tasks/late_interaction/evaluator_worker.py",
        function="evaluate_cache_dataset (excluded ids + aspects)",
        input={"dataset": "toy_aspects", "recall_k": 1, "ndcg_k": 1},
        run=lambda: evaluate_cache_dataset(
            cache_dir=tmp_path,
            program_path=None,
            sample_queries=1,
            recall_k=1,
            ndcg_k=1,
            warmup_queries=0,
            timed_repeats=1,
        ).to_metrics(),
    )

    assert result["toy_aspects_recall_at_1"] == 1.0
    assert result["toy_aspects_ndcg_at_1"] == 1.0
    assert result["toy_aspects_alpha_ndcg_at_1"] == 1.0
    assert result["toy_aspects_aspect_recall_at_1"] == 1.0


def test_run_evaluation_writes_baseline_compatible_json(record_io, tmp_path):
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "toy"
    doc_embeddings = [np.array([[1.0, 0.0]], dtype=np.float32)]
    query_embeddings = [np.array([[1.0, 0.0]], dtype=np.float32)]
    metadata = build_metadata(
        dataset_name="toy",
        benchmark="synthetic",
        model_name="toy-model",
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        dtype="float32",
    )
    write_embedding_cache(
        cache_dir,
        doc_embeddings=doc_embeddings,
        doc_ids=["d1"],
        query_embeddings=query_embeddings,
        query_ids=["q1"],
        qrels={"q1": {"d1": 1}},
        metadata=metadata,
    )
    output = tmp_path / "exact_maxsim.toy.gold.cpu.json"

    result = record_io(
        module="tasks/late_interaction/evaluator.py",
        function="run_evaluation baseline JSON",
        input={"datasets": ["toy"], "recall_k": 1, "ndcg_k": 1},
        run=lambda: run_evaluation(
            program_path=None,
            datasets=["toy"],
            cache_root=cache_root,
            sample_queries=1,
            warmup_queries=0,
            timed_repeats=1,
            recall_k=1,
            ndcg_k=1,
            auto_cache=False,
            output_path=output,
            resume=False,
        ),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.datasets == ["toy"]
    assert payload["toy"]["median_query_latency_ms"] >= 0.0
    assert payload["toy"]["recall_at_1"] == 1.0
    assert payload["_average"]["ndcg_at_1"] == 1.0
    assert payload["toy"]["cache_metadata"]["dataset_name"] == "toy"
