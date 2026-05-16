from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tasks._shared.datasets import BrightProLoader, EvalDataset, OBLIQBenchLoader
from tasks.late_interaction.embedding_cache import load_embedding_cache
from tasks.late_interaction.encode_embeddings import ensure_embedding_cache


def test_obliq_loader_reads_qrels_pooled_and_exclusions(monkeypatch, tmp_path: Path, record_io):
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td2\t2\n", encoding="utf-8")
    pooled = tmp_path / "qrels_pool.tsv"
    pooled.write_text("query-id\tcorpus-id\tscore\nq1\td2\t2\nq1\td3\t1\n", encoding="utf-8")
    excluded = tmp_path / "per_query_excluded_ids.json"
    excluded.write_text(json.dumps({"q1": ["d1"]}), encoding="utf-8")

    def fake_load_dataset(repo, config, split):
        assert repo == "dianetc/OBLIQ-Bench"
        assert config == "math"
        if split == "corpus":
            return [{"_id": "d1", "text": "source"}, {"_id": "d2", "text": "analogue"}]
        if split == "queries":
            return [{"_id": "q1", "text": "find analogues"}]
        raise AssertionError(split)

    def fake_hf_hub_download(*, repo_id, repo_type, filename):
        assert repo_id == "dianetc/OBLIQ-Bench"
        assert repo_type == "dataset"
        return str(tmp_path / Path(filename).name)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    def run():
        ds = OBLIQBenchLoader().load("math")
        return {
            "name": ds.name,
            "qrels": ds.qrels,
            "pooled": ds.qrels_by_mode["pooled"],
            "excluded": ds.excluded_ids,
        }

    result = record_io(
        module="tasks/_shared/datasets.py",
        function="OBLIQBenchLoader.load",
        input={"subset": "math"},
        run=run,
    )
    assert result["name"] == "obliq_math"
    assert result["qrels"] == {"q1": {"d2": 2}}
    assert result["pooled"] == {"q1": {"d2": 2, "d3": 1}}
    assert result["excluded"] == {"q1": ["d1"]}


def test_bright_pro_loader_normalizes_aspects_and_validates_gold(monkeypatch, record_io):
    def fake_load_dataset(repo, config, split):
        assert repo == "yale-nlp/Bright-Pro"
        assert split == "biology"
        if config == "examples":
            return [{"id": 0, "query": "why?", "gold_ids": ["d1", "d2", "d3"], "reference_answer": "a"}]
        if config == "aspects":
            return [
                {"id": "biology-0-a1", "content": "main", "weight": 3, "supporting_docs": ["d1", "d2"]},
                {"id": "biology-0-a2", "content": "side", "weight": 1, "supporting_docs": ["d3"]},
            ]
        if config == "documents":
            return [
                {"id": "d1", "content": "doc one"},
                {"id": "d2", "content": "doc two"},
                {"id": "d3", "content": "doc three"},
            ]
        raise AssertionError(config)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    def run():
        ds = BrightProLoader().load("biology")
        ann = ds.aspect_annotations
        assert ann is not None
        return {
            "name": ds.name,
            "qrels": ds.qrels,
            "weights": ann.query_aspect_weights["biology-0"],
            "doc_to_aspect": ann.query_doc_to_aspect["biology-0"],
        }

    result = record_io(
        module="tasks/_shared/datasets.py",
        function="BrightProLoader.load",
        input={"domain": "biology"},
        run=run,
    )
    assert result["name"] == "bright_pro_biology"
    assert result["qrels"] == {"biology-0": {"d1": 1, "d2": 1, "d3": 1}}
    assert result["weights"] == {"biology-0-a1": pytest.approx(0.75), "biology-0-a2": pytest.approx(0.25)}
    assert result["doc_to_aspect"] == {"d1": "biology-0-a1", "d2": "biology-0-a1", "d3": "biology-0-a2"}


def test_ensure_embedding_cache_uses_fake_encoder(monkeypatch, tmp_path: Path, record_io):
    dataset = EvalDataset(
        name="obliq_toy",
        benchmark="obliq",
        corpus=["doc"],
        corpus_ids=["d1"],
        queries=["query"],
        query_ids=["q1"],
        qrels={"q1": {"d1": 1}},
        excluded_ids={"q1": ["source"]},
    )

    class FakeModel:
        def encode(self, texts, *, is_query, batch_size, show_progress_bar):
            return [np.array([[1.0, 0.0]], dtype=np.float32) for _ in texts]

    monkeypatch.setattr(
        "tasks.late_interaction.encode_embeddings.load_eval_dataset",
        lambda spec, *, beir_data_dir: dataset,
    )

    def run():
        cache_dir = ensure_embedding_cache(
            "obliq_toy",
            cache_root=tmp_path,
            model_name="fake-model",
            dtype="float32",
            model=FakeModel(),
        )
        cache = load_embedding_cache(cache_dir)
        return {
            "cache_dir": cache_dir.name,
            "doc_ids": cache.docs.ids,
            "query_ids": cache.queries.ids,
            "excluded": cache.excluded_ids,
            "metadata": {
                "benchmark": cache.metadata.benchmark,
                "model_name": cache.metadata.model_name,
                "has_excluded_ids": cache.metadata.has_excluded_ids,
            },
        }

    result = record_io(
        module="tasks/late_interaction/encode_embeddings.py",
        function="ensure_embedding_cache",
        input={"dataset": "obliq_toy", "model": "fake-model"},
        run=run,
    )
    assert result["cache_dir"] == "obliq_toy"
    assert result["doc_ids"] == ["d1"]
    assert result["query_ids"] == ["q1"]
    assert result["excluded"] == {"q1": ["source"]}
    assert result["metadata"]["has_excluded_ids"] is True
