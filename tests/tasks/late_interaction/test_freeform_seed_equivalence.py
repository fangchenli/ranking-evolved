"""Equivalence gate: the freeform seed must produce the same retrieval as
`tasks.late_interaction.library.ExactMaxSimRetriever`.

This is the M2 prerequisite — RankEvolve will mutate the seed, but the
*starting* program must agree with the exact-MaxSim correctness anchor on
every query of every dataset to within float32 noise. If this test fails,
either the seed was edited in a way that changed behavior, or the algorithm
itself has drifted.

The test runs on a tiny synthetic cache so it stays fast. The full-corpus
equivalence on the real BEIR/BRIGHT caches is checked at run time by
diffing `tasks/late_interaction/baselines/freeform.cpu.json` against
`exact_maxsim.cpu.json` (proven bit-identical earlier — see
`docs/late_interaction_evaluator_datasets.md` §2).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from tasks.late_interaction.embedding_cache import (
    TokenEmbeddingStore,
    build_metadata,
    write_embedding_cache,
)
from tasks.late_interaction.library import ExactMaxSimRetriever


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_PATH = REPO_ROOT / "tasks" / "late_interaction" / "seeds" / "freeform.py"


def _load_seed_retriever_class():
    """Load the freeform seed file via importlib (mirrors what evaluator does)."""
    import sys
    spec = importlib.util.spec_from_file_location("candidate_freeform_seed", SEED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_freeform_seed"] = module  # required for @dataclass-style introspection
    spec.loader.exec_module(module)
    return module.LateInteractionRetriever


def _make_synthetic(tmp_path: Path, num_docs: int = 32, num_queries: int = 8) -> Path:
    rng = np.random.default_rng(0)
    dim = 16
    doc_embeddings = [
        rng.standard_normal((rng.integers(4, 20), dim)).astype(np.float32)
        for _ in range(num_docs)
    ]
    query_embeddings = [
        rng.standard_normal((rng.integers(2, 8), dim)).astype(np.float32)
        for _ in range(num_queries)
    ]
    qrels = {f"q{i}": {f"d{i % num_docs}": 1} for i in range(num_queries)}
    metadata = build_metadata(
        dataset_name="synthetic",
        benchmark="synthetic",
        model_name="test",
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        dtype="float32",
    )
    write_embedding_cache(
        tmp_path,
        doc_embeddings=doc_embeddings,
        doc_ids=[f"d{i}" for i in range(num_docs)],
        query_embeddings=query_embeddings,
        query_ids=[f"q{i}" for i in range(num_queries)],
        qrels=qrels,
        metadata=metadata,
    )
    return tmp_path


def test_freeform_seed_matches_exact_maxsim_per_query(tmp_path):
    """Per-query rankings must match exact MaxSim within float32 score noise."""
    cache_dir = _make_synthetic(tmp_path)
    from tasks.late_interaction.embedding_cache import load_embedding_cache
    cache = load_embedding_cache(cache_dir)

    SeedCls = _load_seed_retriever_class()
    seed = SeedCls()
    em = ExactMaxSimRetriever()
    seed.build(cache.docs)
    em.build(cache.docs)

    seed_rankings = seed.search(cache.queries, top_k=10)
    em_rankings = em.search(cache.queries, top_k=10)

    assert set(seed_rankings.keys()) == set(em_rankings.keys()), \
        "seed and exact MaxSim returned different query sets"

    for qid in seed_rankings:
        s_rank = seed_rankings[qid]
        e_rank = em_rankings[qid]
        assert len(s_rank) == len(e_rank), f"top-k length mismatch on {qid}"
        # Order must match exactly (deterministic tie-break by doc_id).
        s_ids = [doc_id for doc_id, _ in s_rank]
        e_ids = [doc_id for doc_id, _ in e_rank]
        assert s_ids == e_ids, (
            f"ranking diverged on query {qid}\n"
            f"  seed:  {s_ids}\n"
            f"  exact: {e_ids}"
        )
        # Scores must agree within float32 precision.
        for (sd, ss), (ed, es) in zip(s_rank, e_rank, strict=True):
            assert sd == ed
            assert abs(ss - es) < 1e-4, f"score drift on {qid}/{sd}: seed={ss} exact={es}"
