"""
Cross-validation script for BM25 implementations.

This script compares different BM25 implementations across tokenization methods
to validate correctness and identify performance differences.

Requires:
    - Java 21 for Pyserini
    - uv sync --group benchmark

Usage:
    # Set Java environment
    export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib

    # Run cross-validation
    uv run python -m benchmarks.cross_validation
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from datasets import load_dataset
from ranking_evolved.bm25 import (
    BM25Config,
    BM25Unified,
    Corpus,
)
from ranking_evolved.bm25 import (
    tokenize as simple_tokenize,
)
from ranking_evolved.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def get_lucene_tokenizer() -> Callable[[str], list[str]]:
    """Get Lucene tokenizer via Pyserini (requires Java)."""
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        lucene_analyzer = Analyzer(get_lucene_analyzer())
        return lucene_analyzer.analyze
    except ImportError:
        raise ImportError(
            "Pyserini is required for Lucene tokenization. Install with: uv sync --group benchmark"
        )


def evaluate_bm25_unified(
    corpus: Corpus,
    queries: list[str],
    gold_indices: list[list[int]],
    tokenizer: Callable[[str], list[str]],
    config: BM25Config,
    k: int = 10,
) -> dict:
    """Evaluate BM25Unified with a specific configuration."""
    bm25 = BM25Unified(corpus, config)

    ndcg_scores = []
    all_relevant = []
    all_retrieved = []

    for query_text, gold in zip(queries, gold_indices, strict=False):
        query_tokens = tokenizer(query_text)
        ranked_indices, _ = bm25.rank(query_tokens)

        relevant = np.array(gold, dtype=np.int64)
        retrieved = np.array(ranked_indices, dtype=np.int64)

        all_relevant.append(relevant)
        all_retrieved.append(retrieved)
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))

    return {
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "map": mean_average_precision(all_relevant, all_retrieved),
        "mrr": mean_reciprocal_rank(all_relevant, all_retrieved),
    }


def evaluate_pyserini_raw(
    raw_texts: list[str],
    doc_ids: list[str],
    queries: list[str],
    gold_id_lists: list[list[str]],
    k1: float,
    b: float,
    k: int = 10,
) -> dict:
    """Evaluate Pyserini with raw text (proper usage)."""
    try:
        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        return {"error": "Pyserini not available"}

    index_dir = Path(tempfile.mkdtemp(prefix="pyserini_cv_"))

    try:
        # Build index with raw text
        indexer = LuceneIndexer(str(index_dir / "index"), append=False)
        batch_size = 1000
        for i in range(0, len(doc_ids), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(doc_ids))):
                batch.append({"id": doc_ids[j], "contents": raw_texts[j]})
            indexer.add_batch_dict(batch)
        indexer.close()

        # Build ID mapping
        id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        # Search
        searcher = LuceneSearcher(str(index_dir / "index"))
        searcher.set_bm25(k1, b)

        ndcg_scores = []
        all_relevant = []
        all_retrieved = []

        for query_text, gold_ids in zip(queries, gold_id_lists, strict=False):
            gold_indices = [id_to_idx[gid] for gid in gold_ids if gid in id_to_idx]
            hits = searcher.search(query_text, k=len(doc_ids))

            # Convert to indices
            retrieved = []
            seen = set()
            for hit in hits:
                if hit.docid in id_to_idx:
                    idx = id_to_idx[hit.docid]
                    if idx not in seen:
                        retrieved.append(idx)
                        seen.add(idx)

            # Pad with remaining docs
            for idx in range(len(doc_ids)):
                if idx not in seen:
                    retrieved.append(idx)

            relevant = np.array(gold_indices, dtype=np.int64)
            retrieved = np.array(retrieved, dtype=np.int64)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))

        return {
            "ndcg_at_k": float(np.mean(ndcg_scores)),
            "map": mean_average_precision(all_relevant, all_retrieved),
            "mrr": mean_reciprocal_rank(all_relevant, all_retrieved),
        }

    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


def run_cross_validation(domain: str = "biology", k: int = 10) -> dict:
    """
    Run cross-validation comparing implementations and tokenizers.

    Args:
        domain: BRIGHT dataset domain to evaluate.
        k: Cutoff for @k metrics.

    Returns:
        Dictionary of results.
    """
    print(f"Loading {domain} dataset...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)

    # Extract raw texts and IDs
    raw_texts = []
    doc_ids = []
    for doc in documents:
        content = doc.get("content") or doc.get("text") or ""
        doc_id = doc.get("id") or doc.get("_id")
        raw_texts.append(content)
        doc_ids.append(doc_id)

    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]

    results = {"domain": domain, "k": k, "num_queries": len(queries)}

    # 1. Our BM25 with simple tokenization - Classic TF
    print("Building corpus with simple tokenization...")
    corpus_simple = Corpus.from_huggingface_dataset(documents)
    gold_indices_simple = [corpus_simple.id_to_idx(ids) for ids in gold_id_lists]

    # Test classic TF with simple tokenization
    for k1, b in [(0.9, 0.4), (1.2, 0.75), (1.5, 0.75)]:
        config = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=k1, b=b)
        key = f"simple_classic_k1={k1}_b={b}"
        print(f"Evaluating {key}...")
        results[key] = evaluate_bm25_unified(
            corpus_simple, queries, gold_indices_simple, simple_tokenize, config, k
        )

    # Test evolved TF with simple tokenization
    for k1, b in [(0.9, 0.4), (1.5, 0.75)]:
        config = BM25Config(idf="lucene", tf="evolved", query_mode="unique", k1=k1, b=b)
        key = f"simple_evolved_k1={k1}_b={b}"
        print(f"Evaluating {key}...")
        results[key] = evaluate_bm25_unified(
            corpus_simple, queries, gold_indices_simple, simple_tokenize, config, k
        )

    # 2. Try Lucene tokenization if available
    try:
        tokenize_lucene = get_lucene_tokenizer()

        print("Building corpus with Lucene tokenization...")
        tokenized_docs_lucene = [tokenize_lucene(text) for text in raw_texts]
        corpus_lucene = Corpus(tokenized_docs_lucene, doc_ids)
        gold_indices_lucene = [corpus_lucene.id_to_idx(ids) for ids in gold_id_lists]

        # Test classic TF with Lucene tokenization
        for k1, b in [(0.9, 0.4), (1.2, 0.75)]:
            config = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=k1, b=b)
            key = f"lucene_classic_k1={k1}_b={b}"
            print(f"Evaluating {key}...")
            results[key] = evaluate_bm25_unified(
                corpus_lucene, queries, gold_indices_lucene, tokenize_lucene, config, k
            )

        # Test evolved TF with Lucene tokenization
        for k1, b in [(0.9, 0.4), (1.5, 0.75)]:
            config = BM25Config(idf="lucene", tf="evolved", query_mode="unique", k1=k1, b=b)
            key = f"lucene_evolved_k1={k1}_b={b}"
            print(f"Evaluating {key}...")
            results[key] = evaluate_bm25_unified(
                corpus_lucene, queries, gold_indices_lucene, tokenize_lucene, config, k
            )

        # Test Pyserini-style (sum_all) with Lucene tokenization
        for k1, b in [(0.9, 0.4)]:
            config = BM25Config(idf="lucene", tf="classic", query_mode="sum_all", k1=k1, b=b)
            key = f"lucene_pyserini_style_k1={k1}_b={b}"
            print(f"Evaluating {key}...")
            results[key] = evaluate_bm25_unified(
                corpus_lucene, queries, gold_indices_lucene, tokenize_lucene, config, k
            )

        # 3. Pyserini with raw text (reference implementation)
        for k1, b in [(0.9, 0.4), (1.2, 0.75)]:
            key = f"pyserini_raw_k1={k1}_b={b}"
            print(f"Evaluating {key}...")
            results[key] = evaluate_pyserini_raw(
                raw_texts, doc_ids, queries, gold_id_lists, k1, b, k
            )

    except ImportError as e:
        print(f"Skipping Lucene/Pyserini tests: {e}")

    return results


def print_results(results: dict) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION RESULTS: {results['domain']} Domain")
    print("=" * 80)
    print(f"{'Implementation':<45} {'NDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 80)

    # Sort by NDCG descending
    items = [
        (key, value)
        for key, value in results.items()
        if isinstance(value, dict) and "ndcg_at_k" in value
    ]
    items.sort(key=lambda x: x[1]["ndcg_at_k"], reverse=True)

    for key, value in items:
        print(f"{key:<45} {value['ndcg_at_k']:>10.4f} {value['map']:>10.4f} {value['mrr']:>10.4f}")

    print("-" * 80)
    print(f"Queries: {results['num_queries']}")


if __name__ == "__main__":
    results = run_cross_validation(domain="biology", k=10)
    print_results(results)

    # Save results
    output_path = Path("benchmarks/cross_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
