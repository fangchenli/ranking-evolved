"""
Benchmark all BM25 variants on BRIGHT biology dataset.

Usage:
    uv run python -m benchmarks.run_all_variants
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import BM25Config, BM25Unified, Corpus, tokenize
from ranking_evolved.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k


def evaluate_config(
    corpus: Corpus,
    queries: list[str],
    gold_indices: list[list[int]],
    config: BM25Config,
    k: int = 10,
) -> dict:
    """Evaluate a BM25 configuration."""
    bm25 = BM25Unified(corpus, config)

    ndcg_scores = []
    all_relevant = []
    all_retrieved = []

    for query_text, gold in zip(queries, gold_indices, strict=False):
        query_tokens = tokenize(query_text)
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


def main():
    domain = "biology"
    k = 10

    print(f"Loading {domain} dataset...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)

    print("Building corpus...")
    corpus = Corpus.from_huggingface_dataset(documents)

    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    print(f"Corpus: {len(corpus)} documents, {len(queries)} queries")

    # Define all configurations to test
    configurations = [
        # Preset configurations
        ("Classic (k1=1.2, b=0.75)", BM25Config.classic()),
        ("Classic (k1=1.5, b=0.75)", BM25Config.classic(k1=1.5, b=0.75)),
        ("Lucene (k1=0.9, b=0.4)", BM25Config.lucene()),
        ("Lucene (k1=1.2, b=0.75)", BM25Config.lucene(k1=1.2, b=0.75)),
        ("ATIRE (k1=1.2, b=0.75)", BM25Config.atire()),
        ("BM25L (k1=1.2, b=0.75, δ=0.5)", BM25Config.bm25l()),
        ("BM25+ (k1=1.2, b=0.75, δ=1.0)", BM25Config.bm25_plus()),
        ("Pyserini-style (k1=0.9, b=0.4)", BM25Config.pyserini()),
        ("Evolved (k1=1.5, b=0.75)", BM25Config.evolved()),
        # Evolved TF with Lucene IDF (best combination)
        (
            "Lucene IDF + Evolved TF (k1=0.9, b=0.4)",
            BM25Config(idf="lucene", tf="evolved", query_mode="unique", k1=0.9, b=0.4),
        ),
        (
            "Lucene IDF + Evolved TF (k1=1.5, b=0.75)",
            BM25Config(idf="lucene", tf="evolved", query_mode="unique", k1=1.5, b=0.75),
        ),
        # Clipped IDF variants
        (
            "Clipped IDF + Classic TF (k1=1.5, b=0.75)",
            BM25Config(idf="clipped", tf="classic", query_mode="unique", k1=1.5, b=0.75),
        ),
        # Query mode comparison with Lucene IDF
        (
            "Lucene + sum_all (k1=0.9, b=0.4)",
            BM25Config(idf="lucene", tf="classic", query_mode="sum_all", k1=0.9, b=0.4),
        ),
        (
            "Lucene + saturated (k1=0.9, b=0.4)",
            BM25Config(idf="lucene", tf="classic", query_mode="saturated", k1=0.9, b=0.4),
        ),
    ]

    results = {}

    for name, config in configurations:
        print(f"Evaluating {name}...")
        result = evaluate_config(corpus, queries, gold_indices, config, k)
        results[name] = result
        print(f"  NDCG@{k}: {result['ndcg_at_k']:.4f}, MAP: {result['map']:.4f}, MRR: {result['mrr']:.4f}")

    # Print markdown table
    print("\n" + "=" * 80)
    print("RESULTS TABLE (Markdown)")
    print("=" * 80)
    print("| Configuration | NDCG@10 | MAP | MRR |")
    print("|--------------|---------|-----|-----|")

    # Sort by NDCG descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["ndcg_at_k"], reverse=True)
    for name, r in sorted_results:
        print(f"| {name} | {r['ndcg_at_k']:.4f} | {r['map']:.4f} | {r['mrr']:.4f} |")

    # Save results
    output_path = Path("benchmarks/all_variants_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
