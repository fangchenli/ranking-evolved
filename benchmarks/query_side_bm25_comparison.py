"""
Query-Side BM25 Investigation.

Compares different query term handling strategies on BRIGHT:
- unique: Each unique term contributes once (our default)
- sum_all: Sum over all query term occurrences (Anserini BoW / Pyserini-style)
- saturated: Apply BM25 saturation to query TF (Query-Side BM25 from paper)

Reference: "Lighting the Way for BRIGHT" (Ge et al., 2025)
- BoW = our sum_all mode
- Query-Side BM25 = our saturated mode

Usage:
    uv run python -m benchmarks.query_side_bm25_comparison
    uv run python -m benchmarks.query_side_bm25_comparison --lucene
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import BM25Config, BM25Unified, Corpus, tokenize
from ranking_evolved.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
)


def get_lucene_tokenizer() -> Callable[[str], list[str]]:
    """Get Lucene tokenizer via Pyserini (requires Java 21)."""
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        lucene_analyzer = Analyzer(get_lucene_analyzer())
        return lucene_analyzer.analyze
    except ImportError:
        raise ImportError(
            "Pyserini is required for Lucene tokenization.\n"
            "Install with: uv sync --group benchmark\n"
            "Also requires Java 21:\n"
            "  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home\n"
            "  export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib"
        )


@dataclass
class QueryStats:
    """Statistics about query term repetition."""

    total_queries: int
    queries_with_repeats: int
    avg_query_length: float
    avg_unique_terms: float
    max_term_frequency: int
    queries_over_16_tokens: int
    queries_over_64_tokens: int
    queries_over_256_tokens: int


def analyze_query_repetition(queries: list[list[str]]) -> QueryStats:
    """Analyze query term repetition patterns."""
    queries_with_repeats = 0
    lengths = []
    unique_counts = []
    max_tf = 0

    over_16 = 0
    over_64 = 0
    over_256 = 0

    for query in queries:
        lengths.append(len(query))
        unique = len(set(query))
        unique_counts.append(unique)

        if len(query) > unique:
            queries_with_repeats += 1

        tf_counter = Counter(query)
        if tf_counter:
            max_tf = max(max_tf, max(tf_counter.values()))

        if len(query) > 16:
            over_16 += 1
        if len(query) > 64:
            over_64 += 1
        if len(query) > 256:
            over_256 += 1

    return QueryStats(
        total_queries=len(queries),
        queries_with_repeats=queries_with_repeats,
        avg_query_length=float(np.mean(lengths)) if lengths else 0,
        avg_unique_terms=float(np.mean(unique_counts)) if unique_counts else 0,
        max_term_frequency=max_tf,
        queries_over_16_tokens=over_16,
        queries_over_64_tokens=over_64,
        queries_over_256_tokens=over_256,
    )


@dataclass
class ModeResult:
    """Results for a query term mode."""

    mode: str
    ndcg_at_k: float
    map: float
    mrr: float


def evaluate_mode(
    corpus: Corpus,
    queries: list[list[str]],
    gold_indices: list[list[int]],
    mode: str,
    k1: float,
    b: float,
    k3: float,
    tf: str,
    k: int = 10,
) -> ModeResult:
    """Evaluate a single query term mode."""
    config = BM25Config(
        idf="lucene",
        tf=tf,
        query_mode=mode,
        k1=k1,
        b=b,
        k3=k3,
    )
    bm25 = BM25Unified(corpus, config)

    ndcg_scores = []
    all_relevant = []
    all_retrieved = []

    for query_tokens, gold in zip(queries, gold_indices, strict=False):
        ranked_indices, _ = bm25.rank(query_tokens)

        relevant = np.array(gold, dtype=np.int64)
        retrieved = np.array(ranked_indices, dtype=np.int64)

        all_relevant.append(relevant)
        all_retrieved.append(retrieved)
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))

    return ModeResult(
        mode=mode,
        ndcg_at_k=float(np.mean(ndcg_scores)),
        map=mean_average_precision(all_relevant, all_retrieved),
        mrr=mean_reciprocal_rank(all_relevant, all_retrieved),
    )


def main():
    parser = argparse.ArgumentParser(description="Query-Side BM25 Investigation")
    parser.add_argument(
        "--lucene",
        action="store_true",
        help="Use Lucene tokenizer (requires Java 21 + Pyserini)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="biology",
        help="BRIGHT domain to evaluate (default: biology)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("QUERY-SIDE BM25 INVESTIGATION")
    print("=" * 70)

    # Select tokenizer
    if args.lucene:
        print("\nUsing Lucene tokenizer (via Pyserini)")
        tokenizer_fn = get_lucene_tokenizer()
        tokenizer_name = "Lucene"
    else:
        print("\nUsing simple whitespace tokenizer")
        tokenizer_fn = tokenize
        tokenizer_name = "Simple"

    print(f"Domain: {args.domain}")
    print(f"Tokenizer: {tokenizer_name}")

    # Load data
    print(f"\nLoading {args.domain}...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split=args.domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=args.domain)

    # Build corpus
    raw_texts = []
    doc_ids = []
    for doc in documents:
        content = doc.get("content") or doc.get("text") or ""
        doc_id = doc.get("id") or doc.get("_id")
        raw_texts.append(content)
        doc_ids.append(doc_id)

    print(f"Tokenizing {len(raw_texts)} documents...")
    tokenized_docs = [tokenizer_fn(text) for text in raw_texts]
    corpus = Corpus(tokenized_docs, ids=doc_ids)

    # Tokenize queries
    queries = [tokenizer_fn(example["query"]) for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    # Analyze query patterns
    print("\n" + "=" * 70)
    print("QUERY ANALYSIS")
    print("=" * 70)

    stats = analyze_query_repetition(queries)
    print(f"\nTotal queries: {stats.total_queries}")
    print(
        f"Queries with repeated terms: {stats.queries_with_repeats} ({100 * stats.queries_with_repeats / stats.total_queries:.1f}%)"
    )
    print(f"Average query length: {stats.avg_query_length:.1f} tokens")
    print(f"Average unique terms: {stats.avg_unique_terms:.1f}")
    print(f"Max term frequency in any query: {stats.max_term_frequency}")
    print("\nQuery length distribution:")
    print(
        f"  > 16 tokens: {stats.queries_over_16_tokens} ({100 * stats.queries_over_16_tokens / stats.total_queries:.1f}%)"
    )
    print(
        f"  > 64 tokens: {stats.queries_over_64_tokens} ({100 * stats.queries_over_64_tokens / stats.total_queries:.1f}%)"
    )
    print(
        f"  > 256 tokens: {stats.queries_over_256_tokens} ({100 * stats.queries_over_256_tokens / stats.total_queries:.1f}%)"
    )

    # Paper's recommended range for query-side BM25: 16-256 tokens
    in_paper_range = stats.queries_over_16_tokens - stats.queries_over_256_tokens
    print(
        f"\nIn paper's recommended range (16-256 tokens): {in_paper_range} ({100 * in_paper_range / stats.total_queries:.1f}%)"
    )

    # Test different configurations
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    configurations = [
        # Paper's configuration (k1=0.9, b=0.4)
        {"name": "Paper params (k1=0.9, b=0.4)", "k1": 0.9, "b": 0.4, "tf": "classic"},
        # Our best evolved configuration
        {"name": "Evolved TF (k1=0.9, b=0.4)", "k1": 0.9, "b": 0.4, "tf": "evolved"},
    ]

    modes = ["unique", "sum_all", "saturated"]
    mode_descriptions = {
        "unique": "Unique terms only (our default)",
        "sum_all": "Sum all occurrences (Anserini BoW)",
        "saturated": "Query-Side BM25 (paper's approach)",
    }

    for config in configurations:
        print(f"\n--- {config['name']} ---")
        print(f"{'Mode':<40} {'NDCG@10':>10} {'MAP':>10} {'MRR':>10}")
        print("-" * 75)

        results = []
        for mode in modes:
            result = evaluate_mode(
                corpus=corpus,
                queries=queries,
                gold_indices=gold_indices,
                mode=mode,
                k1=config["k1"],
                b=config["b"],
                k3=8.0,  # Standard k3 for query-side saturation
                tf=config["tf"],
                k=10,
            )
            results.append(result)
            desc = mode_descriptions[mode]
            print(f"{desc:<40} {result.ndcg_at_k:>10.4f} {result.map:>10.4f} {result.mrr:>10.4f}")

        # Find best
        best = max(results, key=lambda r: r.ndcg_at_k)
        print(f"\n  Best mode: {best.mode} (NDCG@10 = {best.ndcg_at_k:.4f})")

    # Compare with paper's Table 1 (Biology, k1=0.9, b=0.4)
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER (Table 1)")
    print("=" * 70)
    print("\nPaper results for Biology (k1=0.9, b=0.4):")
    print("  Anserini BoW:          0.182 NDCG@10")
    print("  Anserini Query-Side:   0.197 NDCG@10")
    print("  BRIGHT Query-Side:     0.189 NDCG@10")
    print("\nNote: Paper uses Lucene tokenization. Our results may differ due to tokenizer.")

    # Additional analysis: impact of k3
    print("\n" + "=" * 70)
    print("K3 SENSITIVITY ANALYSIS (Query-Side BM25 mode)")
    print("=" * 70)

    k3_values = [1.0, 2.0, 4.0, 8.0, 16.0, 100.0]
    print(f"\n{'k3':<10} {'NDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 45)

    for k3 in k3_values:
        result = evaluate_mode(
            corpus=corpus,
            queries=queries,
            gold_indices=gold_indices,
            mode="saturated",
            k1=0.9,
            b=0.4,
            k3=k3,
            tf="evolved",
            k=10,
        )
        print(f"{k3:<10.1f} {result.ndcg_at_k:>10.4f} {result.map:>10.4f} {result.mrr:>10.4f}")

    print("\nNote: Higher k3 = more linear (less saturation)")
    print("      Lower k3 = faster saturation (approaches unique mode)")


if __name__ == "__main__":
    main()
