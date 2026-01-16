"""
Full BRIGHT evaluation using our best BM25 configuration.

Evaluates BM25Unified with lucene IDF + evolved TF (k1=0.9, b=0.4) across all
BRIGHT domains and produces a comprehensive results table.

Usage:
    # With simple tokenizer (default)
    uv run python -m benchmarks.full_bright_evaluation

    # With Lucene tokenizer (requires Java 21 + Pyserini)
    uv run python -m benchmarks.full_bright_evaluation --lucene
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import BM25Config, BM25Unified, Corpus, tokenize
from ranking_evolved.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
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

BRIGHT_SPLITS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]


@dataclass
class DomainResult:
    """Results for a single domain."""

    domain: str
    num_queries: int
    num_documents: int
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    map: float
    mrr: float
    combined: float


def evaluate_domain(
    domain: str,
    config: BM25Config,
    tokenizer_fn: Callable[[str], list[str]],
    k: int = 10,
) -> DomainResult:
    """Evaluate BM25 on a single BRIGHT domain."""
    print(f"  Loading {domain}...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)

    # Build corpus with the specified tokenizer
    raw_texts = []
    doc_ids = []
    for doc in documents:
        content = doc.get("content") or doc.get("text") or ""
        doc_id = doc.get("id") or doc.get("_id")
        raw_texts.append(content)
        doc_ids.append(doc_id)

    # Tokenize documents
    print(f"  Tokenizing {len(raw_texts)} documents...")
    tokenized_docs = [tokenizer_fn(text) for text in raw_texts]
    corpus = Corpus(tokenized_docs, ids=doc_ids)

    bm25 = BM25Unified(corpus, config)

    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    print(f"  Evaluating {len(queries)} queries on {len(corpus)} documents...")

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    all_relevant = []
    all_retrieved = []

    for query_text, gold in zip(queries, gold_indices, strict=False):
        query_tokens = tokenizer_fn(query_text)
        ranked_indices, _ = bm25.rank(query_tokens)

        relevant = np.array(gold, dtype=np.int64)
        retrieved = np.array(ranked_indices, dtype=np.int64)

        all_relevant.append(relevant)
        all_retrieved.append(retrieved)

        precision_scores.append(precision_at_k(relevant, retrieved, k))
        recall_scores.append(recall_at_k(relevant, retrieved, k))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))

    ndcg = float(np.mean(ndcg_scores))
    precision = float(np.mean(precision_scores))
    recall = float(np.mean(recall_scores))
    map_score = mean_average_precision(all_relevant, all_retrieved)
    mrr = mean_reciprocal_rank(all_relevant, all_retrieved)
    combined = float(np.mean([ndcg, precision, recall, map_score, mrr]))

    return DomainResult(
        domain=domain,
        num_queries=len(queries),
        num_documents=len(corpus),
        ndcg_at_k=ndcg,
        precision_at_k=precision,
        recall_at_k=recall,
        map=map_score,
        mrr=mrr,
        combined=combined,
    )


def main():
    parser = argparse.ArgumentParser(description="Full BRIGHT evaluation with best BM25 config")
    parser.add_argument(
        "--lucene",
        action="store_true",
        help="Use Lucene tokenizer (requires Java 21 + Pyserini)",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="unique",
        choices=["unique", "sum_all", "saturated"],
        help="Query term mode (default: unique)",
    )
    parser.add_argument(
        "--k3",
        type=float,
        default=8.0,
        help="k3 parameter for saturated query mode (default: 8.0)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FULL BRIGHT EVALUATION")
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

    # Our best configuration
    config = BM25Config(
        idf="lucene",
        tf="evolved",
        query_mode=args.query_mode,
        k1=0.9,
        b=0.4,
        k3=args.k3,
    )
    k = 10

    print(f"Configuration: {config}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"k = {k}")
    print()

    results: list[DomainResult] = []

    for i, domain in enumerate(BRIGHT_SPLITS):
        print(f"[{i + 1}/{len(BRIGHT_SPLITS)}] {domain}")
        result = evaluate_domain(domain, config, tokenizer_fn, k)
        results.append(result)
        print(f"  NDCG@{k}: {result.ndcg_at_k:.4f}, MAP: {result.map:.4f}, MRR: {result.mrr:.4f}")
        print()

    # Print results table
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print()
    print(
        f"{'Domain':<25} {'Queries':>8} {'Docs':>8} {'NDCG@10':>10} {'P@10':>8} {'R@10':>8} {'MAP':>8} {'MRR':>8} {'Combined':>10}"
    )
    print("-" * 105)

    for r in results:
        print(
            f"{r.domain:<25} {r.num_queries:>8} {r.num_documents:>8} "
            f"{r.ndcg_at_k:>10.4f} {r.precision_at_k:>8.4f} {r.recall_at_k:>8.4f} "
            f"{r.map:>8.4f} {r.mrr:>8.4f} {r.combined:>10.4f}"
        )

    # Compute macro averages
    print("-" * 105)
    macro_ndcg = np.mean([r.ndcg_at_k for r in results])
    macro_precision = np.mean([r.precision_at_k for r in results])
    macro_recall = np.mean([r.recall_at_k for r in results])
    macro_map = np.mean([r.map for r in results])
    macro_mrr = np.mean([r.mrr for r in results])
    macro_combined = np.mean([r.combined for r in results])
    total_queries = sum(r.num_queries for r in results)
    total_docs = sum(r.num_documents for r in results)

    print(
        f"{'**MACRO AVG**':<25} {total_queries:>8} {total_docs:>8} "
        f"{macro_ndcg:>10.4f} {macro_precision:>8.4f} {macro_recall:>8.4f} "
        f"{macro_map:>8.4f} {macro_mrr:>8.4f} {macro_combined:>10.4f}"
    )

    # Print markdown table for README
    print()
    print("=" * 70)
    print("MARKDOWN TABLE FOR README")
    print("=" * 70)
    print()
    print("| Split | Queries | Docs | Combined | P@10 | R@10 | NDCG@10 | MAP | MRR |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for r in results:
        print(
            f"| {r.domain} | {r.num_queries} | {r.num_documents} | "
            f"{r.combined:.4f} | {r.precision_at_k:.4f} | {r.recall_at_k:.4f} | "
            f"{r.ndcg_at_k:.4f} | {r.map:.4f} | {r.mrr:.4f} |"
        )

    print(
        f"| **macro avg** | {total_queries} | {total_docs} | "
        f"**{macro_combined:.4f}** | {macro_precision:.4f} | {macro_recall:.4f} | "
        f"**{macro_ndcg:.4f}** | {macro_map:.4f} | {macro_mrr:.4f} |"
    )

    # Save JSON results
    json_results = {
        "config": {
            "idf": type(config.idf_strategy).__name__,
            "tf": type(config.tf_strategy).__name__,
            "query_mode": config.query_mode.value,
            "k1": config.k1,
            "b": config.b,
            "k": k,
            "tokenizer": tokenizer_name,
        },
        "domains": [
            {
                "domain": r.domain,
                "num_queries": r.num_queries,
                "num_documents": r.num_documents,
                "ndcg_at_k": r.ndcg_at_k,
                "precision_at_k": r.precision_at_k,
                "recall_at_k": r.recall_at_k,
                "map": r.map,
                "mrr": r.mrr,
                "combined": r.combined,
            }
            for r in results
        ],
        "macro_averages": {
            "ndcg_at_k": float(macro_ndcg),
            "precision_at_k": float(macro_precision),
            "recall_at_k": float(macro_recall),
            "map": float(macro_map),
            "mrr": float(macro_mrr),
            "combined": float(macro_combined),
        },
    }

    print()
    print("JSON results:")
    print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
