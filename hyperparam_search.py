"""
Hyperparameter search for BM25 on BRIGHT benchmark.

Searches over k1, b, and tokenizer to find optimal parameters.

Usage:
    uv run python hyperparam_search.py --domain biology
    uv run python hyperparam_search.py --domain biology --tokenizer simple
    uv run python hyperparam_search.py --domain all --sample-queries 50
"""

import argparse
import json
import random
import time
from itertools import product

import numpy as np

from datasets import load_dataset
from ranking_evolved.bm25 import Corpus, LuceneTokenizer, tokenize
from ranking_evolved.bm25_evolved import BM25
from ranking_evolved.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k

# Search grid
K1_VALUES = [0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
B_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
TOKENIZERS = ["simple", "lucene"]

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


def load_bright_data(domain: str):
    """Load BRIGHT documents and examples for a domain."""
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    return documents, examples


def build_corpus(documents, tokenize_fn) -> Corpus:
    """Tokenize documents and build corpus."""
    doc_tokens = [tokenize_fn(doc["content"]) for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    corpus = Corpus(doc_tokens, ids=doc_ids)
    # Trigger sparse matrix build
    _ = corpus.term_doc_matrix
    return corpus


def evaluate_params(
    corpus: Corpus,
    queries: list[list[str]],
    gold_indices: list[list[int]],
    k1: float,
    b: float,
    k: int = 10,
) -> dict:
    """Evaluate BM25 with given parameters."""
    bm25 = BM25(corpus, k1=k1, b=b)

    ndcg_scores = []
    all_relevant = []
    all_retrieved = []

    for query_tokens, gold in zip(queries, gold_indices, strict=False):
        ranked_indices, _ = bm25.rank(query_tokens)
        relevant = np.array(gold, dtype=int)
        retrieved = np.array(ranked_indices, dtype=int)

        all_relevant.append(relevant)
        all_retrieved.append(retrieved)
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))

    return {
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "map": mean_average_precision(all_relevant, all_retrieved),
        "mrr": mean_reciprocal_rank(all_relevant, all_retrieved),
    }


def search_domain(
    domain: str,
    tokenizers: list[str],
    sample_queries: int | None = None,
    seed: int = 42,
    k: int = 10,
) -> dict:
    """Run hyperparameter search on a single domain."""
    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"{'='*60}")

    # Load data
    t0 = time.time()
    documents, examples = load_bright_data(domain)
    print(f"Loaded {len(documents)} documents, {len(examples)} queries in {time.time()-t0:.1f}s")

    # Prepare queries and gold labels
    raw_queries = [ex["query"] for ex in examples]
    gold_id_lists = [ex["gold_ids"] for ex in examples]

    if sample_queries and sample_queries < len(raw_queries):
        rng = random.Random(seed)
        indices = rng.sample(range(len(raw_queries)), sample_queries)
        raw_queries = [raw_queries[i] for i in indices]
        gold_id_lists = [gold_id_lists[i] for i in indices]
        print(f"Sampled {sample_queries} queries")

    results = []
    best_result = None

    for tokenizer_name in tokenizers:
        # Build corpus for this tokenizer
        print(f"\nBuilding corpus with {tokenizer_name} tokenizer...")
        t1 = time.time()

        if tokenizer_name == "lucene":
            tokenize_fn = LuceneTokenizer()
        else:
            tokenize_fn = tokenize

        corpus = build_corpus(documents, tokenize_fn)
        print(f"  Corpus built in {time.time()-t1:.1f}s (vocab: {corpus.vocabulary_size})")

        # Tokenize queries
        queries = [tokenize_fn(q) for q in raw_queries]
        gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

        # Grid search
        print(f"  Searching {len(K1_VALUES)}x{len(B_VALUES)} = {len(K1_VALUES)*len(B_VALUES)} combinations...")
        t2 = time.time()

        for k1, b in product(K1_VALUES, B_VALUES):
            metrics = evaluate_params(corpus, queries, gold_indices, k1, b, k)
            result = {
                "tokenizer": tokenizer_name,
                "k1": k1,
                "b": b,
                **metrics,
            }
            results.append(result)

            if best_result is None or metrics["ndcg_at_k"] > best_result["ndcg_at_k"]:
                best_result = result

        print(f"  Searched in {time.time()-t2:.1f}s")

    # Sort by NDCG
    results.sort(key=lambda x: x["ndcg_at_k"], reverse=True)

    print(f"\nBest: tokenizer={best_result['tokenizer']}, k1={best_result['k1']}, b={best_result['b']}")
    print(f"      NDCG@{k}={best_result['ndcg_at_k']:.4f}, MAP={best_result['map']:.4f}, MRR={best_result['mrr']:.4f}")

    print("\nTop 10 configurations:")
    print(f"{'Tokenizer':<10} {'k1':<6} {'b':<6} {'NDCG@10':<10} {'MAP':<10} {'MRR':<10}")
    print("-" * 60)
    for r in results[:10]:
        print(f"{r['tokenizer']:<10} {r['k1']:<6} {r['b']:<6} {r['ndcg_at_k']:<10.4f} {r['map']:<10.4f} {r['mrr']:<10.4f}")

    return {
        "domain": domain,
        "queries": len(raw_queries),
        "documents": len(documents),
        "best": best_result,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for BM25")
    parser.add_argument(
        "--domain",
        type=str,
        default="biology",
        help="BRIGHT domain to search on, or 'all' for all domains",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["simple", "lucene", "both"],
        default="both",
        help="Tokenizer to use (default: both)",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Sample N queries for faster search (0 = use all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    tokenizers = TOKENIZERS if args.tokenizer == "both" else [args.tokenizer]
    sample_queries = args.sample_queries if args.sample_queries > 0 else None

    domains = BRIGHT_SPLITS if args.domain == "all" else [args.domain]

    all_results = {}
    for domain in domains:
        result = search_domain(domain, tokenizers, sample_queries, args.seed)
        all_results[domain] = result

    # Summary across domains
    if len(domains) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL DOMAINS")
        print(f"{'='*60}")
        print(f"{'Domain':<25} {'Tokenizer':<10} {'k1':<6} {'b':<6} {'NDCG@10':<10}")
        print("-" * 60)
        for domain, result in all_results.items():
            best = result["best"]
            print(f"{domain:<25} {best['tokenizer']:<10} {best['k1']:<6} {best['b']:<6} {best['ndcg_at_k']:<10.4f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
