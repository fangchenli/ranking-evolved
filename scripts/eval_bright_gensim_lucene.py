"""Evaluate gensim's LuceneBM25Model on BRIGHT.

Usage:
    uv run python scripts/eval_bright_gensim_lucene.py --domain biology --k 10 --top-k 10
"""

import argparse
import json
from typing import List

import numpy as np
from datasets import load_dataset
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity

from ranking_evolved.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split()]


def build_corpus(domain: str):
    docs = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    doc_tokens = [tokenize(row["content"]) for row in docs]
    doc_ids = [row["id"] for row in docs]
    queries = [row["query"] for row in examples]
    gold_ids = [row["gold_ids"] for row in examples]
    return doc_tokens, doc_ids, queries, gold_ids


def evaluate(
    domain: str,
    k: int,
    top_k: int | None,
    k1: float,
    b: float,
) -> dict[str, float]:
    docs, doc_ids, queries, gold_ids = build_corpus(domain)
    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    model = LuceneBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=k1, b=b)
    index = SparseMatrixSimilarity(model[bow_corpus], num_features=len(dictionary))

    all_relevant = []
    all_retrieved = []
    prec_scores = []
    rec_scores = []
    ndcg_scores = []
    rr_scores = []
    ap_scores = []

    id_to_idx = {id_: idx for idx, id_ in enumerate(doc_ids)}
    gold_indices = [[id_to_idx[g] for g in gid_list if g in id_to_idx] for gid_list in gold_ids]

    for query, gold in zip(queries, gold_indices):
        q_tokens = tokenize(query)
        q_bow = dictionary.doc2bow(q_tokens)
        q_vec = model[q_bow]
        scores = index[q_vec]
        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[:top_k]
        retrieved = order

        relevant = np.array(gold, dtype=int)
        retrieved_arr = np.array(retrieved, dtype=int)
        all_relevant.append(relevant)
        all_retrieved.append(retrieved_arr)

        prec_scores.append(precision_at_k(relevant, retrieved_arr, k))
        rec_scores.append(recall_at_k(relevant, retrieved_arr, k))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved_arr, k))
        rr_scores.append(reciprocal_rank(relevant, retrieved_arr))
        ap_scores.append(average_precision(relevant, retrieved_arr))

    metrics = {
        "precision_at_k": float(np.mean(prec_scores)),
        "recall_at_k": float(np.mean(rec_scores)),
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "reciprocal_rank": float(np.mean(rr_scores)),
        "mean_average_precision": mean_average_precision(all_relevant, all_retrieved),
        "mean_reciprocal_rank": mean_reciprocal_rank(all_relevant, all_retrieved),
        "k": k,
        "queries": len(all_relevant),
    }
    metrics["combined_score"] = float(
        np.mean(
            [
                metrics["ndcg_at_k"],
                metrics["mean_average_precision"],
                metrics["mean_reciprocal_rank"],
                metrics["precision_at_k"],
                metrics["recall_at_k"],
            ]
        )
    )
    metrics["error"] = 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lucene-style BM25 on BRIGHT.")
    parser.add_argument("--domain", default="biology", help="BRIGHT split to evaluate (default: biology).")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics (default: 10).")
    parser.add_argument("--top-k", type=int, default=None, help="Limit ranking to top-k docs (optional).")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 (default: 1.2).")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b (default: 0.75).")
    args = parser.parse_args()

    results = evaluate(domain=args.domain, k=args.k, top_k=args.top_k, k1=args.k1, b=args.b)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
