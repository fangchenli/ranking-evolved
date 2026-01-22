"""
Evaluator for BM25 variants on BEIR benchmark.

Implements the evaluate(program_path) entrypoint expected by OpenEvolve, using
all metrics from ranking_evolved.metrics (precision/recall@k, AP/MAP, NDCG, and MRR).

BEIR is a heterogeneous benchmark with 18 datasets across 9 tasks:
- Bio-Medical IR: trec-covid, bioasq, nfcorpus
- Question Answering: nq, hotpotqa, fiqa
- News Retrieval: trec-news, robust04
- Argument Retrieval: arguana, webis-touche2020
- Duplicate Question: cqadupstack, quora
- Entity Retrieval: dbpedia-entity
- Citation Prediction: scidocs
- Fact Checking: fever, climate-fever, scifact

Supports:
- Optional query subsampling via --sample-queries flag
- Dataset selection via --dataset flag (single dataset or "all")
- Tokenizer selection via --tokenizer flag (simple or lucene)

For OpenEvolve: The default evaluate() function is called with just program_path.

Reference: https://github.com/beir-cellar/beir
Paper: Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
       of Information Retrieval Models", NeurIPS 2021
"""

import argparse
import importlib.util
import json
import os
import random
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from ranking_evolved.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# BEIR datasets (18 total across 9 tasks)
# Ordered roughly by corpus size (smallest first for faster iteration)
BEIR_DATASETS = [
    # Small datasets (< 10k docs)
    "scifact",  # 5,183 docs - Fact Checking (Scientific)
    "nfcorpus",  # 3,633 docs - Bio-Medical IR
    "arguana",  # 8,674 docs - Argument Retrieval
    # Medium datasets (10k - 100k docs)
    "scidocs",  # 25,657 docs - Citation Prediction
    "fiqa",  # 57,638 docs - Question Answering (Finance)
    "webis-touche2020",  # 382,545 docs - Argument Retrieval
    "trec-covid",  # 171,332 docs - Bio-Medical IR
    "quora",  # 522,931 docs - Duplicate Question
    "cqadupstack",  # 457,199 docs - Duplicate Question (12 StackExchange forums)
    "robust04",  # 528,155 docs - News Retrieval
    "trec-news",  # 594,977 docs - News Retrieval
    # Large datasets (> 1M docs)
    "hotpotqa",  # 5,233,329 docs - Question Answering (Wikipedia)
    "nq",  # 2,681,468 docs - Question Answering (Wikipedia)
    "fever",  # 5,416,568 docs - Fact Checking (Wikipedia)
    "climate-fever",  # 5,416,593 docs - Fact Checking (Wikipedia)
    "dbpedia-entity",  # 4,635,922 docs - Entity Retrieval
    "bioasq",  # 14,914,602 docs - Bio-Medical IR
]

# Default evaluation settings (can be overridden via env vars for OpenEvolve)
DEFAULT_DATASET = os.environ.get("BEIR_DATASET", "scifact")
DEFAULT_SAMPLE_QUERIES = int(os.environ.get("BEIR_SAMPLE_QUERIES", "0")) or None
DEFAULT_SEED = int(os.environ.get("BEIR_SEED", "42"))
DEFAULT_K = int(os.environ.get("BEIR_K", "10"))
DEFAULT_TOKENIZER = os.environ.get("BEIR_TOKENIZER", "lucene")  # simple or lucene
DEFAULT_DATA_DIR = os.environ.get("BEIR_DATA_DIR", "datasets/beir")


def _load_candidate(
    program_path: str,
) -> tuple[type, type, Callable[[str], list[str]], type | None]:
    """
    Dynamically load a BM25 implementation from a file path.

    Returns:
        Tuple of (BM25, Corpus, tokenize, LuceneTokenizer or None)
    """
    spec = importlib.util.spec_from_file_location("candidate_bm25", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Required exports
    if not hasattr(module, "BM25"):
        raise AttributeError("Candidate module must define a BM25 class.")
    if not hasattr(module, "tokenize"):
        raise AttributeError(
            "Candidate module must define a tokenize(text: str) -> list[str] function."
        )
    if not hasattr(module, "Corpus"):
        raise AttributeError("Candidate module must define a Corpus class.")

    # Optional exports
    lucene_tokenizer = getattr(module, "LuceneTokenizer", None)

    return module.BM25, module.Corpus, module.tokenize, lucene_tokenizer


@cache
def _load_beir_dataset(dataset_name: str, data_dir: str = DEFAULT_DATA_DIR):
    """
    Download and load a BEIR dataset.

    Returns:
        Tuple of (corpus, queries, qrels)
        - corpus: dict[doc_id, {"title": str, "text": str}]
        - queries: dict[query_id, query_text]
        - qrels: dict[query_id, dict[doc_id, relevance_score]]
    """
    data_path = Path(data_dir) / dataset_name

    if not data_path.exists():
        # Download dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        print(f"Downloading {dataset_name} from {url}...")
        util.download_and_unzip(url, data_dir)

    # Load dataset
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")
    return corpus, queries, qrels


def evaluate(program_path: str, k: int = DEFAULT_K) -> dict[str, float]:
    """
    Evaluate a BM25 implementation against BEIR.

    This is the main entrypoint for OpenEvolve. Settings can be configured via
    environment variables:
    - BEIR_DATASET: Dataset to evaluate (default: scifact)
    - BEIR_SAMPLE_QUERIES: Number of queries to sample (default: 0 = all)
    - BEIR_SEED: Random seed (default: 42)
    - BEIR_K: Cutoff for @k metrics (default: 10)
    - BEIR_TOKENIZER: Tokenizer to use (simple or lucene, default: lucene)
    - BEIR_DATA_DIR: Directory to store datasets (default: datasets/beir)

    Args:
        program_path: Path to the BM25 implementation file.
        k: Cutoff for @k metrics.

    Returns:
        Dictionary with evaluation metrics. On error, returns combined_score=0.0 and error=1.0.
    """
    try:
        return evaluate_with_options(
            program_path,
            k=k,
            sample_queries=DEFAULT_SAMPLE_QUERIES,
            seed=DEFAULT_SEED,
            dataset=DEFAULT_DATASET,
            tokenizer=DEFAULT_TOKENIZER,
            data_dir=DEFAULT_DATA_DIR,
        )
    except Exception as e:
        # Return error dict so OpenEvolve knows this candidate failed
        return {
            "combined_score": 0.0,
            "ndcg_at_k": 0.0,
            "error": 1.0,
            "error_message": str(e),
        }


def evaluate_with_options(
    program_path: str,
    k: int = 10,
    sample_queries: int | None = None,
    seed: int = 42,
    dataset: str = "scifact",
    tokenizer: str = "lucene",
    data_dir: str = DEFAULT_DATA_DIR,
) -> dict[str, float]:
    """
    Evaluate a BM25 implementation with full configuration options.

    Args:
        program_path: Path to the BM25 implementation file.
        k: Cutoff for @k metrics.
        sample_queries: If set, randomly sample this many queries.
        seed: Seed for reproducible sampling.
        dataset: BEIR dataset name or "all" to aggregate across datasets.
        tokenizer: "simple" or "lucene".
        data_dir: Directory to store downloaded datasets.

    Returns:
        Dictionary with evaluation metrics.
    """
    BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls = _load_candidate(program_path)

    # Select tokenizer
    if tokenizer == "lucene" and LuceneTokenizerCls is not None:
        tokenize_fn = LuceneTokenizerCls()
    # else use simple tokenize_fn from module

    def _eval_single(dataset_name: str) -> dict[str, float]:
        corpus, queries, qrels = _load_beir_dataset(dataset_name, data_dir)

        # Convert corpus to list of tokenized documents
        doc_ids = list(corpus.keys())
        doc_texts = []
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            # Combine title and text (title may be empty)
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            combined = f"{title} {text}".strip() if title else text
            doc_texts.append(combined)

        # Tokenize documents
        doc_tokens = [tokenize_fn(text) for text in doc_texts]

        # Build corpus index
        corpus_index = CorpusCls(doc_tokens, ids=doc_ids)

        # Prepare queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Sample queries if requested
        if sample_queries is not None and sample_queries < len(query_ids):
            rng = random.Random(seed)
            indices = rng.sample(range(len(query_ids)), sample_queries)
            query_ids = [query_ids[i] for i in indices]
            query_texts = [query_texts[i] for i in indices]

        # Create BM25 scorer
        bm25 = BM25Impl(corpus_index)

        # Evaluate
        all_relevant = []
        all_retrieved = []
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rr_scores = []
        ap_scores = []

        for query_id, query_text in zip(query_ids, query_texts, strict=False):
            query_tokens = tokenize_fn(query_text)
            ranked_indices, _ = bm25.rank(query_tokens)

            # Get gold relevance for this query
            gold_doc_ids = qrels.get(query_id, {})
            if not gold_doc_ids:
                continue  # Skip queries with no relevance judgments

            # Convert gold doc IDs to corpus indices
            gold_indices = []
            id_map = corpus_index.map_id_to_idx
            for gid, rel in gold_doc_ids.items():
                if rel > 0 and gid in id_map:
                    gold_indices.append(id_map[gid])

            if not gold_indices:
                continue  # Skip if no relevant documents found in corpus

            relevant = np.array(gold_indices, dtype=int)
            retrieved = np.array(ranked_indices, dtype=int)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

            precision_scores.append(precision_at_k(relevant, retrieved, k))
            recall_scores.append(recall_at_k(relevant, retrieved, k))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))
            rr_scores.append(reciprocal_rank(relevant, retrieved))
            ap_scores.append(average_precision(relevant, retrieved))

        if not all_relevant:
            raise ValueError(f"No valid queries were evaluated for {dataset_name}.")

        metrics = {
            "precision_at_k": float(np.mean(precision_scores)),
            "recall_at_k": float(np.mean(recall_scores)),
            "ndcg_at_k": float(np.mean(ndcg_scores)),
            "reciprocal_rank": float(np.mean(rr_scores)),
            "mean_average_precision": mean_average_precision(all_relevant, all_retrieved),
            "mean_reciprocal_rank": mean_reciprocal_rank(all_relevant, all_retrieved),
            "k": k,
            "queries": len(all_relevant),
            "documents": len(corpus),
            "tokenizer": tokenizer,
            "dataset": dataset_name,
        }

        # Combined score for optimization (weights can be tuned)
        metrics["combined_score"] = float(
            0.4 * metrics["ndcg_at_k"]
            + 0.3 * metrics["mean_average_precision"]
            + 0.2 * metrics["mean_reciprocal_rank"]
            + 0.05 * metrics["precision_at_k"]
            + 0.05 * metrics["recall_at_k"]
        )
        metrics["error"] = 0.0
        return metrics

    if dataset == "all":
        dataset_results = {}
        macro_accumulators = {
            "precision_at_k": [],
            "recall_at_k": [],
            "ndcg_at_k": [],
            "mean_average_precision": [],
            "mean_reciprocal_rank": [],
            "combined_score": [],
        }

        for ds_name in BEIR_DATASETS:
            try:
                print(f"Evaluating {ds_name}...")
                metrics = _eval_single(ds_name)
                dataset_results[ds_name] = metrics
                if metrics.get("error", 1.0) == 0.0:
                    for key in macro_accumulators:
                        macro_accumulators[key].append(metrics[key])
            except Exception as e:
                print(f"  Error: {e}")
                dataset_results[ds_name] = {"error": 1.0, "message": str(e)}

        macro_metrics = {
            f"macro_{key}": float(np.mean(values)) if values else 0.0
            for key, values in macro_accumulators.items()
        }
        macro_metrics["datasets_evaluated"] = sum(
            1 for d in dataset_results.values() if d.get("error", 1.0) == 0.0
        )
        macro_metrics["combined_score"] = macro_metrics.get("macro_combined_score", 0.0)
        macro_metrics["ndcg_at_k"] = macro_metrics.get("macro_ndcg_at_k", 0.0)
        macro_metrics["datasets"] = dataset_results
        macro_metrics["error"] = 0.0 if macro_metrics["datasets_evaluated"] > 0 else 1.0
        return macro_metrics

    return _eval_single(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 on BEIR benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on SciFact (small, fast)
  python evaluator_beir.py src/ranking_evolved/bm25.py --dataset scifact

  # Use Lucene tokenizer
  python evaluator_beir.py src/ranking_evolved/bm25.py --dataset nfcorpus --tokenizer lucene

  # Evaluate across all BEIR datasets
  python evaluator_beir.py src/ranking_evolved/bm25.py --dataset all

  # Fast iteration with query sampling
  python evaluator_beir.py src/ranking_evolved/bm25.py --dataset nq --sample-queries 50

BEIR paper BM25 baselines (nDCG@10):
  SciFact: 0.665, NQ: 0.329, HotpotQA: 0.603, FEVER: 0.753, Quora: 0.789
""",
    )
    parser.add_argument("program_path", help="Path to the BM25 implementation file.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics (default: 10).")
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Randomly sample this many queries (default: 0 = use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="BEIR dataset to evaluate (e.g., scifact). Use 'all' for all datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["simple", "lucene"],
        default="lucene",
        help="Tokenizer to use (default: lucene).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/beir",
        help="Directory to store downloaded datasets (default: datasets/beir).",
    )
    args = parser.parse_args()

    sample_queries = args.sample_queries if args.sample_queries > 0 else None

    results = evaluate_with_options(
        args.program_path,
        k=args.k,
        sample_queries=sample_queries,
        seed=args.seed,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        data_dir=args.data_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
