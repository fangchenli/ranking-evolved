"""
Parallelized Evaluator for Query Likelihood on ALL IR Benchmarks.

Designed for high-core-count machines (e.g., 112 cores, 440GB RAM).

Parallelization Strategy:
=========================
Level 1: Dataset-level parallelism (ProcessPoolExecutor)
  - Each dataset runs in its own process
  - Memory isolation prevents GIL issues
  - Smart scheduling based on dataset size

Level 2: Tokenization parallelism (ThreadPoolExecutor)
  - Within each process, tokenization is parallelized
  - Regex/stemming releases GIL
  - Configurable threads per worker

Level 3: Vectorized QL (NumPy BLAS)
  - NumPy operations use multi-threaded BLAS automatically

Memory-Aware Scheduling:
========================
- Small datasets (<50k docs): Up to 20 concurrent
- Medium datasets (50k-1M): Up to 8 concurrent
- Large datasets (>1M docs): Up to 3 concurrent

Output Format (for OpenEvolve):
===============================
{
    "combined_score": float,  # 0.8 * avg_recall@100 + 0.2 * avg_ndcg@10 (primary optimization target)
    "avg_ndcg@10": float,
    "avg_recall@100": float,
    "total_index_time_ms": float,
    "total_query_time_ms": float,
    "bright_biology_ndcg@10": float,
    "bright_biology_recall@100": float,
    "bright_biology_index_time_ms": float,
    "bright_biology_query_time_ms": float,
    ... (per-dataset metrics for all 31 datasets)
    "datasets_evaluated": int,
    "datasets_failed": int,
    "error": 0.0 or 1.0
}

Usage:
======
    # Full evaluation
    python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py

    # With query sampling for faster iteration
    python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --sample-queries 20

    # Specific benchmarks only
    python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --only-bright

    # Pyserini baseline (LMDirichletSimilarity)
    python evaluator_ql_parallel.py pyserini --save results/baselines/ql_pyserini.json

Environment Variables (for OpenEvolve):
=======================================
    EVAL_SAMPLE_QUERIES=20
    EVAL_TOKENIZER=lucene
    EVAL_MAX_WORKERS=0  # 0 = auto (based on cores/memory)
    EVAL_THREADS_PER_WORKER=8
    EVAL_BENCHMARKS=all  # all, bright, beir, bright+beir (no trec_dl)
    EVAL_EXCLUDE_DATASETS=dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora
        # Comma-separated list of dataset names to skip (for faster iteration)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ranking_evolved.datasets import (
    BEIR_DATASETS,
    BRIGHT_SPLITS,
    DATASET_SIZES,
    TREC_DL_DATASETS,
    BEIRLoader,
    BRIGHTLoader,
    EvalDataset,
    TRECDLLoader,
)
from ranking_evolved.metrics import ndcg_at_k, recall_at_k

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_QUERIES = int(os.environ.get("EVAL_SAMPLE_QUERIES", "0")) or None
DEFAULT_SEED = int(os.environ.get("EVAL_SEED", "42"))
DEFAULT_TOKENIZER = os.environ.get("EVAL_TOKENIZER", "lucene")
DEFAULT_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "0"))  # 0 = auto
DEFAULT_THREADS_PER_WORKER = int(os.environ.get("EVAL_THREADS_PER_WORKER", "8"))

# Dataset exclusion - comma-separated list of dataset names to skip
# Example: "dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora"
EXCLUDE_DATASETS_ENV = os.environ.get("EVAL_EXCLUDE_DATASETS", "")
DEFAULT_EXCLUDE_DATASETS: set[str] = set(
    d.strip() for d in EXCLUDE_DATASETS_ENV.split(",") if d.strip()
)

# Metric cutoffs
NDCG_K = 10
RECALL_K = 100

# Memory thresholds for scheduling (in docs)
# Based on empirical timing analysis:
#   TINY: < 10K docs, < 10s total time
#   SMALL: 10K-50K docs, 10-60s total time  
#   MEDIUM: 50K-200K docs, 1-5 min total time
#   LARGE: 200K-2M docs, 5-30 min total time
#   HUGE: > 2M docs, > 30 min total time (40-65 GB RAM each!)
TINY_THRESHOLD = 10_000
SMALL_THRESHOLD = 50_000
MEDIUM_THRESHOLD = 200_000
LARGE_THRESHOLD = 2_000_000

# Max concurrent workers by size tier (tuned for 440GB RAM, 112 cores)
# Memory estimates per worker: TINY ~1GB, SMALL ~2GB, MEDIUM ~5-15GB, LARGE ~20-40GB, HUGE ~40-65GB
MAX_TINY_WORKERS = 50      # Can run all tiny datasets at once
MAX_SMALL_WORKERS = 25     # Light memory footprint
MAX_MEDIUM_WORKERS = 10    # Moderate memory (5-15 GB each)
MAX_LARGE_WORKERS = 4      # Heavy memory (20-40 GB each)
MAX_HUGE_WORKERS = 1       # Run SOLO (40-65 GB each, risk OOM if parallel)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    sample_queries: int | None = DEFAULT_SAMPLE_QUERIES
    seed: int = DEFAULT_SEED
    tokenizer: str = DEFAULT_TOKENIZER
    max_workers: int = DEFAULT_MAX_WORKERS
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER
    beir_data_dir: str = "datasets/beir"
    trec_dl_data_dir: str = "datasets/trec_dl"
    # Which benchmarks to include
    include_bright: bool = True
    include_beir: bool = True
    include_trec_dl: bool = True
    # Subset of datasets (None = all)
    bright_datasets: list[str] | None = None
    beir_datasets: list[str] | None = None
    trec_dl_datasets: list[str] | None = None
    # Optional: restrict evaluation to these full dataset names only
    include_only_datasets: set[str] | None = None


@dataclass
class DatasetTask:
    """A single dataset evaluation task."""
    benchmark: str
    dataset_name: str
    full_name: str
    estimated_size: int


@dataclass
class DatasetResult:
    """Result from evaluating a single dataset."""
    name: str
    ndcg_at_10: float
    recall_at_100: float
    index_time_ms: float
    query_time_ms: float
    num_docs: int
    num_queries: int
    error: str | None = None
    # Per-query scores for significance testing (populated when --save is used)
    per_query_ids: list[str] | None = None
    per_query_ndcg: list[float] | None = None
    per_query_recall: list[float] | None = None


# =============================================================================
# Candidate Loading
# =============================================================================


def load_candidate(program_path: str) -> tuple[type, type, Callable, type | None]:
    """Load Query Likelihood implementation from file path."""
    import sys

    spec = importlib.util.spec_from_file_location("candidate_ql", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")

    module = importlib.util.module_from_spec(spec)
    # Register module in sys.modules BEFORE exec_module (required for dataclasses)
    sys.modules["candidate_ql"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "QL"):
        raise AttributeError("Module must define QL class")
    if not hasattr(module, "tokenize"):
        raise AttributeError("Module must define tokenize function")
    if not hasattr(module, "Corpus"):
        raise AttributeError("Module must define Corpus class")

    lucene_tokenizer = getattr(module, "LuceneTokenizer", None)
    return module.QL, module.Corpus, module.tokenize, lucene_tokenizer


# =============================================================================
# Parallel Tokenization
# =============================================================================


def tokenize_batch(
    texts: list[str],
    tokenize_fn: Callable[[str], list[str]],
    num_threads: int = 8,
) -> list[list[str]]:
    """
    Tokenize a batch of texts in parallel using ThreadPoolExecutor.
    
    Args:
        texts: List of texts to tokenize
        tokenize_fn: Tokenization function
        num_threads: Number of threads to use
        
    Returns:
        List of tokenized documents
    """
    if len(texts) < 100:
        # For small batches, sequential is faster (thread overhead)
        return [tokenize_fn(text) for text in texts]
    
    # Parallel tokenization
    results = [None] * len(texts)
    
    def tokenize_with_index(idx_text: tuple[int, str]) -> tuple[int, list[str]]:
        idx, text = idx_text
        return idx, tokenize_fn(text)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = executor.map(tokenize_with_index, enumerate(texts))
        for idx, tokens in futures:
            results[idx] = tokens
    
    return results


# =============================================================================
# Pyserini Official Baseline Evaluation
# =============================================================================


def evaluate_pyserini_official(
    benchmark: str,
    dataset_name: str,
    config: EvalConfig,
) -> DatasetResult:
    """
    Evaluate using official Pyserini/Lucene Query Likelihood (LMDirichletSimilarity).

    This uses the actual Pyserini package with Java/Lucene backend for
    ground truth comparison. Uses Pyserini's internal LM Dirichlet defaults
    (mu=2000, no hyperparameters assumed by evaluator).

    OPTIMIZATIONS:
    - Uses batch_search with Java-side multi-threading
    - Only retrieves top-1000 (sufficient for nDCG@10, Recall@100)
    - Larger indexing batch size for better throughput
    - Multi-threaded search (uses all available cores)

    Args:
        benchmark: "bright", "beir", or "trec_dl"
        dataset_name: Dataset name within benchmark
        config: Evaluation configuration

    Returns:
        DatasetResult with metrics and timing
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path
    
    # Set JAVA_HOME if not set
    if "JAVA_HOME" not in os.environ:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    
    full_name = f"{benchmark}_{dataset_name}"
    
    # Top-K for retrieval - 1000 is more than enough for Recall@100 and nDCG@10
    RETRIEVAL_K = 1000

    try:
        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher

        # Load dataset
        if benchmark == "bright":
            loader = BRIGHTLoader()
            dataset = loader.load(dataset_name)
        elif benchmark == "beir":
            loader = BEIRLoader(data_dir=config.beir_data_dir)
            dataset = loader.load(dataset_name)
        elif benchmark == "trec_dl":
            loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
            dataset = loader.load(dataset_name)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        # Reduce parallelism for huge datasets to prevent OOM
        # Number of search threads (Pyserini uses Java-side threading)
        n_docs = len(dataset.corpus)
        if n_docs >= LARGE_THRESHOLD:  # 2M+ docs (beir_fever, beir_hotpotqa, beir_nq)
            num_threads = 1  # Ultra-conservative: single-threaded to avoid OOM
            if config.verbose:
                print(f"    {full_name}: using sequential processing (corpus: {n_docs:,} docs, 1 thread) to prevent OOM", file=sys.stderr, flush=True)
        else:
            num_threads = min(os.cpu_count() or 8, 64)
        
        # Create temp directory for Lucene index
        index_dir = Path(tempfile.mkdtemp(prefix="pyserini_idx_"))
        
        try:
            # === PHASE 1: Index Building ===
            index_start = time.perf_counter()
            
            # Build Lucene index from raw text with larger batches for throughput
            indexer = LuceneIndexer(str(index_dir / "index"), append=False)
            
            # Use larger batch size for better indexing throughput
            batch_size = 10000
            for i in range(0, len(dataset.corpus), batch_size):
                batch = []
                for j in range(i, min(i + batch_size, len(dataset.corpus))):
                    batch.append({
                        "id": dataset.corpus_ids[j],
                        "contents": dataset.corpus[j],
                    })
                indexer.add_batch_dict(batch)
            indexer.close()

            # Create searcher and set to LM Dirichlet similarity (mu=2000 default)
            searcher = LuceneSearcher(str(index_dir / "index"))
            searcher.set_qld()  # Use Query Likelihood with Dirichlet smoothing

            index_end = time.perf_counter()
            index_time_ms = (index_end - index_start) * 1000
            
            # === PHASE 2: Query Evaluation ===
            query_start = time.perf_counter()
            
            # Build ID to index mapping
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dataset.corpus_ids)}
            
            # Get queries (with optional sampling)
            query_ids = dataset.query_ids
            queries = dataset.queries
            
            if config.sample_queries and config.sample_queries < len(queries):
                rng = random.Random(config.seed)
                indices = rng.sample(range(len(queries)), config.sample_queries)
                query_ids = [query_ids[i] for i in indices]
                queries = [queries[i] for i in indices]
            
            # Filter to valid queries (those with relevant docs)
            valid_qids = []
            valid_queries = []
            valid_relevant = []
            
            for qid, query_text in zip(query_ids, queries, strict=False):
                relevant_doc_ids = dataset.get_relevant_docs(qid)
                if not relevant_doc_ids:
                    continue
                
                relevant_indices = [
                    id_to_idx[doc_id]
                    for doc_id in relevant_doc_ids
                    if doc_id in id_to_idx
                ]
                
                if not relevant_indices:
                    continue
                
                valid_qids.append(qid)
                valid_queries.append(query_text)
                valid_relevant.append(np.array(relevant_indices, dtype=int))
            
            if not valid_queries:
                query_end = time.perf_counter()
                return DatasetResult(
                    name=full_name,
                    ndcg_at_10=0.0,
                    recall_at_100=0.0,
                    index_time_ms=index_time_ms,
                    query_time_ms=(query_end - query_start) * 1000,
                    num_docs=len(dataset.corpus),
                    num_queries=0,
                    error="No valid queries",
                )
            
            # OPTIMIZED: Use batch_search with Java-side multi-threading
            # This is much faster than sequential search()
            batch_results = searcher.batch_search(
                queries=valid_queries,
                qids=valid_qids,
                k=RETRIEVAL_K,
                threads=num_threads,
            )
            
            # Process batch results
            all_relevant = valid_relevant
            all_retrieved = []
            
            for qid in valid_qids:
                hits = batch_results.get(qid, [])
                
                # Convert to indices
                retrieved = []
                seen = set()
                for hit in hits:
                    if hit.docid in id_to_idx:
                        idx = id_to_idx[hit.docid]
                        if idx not in seen:
                            retrieved.append(idx)
                            seen.add(idx)
                
                # For recall@100 we only need top-100, but keep all retrieved for safety
                all_retrieved.append(np.array(retrieved, dtype=int))
            
            query_end = time.perf_counter()
            query_time_ms = (query_end - query_start) * 1000
            
            # === Compute Metrics ===
            ndcg_scores = [
                ndcg_at_k(rel, ret, NDCG_K)
                for rel, ret in zip(all_relevant, all_retrieved, strict=False)
            ]
            recall_scores = [
                recall_at_k(rel, ret, RECALL_K)
                for rel, ret in zip(all_relevant, all_retrieved, strict=False)
            ]
            
            return DatasetResult(
                name=full_name,
                ndcg_at_10=float(np.mean(ndcg_scores)),
                recall_at_100=float(np.mean(recall_scores)),
                index_time_ms=index_time_ms,
                query_time_ms=query_time_ms,
                num_docs=len(dataset.corpus),
                num_queries=len(all_relevant),
                per_query_ids=[str(q) for q in valid_qids],
                per_query_ndcg=[float(s) for s in ndcg_scores],
                per_query_recall=[float(s) for s in recall_scores],
            )
            
        finally:
            # Clean up temp directory
            shutil.rmtree(index_dir, ignore_errors=True)
            
    except Exception as e:
        return DatasetResult(
            name=full_name,
            ndcg_at_10=0.0,
            recall_at_100=0.0,
            index_time_ms=0.0,
            query_time_ms=0.0,
            num_docs=0,
            num_queries=0,
            error=str(e),
        )


def evaluate_pyserini_trec_dl_combined(
    config: EvalConfig,
) -> list[DatasetResult]:
    """
    Evaluate both DL19 and DL20 using Pyserini with a SHARED Lucene index.
    
    OPTIMIZATION: Builds Lucene index on MSMARCO corpus once, then evaluates
    both DL19 and DL20 query sets. Saves ~15 minutes of index building time.
    Uses Pyserini's internal BM25 defaults (no hyperparameters assumed).
    
    Args:
        config: Evaluation configuration
        
    Returns:
        List of two DatasetResults: [dl19_result, dl20_result]
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path
    
    # Set JAVA_HOME if not set
    if "JAVA_HOME" not in os.environ:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

    RETRIEVAL_K = 1000

    results = []

    try:
        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher

        # Load shared MSMARCO corpus
        loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
        dl19_dataset = loader.load("dl19")  # This loads the shared corpus

        # Reduce parallelism for huge datasets to prevent OOM
        n_docs = len(dl19_dataset.corpus)
        if n_docs >= LARGE_THRESHOLD:  # 2M+ docs (TREC DL is 8.8M docs)
            num_threads = 1  # Ultra-conservative: single-threaded to avoid OOM
            if config.verbose:
                print(f"    trec_dl_combined: using sequential processing (corpus: {n_docs:,} docs, 1 thread) to prevent OOM", file=sys.stderr, flush=True)
        else:
            num_threads = min(os.cpu_count() or 8, 64)
        
        # Create temp directory for Lucene index
        index_dir = Path(tempfile.mkdtemp(prefix="pyserini_trec_dl_"))
        
        try:
            # === PHASE 1: Index Building (SHARED) ===
            index_start = time.perf_counter()
            
            indexer = LuceneIndexer(str(index_dir / "index"), append=False)
            
            batch_size = 10000
            for i in range(0, len(dl19_dataset.corpus), batch_size):
                batch = []
                for j in range(i, min(i + batch_size, len(dl19_dataset.corpus))):
                    batch.append({
                        "id": dl19_dataset.corpus_ids[j],
                        "contents": dl19_dataset.corpus[j],
                    })
                indexer.add_batch_dict(batch)
            indexer.close()

            # Create searcher and set to LM Dirichlet similarity (mu=2000 default)
            searcher = LuceneSearcher(str(index_dir / "index"))
            searcher.set_qld()  # Use Query Likelihood with Dirichlet smoothing

            index_end = time.perf_counter()
            shared_index_time_ms = (index_end - index_start) * 1000
            
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dl19_dataset.corpus_ids)}
            
            # === PHASE 2: Evaluate DL19 ===
            dl19_result = _evaluate_pyserini_queries(
                searcher, dl19_dataset, id_to_idx, config,
                "trec_dl_dl19", shared_index_time_ms, RETRIEVAL_K, num_threads
            )
            results.append(dl19_result)
            
            # === PHASE 3: Evaluate DL20 (reuse index!) ===
            dl20_dataset = loader.load("dl20")
            dl20_result = _evaluate_pyserini_queries(
                searcher, dl20_dataset, id_to_idx, config,
                "trec_dl_dl20", 0.0, RETRIEVAL_K, num_threads
            )
            results.append(dl20_result)
            
            return results
            
        finally:
            shutil.rmtree(index_dir, ignore_errors=True)
            
    except Exception as e:
        error_msg = str(e)
        return [
            DatasetResult(
                name="trec_dl_dl19",
                ndcg_at_10=0.0, recall_at_100=0.0,
                index_time_ms=0.0, query_time_ms=0.0,
                num_docs=0, num_queries=0, error=error_msg,
            ),
            DatasetResult(
                name="trec_dl_dl20",
                ndcg_at_10=0.0, recall_at_100=0.0,
                index_time_ms=0.0, query_time_ms=0.0,
                num_docs=0, num_queries=0, error=error_msg,
            ),
        ]


def _evaluate_pyserini_queries(
    searcher,
    dataset: EvalDataset,
    id_to_idx: dict[str, int],
    config: EvalConfig,
    full_name: str,
    index_time_ms: float,
    retrieval_k: int,
    num_threads: int,
) -> DatasetResult:
    """
    Evaluate queries using a Pyserini searcher on an already-built index.
    
    Helper for evaluate_pyserini_trec_dl_combined() to avoid code duplication.
    """
    query_start = time.perf_counter()
    
    query_ids = dataset.query_ids
    queries = dataset.queries
    
    if config.sample_queries and config.sample_queries < len(queries):
        rng = random.Random(config.seed)
        indices = rng.sample(range(len(queries)), config.sample_queries)
        query_ids = [query_ids[i] for i in indices]
        queries = [queries[i] for i in indices]
    
    # Filter to valid queries
    valid_qids = []
    valid_queries = []
    valid_relevant = []
    
    for qid, query_text in zip(query_ids, queries, strict=False):
        relevant_doc_ids = dataset.get_relevant_docs(qid)
        if not relevant_doc_ids:
            continue
        
        relevant_indices = [
            id_to_idx[doc_id]
            for doc_id in relevant_doc_ids
            if doc_id in id_to_idx
        ]
        
        if not relevant_indices:
            continue
        
        valid_qids.append(qid)
        valid_queries.append(query_text)
        valid_relevant.append(np.array(relevant_indices, dtype=int))
    
    if not valid_queries:
        query_end = time.perf_counter()
        return DatasetResult(
            name=full_name,
            ndcg_at_10=0.0, recall_at_100=0.0,
            index_time_ms=index_time_ms,
            query_time_ms=(query_end - query_start) * 1000,
            num_docs=len(dataset.corpus), num_queries=0,
            error="No valid queries",
        )
    
    # Batch search
    batch_results = searcher.batch_search(
        queries=valid_queries,
        qids=valid_qids,
        k=retrieval_k,
        threads=num_threads,
    )
    
    all_relevant = valid_relevant
    all_retrieved = []
    
    for qid in valid_qids:
        hits = batch_results.get(qid, [])
        retrieved = []
        seen = set()
        for hit in hits:
            if hit.docid in id_to_idx:
                idx = id_to_idx[hit.docid]
                if idx not in seen:
                    retrieved.append(idx)
                    seen.add(idx)
        all_retrieved.append(np.array(retrieved, dtype=int))
    
    query_end = time.perf_counter()
    query_time_ms = (query_end - query_start) * 1000
    
    ndcg_scores = [
        ndcg_at_k(rel, ret, NDCG_K)
        for rel, ret in zip(all_relevant, all_retrieved, strict=False)
    ]
    recall_scores = [
        recall_at_k(rel, ret, RECALL_K)
        for rel, ret in zip(all_relevant, all_retrieved, strict=False)
    ]
    
    return DatasetResult(
        name=full_name,
        ndcg_at_10=float(np.mean(ndcg_scores)),
        recall_at_100=float(np.mean(recall_scores)),
        index_time_ms=index_time_ms,
        query_time_ms=query_time_ms,
        num_docs=len(dataset.corpus),
        num_queries=len(all_relevant),
        per_query_ids=[str(q) for q in valid_qids],
        per_query_ndcg=[float(s) for s in ndcg_scores],
        per_query_recall=[float(s) for s in recall_scores],
    )


# =============================================================================
# Single Dataset Evaluation
# =============================================================================


def evaluate_single_dataset(
    program_path: str,
    benchmark: str,
    dataset_name: str,
    config: EvalConfig,
) -> DatasetResult:
    """
    Evaluate Query Likelihood on a single dataset.

    This function runs in a separate process for isolation.

    Args:
        program_path: Path to QL implementation
        benchmark: "bright", "beir", or "trec_dl"
        dataset_name: Dataset name within benchmark
        config: Evaluation configuration

    Returns:
        DatasetResult with metrics and timing
    """
    full_name = f"{benchmark}_{dataset_name}"

    try:
        # Load candidate QL implementation
        QLImpl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)
        
        # Select tokenizer
        if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
            tokenize_fn = LuceneTokenizerCls()
        
        # Load dataset
        if benchmark == "bright":
            loader = BRIGHTLoader()
            dataset = loader.load(dataset_name)
        elif benchmark == "beir":
            loader = BEIRLoader(data_dir=config.beir_data_dir)
            dataset = loader.load(dataset_name)
        elif benchmark == "trec_dl":
            loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
            dataset = loader.load(dataset_name)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        
        # === PHASE 1: Tokenization + Index Building ===
        index_start = time.perf_counter()
        
        # Parallel tokenization
        doc_tokens = tokenize_batch(
            dataset.corpus,
            tokenize_fn,
            num_threads=config.threads_per_worker,
        )
        
        # Build corpus and QL index
        corpus = CorpusCls(doc_tokens, ids=dataset.corpus_ids)
        ql = QLImpl(corpus)
        
        # Force lazy property computation (if available)
        if hasattr(corpus, 'vocabulary_size'):
            _ = corpus.vocabulary_size
        if hasattr(corpus, 'idf_array'):
            _ = corpus.idf_array
        if hasattr(corpus, 'term_doc_matrix'):
            _ = corpus.term_doc_matrix
        
        index_end = time.perf_counter()
        index_time_ms = (index_end - index_start) * 1000
        
        # Free tokenized docs - no longer needed after index construction (~6GB for 5M docs)
        import gc
        del doc_tokens
        gc.collect()
        
        # === PHASE 2: Query Evaluation ===
        query_start = time.perf_counter()
        
        # Get queries (with optional sampling)
        query_ids = dataset.query_ids
        queries = dataset.queries
        
        if config.sample_queries and config.sample_queries < len(queries):
            rng = random.Random(config.seed)
            indices = rng.sample(range(len(queries)), config.sample_queries)
            query_ids = [query_ids[i] for i in indices]
            queries = [queries[i] for i in indices]
        
        # Build ID to index mapping
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dataset.corpus_ids)}
        
        # Evaluate queries
        all_relevant = []
        all_retrieved = []
        valid_qids = []
        
        for qid, query_text in zip(query_ids, queries, strict=False):
            query_tokens = tokenize_fn(query_text)
            # Pass top_k to avoid returning full N-element arrays
            ranked_indices, _ = ql.rank(query_tokens, top_k=RECALL_K)
            
            # Get relevant documents
            relevant_doc_ids = dataset.get_relevant_docs(qid)
            if not relevant_doc_ids:
                continue
            
            relevant_indices = [
                id_to_idx[doc_id] 
                for doc_id in relevant_doc_ids 
                if doc_id in id_to_idx
            ]
            
            if not relevant_indices:
                continue
            
            valid_qids.append(str(qid))
            all_relevant.append(np.array(relevant_indices, dtype=int))
            # CRITICAL: .copy() releases the numpy view that holds the full N-element
            # backing array (~42MB for 5M docs) instead of just top_k elements (~800 bytes)
            all_retrieved.append(ranked_indices[:RECALL_K].copy())
        
        query_end = time.perf_counter()
        query_time_ms = (query_end - query_start) * 1000
        
        # === Compute Metrics ===
        if not all_relevant:
            return DatasetResult(
                name=full_name,
                ndcg_at_10=0.0,
                recall_at_100=0.0,
                index_time_ms=index_time_ms,
                query_time_ms=query_time_ms,
                num_docs=len(dataset.corpus),
                num_queries=0,
                error="No valid queries",
            )
        
        ndcg_scores = [
            ndcg_at_k(rel, ret, NDCG_K) 
            for rel, ret in zip(all_relevant, all_retrieved, strict=False)
        ]
        recall_scores = [
            recall_at_k(rel, ret, RECALL_K)
            for rel, ret in zip(all_relevant, all_retrieved, strict=False)
        ]
        
        return DatasetResult(
            name=full_name,
            ndcg_at_10=float(np.mean(ndcg_scores)),
            recall_at_100=float(np.mean(recall_scores)),
            index_time_ms=index_time_ms,
            query_time_ms=query_time_ms,
            num_docs=len(dataset.corpus),
            num_queries=len(all_relevant),
            per_query_ids=valid_qids,
            per_query_ndcg=[float(s) for s in ndcg_scores],
            per_query_recall=[float(s) for s in recall_scores],
        )
        
    except Exception as e:
        return DatasetResult(
            name=full_name,
            ndcg_at_10=0.0,
            recall_at_100=0.0,
            index_time_ms=0.0,
            query_time_ms=0.0,
            num_docs=0,
            num_queries=0,
            error=str(e),
        )


def evaluate_trec_dl_combined(
    program_path: str,
    config: EvalConfig,
) -> list[DatasetResult]:
    """
    Evaluate both DL19 and DL20 with a SHARED corpus and index.

    OPTIMIZATION: DL19 and DL20 share the same MSMARCO passage corpus (~8.8M passages).
    By building the index once, we save ~30 minutes of redundant indexing time.

    Args:
        program_path: Path to QL implementation
        config: Evaluation configuration

    Returns:
        List of two DatasetResults: [dl19_result, dl20_result]
    """
    results = []

    try:
        # Load candidate QL implementation
        QLImpl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)
        
        # Select tokenizer
        if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
            tokenize_fn = LuceneTokenizerCls()
        
        # Load shared MSMARCO corpus (only once!)
        loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
        
        # Load DL19 first to get the corpus
        dl19_dataset = loader.load("dl19")
        
        # === PHASE 1: Tokenization + Index Building (SHARED) ===
        index_start = time.perf_counter()
        
        # Parallel tokenization of the shared corpus
        doc_tokens = tokenize_batch(
            dl19_dataset.corpus,
            tokenize_fn,
            num_threads=config.threads_per_worker,
        )
        
        # Build corpus and QL index (ONCE for both DL19 and DL20)
        corpus = CorpusCls(doc_tokens, ids=dl19_dataset.corpus_ids)
        ql = QLImpl(corpus)
        
        # Force lazy property computation
        if hasattr(corpus, 'vocabulary_size'):
            _ = corpus.vocabulary_size
        if hasattr(corpus, 'idf_array'):
            _ = corpus.idf_array
        if hasattr(corpus, 'term_doc_matrix'):
            _ = corpus.term_doc_matrix
        
        index_end = time.perf_counter()
        shared_index_time_ms = (index_end - index_start) * 1000
        
        # Free tokenized docs - no longer needed after index construction
        import gc
        del doc_tokens
        gc.collect()
        
        # Build ID to index mapping (shared)
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dl19_dataset.corpus_ids)}
        
        # === PHASE 2: Evaluate DL19 ===
        dl19_result = _evaluate_queries_on_index(
            ql, dl19_dataset, id_to_idx, tokenize_fn, config,
            "trec_dl_dl19", shared_index_time_ms
        )
        results.append(dl19_result)

        # === PHASE 3: Evaluate DL20 (reuse index!) ===
        # Load DL20 queries/qrels only (corpus is already loaded)
        dl20_dataset = loader.load("dl20")

        dl20_result = _evaluate_queries_on_index(
            ql, dl20_dataset, id_to_idx, tokenize_fn, config,
            "trec_dl_dl20", 0.0  # Index time is 0 for DL20 (reused)
        )
        results.append(dl20_result)
        
        return results
        
    except Exception as e:
        # Return error results for both datasets
        error_msg = str(e)
        return [
            DatasetResult(
                name="trec_dl_dl19",
                ndcg_at_10=0.0, recall_at_100=0.0,
                index_time_ms=0.0, query_time_ms=0.0,
                num_docs=0, num_queries=0, error=error_msg,
            ),
            DatasetResult(
                name="trec_dl_dl20",
                ndcg_at_10=0.0, recall_at_100=0.0,
                index_time_ms=0.0, query_time_ms=0.0,
                num_docs=0, num_queries=0, error=error_msg,
            ),
        ]


def _evaluate_queries_on_index(
    ql,
    dataset: EvalDataset,
    id_to_idx: dict[str, int],
    tokenize_fn,
    config: EvalConfig,
    full_name: str,
    index_time_ms: float,
) -> DatasetResult:
    """
    Evaluate queries on an already-built QL index.

    Helper for evaluate_trec_dl_combined() to avoid code duplication.
    """
    query_start = time.perf_counter()
    
    # Get queries (with optional sampling)
    query_ids = dataset.query_ids
    queries = dataset.queries
    
    if config.sample_queries and config.sample_queries < len(queries):
        rng = random.Random(config.seed)
        indices = rng.sample(range(len(queries)), config.sample_queries)
        query_ids = [query_ids[i] for i in indices]
        queries = [queries[i] for i in indices]
    
    # Evaluate queries
    all_relevant = []
    all_retrieved = []
    valid_qids = []
    
    for qid, query_text in zip(query_ids, queries, strict=False):
        query_tokens = tokenize_fn(query_text)
        # Pass top_k to avoid returning full N-element arrays
        ranked_indices, _ = ql.rank(query_tokens, top_k=RECALL_K)

        # Get relevant documents
        relevant_doc_ids = dataset.get_relevant_docs(qid)
        if not relevant_doc_ids:
            continue
        
        relevant_indices = [
            id_to_idx[doc_id]
            for doc_id in relevant_doc_ids
            if doc_id in id_to_idx
        ]
        
        if not relevant_indices:
            continue
        
        valid_qids.append(str(qid))
        all_relevant.append(np.array(relevant_indices, dtype=int))
        # .copy() releases the numpy view that holds the full N-element backing array
        all_retrieved.append(ranked_indices[:RECALL_K].copy())
    
    query_end = time.perf_counter()
    query_time_ms = (query_end - query_start) * 1000
    
    # Compute metrics
    if not all_relevant:
        return DatasetResult(
            name=full_name,
            ndcg_at_10=0.0,
            recall_at_100=0.0,
            index_time_ms=index_time_ms,
            query_time_ms=query_time_ms,
            num_docs=len(dataset.corpus),
            num_queries=0,
            error="No valid queries with relevant docs",
        )
    
    ndcg_scores = [
        ndcg_at_k(rel, ret, NDCG_K)
        for rel, ret in zip(all_relevant, all_retrieved, strict=False)
    ]
    recall_scores = [
        recall_at_k(rel, ret, RECALL_K)
        for rel, ret in zip(all_relevant, all_retrieved, strict=False)
    ]
    
    return DatasetResult(
        name=full_name,
        ndcg_at_10=float(np.mean(ndcg_scores)),
        recall_at_100=float(np.mean(recall_scores)),
        index_time_ms=index_time_ms,
        query_time_ms=query_time_ms,
        num_docs=len(dataset.corpus),
        num_queries=len(all_relevant),
        per_query_ids=valid_qids,
        per_query_ndcg=[float(s) for s in ndcg_scores],
        per_query_recall=[float(s) for s in recall_scores],
    )


# Import worker from a stable module name so ProcessPoolExecutor workers can unpickle
# when OpenEvolve loads this file as "evaluation_module" (child processes would otherwise
# fail with ModuleNotFoundError: No module named 'evaluation_module').
from evaluator_ql_parallel_worker import _worker_evaluate


# =============================================================================
# Incremental Evaluation Support
# =============================================================================

PER_DATASET_SUFFIXES = ("_ndcg@10", "_recall@100", "_index_time_ms", "_query_time_ms", "_error")
AGGREGATE_KEYS = (
    "avg_ndcg@10", "avg_recall@100", "combined_score", "average_score",
    "datasets_evaluated", "datasets_failed",
    "total_index_time_ms", "total_query_time_ms", "total_time_ms",
)


def _get_failed_datasets(data: dict) -> set[str]:
    """Full dataset names that have an _error key."""
    failed = set()
    for key in data:
        if key.endswith("_error"):
            failed.add(key[: -len("_error")])
    return failed


def _get_all_prefixes(data: dict) -> set[str]:
    """Dataset prefixes from _ndcg@10 keys."""
    return {key[: -len("_ndcg@10")] for key in data if key.endswith("_ndcg@10") and key != "avg_ndcg@10"}


def _recompute_aggregates(data: dict) -> None:
    """Recompute aggregate fields from per-dataset keys. Modifies in place."""
    prefixes = _get_all_prefixes(data)
    all_ndcg, all_recall = [], []
    total_index = total_query = 0.0
    evaluated = failed = 0
    for prefix in prefixes:
        if data.get(f"{prefix}_error"):
            failed += 1
            continue
        ndcg_key, recall_key = f"{prefix}_ndcg@10", f"{prefix}_recall@100"
        if ndcg_key in data and recall_key in data:
            all_ndcg.append(float(data[ndcg_key]))
            all_recall.append(float(data[recall_key]))
            total_index += float(data.get(f"{prefix}_index_time_ms", 0))
            total_query += float(data.get(f"{prefix}_query_time_ms", 0))
            evaluated += 1
    data["avg_ndcg@10"] = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0
    data["avg_recall@100"] = sum(all_recall) / len(all_recall) if all_recall else 0.0
    data["datasets_evaluated"] = evaluated
    data["datasets_failed"] = failed
    data["total_index_time_ms"] = total_index
    data["total_query_time_ms"] = total_query
    data["total_time_ms"] = total_index + total_query
    data["combined_score"] = 0.0 if failed > 0 else (0.8 * data["avg_recall@100"] + 0.2 * data["avg_ndcg@10"])
    data["average_score"] = 0.0 if failed > 0 else (0.5 * data["avg_ndcg@10"] + 0.5 * data["avg_recall@100"])
    data["error"] = 0.0 if evaluated > 0 else 1.0


def _merge_partial_into(existing: dict, partial: dict) -> None:
    """Update existing with per-dataset keys from partial; remove _error when partial succeeded; recompute aggregates."""
    for key in list(partial.keys()):
        if key in ("_metadata", "error") or key in AGGREGATE_KEYS:
            continue
        for suffix in PER_DATASET_SUFFIXES:
            if key.endswith(suffix):
                existing[key] = partial[key]
                break
    for key in list(existing.keys()):
        if key.endswith("_error"):
            prefix = key[: -len("_error")]
            if f"{prefix}_ndcg@10" in partial and f"{prefix}_error" not in partial:
                del existing[key]
    _recompute_aggregates(existing)


# =============================================================================
# Memory-Aware Scheduling
# =============================================================================


def get_dataset_tasks(config: EvalConfig, exclude_datasets: set[str] | None = None) -> list[DatasetTask]:
    """
    Get list of dataset tasks to evaluate.
    
    Args:
        config: Evaluation configuration
        exclude_datasets: Set of dataset names to skip (e.g., {"dl19", "fever", "hotpotqa"})
    """
    if exclude_datasets is None:
        exclude_datasets = DEFAULT_EXCLUDE_DATASETS
    
    only = config.include_only_datasets  # None = include all; set = restrict to these
    
    tasks = []
    
    if config.include_bright:
        datasets = config.bright_datasets or BRIGHT_SPLITS
        for ds in datasets:
            if ds in exclude_datasets:
                continue
            full_name = f"bright_{ds}"
            if only is not None and full_name not in only:
                continue
            size = DATASET_SIZES.get(full_name, 100_000)
            tasks.append(DatasetTask("bright", ds, full_name, size))
    
    if config.include_beir:
        datasets = config.beir_datasets or BEIR_DATASETS
        for ds in datasets:
            if ds in exclude_datasets:
                continue
            full_name = f"beir_{ds}"
            if only is not None and full_name not in only:
                continue
            size = DATASET_SIZES.get(full_name, 100_000)
            tasks.append(DatasetTask("beir", ds, full_name, size))
    
    if config.include_trec_dl:
        datasets = config.trec_dl_datasets or TREC_DL_DATASETS
        # Filter out excluded TREC DL datasets
        datasets = [ds for ds in datasets if ds not in exclude_datasets]
        
        # If include_only_datasets is set, check if any TREC DL tasks are wanted
        if only is not None:
            want_trec = "trec_dl_combined" in only or any(f"trec_dl_{ds}" in only for ds in datasets)
            if not want_trec:
                datasets = []
        
        if not datasets:
            pass  # All TREC DL excluded
        # OPTIMIZATION: If both DL19 and DL20 are included, use combined evaluation
        # This builds the MSMARCO corpus index only ONCE, saving ~30 minutes
        elif set(datasets) == {"dl19", "dl20"} or datasets == TREC_DL_DATASETS:
            # Combined task: builds index once, evaluates both query sets
            tasks.append(DatasetTask(
                "trec_dl_combined", "dl19_dl20", "trec_dl_combined", 
                8_800_000  # MSMARCO passage corpus size
            ))
        else:
            # If only one is requested, use separate evaluation
            for ds in datasets:
                full_name = f"trec_dl_{ds}"
                if only is not None and full_name not in only:
                    continue
                size = DATASET_SIZES.get(full_name, 8_000_000)
                tasks.append(DatasetTask("trec_dl", ds, full_name, size))
    
    return tasks


def schedule_tasks(tasks: list[DatasetTask], max_workers: int = 0) -> list[list[DatasetTask]]:
    """
    Schedule tasks into batches based on size for memory-aware execution.
    
    Strategy (based on empirical timing data):
    - TINY (< 10K docs): Run all at once, fast completion
    - SMALL (10-50K): High parallelism (25 workers)
    - MEDIUM (50-200K): Moderate parallelism (10 workers)
    - LARGE (200K-2M): Limited parallelism (4 workers)
    - HUGE (> 2M docs): Run SOLO to prevent OOM (40-65 GB each!)
    
    Args:
        tasks: List of dataset tasks
        max_workers: Maximum workers (0 = auto based on CPU cores)
        
    Returns:
        List of batches, where each batch can run concurrently
    """
    if max_workers == 0:
        cpu_count = mp.cpu_count()
        max_workers = min(cpu_count, 56)  # Cap at 56 for reasonable batching
    
    # Categorize by size (5 tiers based on memory/timing analysis)
    tiny_tasks = [t for t in tasks if t.estimated_size < TINY_THRESHOLD]
    small_tasks = [t for t in tasks if TINY_THRESHOLD <= t.estimated_size < SMALL_THRESHOLD]
    medium_tasks = [t for t in tasks if SMALL_THRESHOLD <= t.estimated_size < MEDIUM_THRESHOLD]
    large_tasks = [t for t in tasks if MEDIUM_THRESHOLD <= t.estimated_size < LARGE_THRESHOLD]
    huge_tasks = [t for t in tasks if t.estimated_size >= LARGE_THRESHOLD]
    
    batches = []
    
    # TINY: Run all at once (< 10s each, ~1GB RAM each)
    if tiny_tasks:
        tiny_batch_size = min(MAX_TINY_WORKERS, max_workers, len(tiny_tasks))
        for i in range(0, len(tiny_tasks), tiny_batch_size):
            batches.append(tiny_tasks[i:i + tiny_batch_size])
    
    # SMALL: High parallelism (10-60s each, ~2GB RAM each)
    if small_tasks:
        small_batch_size = min(MAX_SMALL_WORKERS, max_workers)
        for i in range(0, len(small_tasks), small_batch_size):
            batches.append(small_tasks[i:i + small_batch_size])
    
    # MEDIUM: Moderate parallelism (1-5 min each, 5-15 GB RAM each)
    if medium_tasks:
        medium_batch_size = min(MAX_MEDIUM_WORKERS, max_workers)
        for i in range(0, len(medium_tasks), medium_batch_size):
            batches.append(medium_tasks[i:i + medium_batch_size])
    
    # LARGE: Limited parallelism (5-30 min each, 20-40 GB RAM each)
    if large_tasks:
        large_batch_size = min(MAX_LARGE_WORKERS, max_workers)
        for i in range(0, len(large_tasks), large_batch_size):
            batches.append(large_tasks[i:i + large_batch_size])
    
    # HUGE: Run SOLO (> 30 min each, 40-65 GB RAM each - OOM risk if parallel)
    for task in huge_tasks:
        batches.append([task])  # Each huge dataset in its own batch
    
    return batches


# =============================================================================
# Main Parallel Evaluation
# =============================================================================


def _collect_per_query(result: DatasetResult, per_query_data: dict) -> None:
    """Add per-query scores from a DatasetResult to the per_query_data dict."""
    if result.per_query_ids is not None and result.error is None:
        per_query_data[result.name] = {
            "query_ids": result.per_query_ids,
            "ndcg@10": result.per_query_ndcg,
            "recall@100": result.per_query_recall,
        }


def _save_per_query_incremental(save_path: Path, per_query_data: dict) -> None:
    """Atomically save per-query data to companion file."""
    pq_path = save_path.with_name(save_path.stem + "_perquery.json")
    tmp_path = pq_path.with_suffix(pq_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(per_query_data, f, indent=2)
    tmp_path.replace(pq_path)


def evaluate_parallel(
    program_path: str,
    config: EvalConfig,
    verbose: bool = False,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate QL on all datasets in parallel with incremental saving.

    Args:
        program_path: Path to QL implementation
        config: Evaluation configuration
        verbose: Print progress
        save_path: Optional path to incrementally save results

    Returns:
        Dictionary with all metrics (flat format for OpenEvolve)
    """
    from pathlib import Path

    # Load existing results if save_path exists
    existing: dict[str, Any] = {}
    per_query_data: dict[str, Any] = {}
    if save_path and save_path.is_file():
        with open(save_path) as f:
            existing = json.load(f)
        if verbose:
            print(f"Loaded existing results from {save_path}", file=sys.stderr)
        # Load existing per-query data if available
        pq_path = save_path.with_name(save_path.stem + "_perquery.json")
        if pq_path.is_file():
            try:
                with open(pq_path) as f:
                    per_query_data = json.load(f)
            except Exception:
                per_query_data = {}

    # Get and schedule tasks
    tasks = get_dataset_tasks(config)
    batches = schedule_tasks(tasks, config.max_workers)

    if verbose:
        total_tasks = sum(len(b) for b in batches)
        print(f"Evaluating {total_tasks} datasets in {len(batches)} batches")
        print(f"  Tokenizer: {config.tokenizer}")
        print(f"  Sample queries: {config.sample_queries or 'all'}")
        print(f"  Threads per worker: {config.threads_per_worker}")
        print()

    # Config dict for serialization to workers
    config_dict = {
        "sample_queries": config.sample_queries,
        "seed": config.seed,
        "tokenizer": config.tokenizer,
        "threads_per_worker": config.threads_per_worker,
        "beir_data_dir": config.beir_data_dir,
        "trec_dl_data_dir": config.trec_dl_data_dir,
    }

    results: list[DatasetResult] = []

    # Process batches
    for batch_idx, batch in enumerate(batches):
        if verbose:
            batch_names = [t.full_name for t in batch]
            print(f"Batch {batch_idx + 1}/{len(batches)}: {', '.join(batch_names)}")

        # Prepare worker arguments - also track task info for error handling
        worker_args = [
            (program_path, task.benchmark, task.dataset_name, config_dict)
            for task in batch
        ]
        task_info = {task.dataset_name: task for task in batch}

        # Run batch in parallel
        batch_results = []
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(_worker_evaluate, args): args[2]  # dataset_name
                for args in worker_args
            }

            for future in as_completed(futures):
                dataset_name = futures[future]
                task = task_info[dataset_name]
                try:
                    result = future.result(timeout=1800)  # 30 min timeout

                    # Handle combined results (returns list of DatasetResults)
                    if isinstance(result, list):
                        for r in result:
                            batch_results.append(r)
                            results.append(r)
                            _collect_per_query(r, per_query_data)
                            if verbose:
                                status = "OK" if r.error is None else f"ERROR: {r.error}"
                                print(f"  {r.name}: {status}")

                            # Incremental save: save each dataset result immediately
                            if save_path:
                                partial = aggregate_results([r])
                                _merge_partial_into(existing, partial)
                                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                                with open(tmp_path, "w") as f:
                                    json.dump(existing, f, indent=2)
                                tmp_path.replace(save_path)
                                _save_per_query_incremental(save_path, per_query_data)
                    else:
                        batch_results.append(result)
                        results.append(result)
                        _collect_per_query(result, per_query_data)
                        if verbose:
                            status = "OK" if result.error is None else f"ERROR: {result.error}"
                            print(f"  {result.name}: {status}")

                        # Incremental save: save each dataset result immediately
                        if save_path:
                            partial = aggregate_results([result])
                            _merge_partial_into(existing, partial)
                            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                            with open(tmp_path, "w") as f:
                                json.dump(existing, f, indent=2)
                            tmp_path.replace(save_path)
                            _save_per_query_incremental(save_path, per_query_data)
                except Exception as e:
                    # Handle timeout or other errors - use correct prefix
                    if task.benchmark == "trec_dl_combined":
                        # Combined task failed - return error for both DL19 and DL20
                        for name in ["trec_dl_dl19", "trec_dl_dl20"]:
                            error_result = DatasetResult(
                                name=name,
                                ndcg_at_10=0.0,
                                recall_at_100=0.0,
                                index_time_ms=0.0,
                                query_time_ms=0.0,
                                num_docs=0,
                                num_queries=0,
                                error=f"Worker failed: {e}",
                            )
                            batch_results.append(error_result)
                            results.append(error_result)
                            if verbose:
                                print(f"  {error_result.name}: ERROR: {error_result.error}")

                            # Incremental save: save error result immediately
                            if save_path:
                                partial = aggregate_results([error_result])
                                _merge_partial_into(existing, partial)
                                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                                with open(tmp_path, "w") as f:
                                    json.dump(existing, f, indent=2)
                                tmp_path.replace(save_path)
                    else:
                        full_name = f"{task.benchmark}_{dataset_name}"
                        error_result = DatasetResult(
                            name=full_name,
                            ndcg_at_10=0.0,
                            recall_at_100=0.0,
                            index_time_ms=0.0,
                            query_time_ms=0.0,
                            num_docs=0,
                            num_queries=0,
                            error=f"Worker failed: {e}",
                        )
                        batch_results.append(error_result)
                        results.append(error_result)
                        if verbose:
                            print(f"  {error_result.name}: ERROR: {error_result.error}")

                        # Incremental save: save error result immediately
                        if save_path:
                            partial = aggregate_results([error_result])
                            _merge_partial_into(existing, partial)
                            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                            with open(tmp_path, "w") as f:
                                json.dump(existing, f, indent=2)
                            tmp_path.replace(save_path)

    # Return the accumulated existing dict if we were saving, otherwise aggregate from results
    if save_path and existing:
        return existing
    else:
        return aggregate_results(results)


def aggregate_results(results: list[DatasetResult]) -> dict[str, Any]:
    """
    Aggregate dataset results into a flat dictionary for OpenEvolve.
    
    Args:
        results: List of per-dataset results
        
    Returns:
        Flat dictionary with all metrics
    """
    output: dict[str, Any] = {}
    
    all_ndcg = []
    all_recall = []
    total_index_time = 0.0
    total_query_time = 0.0
    datasets_evaluated = 0
    datasets_failed = 0
    
    for result in results:
        prefix = result.name
        
        # Per-dataset metrics (no num_docs/num_queries to keep trace/checkpoints smaller)
        output[f"{prefix}_ndcg@10"] = result.ndcg_at_10
        output[f"{prefix}_recall@100"] = result.recall_at_100
        output[f"{prefix}_index_time_ms"] = result.index_time_ms
        output[f"{prefix}_query_time_ms"] = result.query_time_ms
        
        if result.error:
            output[f"{prefix}_error"] = result.error
            datasets_failed += 1
        else:
            all_ndcg.append(result.ndcg_at_10)
            all_recall.append(result.recall_at_100)
            total_index_time += result.index_time_ms
            total_query_time += result.query_time_ms
            datasets_evaluated += 1
    
    # Aggregate metrics
    avg_ndcg = float(np.mean(all_ndcg)) if all_ndcg else 0.0
    avg_recall = float(np.mean(all_recall)) if all_recall else 0.0
    
    output["avg_ndcg@10"] = avg_ndcg
    output["avg_recall@100"] = avg_recall
    # Zero score if any dataset failed (avoids reward hacking from partial/crashed runs)
    output["combined_score"] = 0.0 if datasets_failed > 0 else (0.8 * avg_recall + 0.2 * avg_ndcg)
    output["average_score"] = 0.0 if datasets_failed > 0 else (0.5 * avg_ndcg + 0.5 * avg_recall)
    
    # Timing
    output["total_index_time_ms"] = total_index_time
    output["total_query_time_ms"] = total_query_time
    output["total_time_ms"] = total_index_time + total_query_time
    
    # Metadata
    output["datasets_evaluated"] = datasets_evaluated
    output["datasets_failed"] = datasets_failed
    output["error"] = 0.0 if datasets_evaluated > 0 else 1.0
    
    return output


# =============================================================================
# OpenEvolve Entrypoint
# =============================================================================


def evaluate(program_path: str) -> dict[str, float]:
    """
    OpenEvolve entrypoint: Evaluate Query Likelihood on all benchmarks.

    Configuration via environment variables:
    - EVAL_SAMPLE_QUERIES: Sample N queries per dataset (0 = all)
    - EVAL_TOKENIZER: simple or lucene
    - EVAL_MAX_WORKERS: Max parallel workers (0 = auto)
    - EVAL_THREADS_PER_WORKER: Threads for tokenization
    - EVAL_BENCHMARKS: Which benchmarks to run:
        - "all": BRIGHT + BEIR + TREC_DL (default)
        - "bright": Only BRIGHT (12 datasets)
        - "beir": Only BEIR (17 datasets)
        - "bright+beir": BRIGHT + BEIR (29 datasets, no TREC_DL)

    Returns:
        Flat dictionary with combined_score and per-dataset metrics
    """
    try:
        # Parse benchmark selection from env
        benchmarks = os.environ.get("EVAL_BENCHMARKS", "all").lower().strip()
        
        include_bright = True
        include_beir = True
        include_trec_dl = True
        
        if benchmarks == "bright":
            include_beir = False
            include_trec_dl = False
        elif benchmarks == "beir":
            include_bright = False
            include_trec_dl = False
        elif benchmarks == "bright+beir" or benchmarks == "beir+bright":
            include_trec_dl = False
        # else "all" - include everything
        
        config = EvalConfig(
            include_bright=include_bright,
            include_beir=include_beir,
            include_trec_dl=include_trec_dl,
        )
        return evaluate_parallel(program_path, config, verbose=False)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "avg_ndcg@10": 0.0,
            "avg_recall@100": 0.0,
            "error": 1.0,
            "error_message": str(e),
        }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallelized Query Likelihood evaluation on ALL IR benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (31 datasets)
  python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py

  # Fast iteration with sampling
  python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --sample-queries 20

  # BRIGHT only
  python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --only-bright

  # Pyserini baseline (LMDirichletSimilarity)
  python evaluator_ql_parallel.py pyserini --save results/baselines/ql_pyserini.json

  # Save results to file
  python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --save results/baseline.json

  # Control parallelism
  python evaluator_ql_parallel.py src/ranking_evolved/ql_freeform_fast.py --max-workers 16

Parallelization:
  - Dataset-level: ProcessPoolExecutor (isolated memory)
  - Tokenization: ThreadPoolExecutor within each worker
  - Memory-aware: Small/medium/large batching

Output includes:
  - Per-dataset: nDCG@10, Recall@100, index_time_ms, query_time_ms
  - Aggregate: avg_nDCG@10, avg_Recall@100, combined_score
  - Timing: total_index_time_ms, total_query_time_ms
""",
    )
    parser.add_argument("program_path", help="Path to QL implementation file")
    parser.add_argument(
        "--sample-queries", type=int, default=0,
        help="Sample N queries per dataset (0 = all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--tokenizer", choices=["simple", "lucene"], default="lucene",
        help="Tokenizer to use"
    )
    parser.add_argument(
        "--max-workers", type=int, default=0,
        help="Max parallel workers (0 = auto)"
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=8,
        help="Threads for tokenization per worker"
    )
    parser.add_argument(
        "--only-bright", action="store_true",
        help="Only evaluate BRIGHT datasets"
    )
    parser.add_argument(
        "--only-beir", action="store_true",
        help="Only evaluate BEIR datasets"
    )
    parser.add_argument(
        "--only-trec-dl", action="store_true",
        help="Only evaluate TREC DL datasets"
    )
    parser.add_argument(
        "--save", "-s", type=str, default=None,
        help="Save results to JSON file (creates parent dirs if needed)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )
    args = parser.parse_args()

    from pathlib import Path

    config = EvalConfig(
        sample_queries=args.sample_queries if args.sample_queries > 0 else None,
        seed=args.seed,
        tokenizer=args.tokenizer,
        max_workers=args.max_workers,
        threads_per_worker=args.threads_per_worker,
        include_bright=not (args.only_beir or args.only_trec_dl),
        include_beir=not (args.only_bright or args.only_trec_dl),
        include_trec_dl=not (args.only_bright or args.only_beir),
    )

    save_path = Path(args.save) if args.save else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Smart incremental evaluation: detect missing and failed datasets
    if save_path and save_path.is_file():
        with open(save_path) as f:
            existing = json.load(f)

        # Get all expected datasets for this evaluation
        all_tasks = get_dataset_tasks(config)
        expected_datasets = {task.full_name for task in all_tasks}

        # trec_dl_combined is a virtual task that produces trec_dl_dl19 + trec_dl_dl20.
        # Expand it so we compare against the actual result keys.
        if "trec_dl_combined" in expected_datasets:
            expected_datasets.discard("trec_dl_combined")
            expected_datasets.update({"trec_dl_dl19", "trec_dl_dl20"})

        # Get completed and failed datasets from existing results
        completed_datasets = _get_all_prefixes(existing)
        failed_datasets = _get_failed_datasets(existing)
        completed_datasets -= failed_datasets

        # Datasets to run = (expected but not completed) union failed
        missing_datasets = expected_datasets - completed_datasets - failed_datasets
        to_run = missing_datasets | failed_datasets

        # Also check per-query companion file: if a dataset is "completed" in the
        # main JSON but missing from the per-query file, we need to rerun it so
        # that per-query scores are saved for significance testing.
        pq_path = save_path.with_name(save_path.stem + "_perquery.json")
        pq_completed: set[str] = set()
        if pq_path.is_file():
            try:
                with open(pq_path) as f:
                    pq_data = json.load(f)
                pq_completed = set(pq_data.keys())
            except Exception:
                pass
        missing_perquery = (completed_datasets - failed_datasets) - pq_completed
        if missing_perquery:
            to_run |= missing_perquery

        if to_run:
            # Split full dataset names into benchmark-specific lists
            bright_to_run = []
            beir_to_run = []
            trec_dl_to_run = []

            for full_name in to_run:
                if full_name.startswith("bright_"):
                    bright_to_run.append(full_name[7:])  # Remove "bright_" prefix
                elif full_name.startswith("beir_"):
                    beir_to_run.append(full_name[5:])  # Remove "beir_" prefix
                elif full_name.startswith("trec_dl_"):
                    # Handle both "trec_dl_combined" and "trec_dl_dl19", etc.
                    if full_name == "trec_dl_combined":
                        trec_dl_to_run = ["dl19", "dl20"]
                    else:
                        trec_dl_to_run.append(full_name[8:])  # Remove "trec_dl_" prefix

            # Create modified config with only the datasets to run
            config = EvalConfig(
                sample_queries=args.sample_queries if args.sample_queries > 0 else None,
                seed=args.seed,
                tokenizer=args.tokenizer,
                max_workers=args.max_workers,
                threads_per_worker=args.threads_per_worker,
                include_bright=bool(bright_to_run),
                include_beir=bool(beir_to_run),
                include_trec_dl=bool(trec_dl_to_run),
                bright_datasets=bright_to_run if bright_to_run else None,
                beir_datasets=beir_to_run if beir_to_run else None,
                trec_dl_datasets=trec_dl_to_run if trec_dl_to_run else None,
            )

            if args.verbose:
                print(f"Loaded existing results with {len(completed_datasets)} completed dataset(s)", file=sys.stderr)
                print(f"Running {len(to_run)} dataset(s):", file=sys.stderr)
                if failed_datasets:
                    print(f"  - {len(failed_datasets)} failed (rerunning): {sorted(failed_datasets)}", file=sys.stderr)
                if missing_datasets:
                    print(f"  - {len(missing_datasets)} missing (new): {sorted(missing_datasets)}", file=sys.stderr)
                if missing_perquery:
                    print(f"  - {len(missing_perquery)} missing per-query data: {sorted(missing_perquery)}", file=sys.stderr)
        else:
            if args.verbose:
                print(f"Results exist with all {len(expected_datasets)} dataset(s) completed successfully.", file=sys.stderr)
                if pq_path.is_file():
                    print(f"Per-query data also complete ({len(pq_completed)} datasets).", file=sys.stderr)
            print(json.dumps(existing, indent=2))
            sys.exit(0)

    results = evaluate_parallel(args.program_path, config, verbose=args.verbose, save_path=save_path)

    # Print summary
    if args.verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Combined Score: {results['combined_score']:.4f}", file=sys.stderr)
        print(f"Average Score:  {results.get('average_score', 0.0):.4f}", file=sys.stderr)
        print(f"  avg_nDCG@10:    {results['avg_ndcg@10']:.4f}", file=sys.stderr)
        print(f"  avg_Recall@100: {results['avg_recall@100']:.4f}", file=sys.stderr)
        print("Timing:", file=sys.stderr)
        print(f"  Index: {results['total_index_time_ms'] / 1000:.1f}s", file=sys.stderr)
        print(f"  Query: {results['total_query_time_ms'] / 1000:.1f}s", file=sys.stderr)
        print(f"  Total: {results['total_time_ms'] / 1000:.1f}s", file=sys.stderr)
        print(f"Datasets: {results['datasets_evaluated']} OK, {results['datasets_failed']} failed", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    # Final save with metadata
    if save_path:
        # Results were already written incrementally during evaluation
        # Final write ensures aggregates are up-to-date and metadata is added
        if save_path.is_file():
            with open(save_path) as f:
                final_existing = json.load(f)
            # Merge any remaining results (should be none, but ensure consistency)
            if results != final_existing:
                for key, value in results.items():
                    if key not in ("_metadata",):
                        final_existing[key] = value
        else:
            final_existing = results
        # Add/update metadata
        final_existing["_metadata"] = {
            "program_path": args.program_path,
            "tokenizer": args.tokenizer,
            "sample_queries": args.sample_queries if args.sample_queries > 0 else "all",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluator": "evaluator_ql_parallel",
        }
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(final_existing, f, indent=2)
        tmp_path.replace(save_path)
        if args.verbose:
            pq_path = save_path.with_name(save_path.stem + "_perquery.json")
            print(f"Final save: {save_path}", file=sys.stderr)
            if pq_path.is_file():
                print(f"Per-query data: {pq_path}", file=sys.stderr)
        print(json.dumps(final_existing, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Required for CUDA/large memory
    main()
