"""
Wave HPC parallel evaluator for 96 CPUs and 600G RAM.

Tuned for WAVE HPC:
- 96 CPUs, 600G: many datasets in parallel per batch.
- Query-level progress to stderr (every 100 queries or 10% steps).
- With --save <path>: if path exists, rerun only failed datasets and refresh;
  if path does not exist, run full evaluation and save.

Concurrency: Do not run two processes with the same --save path; results would
race. ProcessPoolExecutor workers are independent (no shared mutable state);
main process does all file I/O. Saves are atomic (write to temp then rename).

Indexing bottleneck: For large corpora (e.g. beir_fever 5.4M, TREC DL 8.8M docs),
the indexing phase (tokenize + build corpus + build BM25) dominates runtime.
- Tokenization uses --threads-per-worker (default 32 on Wave); more threads = faster.
- Corpus and BM25 construction are single-threaded, so extra CPU cores do not help
  that phase. Query scoring: we use BM25.batch_rank() for all datasets (bright, beir,
  TREC DL) when available, so many queries are scored in parallel (threads share
  the same index in memory).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from evaluator_parallel import (
    BEIR_DATASETS,
    BRIGHT_SPLITS,
    NDCG_K,
    RECALL_K,
    LARGE_THRESHOLD,
    MEDIUM_THRESHOLD,
    SMALL_THRESHOLD,
    TINY_THRESHOLD,
    BEIRLoader,
    BRIGHTLoader,
    EvalDataset,
    TRECDLLoader,
    DatasetResult,
    DatasetTask,
    EvalConfig,
    aggregate_results,
    get_dataset_tasks,
    load_candidate,
    tokenize_batch,
)
from ranking_evolved.metrics import ndcg_at_k, recall_at_k

# -----------------------------------------------------------------------------
# Wave HPC tuning: 96 CPUs, 600G RAM
# -----------------------------------------------------------------------------
WAVE_MAX_WORKERS = 96
WAVE_MAX_TINY_WORKERS = 80
WAVE_MAX_SMALL_WORKERS = 48
WAVE_MAX_MEDIUM_WORKERS = 24
WAVE_MAX_LARGE_WORKERS = 12
WAVE_MAX_HUGE_WORKERS = 4

QUERY_PROGRESS_INTERVAL = 100

PER_DATASET_SUFFIXES = ("_ndcg@10", "_recall@100", "_index_time_ms", "_query_time_ms", "_error")
AGGREGATE_KEYS = (
    "avg_ndcg@10", "avg_recall@100", "combined_score",
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
    return {key[: -len("_ndcg@10")] for key in data if key.endswith("_ndcg@10")}


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


# Chunk size for tokenization progress (large corpora get periodic stderr updates)
TOKENIZE_CHUNK_FOR_PROGRESS = 200_000


def _tokenize_batch_with_progress(
    corpus: list[str],
    tokenize_fn,
    num_threads: int,
    full_name: str,
    verbose: bool,
) -> list[list[str]]:
    """Tokenize corpus in chunks and print progress for large corpora so runs don't appear to hang."""
    n = len(corpus)
    if n <= TOKENIZE_CHUNK_FOR_PROGRESS or not verbose:
        return tokenize_batch(corpus, tokenize_fn, num_threads=num_threads)
    out: list[list[str]] = []
    for start in range(0, n, TOKENIZE_CHUNK_FOR_PROGRESS):
        end = min(start + TOKENIZE_CHUNK_FOR_PROGRESS, n)
        chunk = tokenize_batch(corpus[start:end], tokenize_fn, num_threads=num_threads)
        out.extend(chunk)
        print(f"    {full_name}: tokenized {end}/{n} ({100 * end // n}%)", file=sys.stderr, flush=True)
    return out


def _print_query_progress(full_name: str, current: int, total: int, verbose: bool) -> None:
    if not verbose or total <= 0:
        return
    step = max(QUERY_PROGRESS_INTERVAL, total // 10, 1)
    if current % step == 0 or current == total:
        pct = 100 * current // total if total else 0
        print(f"    {full_name}: query {current}/{total} ({pct}%)", file=sys.stderr, flush=True)


def schedule_tasks_wave(
    tasks: list[DatasetTask],
    max_workers: int = 0,
) -> list[list[DatasetTask]]:
    """Schedule for 96 CPUs / 600G."""
    if max_workers == 0:
        max_workers = min(mp.cpu_count() or 96, WAVE_MAX_WORKERS)
    tiny = [t for t in tasks if t.estimated_size < TINY_THRESHOLD]
    small = [t for t in tasks if TINY_THRESHOLD <= t.estimated_size < SMALL_THRESHOLD]
    medium = [t for t in tasks if SMALL_THRESHOLD <= t.estimated_size < MEDIUM_THRESHOLD]
    large = [t for t in tasks if MEDIUM_THRESHOLD <= t.estimated_size < LARGE_THRESHOLD]
    huge = [t for t in tasks if t.estimated_size >= LARGE_THRESHOLD]
    batches = []
    if tiny:
        for i in range(0, len(tiny), min(WAVE_MAX_TINY_WORKERS, max_workers, len(tiny))):
            batches.append(tiny[i : i + min(WAVE_MAX_TINY_WORKERS, max_workers, len(tiny))])
    if small:
        for i in range(0, len(small), min(WAVE_MAX_SMALL_WORKERS, max_workers)):
            batches.append(small[i : i + min(WAVE_MAX_SMALL_WORKERS, max_workers)])
    if medium:
        for i in range(0, len(medium), min(WAVE_MAX_MEDIUM_WORKERS, max_workers)):
            batches.append(medium[i : i + min(WAVE_MAX_MEDIUM_WORKERS, max_workers)])
    if large:
        for i in range(0, len(large), min(WAVE_MAX_LARGE_WORKERS, max_workers)):
            batches.append(large[i : i + min(WAVE_MAX_LARGE_WORKERS, max_workers)])
    for task in huge:
        batches.append([task])
    return batches


def evaluate_single_dataset_wave(
    program_path: str,
    benchmark: str,
    dataset_name: str,
    config: EvalConfig,
    verbose: bool = False,
) -> DatasetResult:
    """Evaluate one dataset with query progress to stderr."""
    full_name = f"{benchmark}_{dataset_name}"
    try:
        BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)
        if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
            tokenize_fn = LuceneTokenizerCls()
        if benchmark == "bright":
            dataset = BRIGHTLoader().load(dataset_name)
        elif benchmark == "beir":
            dataset = BEIRLoader(data_dir=config.beir_data_dir).load(dataset_name)
        elif benchmark == "trec_dl":
            dataset = TRECDLLoader(data_dir=config.trec_dl_data_dir).load(dataset_name)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        index_start = time.perf_counter()
        n_docs = len(dataset.corpus)
        if verbose and n_docs > 50_000:
            print(f"    {full_name}: tokenizing {n_docs} documents (may take 20–60 min for 5M+ docs)...", file=sys.stderr, flush=True)
        doc_tokens = _tokenize_batch_with_progress(
            dataset.corpus, tokenize_fn, config.threads_per_worker, full_name, verbose
        )
        if verbose and n_docs > 50_000:
            print(f"    {full_name}: building corpus and BM25 index...", file=sys.stderr, flush=True)
        corpus = CorpusCls(doc_tokens, ids=dataset.corpus_ids)
        bm25 = BM25Impl(corpus)
        for attr in ("vocabulary_size", "idf_array", "term_doc_matrix"):
            if hasattr(corpus, attr):
                _ = getattr(corpus, attr)
        index_time_ms = (time.perf_counter() - index_start) * 1000

        query_ids = list(dataset.query_ids)
        queries = list(dataset.queries)
        if config.sample_queries and config.sample_queries < len(queries):
            rng = random.Random(config.seed)
            idx = rng.sample(range(len(queries)), config.sample_queries)
            query_ids = [query_ids[i] for i in idx]
            queries = [queries[i] for i in idx]
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dataset.corpus_ids)}
        all_relevant, all_retrieved = [], []
        total = len(query_ids)
        query_start = time.perf_counter()

        # Use batch_rank for all datasets when available (large corpus + long query list)
        batch_rank_fn = getattr(bm25, "batch_rank", None)
        if batch_rank_fn is not None:
            if verbose and total > 100:
                print(f"    {full_name}: tokenizing {total} queries...", file=sys.stderr, flush=True)
            query_tokens_list = [tokenize_fn(q) for q in queries]
            if verbose and total > 100:
                print(f"    {full_name}: batch_rank (top_k={RECALL_K})...", file=sys.stderr, flush=True)
            batch_results = batch_rank_fn(query_tokens_list, top_k=RECALL_K)
            for idx, (qid, (ranked_indices, _)) in enumerate(zip(query_ids, batch_results, strict=False)):
                relevant_doc_ids = dataset.get_relevant_docs(qid)
                if not relevant_doc_ids:
                    _print_query_progress(full_name, idx + 1, total, verbose)
                    continue
                relevant_indices = [id_to_idx[d] for d in relevant_doc_ids if d in id_to_idx]
                if not relevant_indices:
                    _print_query_progress(full_name, idx + 1, total, verbose)
                    continue
                all_relevant.append(np.array(relevant_indices, dtype=int))
                all_retrieved.append(np.array(ranked_indices, dtype=int))
                _print_query_progress(full_name, idx + 1, total, verbose)
            # Ensure final progress is printed even if loop finished quickly
            if verbose and total > 0:
                _print_query_progress(full_name, total, total, verbose)
        else:
            for idx, (qid, query_text) in enumerate(zip(query_ids, queries, strict=False)):
                query_tokens = tokenize_fn(query_text)
                ranked_indices, _ = bm25.rank(query_tokens)
                relevant_doc_ids = dataset.get_relevant_docs(qid)
                if not relevant_doc_ids:
                    _print_query_progress(full_name, idx + 1, total, verbose)
                    continue
                relevant_indices = [id_to_idx[d] for d in relevant_doc_ids if d in id_to_idx]
                if not relevant_indices:
                    _print_query_progress(full_name, idx + 1, total, verbose)
                    continue
                all_relevant.append(np.array(relevant_indices, dtype=int))
                all_retrieved.append(np.array(ranked_indices, dtype=int))
                _print_query_progress(full_name, idx + 1, total, verbose)

        query_time_ms = (time.perf_counter() - query_start) * 1000
        if not all_relevant:
            return DatasetResult(name=full_name, ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=index_time_ms,
                query_time_ms=query_time_ms, num_docs=len(dataset.corpus), num_queries=0, error="No valid queries")
        ndcg_scores = [ndcg_at_k(rel, ret, NDCG_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)]
        recall_scores = [recall_at_k(rel, ret, RECALL_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)]
        return DatasetResult(name=full_name, ndcg_at_10=float(np.mean(ndcg_scores)), recall_at_100=float(np.mean(recall_scores)),
            index_time_ms=index_time_ms, query_time_ms=query_time_ms, num_docs=len(dataset.corpus), num_queries=len(all_relevant))
    except Exception as e:
        return DatasetResult(name=full_name, ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=0.0, query_time_ms=0.0,
            num_docs=0, num_queries=0, error=str(e))


def _evaluate_queries_on_index_wave(
    bm25, dataset: EvalDataset, id_to_idx: dict, tokenize_fn, config: EvalConfig,
    full_name: str, index_time_ms: float, verbose: bool = False,
) -> DatasetResult:
    query_start = time.perf_counter()
    query_ids = list(dataset.query_ids)
    queries = list(dataset.queries)
    if config.sample_queries and config.sample_queries < len(queries):
        rng = random.Random(config.seed)
        idx = rng.sample(range(len(queries)), config.sample_queries)
        query_ids = [query_ids[i] for i in idx]
        queries = [queries[i] for i in idx]
    total = len(query_ids)

    # Use batch_rank when available for any dataset (large corpus + long query list)
    batch_rank_fn = getattr(bm25, "batch_rank", None)
    if batch_rank_fn is not None:
        if verbose and total > 100:
            print(f"    {full_name}: tokenizing {total} queries...", file=sys.stderr, flush=True)
        query_tokens_list = [tokenize_fn(q) for q in queries]
        if verbose and total > 100:
            print(f"    {full_name}: batch_rank (top_k={RECALL_K})...", file=sys.stderr, flush=True)
        batch_results = batch_rank_fn(query_tokens_list, top_k=RECALL_K)
        all_relevant, all_retrieved = [], []
        for i, (qid, (ranked_indices, _)) in enumerate(zip(query_ids, batch_results, strict=False)):
            relevant_doc_ids = dataset.get_relevant_docs(qid)
            if not relevant_doc_ids:
                _print_query_progress(full_name, i + 1, total, verbose)
                continue
            relevant_indices = [id_to_idx[d] for d in relevant_doc_ids if d in id_to_idx]
            if not relevant_indices:
                _print_query_progress(full_name, i + 1, total, verbose)
                continue
            all_relevant.append(np.array(relevant_indices, dtype=int))
            all_retrieved.append(np.array(ranked_indices, dtype=int))
            _print_query_progress(full_name, i + 1, total, verbose)
        # Ensure final progress is printed even if loop finished quickly
        if verbose and total > 0:
            _print_query_progress(full_name, total, total, verbose)
    else:
        all_relevant, all_retrieved = [], []
        for i, (qid, query_text) in enumerate(zip(query_ids, queries, strict=False)):
            query_tokens = tokenize_fn(query_text)
            ranked_indices, _ = bm25.rank(query_tokens)
            relevant_doc_ids = dataset.get_relevant_docs(qid)
            if not relevant_doc_ids:
                _print_query_progress(full_name, i + 1, total, verbose)
                continue
            relevant_indices = [id_to_idx[d] for d in relevant_doc_ids if d in id_to_idx]
            if not relevant_indices:
                _print_query_progress(full_name, i + 1, total, verbose)
                continue
            all_relevant.append(np.array(relevant_indices, dtype=int))
            all_retrieved.append(np.array(ranked_indices, dtype=int))
            _print_query_progress(full_name, i + 1, total, verbose)

    query_time_ms = (time.perf_counter() - query_start) * 1000
    if not all_relevant:
        return DatasetResult(name=full_name, ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=index_time_ms,
            query_time_ms=query_time_ms, num_docs=len(dataset.corpus), num_queries=0, error="No valid queries with relevant docs")
    ndcg_scores = [ndcg_at_k(rel, ret, NDCG_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)]
    recall_scores = [recall_at_k(rel, ret, RECALL_K) for rel, ret in zip(all_relevant, all_retrieved, strict=False)]
    return DatasetResult(name=full_name, ndcg_at_10=float(np.mean(ndcg_scores)), recall_at_100=float(np.mean(recall_scores)),
        index_time_ms=index_time_ms, query_time_ms=query_time_ms, num_docs=len(dataset.corpus), num_queries=len(all_relevant))


def evaluate_trec_dl_combined_wave(
    program_path: str, config: EvalConfig, verbose: bool = False,
) -> list[DatasetResult]:
    results = []
    try:
        BM25Impl, CorpusCls, tokenize_fn, LuceneTokenizerCls = load_candidate(program_path)
        if config.tokenizer == "lucene" and LuceneTokenizerCls is not None:
            tokenize_fn = LuceneTokenizerCls()
        loader = TRECDLLoader(data_dir=config.trec_dl_data_dir)
        dl19 = loader.load("dl19")
        index_start = time.perf_counter()
        n_docs = len(dl19.corpus)
        if verbose and n_docs > 50_000:
            print(f"    trec_dl_dl19: tokenizing {n_docs} documents (may take 20–60 min for 8M+ docs)...", file=sys.stderr, flush=True)
        doc_tokens = _tokenize_batch_with_progress(
            dl19.corpus, tokenize_fn, config.threads_per_worker, "trec_dl_dl19", verbose
        )
        if verbose and n_docs > 50_000:
            print(f"    trec_dl_dl19: building corpus and BM25 index...", file=sys.stderr, flush=True)
        corpus = CorpusCls(doc_tokens, ids=dl19.corpus_ids)
        bm25 = BM25Impl(corpus)
        for attr in ("vocabulary_size", "idf_array", "term_doc_matrix"):
            if hasattr(corpus, attr):
                _ = getattr(corpus, attr)
        shared_index_ms = (time.perf_counter() - index_start) * 1000
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(dl19.corpus_ids)}
        results.append(_evaluate_queries_on_index_wave(bm25, dl19, id_to_idx, tokenize_fn, config, "trec_dl_dl19", shared_index_ms, verbose))
        dl20 = loader.load("dl20")
        results.append(_evaluate_queries_on_index_wave(bm25, dl20, id_to_idx, tokenize_fn, config, "trec_dl_dl20", 0.0, verbose))
        return results
    except Exception as e:
        return [
            DatasetResult(name="trec_dl_dl19", ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=0.0, query_time_ms=0.0, num_docs=0, num_queries=0, error=str(e)),
            DatasetResult(name="trec_dl_dl20", ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=0.0, query_time_ms=0.0, num_docs=0, num_queries=0, error=str(e)),
        ]


from evaluator_parallel_wave_worker import _worker_evaluate_wave  # noqa: E402


def evaluate_parallel_wave(
    program_path: str, config: EvalConfig, verbose: bool = False, save_path: Path | None = None,
) -> dict[str, Any]:
    """Run evaluation with wave scheduling (96 CPUs / 600G) and query progress."""
    tasks = get_dataset_tasks(config)
    batches = schedule_tasks_wave(tasks, config.max_workers or WAVE_MAX_WORKERS)
    if verbose:
        total_tasks = sum(len(b) for b in batches)
        print(f"Evaluating {total_tasks} datasets in {len(batches)} batches (wave: 96 CPUs / 600G)", file=sys.stderr)
        print(f"  Tokenizer: {config.tokenizer}", file=sys.stderr)
        print(f"  Sample queries: {config.sample_queries or 'all'}", file=sys.stderr)
        print(file=sys.stderr)
    config_dict = {
        "sample_queries": config.sample_queries, "seed": config.seed, "tokenizer": config.tokenizer,
        "threads_per_worker": config.threads_per_worker, "beir_data_dir": config.beir_data_dir,
        "trec_dl_data_dir": config.trec_dl_data_dir, "verbose": verbose,
    }
    results: list[DatasetResult] = []
    existing: dict[str, Any] = {}
    if save_path and save_path.is_file():
        try:
            with open(save_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    
    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"Batch {batch_idx + 1}/{len(batches)}: {', '.join(t.full_name for t in batch)}", file=sys.stderr)
        worker_args = [(program_path, t.benchmark, t.dataset_name, config_dict) for t in batch]
        task_info = {t.dataset_name: t for t in batch}
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(_worker_evaluate_wave, a): a[2] for a in worker_args}
            for future in as_completed(futures):
                dataset_name = futures[future]
                task = task_info[dataset_name]
                try:
                    result = future.result(timeout=1800)
                    if isinstance(result, list):
                        for r in result:
                            results.append(r)
                            if verbose:
                                print(f"  {r.name}: {'OK' if r.error is None else f'ERROR: {r.error}'}", file=sys.stderr, flush=True)
                            # Incremental write: save each dataset result immediately
                            if save_path:
                                partial = aggregate_results([r])
                                if existing:
                                    _merge_partial_into(existing, partial)
                                    out = existing
                                else:
                                    out = partial
                                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                                with open(tmp_path, "w") as f:
                                    json.dump(out, f, indent=2)
                                tmp_path.replace(save_path)
                    else:
                        results.append(result)
                        if verbose:
                            print(f"  {result.name}: {'OK' if result.error is None else f'ERROR: {result.error}'}", file=sys.stderr, flush=True)
                        # Incremental write: save each dataset result immediately
                        if save_path:
                            partial = aggregate_results([result])
                            if existing:
                                _merge_partial_into(existing, partial)
                                out = existing
                            else:
                                out = partial
                            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                            with open(tmp_path, "w") as f:
                                json.dump(out, f, indent=2)
                            tmp_path.replace(save_path)
                except Exception as e:
                    if task.benchmark == "trec_dl_combined":
                        for name in ("trec_dl_dl19", "trec_dl_dl20"):
                            error_result = DatasetResult(name=name, ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=0.0, query_time_ms=0.0, num_docs=0, num_queries=0, error=f"Worker failed: {e}")
                            results.append(error_result)
                            if verbose:
                                print(f"  {error_result.name}: ERROR: {error_result.error}", file=sys.stderr, flush=True)
                            # Incremental write: save error result immediately
                            if save_path:
                                partial = aggregate_results([error_result])
                                if existing:
                                    _merge_partial_into(existing, partial)
                                    out = existing
                                else:
                                    out = partial
                                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                                with open(tmp_path, "w") as f:
                                    json.dump(out, f, indent=2)
                                tmp_path.replace(save_path)
                    else:
                        error_result = DatasetResult(name=f"{task.benchmark}_{task.dataset_name}", ndcg_at_10=0.0, recall_at_100=0.0, index_time_ms=0.0, query_time_ms=0.0, num_docs=0, num_queries=0, error=f"Worker failed: {e}")
                        results.append(error_result)
                        if verbose:
                            print(f"  {error_result.name}: ERROR: {error_result.error}", file=sys.stderr, flush=True)
                        # Incremental write: save error result immediately
                        if save_path:
                            partial = aggregate_results([error_result])
                            if existing:
                                _merge_partial_into(existing, partial)
                                out = existing
                            else:
                                out = partial
                            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                            with open(tmp_path, "w") as f:
                                json.dump(out, f, indent=2)
                            tmp_path.replace(save_path)
    return aggregate_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wave HPC evaluator (96 CPUs, 600G). With --save: full eval or rerun failed and refresh.")
    parser.add_argument("program_path", help="Path to BM25 implementation")
    parser.add_argument("--sample-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=["simple", "lucene"], default="lucene")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=int(os.environ.get("EVAL_THREADS_PER_WORKER", "32")),
        help="Threads for tokenization during indexing (default 32; corpus/BM25 build is single-threaded)",
    )
    parser.add_argument("--only-datasets", type=str, default=None)
    parser.add_argument("--only-bright", action="store_true")
    parser.add_argument("--only-beir", action="store_true")
    parser.add_argument("--only-trec-dl", action="store_true")
    parser.add_argument("--save", "-s", type=str, default=None, help="If file exists: rerun failed and refresh; else: full eval and save.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    only_datasets = None
    if args.only_datasets:
        only_datasets = {d.strip() for d in args.only_datasets.split(",") if d.strip()}

    save_path = Path(args.save) if args.save else None
    run_full = True
    existing: dict[str, Any] = {}

    if save_path and save_path.is_file():
        with open(save_path) as f:
            existing = json.load(f)
        failed = _get_failed_datasets(existing)
        if failed:
            only_datasets = failed
            run_full = False
            if args.verbose:
                print(f"Results exist; rerunning {len(failed)} failed datasets: {sorted(failed)}", file=sys.stderr)
        else:
            if args.verbose:
                print("Results exist and no failed datasets; nothing to run.", file=sys.stderr)
            print(json.dumps(existing, indent=2))
            sys.exit(0)

    config = EvalConfig(
        sample_queries=args.sample_queries if args.sample_queries > 0 else None,
        seed=args.seed, tokenizer=args.tokenizer,
        max_workers=args.max_workers or WAVE_MAX_WORKERS,
        threads_per_worker=args.threads_per_worker,
        include_bright=not (args.only_beir or args.only_trec_dl),
        include_beir=not (args.only_bright or args.only_trec_dl),
        include_trec_dl=not (args.only_bright or args.only_beir),
        include_only_datasets=only_datasets,
    )

    results = evaluate_parallel_wave(args.program_path, config, verbose=args.verbose, save_path=save_path)
    results["_metadata"] = {
        "program_path": args.program_path,
        "tokenizer": args.tokenizer,
        "sample_queries": args.sample_queries if args.sample_queries > 0 else "all",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator": "evaluator_parallel_wave",
    }

    if args.verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Combined Score: {results['combined_score']:.4f}", file=sys.stderr)
        print(f"  avg_nDCG@10: {results['avg_ndcg@10']:.4f}  avg_Recall@100: {results['avg_recall@100']:.4f}", file=sys.stderr)
        print(f"Datasets: {results['datasets_evaluated']} OK, {results['datasets_failed']} failed", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    if save_path:
        # Results were already written incrementally during evaluation
        # Final write ensures aggregates are up-to-date and metadata is added
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.is_file():
            with open(save_path) as f:
                final_existing = json.load(f)
            # Merge any remaining results (should be none, but ensure consistency)
            _merge_partial_into(final_existing, results)
            out = final_existing
        else:
            out = results
        # Add/update metadata
        out["_metadata"] = {
            "program_path": args.program_path,
            "tokenizer": args.tokenizer,
            "sample_queries": args.sample_queries if args.sample_queries > 0 else "all",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluator": "evaluator_parallel_wave",
        }
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(out, f, indent=2)
        tmp_path.replace(save_path)
        if args.verbose:
            print(f"Final save: {save_path}", file=sys.stderr)
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(results, indent=2))
