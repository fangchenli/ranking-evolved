"""
Worker entrypoint for ProcessPoolExecutor used by evaluator_ql_parallel.py.

This module exists so that when OpenEvolve (or any loader) imports evaluator_ql_parallel
under a different name (e.g. "evaluation_module"), spawned worker processes can still
unpickle the task: the task references evaluator_ql_parallel_worker._worker_evaluate,
which is a stable module name that resolves when the child imports it.
"""

from __future__ import annotations


def _worker_evaluate(args: tuple) -> "DatasetResult | list[DatasetResult]":
    """Worker function for ProcessPoolExecutor. Imports inside to avoid circular import."""
    from evaluator_ql_parallel import (
        EvalConfig,
        DatasetResult,
        evaluate_single_dataset,
        evaluate_pyserini_official,
        evaluate_trec_dl_combined,
        evaluate_pyserini_trec_dl_combined,
    )

    program_path, benchmark, dataset_name, config_dict = args

    config = EvalConfig(
        sample_queries=config_dict.get("sample_queries"),
        seed=config_dict.get("seed", 42),
        tokenizer=config_dict.get("tokenizer", "lucene"),
        threads_per_worker=config_dict.get("threads_per_worker", 8),
        beir_data_dir=config_dict.get("beir_data_dir", "datasets/beir"),
        trec_dl_data_dir=config_dict.get("trec_dl_data_dir", "datasets/trec_dl"),
    )

    if benchmark == "trec_dl_combined":
        if program_path == "pyserini":
            return evaluate_pyserini_trec_dl_combined(config)
        return evaluate_trec_dl_combined(program_path, config)

    if program_path == "pyserini":
        return evaluate_pyserini_official(benchmark, dataset_name, config)

    return evaluate_single_dataset(program_path, benchmark, dataset_name, config)
