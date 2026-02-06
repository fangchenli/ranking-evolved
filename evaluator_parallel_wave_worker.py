"""
Worker for ProcessPoolExecutor used by evaluator_parallel_wave.py.

Stable module name so child processes can unpickle and resolve
evaluator_parallel_wave_worker._worker_evaluate_wave.
"""

from __future__ import annotations


def _worker_evaluate_wave(args: tuple) -> "DatasetResult | list[DatasetResult]":
    """Worker: run wave single-dataset or TREC DL combined with query progress."""
    from evaluator_parallel import (
        EvalConfig,
        evaluate_pyserini_official,
        evaluate_pyserini_trec_dl_combined,
    )
    from evaluator_parallel_wave import (
        DatasetResult,
        evaluate_single_dataset_wave,
        evaluate_trec_dl_combined_wave,
    )

    program_path, benchmark, dataset_name, config_dict = args
    verbose = config_dict.get("verbose", False)

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
        return evaluate_trec_dl_combined_wave(program_path, config, verbose=verbose)

    if program_path == "pyserini":
        return evaluate_pyserini_official(benchmark, dataset_name, config)

    return evaluate_single_dataset_wave(
        program_path, benchmark, dataset_name, config, verbose=verbose
    )
