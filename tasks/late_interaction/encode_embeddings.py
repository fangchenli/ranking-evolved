"""Precompute LateOn/PyLate contextualized token embedding caches.

Run from the repository root, for example:

    uv run python -m tasks.late_interaction.encode_embeddings --suite smoke

For the planned small/medium evaluator datasets:

    uv run python -m tasks.late_interaction.encode_embeddings --suite full

This module intentionally imports PyLate lazily so the NumPy-only oracle and tests
do not require neural retrieval dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tasks._shared.datasets import (
    BEIRLoader,
    BRIGHTLoader,
    BrightProLoader,
    EvalDataset,
    OBLIQBenchLoader,
)
from tasks.late_interaction.embedding_cache import (
    CacheMetadata,
    write_embedding_cache,
)

DEFAULT_MODEL = "lightonai/LateOn"

MAIN_SUITE = [
    "beir:scifact",
    "beir:nfcorpus",
    "beir:arguana",
    "beir:scidocs",
    "beir:fiqa",
    "beir:trec-covid",
    "bright:biology",
    "bright:earth_science",
]

NEW_BENCHMARKS_CURATED_SUITE = [
    "bright_pro:economics",
    "bright_pro:stackoverflow",
    "bright_pro:earth_science",
    "obliq:twitter",
    "obliq:congress",
]

NEW_BENCHMARKS_ALL_SUITE = [
    "bright_pro:biology",
    "bright_pro:earth_science",
    "bright_pro:economics",
    "bright_pro:psychology",
    "bright_pro:robotics",
    "bright_pro:stackoverflow",
    "bright_pro:sustainable_living",
    "obliq:math",
    "obliq:writing",
    "obliq:twitter",
    "obliq:wildchat",
    "obliq:congress",
]

SUITES = {
    "smoke": ["beir:scifact"],
    "early": ["beir:scifact", "beir:nfcorpus", "beir:arguana"],
    "main": MAIN_SUITE,
    "full": MAIN_SUITE,
    "new-benchmarks-curated": NEW_BENCHMARKS_CURATED_SUITE,
    "new-benchmarks-all": NEW_BENCHMARKS_ALL_SUITE,
}


@dataclass(frozen=True)
class EncodedChunkCollection:
    """Re-iterable view over temporary per-batch encoded embedding chunks."""

    chunk_paths: list[Path]
    lengths: list[int]
    embedding_dim: int
    dtype: str

    @property
    def num_items(self) -> int:
        return len(self.lengths)

    @property
    def total_tokens(self) -> int:
        return int(sum(self.lengths))

    @property
    def max_tokens(self) -> int:
        return max(self.lengths, default=0)

    def __iter__(self) -> Iterator[np.ndarray]:
        for chunk_path in self.chunk_paths:
            chunk = np.load(chunk_path, allow_pickle=True)
            for item in chunk:
                yield np.asarray(item, dtype=np.dtype(self.dtype))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_specs = resolve_dataset_specs(args)
    output_root = Path(args.output_dir)

    print(
        "[late_interaction] laptop note: encode one dataset at a time, "
        "write float16 caches, and lower --batch-size if memory pressure appears.",
        flush=True,
    )
    print(
        f"[late_interaction] suite={args.suite}, datasets={','.join(dataset_specs)}, "
        f"model={args.model}, dtype={args.dtype}, batch_size={args.batch_size}",
        flush=True,
    )

    model: Any | None = None

    for dataset_number, spec in enumerate(dataset_specs, start=1):
        print(
            f"[late_interaction] ({dataset_number}/{len(dataset_specs)}) loading dataset {spec}",
            flush=True,
        )
        dataset = load_eval_dataset(spec, beir_data_dir=args.beir_data_dir)
        if args.max_docs is not None:
            dataset = subset_docs(dataset, args.max_docs)
        if args.max_queries is not None:
            dataset = subset_queries(dataset, args.max_queries)

        cache_dir = output_root / safe_model_name(args.model) / dataset.name
        print(
            f"[late_interaction] {dataset.name}: docs={dataset.corpus_size}, "
            f"queries={dataset.num_queries}, cache={cache_dir}",
            flush=True,
        )

        if args.dry_run:
            continue

        cache_status = check_existing_cache(
            cache_dir,
            dataset=dataset,
            model_name=args.model,
            dtype=args.dtype,
        )
        if cache_status == "complete" and not args.overwrite:
            print(f"[late_interaction] skip {dataset.name}: complete cache already exists", flush=True)
            continue
        if cache_status == "incomplete" and not args.overwrite:
            raise SystemExit(
                f"Cache directory exists but is incomplete or incompatible: {cache_dir}\n"
                "Pass --overwrite to rebuild it."
            )

        if model is None:
            model = load_pylate_model(args.model)

        with tempfile.TemporaryDirectory(prefix=f"late_interaction_{dataset.name}_") as tmp:
            tmp_dir = Path(tmp)
            doc_chunks = encode_texts_to_chunks(
                model,
                dataset.corpus,
                is_query=False,
                batch_size=args.batch_size,
                dtype=args.dtype,
                max_tokens=args.max_doc_tokens,
                label=f"{dataset.name} documents",
                chunk_dir=tmp_dir / "docs",
            )
            query_chunks = encode_texts_to_chunks(
                model,
                dataset.queries,
                is_query=True,
                batch_size=args.batch_size,
                dtype=args.dtype,
                max_tokens=args.max_query_tokens,
                label=f"{dataset.name} queries",
                chunk_dir=tmp_dir / "queries",
            )

            metadata = build_metadata_from_chunks(
                dataset_name=dataset.name,
                benchmark=dataset.benchmark,
                model_name=args.model,
                doc_chunks=doc_chunks,
                query_chunks=query_chunks,
                dtype=args.dtype,
                qrels_modes=sorted(dataset.qrels_by_mode or {"gold": dataset.qrels}),
                has_excluded_ids=bool(dataset.excluded_ids),
                has_aspect_annotations=dataset.aspect_annotations is not None,
            )
            bytes_per_value = np.dtype(args.dtype).itemsize
            estimated_bytes = (
                metadata.total_doc_tokens + metadata.total_query_tokens
            ) * metadata.embedding_dim * bytes_per_value
            print(
                f"[late_interaction] writing cache for {dataset.name}: "
                f"doc_tokens={metadata.total_doc_tokens}, "
                f"query_tokens={metadata.total_query_tokens}, "
                f"estimated_embedding_bytes={estimated_bytes:,}",
                flush=True,
            )
            write_embedding_cache(
                cache_dir,
                doc_embeddings=doc_chunks,
                doc_ids=dataset.corpus_ids,
                query_embeddings=query_chunks,
                query_ids=dataset.query_ids,
                qrels=dataset.qrels,
                qrels_by_mode=dataset.qrels_by_mode,
                excluded_ids=dataset.excluded_ids,
                aspect_annotations=serialize_aspect_annotations(dataset),
                metadata=metadata,
                overwrite=args.overwrite,
            )
            print(
                f"[late_interaction] wrote {dataset.name}: "
                f"doc_tokens={metadata.total_doc_tokens}, "
                f"query_tokens={metadata.total_query_tokens}",
                flush=True,
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="PyLate/ColBERT model name")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="smoke",
        help="Dataset suite to encode. 'full' is the planned small BEIR+BRIGHT suite.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec such as beir:scifact or bright:biology. May be repeated.",
    )
    parser.add_argument("--output-dir", default="cache/late_interaction")
    parser.add_argument("--beir-data-dir", default="datasets/beir")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Encoding batch size. Laptop default is conservative; increase if memory allows.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-docs", type=int, default=None, help="Debug-only corpus truncation")
    parser.add_argument("--max-queries", type=int, default=None, help="Debug-only query truncation")
    parser.add_argument("--max-doc-tokens", type=int, default=None)
    parser.add_argument("--max-query-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets and print target cache paths without encoding",
    )
    return parser


def resolve_dataset_specs(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        return args.dataset
    return SUITES[args.suite]


def ensure_embedding_cache(
    dataset_id: str,
    *,
    cache_root: str | Path,
    model_name: str = DEFAULT_MODEL,
    beir_data_dir: str = "datasets/beir",
    batch_size: int = 16,
    dtype: str = "float16",
    max_doc_tokens: int | None = None,
    max_query_tokens: int | None = None,
    overwrite: bool = False,
    model: Any | None = None,
) -> Path:
    """Create one cache if missing; return the cache directory.

    `cache_root` is the model-specific evaluator root, e.g.
    `cache/late_interaction/lightonai__LateOn`, not the parent
    `cache/late_interaction` used by the standalone encoder CLI.
    """
    dataset = load_eval_dataset(dataset_id, beir_data_dir=beir_data_dir)
    cache_dir = Path(cache_root) / dataset.name
    status = check_existing_cache(cache_dir, dataset=dataset, model_name=model_name, dtype=dtype)
    if status == "complete" and not overwrite:
        return cache_dir
    if status == "incomplete" and not overwrite:
        raise RuntimeError(
            f"Cache directory exists but is incomplete or incompatible: {cache_dir}. "
            "Pass overwrite=True or remove the cache directory."
        )

    local_model = model if model is not None else load_pylate_model(model_name)
    with tempfile.TemporaryDirectory(prefix=f"late_interaction_{dataset.name}_") as tmp:
        tmp_dir = Path(tmp)
        doc_chunks = encode_texts_to_chunks(
            local_model,
            dataset.corpus,
            is_query=False,
            batch_size=batch_size,
            dtype=dtype,
            max_tokens=max_doc_tokens,
            label=f"{dataset.name} documents",
            chunk_dir=tmp_dir / "docs",
        )
        query_chunks = encode_texts_to_chunks(
            local_model,
            dataset.queries,
            is_query=True,
            batch_size=batch_size,
            dtype=dtype,
            max_tokens=max_query_tokens,
            label=f"{dataset.name} queries",
            chunk_dir=tmp_dir / "queries",
        )
        metadata = build_metadata_from_chunks(
            dataset_name=dataset.name,
            benchmark=dataset.benchmark,
            model_name=model_name,
            doc_chunks=doc_chunks,
            query_chunks=query_chunks,
            dtype=dtype,
            qrels_modes=sorted(dataset.qrels_by_mode or {"gold": dataset.qrels}),
            has_excluded_ids=bool(dataset.excluded_ids),
            has_aspect_annotations=dataset.aspect_annotations is not None,
        )
        write_embedding_cache(
            cache_dir,
            doc_embeddings=doc_chunks,
            doc_ids=dataset.corpus_ids,
            query_embeddings=query_chunks,
            query_ids=dataset.query_ids,
            qrels=dataset.qrels,
            qrels_by_mode=dataset.qrels_by_mode,
            excluded_ids=dataset.excluded_ids,
            aspect_annotations=serialize_aspect_annotations(dataset),
            metadata=metadata,
            overwrite=True,
        )
    return cache_dir


def check_existing_cache(
    cache_dir: Path,
    *,
    dataset: EvalDataset,
    model_name: str,
    dtype: str,
) -> str:
    """Return ``complete``, ``incomplete``, or ``missing`` for a cache directory."""

    required_files = [
        "metadata.json",
        "docs.embeddings.npy",
        "docs.lengths.npy",
        "docs.offsets.npy",
        "docs.ids.jsonl",
        "queries.embeddings.npy",
        "queries.lengths.npy",
        "queries.offsets.npy",
        "queries.ids.jsonl",
        "qrels.jsonl",
    ]
    if not cache_dir.exists():
        return "missing"
    if not cache_dir.is_dir():
        return "incomplete"
    if not any(cache_dir.iterdir()):
        return "missing"
    if any(not (cache_dir / name).is_file() for name in required_files):
        return "incomplete"

    try:
        metadata = CacheMetadata.from_json(cache_dir / "metadata.json")
        if metadata.dataset_name != dataset.name:
            return "incomplete"
        if metadata.benchmark != dataset.benchmark:
            return "incomplete"
        if metadata.model_name != model_name:
            return "incomplete"
        if np.dtype(metadata.dtype) != np.dtype(dtype):
            return "incomplete"
        if metadata.num_docs != dataset.corpus_size:
            return "incomplete"
        if metadata.num_queries != dataset.num_queries:
            return "incomplete"
        expected_modes = sorted(dataset.qrels_by_mode or {"gold": dataset.qrels})
        if sorted(metadata.qrels_modes or ["gold"]) != expected_modes:
            return "incomplete"
        if metadata.has_excluded_ids != bool(dataset.excluded_ids):
            return "incomplete"
        if metadata.has_aspect_annotations != (dataset.aspect_annotations is not None):
            return "incomplete"
        for mode in expected_modes:
            if mode != "gold" and not (cache_dir / f"qrels_{mode}.jsonl").is_file():
                return "incomplete"
        if metadata.has_excluded_ids and not (cache_dir / "excluded_ids.json").is_file():
            return "incomplete"
        if metadata.has_aspect_annotations and not (cache_dir / "aspect_annotations.json").is_file():
            return "incomplete"
        if not store_files_match_metadata(cache_dir, "docs", metadata):
            return "incomplete"
        if not store_files_match_metadata(cache_dir, "queries", metadata):
            return "incomplete"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return "incomplete"

    return "complete"


def store_files_match_metadata(cache_dir: Path, prefix: str, metadata: CacheMetadata) -> bool:
    expected_count = metadata.num_docs if prefix == "docs" else metadata.num_queries
    expected_tokens = metadata.total_doc_tokens if prefix == "docs" else metadata.total_query_tokens

    lengths = np.load(cache_dir / f"{prefix}.lengths.npy", mmap_mode="r")
    offsets = np.load(cache_dir / f"{prefix}.offsets.npy", mmap_mode="r")
    embeddings = np.load(cache_dir / f"{prefix}.embeddings.npy", mmap_mode="r")

    if len(lengths) != expected_count or len(offsets) != expected_count:
        return False
    if int(np.sum(lengths, dtype=np.int64)) != expected_tokens:
        return False
    if embeddings.shape != (expected_tokens, metadata.embedding_dim):
        return False
    if np.dtype(embeddings.dtype) != np.dtype(metadata.dtype):
        return False
    return True


def load_eval_dataset(spec: str, *, beir_data_dir: str) -> EvalDataset:
    spec = normalize_dataset_spec(spec)
    try:
        benchmark, name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Dataset spec must look like beir:scifact or bright:biology: {spec}") from exc

    if benchmark == "beir":
        return BEIRLoader(data_dir=beir_data_dir).load(name)
    if benchmark == "bright":
        return BRIGHTLoader().load(name)
    if benchmark == "obliq":
        return OBLIQBenchLoader().load(name)
    if benchmark == "bright_pro":
        return BrightProLoader().load(name)
    raise ValueError(f"Unsupported dataset benchmark: {benchmark}")


def normalize_dataset_spec(spec: str) -> str:
    """Accept either `benchmark:name` specs or canonical cache dataset ids."""
    if ":" in spec:
        return spec
    if spec.startswith("beir_"):
        return "beir:" + spec[len("beir_"):]
    if spec.startswith("bright_pro_"):
        return "bright_pro:" + spec[len("bright_pro_"):]
    if spec.startswith("bright_"):
        return "bright:" + spec[len("bright_"):]
    if spec.startswith("obliq_"):
        return "obliq:" + spec[len("obliq_"):]
    raise ValueError(f"Unsupported dataset identifier: {spec!r}")


def subset_docs(dataset: EvalDataset, max_docs: int) -> EvalDataset:
    kept_doc_ids = set(dataset.corpus_ids[:max_docs])
    qrels = {
        qid: {doc_id: rel for doc_id, rel in rels.items() if doc_id in kept_doc_ids}
        for qid, rels in dataset.qrels.items()
    }
    return EvalDataset(
        name=f"{dataset.name}_docs{max_docs}",
        benchmark=dataset.benchmark,
        corpus=dataset.corpus[:max_docs],
        corpus_ids=dataset.corpus_ids[:max_docs],
        queries=dataset.queries,
        query_ids=dataset.query_ids,
        qrels=qrels,
        qrels_by_mode={mode: {
            qid: {doc_id: rel for doc_id, rel in rels.items() if doc_id in kept_doc_ids}
            for qid, rels in mode_qrels.items()
        } for mode, mode_qrels in dataset.qrels_by_mode.items()},
        excluded_ids=dataset.excluded_ids,
        aspect_annotations=dataset.aspect_annotations,
    )


def subset_queries(dataset: EvalDataset, max_queries: int) -> EvalDataset:
    kept_query_ids = dataset.query_ids[:max_queries]
    return EvalDataset(
        name=f"{dataset.name}_queries{max_queries}",
        benchmark=dataset.benchmark,
        corpus=dataset.corpus,
        corpus_ids=dataset.corpus_ids,
        queries=dataset.queries[:max_queries],
        query_ids=kept_query_ids,
        qrels={qid: dataset.qrels.get(qid, {}) for qid in kept_query_ids},
        qrels_by_mode={
            mode: {qid: mode_qrels.get(qid, {}) for qid in kept_query_ids}
            for mode, mode_qrels in dataset.qrels_by_mode.items()
        },
        excluded_ids={qid: dataset.excluded_ids.get(qid, []) for qid in kept_query_ids},
        aspect_annotations=_subset_aspect_annotations(dataset, kept_query_ids),
    )


def load_pylate_model(model_name: str) -> Any:
    try:
        from pylate import models
    except ImportError as exc:
        raise SystemExit(
            "PyLate is required to encode LateOn embeddings.\n"
            "Install the neural retrieval dependencies with uv, for example:\n"
            "  uv add --group late-interaction pylate torch transformers\n"
            "Then rerun this module."
        ) from exc

    print(f"[late_interaction] loading PyLate model: {model_name}", flush=True)
    try:
        return models.ColBERT(model_name_or_path=model_name)
    except TypeError:
        return models.ColBERT(model_name)


def encode_texts_to_chunks(
    model: Any,
    texts: list[str],
    *,
    is_query: bool,
    batch_size: int,
    dtype: str,
    max_tokens: int | None,
    label: str,
    chunk_dir: Path,
) -> EncodedChunkCollection:
    print(
        f"[late_interaction] encoding {label}: items={len(texts)}, "
        f"batch_size={batch_size}, dtype={dtype}",
        flush=True,
    )
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    lengths: list[int] = []
    embedding_dim: int | None = None
    total_batches = (len(texts) + batch_size - 1) // batch_size

    batch_starts = range(0, len(texts), batch_size)
    for batch_index, start in enumerate(
        progress_batches(batch_starts, total=total_batches, label=f"saving {label} batches"),
        start=1,
    ):
        end = min(start + batch_size, len(texts))
        encoded = model.encode(
            texts[start:end],
            is_query=is_query,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        arrays = [to_numpy_2d(item, dtype=dtype) for item in encoded]
        if max_tokens is not None:
            arrays = [arr[:max_tokens] for arr in arrays]
        if arrays:
            current_dim = int(arrays[0].shape[1])
            if embedding_dim is None:
                embedding_dim = current_dim
            elif embedding_dim != current_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension for {label}: {embedding_dim} vs {current_dim}"
                )

        chunk_path = chunk_dir / f"chunk_{batch_index:06d}.npy"
        save_object_chunk(chunk_path, arrays)
        chunk_paths.append(chunk_path)
        lengths.extend(int(arr.shape[0]) for arr in arrays)

    if embedding_dim is None:
        raise ValueError(f"Cannot encode empty text collection for {label}")

    return EncodedChunkCollection(
        chunk_paths=chunk_paths,
        lengths=lengths,
        embedding_dim=embedding_dim,
        dtype=np.dtype(dtype).name,
    )


def progress_batches(batch_starts: range, *, total: int, label: str) -> Iterator[int]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from batch_starts
        return

    yield from tqdm(batch_starts, total=total, desc=f"[late_interaction] {label}", unit="batch")


def save_object_chunk(path: Path, arrays: list[np.ndarray]) -> None:
    chunk = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        chunk[index] = array
    np.save(path, chunk, allow_pickle=True)


def build_metadata_from_chunks(
    *,
    dataset_name: str,
    benchmark: str,
    model_name: str,
    doc_chunks: EncodedChunkCollection,
    query_chunks: EncodedChunkCollection,
    dtype: str,
    qrels_modes: list[str] | None = None,
    has_excluded_ids: bool = False,
    has_aspect_annotations: bool = False,
) -> CacheMetadata:
    if doc_chunks.embedding_dim != query_chunks.embedding_dim:
        raise ValueError("document and query embedding dimensions differ")

    return CacheMetadata(
        dataset_name=dataset_name,
        benchmark=benchmark,
        model_name=model_name,
        embedding_dim=doc_chunks.embedding_dim,
        dtype=np.dtype(dtype).name,
        num_docs=doc_chunks.num_items,
        num_queries=query_chunks.num_items,
        total_doc_tokens=doc_chunks.total_tokens,
        total_query_tokens=query_chunks.total_tokens,
        max_doc_tokens=doc_chunks.max_tokens,
        max_query_tokens=query_chunks.max_tokens,
        qrels_modes=qrels_modes or ["gold"],
        has_excluded_ids=has_excluded_ids,
        has_aspect_annotations=has_aspect_annotations,
    )


def serialize_aspect_annotations(dataset: EvalDataset) -> dict[str, Any] | None:
    annotations = dataset.aspect_annotations
    if annotations is None:
        return None
    return {
        "query_aspect_weights": annotations.query_aspect_weights,
        "query_doc_to_aspect": annotations.query_doc_to_aspect,
        "query_aspect_content": annotations.query_aspect_content,
    }


def _subset_aspect_annotations(dataset: EvalDataset, query_ids: list[str]):
    annotations = dataset.aspect_annotations
    if annotations is None:
        return None
    from tasks._shared.datasets import AspectAnnotations

    kept = set(query_ids)
    return AspectAnnotations(
        query_aspect_weights={
            qid: weights
            for qid, weights in annotations.query_aspect_weights.items()
            if qid in kept
        },
        query_doc_to_aspect={
            qid: mapping
            for qid, mapping in annotations.query_doc_to_aspect.items()
            if qid in kept
        },
        query_aspect_content={
            qid: content
            for qid, content in annotations.query_aspect_content.items()
            if qid in kept
        },
    )


def to_numpy_2d(value: Any, *, dtype: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.ndim != 2:
        raise ValueError(f"Expected one 2D token embedding matrix, got shape {array.shape}")
    return array


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
