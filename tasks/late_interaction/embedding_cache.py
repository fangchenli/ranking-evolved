"""Memmap-backed embedding cache for late-interaction retrieval.

The cache stores variable-length token embeddings in one contiguous matrix plus
per-item offsets and lengths. This keeps evaluator runs independent from the
encoder and makes exact MaxSim / approximate retrieval code operate on a stable
NumPy-only interface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CACHE_VERSION = 1


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata required to validate and reopen an embedding cache."""

    dataset_name: str
    benchmark: str
    model_name: str
    embedding_dim: int
    dtype: str
    num_docs: int
    num_queries: int
    total_doc_tokens: int
    total_query_tokens: int
    max_doc_tokens: int
    max_query_tokens: int
    cache_version: int = CACHE_VERSION

    @classmethod
    def from_json(cls, path: Path) -> CacheMetadata:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class TokenEmbeddingStore:
    """Variable-length token embedding store backed by a contiguous matrix."""

    ids: list[str]
    embeddings: NDArray[np.float32] | np.memmap
    lengths: NDArray[np.int64]
    offsets: NDArray[np.int64]

    def __post_init__(self) -> None:
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self._validate()
        self._id_to_index = {item_id: idx for idx, item_id in enumerate(self.ids)}

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        return int(self.embeddings.shape[1])

    @property
    def total_tokens(self) -> int:
        return int(self.embeddings.shape[0])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> NDArray[np.float32]:
        return self.get(index)

    def get(self, index: int) -> NDArray[np.float32]:
        start = int(self.offsets[index])
        end = start + int(self.lengths[index])
        return np.asarray(self.embeddings[start:end])

    def get_by_id(self, item_id: str) -> NDArray[np.float32]:
        return self.get(self.index_of(item_id))

    def index_of(self, item_id: str) -> int:
        try:
            return self._id_to_index[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown embedding id: {item_id}") from exc

    def _validate(self) -> None:
        if len(self.ids) != len(self.lengths) or len(self.ids) != len(self.offsets):
            raise ValueError("ids, lengths, and offsets must have identical lengths")
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique")
        if np.any(self.lengths < 0):
            raise ValueError("lengths must be non-negative")
        if np.any(self.offsets < 0):
            raise ValueError("offsets must be non-negative")
        if len(self.offsets) == 0:
            if self.embeddings.shape[0] != 0:
                raise ValueError("empty store cannot contain token embeddings")
            return

        expected_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(self.lengths[:-1], dtype=np.int64)]
        )
        if not np.array_equal(self.offsets, expected_offsets):
            raise ValueError("offsets must be contiguous cumulative sums of lengths")
        expected_total = int(np.sum(self.lengths, dtype=np.int64))
        if expected_total != int(self.embeddings.shape[0]):
            raise ValueError(
                f"lengths sum to {expected_total}, but embeddings contain {self.embeddings.shape[0]} rows"
            )


@dataclass
class LateInteractionCache:
    """Loaded query/document embedding cache."""

    metadata: CacheMetadata
    docs: TokenEmbeddingStore
    queries: TokenEmbeddingStore
    qrels: dict[str, dict[str, int]]


def write_embedding_cache(
    cache_dir: str | Path,
    *,
    doc_embeddings: Iterable[NDArray[Any]],
    doc_ids: list[str],
    query_embeddings: Iterable[NDArray[Any]],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    metadata: CacheMetadata,
    overwrite: bool = False,
) -> None:
    """Write a complete late-interaction cache.

    Args:
        cache_dir: Target directory.
        doc_embeddings: One ``(doc_tokens, dim)`` array per document.
        doc_ids: Document IDs aligned with ``doc_embeddings``.
        query_embeddings: One ``(query_tokens, dim)`` array per query.
        query_ids: Query IDs aligned with ``query_embeddings``.
        qrels: Relevance judgments.
        metadata: Cache metadata. Counts and token totals are validated.
        overwrite: Whether to replace an existing cache directory.
    """

    path = Path(cache_dir)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Cache already exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)

    doc_lengths = _scan_arrays(doc_embeddings, metadata.embedding_dim, metadata.dtype)
    query_lengths = _scan_arrays(query_embeddings, metadata.embedding_dim, metadata.dtype)

    _validate_metadata_against_lengths(metadata, doc_lengths, doc_ids, query_lengths, query_ids)

    _write_store(path, "docs", doc_embeddings, doc_lengths, doc_ids, metadata.dtype, metadata.embedding_dim)
    _write_store(
        path,
        "queries",
        query_embeddings,
        query_lengths,
        query_ids,
        metadata.dtype,
        metadata.embedding_dim,
    )
    _write_qrels(path / "qrels.jsonl", qrels)
    metadata.to_json(path / "metadata.json")


def load_embedding_cache(cache_dir: str | Path, mmap_mode: str = "r") -> LateInteractionCache:
    """Load a complete late-interaction cache."""

    path = Path(cache_dir)
    metadata = CacheMetadata.from_json(path / "metadata.json")
    docs = _load_store(
        path,
        "docs",
        dtype=metadata.dtype,
        embedding_dim=metadata.embedding_dim,
        expected_total_tokens=metadata.total_doc_tokens,
        mmap_mode=mmap_mode,
    )
    queries = _load_store(
        path,
        "queries",
        dtype=metadata.dtype,
        embedding_dim=metadata.embedding_dim,
        expected_total_tokens=metadata.total_query_tokens,
        mmap_mode=mmap_mode,
    )
    qrels = _read_qrels(path / "qrels.jsonl")

    if len(docs) != metadata.num_docs:
        raise ValueError("metadata num_docs does not match loaded document store")
    if len(queries) != metadata.num_queries:
        raise ValueError("metadata num_queries does not match loaded query store")

    return LateInteractionCache(metadata=metadata, docs=docs, queries=queries, qrels=qrels)


def build_metadata(
    *,
    dataset_name: str,
    benchmark: str,
    model_name: str,
    doc_embeddings: Iterable[NDArray[Any]],
    query_embeddings: Iterable[NDArray[Any]],
    dtype: str,
) -> CacheMetadata:
    """Build metadata from embedding arrays."""

    doc_arrays = list(doc_embeddings)
    query_arrays = list(query_embeddings)
    embedding_dim = _infer_embedding_dim(doc_arrays, query_arrays)
    doc_lengths = [int(np.asarray(arr).shape[0]) for arr in doc_arrays]
    query_lengths = [int(np.asarray(arr).shape[0]) for arr in query_arrays]

    return CacheMetadata(
        dataset_name=dataset_name,
        benchmark=benchmark,
        model_name=model_name,
        embedding_dim=embedding_dim,
        dtype=np.dtype(dtype).name,
        num_docs=len(doc_arrays),
        num_queries=len(query_arrays),
        total_doc_tokens=int(sum(doc_lengths)),
        total_query_tokens=int(sum(query_lengths)),
        max_doc_tokens=max(doc_lengths, default=0),
        max_query_tokens=max(query_lengths, default=0),
    )


def _as_2d_array(array: NDArray[Any], embedding_dim: int, dtype: str) -> NDArray[Any]:
    arr = np.asarray(array, dtype=np.dtype(dtype))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D token embedding array, got shape {arr.shape}")
    if arr.shape[1] != embedding_dim:
        raise ValueError(f"expected embedding dim {embedding_dim}, got {arr.shape[1]}")
    return arr


def _infer_embedding_dim(
    doc_arrays: list[NDArray[Any]], query_arrays: list[NDArray[Any]]
) -> int:
    for arr in [*doc_arrays, *query_arrays]:
        as_array = np.asarray(arr)
        if as_array.ndim != 2:
            raise ValueError(f"expected 2D token embedding array, got shape {as_array.shape}")
        return int(as_array.shape[1])
    raise ValueError("cannot infer embedding dimension from an empty cache")


def _validate_metadata_against_lengths(
    metadata: CacheMetadata,
    doc_lengths: NDArray[np.int64],
    doc_ids: list[str],
    query_lengths: NDArray[np.int64],
    query_ids: list[str],
) -> None:
    if metadata.num_docs != len(doc_lengths) or metadata.num_docs != len(doc_ids):
        raise ValueError("metadata num_docs does not match document arrays/ids")
    if metadata.num_queries != len(query_lengths) or metadata.num_queries != len(query_ids):
        raise ValueError("metadata num_queries does not match query arrays/ids")
    if metadata.total_doc_tokens != int(np.sum(doc_lengths, dtype=np.int64)):
        raise ValueError("metadata total_doc_tokens does not match document arrays")
    if metadata.total_query_tokens != int(np.sum(query_lengths, dtype=np.int64)):
        raise ValueError("metadata total_query_tokens does not match query arrays")
    if metadata.max_doc_tokens != int(np.max(doc_lengths, initial=0)):
        raise ValueError("metadata max_doc_tokens does not match document arrays")
    if metadata.max_query_tokens != int(np.max(query_lengths, initial=0)):
        raise ValueError("metadata max_query_tokens does not match query arrays")


def _scan_arrays(
    arrays: Iterable[NDArray[Any]],
    embedding_dim: int,
    dtype: str,
) -> NDArray[np.int64]:
    lengths: list[int] = []
    for arr in arrays:
        checked = _as_2d_array(arr, embedding_dim, dtype)
        lengths.append(int(checked.shape[0]))
    return np.asarray(lengths, dtype=np.int64)


def _write_store(
    cache_dir: Path,
    prefix: str,
    arrays: Iterable[NDArray[Any]],
    lengths: NDArray[np.int64],
    ids: list[str],
    dtype: str,
    embedding_dim: int,
) -> None:
    offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    total_tokens = int(np.sum(lengths, dtype=np.int64))

    embeddings = np.lib.format.open_memmap(
        cache_dir / f"{prefix}.embeddings.npy",
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(total_tokens, embedding_dim),
    )
    cursor = 0
    count = 0
    for arr in arrays:
        checked = _as_2d_array(arr, embedding_dim, dtype)
        next_cursor = cursor + int(checked.shape[0])
        embeddings[cursor:next_cursor] = checked
        cursor = next_cursor
        count += 1
    embeddings.flush()

    if cursor != total_tokens:
        raise ValueError(f"{prefix} embeddings yielded {cursor} tokens, expected {total_tokens}")
    if count != len(ids):
        raise ValueError(f"{prefix} embeddings yielded {count} arrays, expected {len(ids)}")

    np.save(cache_dir / f"{prefix}.lengths.npy", lengths)
    np.save(cache_dir / f"{prefix}.offsets.npy", offsets)
    _write_ids(cache_dir / f"{prefix}.ids.jsonl", ids)


def _load_store(
    cache_dir: Path,
    prefix: str,
    *,
    dtype: str,
    embedding_dim: int,
    expected_total_tokens: int,
    mmap_mode: str,
) -> TokenEmbeddingStore:
    embeddings = np.load(cache_dir / f"{prefix}.embeddings.npy", mmap_mode=mmap_mode)
    if embeddings.shape != (expected_total_tokens, embedding_dim):
        raise ValueError(
            f"{prefix} embeddings shape {embeddings.shape} does not match "
            f"expected {(expected_total_tokens, embedding_dim)}"
        )
    if np.dtype(embeddings.dtype) != np.dtype(dtype):
        raise ValueError(f"{prefix} embeddings dtype {embeddings.dtype} does not match {dtype}")

    return TokenEmbeddingStore(
        ids=_read_ids(cache_dir / f"{prefix}.ids.jsonl"),
        embeddings=embeddings,
        lengths=np.load(cache_dir / f"{prefix}.lengths.npy"),
        offsets=np.load(cache_dir / f"{prefix}.offsets.npy"),
    )


def _write_ids(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item_id in ids:
            f.write(json.dumps({"id": item_id}, ensure_ascii=False) + "\n")


def _read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(str(json.loads(line)["id"]))
    return ids


def _write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for query_id in sorted(qrels):
            for doc_id, relevance in sorted(qrels[query_id].items()):
                f.write(
                    json.dumps(
                        {"query_id": query_id, "doc_id": doc_id, "relevance": int(relevance)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row["relevance"])
    return qrels
