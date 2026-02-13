"""
Constrained BM25 seed — safe search over known primitives.

Stay within BM25: tune hyperparameters (k1, b, k3), swap IDF/TF/length-norm
formulas for known alternatives, and combine them meaningfully. No exploration
of novel retrieval ideas; efficient grid-search over a known search space.

Evaluator contract: BM25, Corpus, tokenize, LuceneTokenizer; BM25.rank(), BM25.score().
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from ranking_evolved.bm25 import LuceneTokenizer as _BaseLuceneTokenizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_QUERY_WORKERS = min(int(os.environ.get("BM25_QUERY_WORKERS", 32)), 64)
MIN_QUERIES_FOR_PARALLEL = 10


# -----------------------------------------------------------------------------
# Parameters — EVOLVE: k1, b, k3 (e.g. k1 in [0.5, 2.0], b in [0, 1])
# -----------------------------------------------------------------------------

class Parameters:
    k1: float = 0.9
    b: float = 0.4
    k3: float = 8.0


# -----------------------------------------------------------------------------
# IDF — EVOLVE: swap for known formulas (Lucene, Robertson, ATIRE, BM25L, BM25+)
# -----------------------------------------------------------------------------

def idf(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    """EVOLVE: Lucene default; alternatives: Robertson log((N-df+0.5)/(df+0.5)), ATIRE log(N/df), BM25L log((N+1)/(df+0.5)), BM25+ log((N+1)/df)."""
    return np.log(1.0 + (N - df + 0.5) / (df + 0.5))


# -----------------------------------------------------------------------------
# TF saturation — EVOLVE: swap for known formulas (Lucene, Robertson, log, etc.)
# -----------------------------------------------------------------------------

def tf_saturated(tf: NDArray[np.float64], k1: float, norm: NDArray[np.float64]) -> NDArray[np.float64]:
    """EVOLVE: Lucene tf/(tf+k1*norm); Robertson (k1+1)*tf/(tf+k1*norm); log log(1+tf)/(tf+k1*norm)."""
    return tf / (tf + k1 * norm + 1e-9)


def tf_saturated_scalar(tf: float, k1: float, norm: float) -> float:
    return tf / (tf + k1 * norm + 1e-9)


# -----------------------------------------------------------------------------
# Length norm — EVOLVE: swap for known formulas (pivoted, none, log, sqrt)
# -----------------------------------------------------------------------------

def length_norm(doc_lengths: NDArray[np.float64], avgdl: float, b: float) -> NDArray[np.float64]:
    """EVOLVE: Pivoted 1-b+b*dl/avgdl; none 1.0; log 1/log(e+dl); sqrt 1/sqrt(dl)."""
    return 1.0 - b + b * (doc_lengths / max(avgdl, 1.0))


# -----------------------------------------------------------------------------
# Query term weights — EVOLVE: unique (1 per term), count (qtf), saturated (k3)
# -----------------------------------------------------------------------------

def query_weights(
    query: list[str], k3: float, mode: str = "count"
) -> tuple[list[str], NDArray[np.float64]]:
    """EVOLVE: unique (bag-of-words), count (qtf), saturated (k3+1)*qtf/(k3+qtf). Default count matches Pyserini."""
    if not query:
        return [], np.array([], dtype=np.float64)
    cnt = Counter(query)
    terms = list(cnt.keys())
    if mode == "unique":
        w = np.ones(len(terms), dtype=np.float64)
    elif mode == "count":
        w = np.array([float(cnt[t]) for t in terms], dtype=np.float64)
    elif mode == "saturated":
        qtf = np.array([float(cnt[t]) for t in terms], dtype=np.float64)
        w = (k3 + 1.0) * qtf / (k3 + qtf)
    else:
        w = np.array([float(cnt[t]) for t in terms], dtype=np.float64)
    return terms, w


# -----------------------------------------------------------------------------
# Tokenization (fixed for evaluator)
# -----------------------------------------------------------------------------

_TOKENIZER: _BaseLuceneTokenizer | None = None
_TOKENIZER_LOCK = threading.Lock()

def _get_tokenizer() -> _BaseLuceneTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        with _TOKENIZER_LOCK:
            if _TOKENIZER is None:
                _TOKENIZER = _BaseLuceneTokenizer()
    return _TOKENIZER

def tokenize(text: str) -> list[str]:
    return _get_tokenizer()(text)

class LuceneTokenizer:
    def __init__(self):
        self._tokenizer = _BaseLuceneTokenizer()
    def __call__(self, text: str) -> list[str]:
        return self._tokenizer(text)


# -----------------------------------------------------------------------------
# Corpus (interface fixed for evaluator)
# -----------------------------------------------------------------------------

class Corpus:
    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        # MEMORY OPTIMIZATION: Don't store documents - only needed during construction
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0

        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        V = len(self._vocab)

        tf_lil = lil_matrix((V, self.N), dtype=np.float64)
        self._posting_lists: dict[int, list[int]] = {i: [] for i in range(V)}
        self._df = np.zeros(V, dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            seen = set()
            for term, count in Counter(doc).items():
                tid = self._vocab[term]
                tf_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._posting_lists[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_lil)
        self._posting_lists = {
            tid: np.array(doc_ids, dtype=np.int64)
            for tid, doc_ids in self._posting_lists.items()
            if doc_ids
        }

    def __len__(self) -> int:
        return self.N

    def get_df(self, term: str) -> int:
        tid = self._vocab.get(term)
        return int(self._df[tid]) if tid is not None else 0

    def get_df_by_id(self, term_id: int) -> int:
        return int(self._df[term_id])

    def get_tf(self, doc_idx: int, term: str) -> int:
        tid = self._vocab.get(term)
        return int(self.tf_matrix[tid, doc_idx]) if tid is not None else 0

    def get_tf_by_id(self, term_id: int, doc_idx: int) -> float:
        return float(self.tf_matrix[term_id, doc_idx])

    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        tid = self._vocab.get(term)
        return self._posting_lists.get(tid, np.array([], dtype=np.int64)) if tid is not None else np.array([], dtype=np.int64)

    def get_posting_list_by_id(self, term_id: int) -> NDArray[np.int64]:
        return self._posting_lists.get(term_id, np.array([], dtype=np.int64))

    def get_term_id(self, term: str) -> int | None:
        return self._vocab.get(term)

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self._id_to_idx[i] for i in ids if i in self._id_to_idx]

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def idf_array(self) -> NDArray[np.float64]:
        return idf(self._df, self.N)

    @property
    def term_doc_matrix(self) -> None:
        return None


# -----------------------------------------------------------------------------
# BM25 (interface fixed for evaluator)
# -----------------------------------------------------------------------------

class BM25:
    def __init__(
        self,
        corpus: Corpus,
        k1: float | None = None,
        b: float | None = None,
        k3: float | None = None,
    ):
        self.corpus = corpus
        self.k1 = k1 if k1 is not None else Parameters.k1
        self.b = b if b is not None else Parameters.b
        self.k3 = k3 if k3 is not None else Parameters.k3
        self.idf_array = idf(corpus._df, corpus.N)
        self.norm_array = length_norm(corpus.doc_lengths, corpus.avgdl, self.b)
        self._idf_by_term = {term: float(self.idf_array[tid]) for term, tid in corpus._vocab.items()}

    def score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """EVOLVE: same formula as vectorized path (IDF × saturated TF, sum)."""
        norm = self.norm_array[doc_idx]
        s = 0.0
        for term in query_terms:
            idf_val = self._idf_by_term.get(term, 0.0)
            if idf_val == 0:
                continue
            tf_val = self.corpus.get_tf(doc_idx, term)
            if tf_val == 0:
                continue
            s += idf_val * tf_saturated_scalar(float(tf_val), self.k1, norm)
        return s

    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
        query_term_weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)
        norms = self.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        for i, tid in enumerate(query_term_ids):
            idf_val = self.idf_array[tid]
            if idf_val <= 0:
                continue
            w = query_term_weights[i] if query_term_weights is not None else 1.0
            tf_row = self.corpus.tf_matrix[tid, candidate_docs].toarray().flatten()
            scores += w * idf_val * tf_saturated(tf_row, self.k1, norms)
        return scores

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        if not query:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)
        terms, weights = query_weights(query, self.k3, "count")
        term_ids = []
        w_arr = []
        for t, w in zip(terms, weights):
            tid = self.corpus.get_term_id(t)
            if tid is not None:
                term_ids.append(tid)
                w_arr.append(w)
        if not term_ids:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)
        w_arr = np.array(w_arr, dtype=np.float64)

        # For large corpora, use NumPy operations instead of Python sets to avoid memory overhead
        posting_lists = []
        for tid in term_ids:
            pl = self.corpus.get_posting_list_by_id(tid)
            if len(pl) > 0:
                posting_lists.append(pl)

        if not posting_lists:
            candidate_docs = np.array([], dtype=np.int64)
        elif len(posting_lists) == 1:
            candidate_docs = posting_lists[0]  # Already sorted in posting list
        else:
            # np.unique sorts and deduplicates - more memory efficient than Python set for large arrays
            candidate_docs = np.unique(np.concatenate(posting_lists))
        cand_scores = self._score_candidates_vectorized(term_ids, candidate_docs, w_arr)
        all_scores = np.zeros(self.corpus.N, dtype=np.float64)
        all_scores[candidate_docs] = cand_scores
        sorted_indices = np.argsort(-all_scores).astype(np.int64)
        sorted_scores = all_scores[sorted_indices]
        if top_k is not None:
            sorted_indices, sorted_scores = sorted_indices[:top_k], sorted_scores[:top_k]
        return sorted_indices, sorted_scores

    def batch_rank(
        self,
        queries: list[list[str]],
        top_k: int | None = None,
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        if len(queries) < MIN_QUERIES_FOR_PARALLEL:
            return [self.rank(q, top_k) for q in queries]
        with ThreadPoolExecutor(max_workers=NUM_QUERY_WORKERS) as ex:
            return list(ex.map(lambda q: self.rank(q, top_k), queries))

    def score(self, query: list[str], doc_idx: int) -> float:
        return self.score_document(query, doc_idx)


__all__ = [
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "Parameters",
    "idf",
    "tf_saturated",
    "length_norm",
    "query_weights",
]
