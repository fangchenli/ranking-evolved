"""
BM25T — BM25 with term-specific k1 via log-logistic estimation.

Reference: Lv & Zhai, "A Log-Logistic Model-Based Interpretation of TF
Normalization of BM25", ECIR 2012 / KAIS 2012.
https://hal.science/hal-00617973v1/file/KAIS.pdf

Also described in: Trotman, Puurula, Burgess, "Improvements to BM25 and
Language Models Examined", ADCS 2014.
https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Key idea: Rather than using a single global k1 for all terms, BM25T
estimates a term-specific k1' from the distribution of length-normalized
term frequencies using the log-moment estimation method.

Procedure per term t:
    1. Compute ctd = tf / (1 - b + b * dl / avgdl) for all docs containing t
       (the "elite set" C_w).
    2. Compute target = mean(log(ctd + 1)) over the elite set.
    3. Find k1' such that g(k1') = target, where:
       g(k1) = k1/(k1-1) * log(k1)   if k1 != 1
       g(1)  = 1
    4. Substitute k1' into BM25 for this term.

Score (BM25T):
    IDF:  log((N + 1) / (df + 0.5))   [BM25L-style IDF]
    norm: 1 - b + b * dl / avgdl
    TF:   (k1'_t + 1) * tf / (k1'_t * norm + tf)
    Score = sum_t  IDF_t * TF_t

Default parameter: b=0.3 (from Trotman et al. training results)
"""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from ranking_evolved.bm25 import (
    ENGLISH_STOPWORDS,
    LUCENE_STOPWORDS,
    LuceneTokenizer as _BaseLuceneTokenizer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_QUERY_WORKERS = min(int(os.environ.get("BM25_QUERY_WORKERS", 32)), 64)
MIN_QUERIES_FOR_PARALLEL = 10


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

class Config:
    b: float = 0.3        # length normalization (from Trotman training)
    k1_min: float = 0.1   # minimum allowed k1' per term
    k1_max: float = 10.0  # maximum allowed k1' per term
    k1_default: float = 1.2  # fallback k1' if estimation fails
    epsilon: float = 1e-9


# -----------------------------------------------------------------------------
# IDF — BM25T uses BM25L-style IDF: log((N + 1) / (df + 0.5))
# -----------------------------------------------------------------------------

def idf(df: float | NDArray[np.float64], N: int) -> float | NDArray[np.float64]:
    """BM25T IDF: log((N + 1) / (df + 0.5)).  Always non-negative."""
    return np.log((N + 1.0) / (df + 0.5))


# -----------------------------------------------------------------------------
# Log-logistic k1 estimation
# -----------------------------------------------------------------------------

def _g_func(k1: float) -> float:
    """
    g(k1) = k1/(k1-1) * log(k1) if k1 != 1, else 1.

    This is the expected value of log(X+1) where X follows a log-logistic
    distribution with shape parameter k1.
    """
    if abs(k1 - 1.0) < 1e-10:
        return 1.0
    return k1 / (k1 - 1.0) * math.log(k1)


def _g_derivative(k1: float) -> float:
    """
    Derivative of g(k1) w.r.t. k1, for Newton-Raphson.

    g(k1) = k1/(k1-1) * ln(k1)
    g'(k1) = d/dk1 [k1/(k1-1) * ln(k1)]
           = [(k1-1)*ln(k1) + k1/(k1-1)*(k1-1) - k1*ln(k1)] / (k1-1)^2
           Wait, let me do this properly.

    Let f = k1/(k1-1), h = ln(k1)
    g = f * h
    g' = f' * h + f * h'
    f' = -1/(k1-1)^2
    h' = 1/k1
    g' = -ln(k1)/(k1-1)^2 + 1/(k1-1)
    """
    if abs(k1 - 1.0) < 1e-10:
        return 0.5  # limit as k1 -> 1
    return -math.log(k1) / (k1 - 1.0) ** 2 + 1.0 / (k1 - 1.0)


def _estimate_k1_newton(target: float, max_iter: int = 50) -> float:
    """
    Solve g(k1) = target using Newton-Raphson.

    g(k1) is monotonically increasing from 0 (at k1->0+) to infinity.
    For target > 0, there is always a unique solution.
    """
    if target <= 0:
        return Config.k1_default

    # Initial guess: k1 = 1.5 (reasonable starting point)
    k1 = 1.5

    for _ in range(max_iter):
        g_val = _g_func(k1)
        g_der = _g_derivative(k1)
        if abs(g_der) < 1e-15:
            break
        step = (g_val - target) / g_der
        k1_new = k1 - step
        # Clamp to valid range
        k1_new = max(Config.k1_min, min(Config.k1_max, k1_new))
        if abs(k1_new - k1) < 1e-10:
            break
        k1 = k1_new

    return max(Config.k1_min, min(Config.k1_max, k1))


def _compute_term_k1(
    ctd_values: NDArray[np.float64],
    df_t: int,
) -> float:
    """
    Compute term-specific k1' using log-moment estimation.

    Args:
        ctd_values: Length-normalized TF values for all docs containing the term.
        df_t: Document frequency of the term.

    Returns:
        k1' for this term.
    """
    if df_t == 0 or len(ctd_values) == 0:
        return Config.k1_default

    # Target = mean(log(ctd + 1)) over the elite set
    target = float(np.mean(np.log(ctd_values + 1.0)))

    if target <= 0:
        return Config.k1_default

    return _estimate_k1_newton(target)


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
# Corpus
# -----------------------------------------------------------------------------

class Corpus:
    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.document_count = self.N
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        # Build vocabulary
        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        # Build sparse TF matrix and inverted index
        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            term_counts = Counter(doc)
            seen: set[int] = set()
            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_matrix_lil)

        # IDF: log((N+1)/(df+0.5))
        self.idf_array = np.asarray(idf(self._df, self.N), dtype=np.float64)

        # Precompute length normalization: 1 - b + b * dl / avgdl
        b = Config.b
        self.norm_array = 1.0 - b + b * (self.doc_lengths / max(self.avgdl, 1.0))

        self.document_frequency = Counter(
            {term: max(1, int(self._df[tid])) for term, tid in self._vocab.items()}
        )

        self._posting_lists: dict[int, NDArray[np.int64]] = {
            tid: np.array(doc_ids, dtype=np.int64)
            for tid, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index

        # ---------------------------------------------------------------
        # Compute per-term k1' using log-logistic estimation
        # ---------------------------------------------------------------
        self._term_k1 = np.full(self.vocab_size, Config.k1_default, dtype=np.float64)

        for term, tid in self._vocab.items():
            df_t = int(self._df[tid])
            if df_t == 0:
                continue
            posting = self._posting_lists.get(tid)
            if posting is None or len(posting) == 0:
                continue
            # Get raw TF values for all docs containing this term
            tf_vals = self.tf_matrix[tid, posting].toarray().flatten()
            # Compute ctd (length-normalized TF)
            norms = self.norm_array[posting]
            ctd_vals = tf_vals / (norms + Config.epsilon)

            self._term_k1[tid] = _compute_term_k1(ctd_vals, df_t)

        self.document_length = self.doc_lengths

    def __len__(self) -> int:
        return self.N

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> Corpus:
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)

    def get_df(self, term: str) -> int:
        tid = self._vocab.get(term)
        return max(1, int(self._df[tid])) if tid is not None else 1

    def get_tf(self, doc_idx: int, term: str) -> int:
        tid = self._vocab.get(term)
        return int(self.tf_matrix[tid, doc_idx]) if tid is not None else 0

    def get_term_frequencies(self, doc_idx: int) -> Counter[str]:
        result = Counter()
        for term, tid in self._vocab.items():
            tf = int(self.tf_matrix[tid, doc_idx])
            if tf > 0:
                result[term] = tf
        return result

    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        tid = self._vocab.get(term)
        if tid is not None:
            return self._posting_lists.get(tid, np.array([], dtype=np.int64))
        return np.array([], dtype=np.int64)

    def get_term_id(self, term: str) -> int | None:
        return self._vocab.get(term)

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self._id_to_idx[i] for i in ids if i in self._id_to_idx]

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx

    @property
    def term_frequency(self) -> list[Counter[str]]:
        return [self.get_term_frequencies(i) for i in range(self.N)]

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

    @property
    def term_doc_matrix(self) -> None:
        return None


# -----------------------------------------------------------------------------
# BM25T scoring
# -----------------------------------------------------------------------------

def retrieval_score(
    query_terms: list[str],
    doc_tf: Counter[str],
    doc_length: float,
    corpus: Corpus,
) -> float:
    """Score one document for one query using BM25T."""
    b, eps = Config.b, Config.epsilon
    norm = 1.0 - b + b * (doc_length / (corpus.avgdl + eps)) if corpus.avgdl > 0 else 1.0
    score = 0.0
    for term in query_terms:
        tf = float(doc_tf.get(term, 0))
        if tf <= 0:
            continue
        tid = corpus.get_term_id(term)
        if tid is None:
            continue
        idf_val = float(corpus.idf_array[tid])
        if idf_val <= 0:
            continue
        k1 = corpus._term_k1[tid]
        tf_part = (k1 + 1.0) * tf / (k1 * norm + tf + eps)
        score += idf_val * tf_part
    return score


class BM25:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: list[str], index: int) -> float:
        if not query:
            return 0.0
        doc_tf = self.corpus.get_term_frequencies(index)
        doc_length = float(self.corpus.doc_lengths[index])
        return retrieval_score(query, doc_tf, doc_length, self.corpus)

    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
        query_term_weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Vectorized BM25T scoring."""
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        eps = Config.epsilon
        norms = self.corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            idf_val = self.corpus.idf_array[term_id]
            if idf_val <= 0:
                continue
            k1 = self.corpus._term_k1[term_id]
            w = query_term_weights[i] if query_term_weights is not None else 1.0
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().flatten()
            # BM25T: IDF * (k1'+1)*tf / (k1'*norm + tf)
            tf_part = (k1 + 1.0) * tf_row / (k1 * norms + tf_row + eps)
            scores += w * idf_val * tf_part

        return scores

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not query:
            return (
                np.arange(self.corpus.N, dtype=np.int64),
                np.zeros(self.corpus.N, dtype=np.float64),
            )
        term_counts = Counter(query)
        query_term_ids: list[int] = []
        query_term_weights: list[float] = []
        for term, count in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                query_term_weights.append(float(count))
        if not query_term_ids:
            return (
                np.arange(self.corpus.N, dtype=np.int64),
                np.zeros(self.corpus.N, dtype=np.float64),
            )
        qtf = np.array(query_term_weights, dtype=np.float64)

        # Gather candidate documents from posting lists
        posting_lists = []
        for tid in query_term_ids:
            pl = self.corpus._posting_lists.get(tid, np.array([], dtype=np.int64))
            if len(pl) > 0:
                posting_lists.append(pl)

        if not posting_lists:
            candidate_docs = np.array([], dtype=np.int64)
        elif len(posting_lists) == 1:
            candidate_docs = posting_lists[0]
        else:
            candidate_docs = np.unique(np.concatenate(posting_lists))

        candidate_scores = self._score_candidates_vectorized(
            query_term_ids, candidate_docs, qtf,
        )
        all_scores = np.zeros(self.corpus.N, dtype=np.float64)
        all_scores[candidate_docs] = candidate_scores
        sorted_indices = np.argsort(-all_scores).astype(np.int64)
        sorted_scores = all_scores[sorted_indices]
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
        return sorted_indices, sorted_scores

    def batch_rank(
        self,
        queries: list[list[str]],
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(queries) < MIN_QUERIES_FOR_PARALLEL:
            return [self.rank(q, top_k) for q in queries]
        with ThreadPoolExecutor(max_workers=NUM_QUERY_WORKERS) as ex:
            return list(ex.map(lambda q: self.rank(q, top_k), queries))


__all__ = [
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "LUCENE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "Config",
    "idf",
    "retrieval_score",
]
