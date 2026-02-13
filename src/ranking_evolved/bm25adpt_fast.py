"""
BM25-adpt — BM25 with adaptive, term-specific k1 via information gain.

Reference: Lv & Zhai, "Adaptive Term Frequency Normalization for BM25", CIKM 2011.
https://dl.acm.org/doi/10.1145/2063576.2063871

Also described in: Trotman, Puurula, Burgess, "Improvements to BM25 and
Language Models Examined", ADCS 2014.
https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Key idea: A global k1 applied to all query terms is sub-optimal because
different terms have different TF saturation curves.  BM25-adpt computes a
term-specific k1' by aligning BM25's TF saturation function to the empirical
information gain curve derived from the index.

Procedure per term t:
    1. Compute ctd = tf / (1 - b + b * dl / avgdl) for all docs containing t.
    2. Compute df_r = |{d : ctd_d >= r - 0.5}| for r = 0, 1, 2, ...
       (df_0 = N, df_1 = df_t)
    3. Information gain at r:
       G_r = log2((df_{r+1} + 0.5)/(df_r + 1)) - log2((df_t + 0.5)/(N + 1))
    4. Fit k1' by minimizing:
       sum_r ((G_r / G_1) - (k1+1)*r/(k1+r))^2

Score:
    IDF replaced by G_1 (information gain at first occurrence)
    rsv = sum_t  G_1_t * (k1'_t + 1) * tf / (k1'_t * norm + tf)

Default parameter: b=0.3 (from Trotman et al. training results)
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar
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
    max_r: int = 20       # maximum r for information gain computation
    epsilon: float = 1e-9


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
# Per-term k1 estimation via information gain
# -----------------------------------------------------------------------------

def _compute_term_k1_and_g1(
    ctd_values: NDArray[np.float64],
    df_t: int,
    N: int,
) -> tuple[float, float]:
    """
    Compute term-specific k1' and G_1 (information gain at first occurrence).

    Args:
        ctd_values: Length-normalized TF values for all docs containing the term.
        df_t: Document frequency of the term.
        N: Total number of documents.

    Returns:
        (k1_prime, G_1)
    """
    if df_t == 0 or len(ctd_values) == 0:
        return Config.k1_default, 0.0

    # Sort ctd values in ascending order for efficient df_r computation
    sorted_ctd_asc = np.sort(ctd_values)

    # Compute df_r for r = 0, 1, 2, ..., max_r
    # df_0 = N, df_1 = df_t, df_r = |{d : ctd >= r - 0.5}| for r > 1
    max_r = min(Config.max_r, int(np.max(ctd_values) + 1.5))
    max_r = max(max_r, 2)  # need at least r=0,1 for G_1

    df_r = np.zeros(max_r + 2, dtype=np.float64)
    df_r[0] = float(N)
    df_r[1] = float(df_t)
    for r in range(2, max_r + 2):
        threshold = r - 0.5
        # Count docs with ctd >= threshold using searchsorted on ascending array
        count = len(sorted_ctd_asc) - int(np.searchsorted(sorted_ctd_asc, threshold, side='left'))
        df_r[r] = float(count)
        if count == 0:
            # All subsequent df_r will also be 0
            break

    # Base probability: log2((df_t + 0.5) / (N + 1))
    base_log = np.log2((df_t + 0.5) / (N + 1.0))

    # Compute information gain G_r for r = 0, 1, ..., max_r
    # G_r = log2((df_{r+1} + 0.5) / (df_r + 1)) - base_log
    G = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        if df_r[r] + 1 > 0 and df_r[r + 1] + 0.5 > 0:
            G[r] = np.log2((df_r[r + 1] + 0.5) / (df_r[r] + 1.0)) - base_log
        else:
            G[r] = 0.0

    G_1 = G[1] if len(G) > 1 else 0.0
    if G_1 <= 0:
        return Config.k1_default, max(G_1, 0.0)

    # Normalize: target_r = G_r / G_1
    # Find the range of r values where G_r > 0
    valid_rs = []
    targets = []
    for r in range(max_r + 1):
        if r == 0:
            continue  # skip r=0 (it's the base case)
        ratio = G[r] / G_1 if G_1 > 0 else 0.0
        if ratio >= 0:
            valid_rs.append(r)
            targets.append(ratio)

    if len(valid_rs) < 2:
        return Config.k1_default, G_1

    rs = np.array(valid_rs, dtype=np.float64)
    tgts = np.array(targets, dtype=np.float64)

    # Fit k1' by minimizing sum_r ((G_r/G_1) - (k1+1)*r/(k1+r))^2
    def objective(k1: float) -> float:
        bm25_curve = (k1 + 1.0) * rs / (k1 + rs)
        return float(np.sum((tgts - bm25_curve) ** 2))

    result = minimize_scalar(
        objective,
        bounds=(Config.k1_min, Config.k1_max),
        method="bounded",
    )
    k1_prime = float(result.x)
    return k1_prime, G_1


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
        # Compute per-term k1' and G_1 (information gain at 1st occurrence)
        # ---------------------------------------------------------------
        self._term_k1 = np.full(self.vocab_size, Config.k1_default, dtype=np.float64)
        self._term_g1 = np.zeros(self.vocab_size, dtype=np.float64)

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

            k1_prime, g1 = _compute_term_k1_and_g1(ctd_vals, df_t, self.N)
            self._term_k1[tid] = k1_prime
            self._term_g1[tid] = g1

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
# BM25-adpt scoring
# -----------------------------------------------------------------------------

def retrieval_score(
    query_terms: list[str],
    doc_tf: Counter[str],
    doc_length: float,
    corpus: Corpus,
) -> float:
    """Score one document for one query using BM25-adpt."""
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
        g1 = corpus._term_g1[tid]
        if g1 <= 0:
            continue
        k1 = corpus._term_k1[tid]
        tf_part = (k1 + 1.0) * tf / (k1 * norm + tf + eps)
        score += g1 * tf_part
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
        """Vectorized BM25-adpt scoring."""
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        eps = Config.epsilon
        norms = self.corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            g1 = self.corpus._term_g1[term_id]
            if g1 <= 0:
                continue
            k1 = self.corpus._term_k1[term_id]
            w = query_term_weights[i] if query_term_weights is not None else 1.0
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().flatten()
            # BM25-adpt: G_1 * (k1'+1)*tf / (k1'*norm + tf)
            tf_part = (k1 + 1.0) * tf_row / (k1 * norms + tf_row + eps)
            scores += w * g1 * tf_part

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
    "retrieval_score",
]
