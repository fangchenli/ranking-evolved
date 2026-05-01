"""
Freeform lexical retrieval seed — maximum freedom for discovering a new retrieval method.

Core idea: document representation + query representation + scoring method.
The evaluator requires: BM25, Corpus, tokenize, LuceneTokenizer; BM25 must have rank() and score().
Everything else is evolvable. Default behavior: Lucene BM25 (same as current seed).
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
# Config — EVOLVE: add parameters for your retrieval method
# -----------------------------------------------------------------------------

class Config:
    # BM25 core parameters (kept from the current best-performing island variant).
    k1: float = 0.9
    b: float = 0.4
    epsilon: float = 1e-9

    # BM25+: score_t = idf(t) * ( (tf*(k1+1))/(tf + k1*norm) + delta )
    delta: float = 0.5

    # Emphasize discriminative terms a bit more than plain BM25 by applying a mild
    # super-linear transform to IDF. This often helps early precision (nDCG@10)
    # without changing candidate generation.
    idf_power: float = 1.3

    # Verbosity-robust length: mix raw token length with unique-term count.
    # Helps avoid over-penalizing long but repetitive documents.
    lenmix: float = 0.5

    # Query-time term filtering to reduce candidate explosions from very common terms.
    # Often improves both latency and nDCG@10 without much recall@1000 loss.
    drop_stopwords_in_query: bool = True
    max_df_ratio: float = 0.80  # drop terms occurring in >80% of documents

    # Keep coordination for compatibility (default off).
    coord_weight: float = 0.0
    coord_k: float = 1.0


# -----------------------------------------------------------------------------
# IDF — EVOLVE: fundamental term importance (e.g. rarity, discriminativity)
# -----------------------------------------------------------------------------

def idf(df: float | NDArray[np.float64], N: int) -> float | NDArray[np.float64]:
    """Term importance from document frequency. EVOLVE: try other formulations."""
    return np.log(1.0 + (N - df + 0.5) / (df + 0.5))


# -----------------------------------------------------------------------------
# Document representation — EVOLVE: what to store per document
# -----------------------------------------------------------------------------

class DocumentRepr:
    def __init__(self, term_frequencies: Counter[str], length: float):
        self.term_frequencies = term_frequencies
        self.length = length

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> DocumentRepr:
        """EVOLVE: different document views (e.g. positions, fields)."""
        return cls(term_frequencies=Counter(tokens), length=float(len(tokens)))


# -----------------------------------------------------------------------------
# Query representation — EVOLVE: how to represent the query
# -----------------------------------------------------------------------------

class QueryRepr:
    def __init__(self, terms: list[str], term_weights: dict[str, float] | None = None):
        # `terms` is treated as a unique term list; weights carry any repetition signal.
        self.terms = terms
        self.term_weights = term_weights or {t: 1.0 for t in terms}

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> QueryRepr:
        """
        Represent the query as unique terms + a lightweight qtf weight.

        Important: rank() uses a vectorized path that works over unique term ids and a parallel
        weight vector. To keep rank() and score() consistent, we *must not* iterate over
        duplicate query tokens during scoring.

        Weighting: w_q(t) = 1 + log(qtf). This is cheap, robust, and avoids overweighting
        repeated tokens while still capturing emphasis.
        """
        if not tokens:
            return cls(terms=[], term_weights={})
        counts = Counter(tokens)
        terms = list(counts.keys())
        weights = {t: 1.0 + math.log(float(c)) for t, c in counts.items()}
        return cls(terms=terms, term_weights=weights)


# -----------------------------------------------------------------------------
# Lexical retrieval score — EVOLVE: the core relevance formula
# -----------------------------------------------------------------------------

def retrieval_score(
    query_repr: QueryRepr,
    doc_tf: Counter[str],
    doc_length: float,
    N: int,
    avgdl: float,
    corpus_df: Counter[str],
) -> float:
    """
    Score one document for one query.

    Must stay semantically aligned with BM25._score_candidates_vectorized().

    Core change: BM25+ (additive delta).
      score_t = idf(t) * ( (tf*(k1+1))/(tf + k1*norm) + delta )
    This is still purely lexical, cheap, and often improves robustness on verbose documents.
    """
    k1, b, eps = Config.k1, Config.b, Config.epsilon
    delta = float(Config.delta)

    score = 0.0
    matched_terms = 0.0

    norm = 1.0 - b + b * (doc_length / (avgdl + eps)) if avgdl > 0 else 1.0

    for term, w in query_repr.term_weights.items():
        tf = float(doc_tf.get(term, 0))
        if tf <= 0:
            continue
        matched_terms += 1.0
        df = float(corpus_df.get(term, 1))
        term_idf = float(idf(df, N))
        if term_idf <= 0:
            continue
        term_w = float(term_idf) ** float(Config.idf_power)

        tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * norm + eps)
        score += float(w) * term_w * (tf_norm + delta)

    if score <= 0.0:
        return 0.0

    # Optional coordination (default off)
    if Config.coord_weight != 0.0:
        qn = float(len(query_repr.term_weights))
        coord = matched_terms / (qn + Config.coord_k)
        score *= (1.0 + Config.coord_weight * coord)

    return float(score)


def score_document(query: list[str], doc_idx: int, corpus: "Corpus") -> float:
    """Entry point used by BM25.score(). EVOLVE: change pipeline if needed."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0
    doc_tf = corpus.get_term_frequencies(doc_idx)
    doc_length = float(corpus.doc_len_eff[doc_idx])
    return retrieval_score(q, doc_tf, doc_length, corpus.N, corpus.avgdl, corpus.document_frequency)


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
# Corpus (interface fixed for evaluator; internals can evolve if needed)
# -----------------------------------------------------------------------------

class Corpus:
    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        # MEMORY OPTIMIZATION: Don't store documents - only needed during construction
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.document_count = self.N

        # Precompute a fast stopword set for query-time filtering.
        self._stopword_set = set(LUCENE_STOPWORDS) | set(ENGLISH_STOPWORDS)

        # Keep raw lengths, but use an "effective" verbosity-robust length for normalization.
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self._doc_unique_lengths = np.zeros(self.N, dtype=np.float64)

        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            term_counts = Counter(doc)
            self._doc_unique_lengths[doc_idx] = float(len(term_counts))
            seen = set()
            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        # Effective length: interpolate between raw token length and unique-term count.
        mix = float(Config.lenmix)
        self.doc_len_eff = (1.0 - mix) * self.doc_lengths + mix * self._doc_unique_lengths
        self.avgdl = float(np.mean(self.doc_len_eff)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        self.tf_matrix = csr_matrix(tf_matrix_lil)
        self.idf_array = np.asarray(idf(self._df, self.N), dtype=np.float64)
        # Precompute IDF^p once to avoid per-query pow().
        self.idf_pow_array = np.power(self.idf_array, float(Config.idf_power), dtype=np.float64)

        # Handy for query-time filtering of extremely common terms
        self.df_ratio_array = (self._df / max(float(self.N), 1.0)).astype(np.float64)

        b = Config.b
        self.norm_array = 1.0 - b + b * (self.doc_len_eff / max(self.avgdl, 1.0))
        self.document_frequency = Counter(
            {term: max(1, int(self._df[tid])) for term, tid in self._vocab.items()}
        )

        self._posting_lists: dict[int, NDArray[np.int64]] = {
            tid: np.array(doc_ids, dtype=np.int64)
            for tid, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index

        # Preserve expected attribute name, but point it at the effective length used in scoring.
        self.document_length = self.doc_len_eff

    def __len__(self) -> int:
        return self.N

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> "Corpus":
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
        # MEMORY OPTIMIZATION: Reconstruct Counter on-demand from sparse matrix
        # This is only called by .score() which is rarely used (evaluator uses .rank())
        result = Counter()
        for term, tid in self._vocab.items():
            tf = int(self.tf_matrix[tid, doc_idx])
            if tf > 0:
                result[term] = tf
        return result

    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        tid = self._vocab.get(term)
        if tid is None:
            return np.array([], dtype=np.int64)
        return self._posting_lists.get(tid, np.array([], dtype=np.int64))

    def get_term_id(self, term: str) -> int | None:
        return self._vocab.get(term)

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self._id_to_idx[i] for i in ids if i in self._id_to_idx]

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx

    @property
    def term_frequency(self) -> list[Counter[str]]:
        # MEMORY OPTIMIZATION: Reconstruct on-demand if needed (rarely used)
        return [self.get_term_frequencies(i) for i in range(self.N)]

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

    @property
    def term_doc_matrix(self) -> None:
        return None


# -----------------------------------------------------------------------------
# BM25 (interface fixed for evaluator)
# -----------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: list[str], index: int) -> float:
        return score_document(query, index, self.corpus)

    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
        query_term_weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Vectorized scoring for rank(); must match retrieval_score formula."""
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        k1, eps = Config.k1, Config.epsilon
        delta = float(Config.delta)

        norms = self.corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        match_counts = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            idf_w = self.corpus.idf_pow_array[term_id]
            if idf_w <= 0:
                continue
            w = query_term_weights[i] if query_term_weights is not None else 1.0

            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            match_counts += (tf_row > 0).astype(np.float64)

            tf_norm = (tf_row * (k1 + 1.0)) / (tf_row + k1 * norms + eps)
            scores += w * idf_w * (tf_norm + delta)

        # Optional coordination (default off)
        if Config.coord_weight != 0.0:
            qn = float(len(query_term_ids))
            coord = match_counts / (qn + Config.coord_k)
            scores *= (1.0 + Config.coord_weight * coord)

        return scores

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Efficient top-k ranking (posting-accumulation).

        Compared to candidate-union (np.unique(concat(postings)) + dense slicing),
        we traverse each query-term posting list and accumulate document scores.

        Benefits:
          - avoids materializing the full candidate set upfront
          - avoids per-term dense toarray() over candidate_docs
          - often improves latency on large corpora and/or common terms
        """
        N = self.corpus.N
        if not query:
            idx = np.arange(N, dtype=np.int64)
            scores = np.zeros(N, dtype=np.float64)
            return (idx[:top_k], scores[:top_k]) if top_k is not None else (idx, scores)

        term_counts = Counter(query)
        query_term_ids: list[int] = []
        query_term_weights: list[float] = []

        drop_stop = bool(Config.drop_stopwords_in_query)
        max_df_ratio = float(Config.max_df_ratio)

        for term, count in term_counts.items():
            if drop_stop and term in self.corpus._stopword_set:
                continue
            tid = self.corpus.get_term_id(term)
            if tid is None:
                continue
            if max_df_ratio < 1.0 and self.corpus.df_ratio_array[tid] > max_df_ratio:
                continue
            query_term_ids.append(tid)
            query_term_weights.append(1.0 + math.log(float(count)))

        if not query_term_ids:
            idx = np.arange(N, dtype=np.int64)
            scores = np.zeros(N, dtype=np.float64)
            return (idx[:top_k], scores[:top_k]) if top_k is not None else (idx, scores)

        k1, eps = Config.k1, Config.epsilon
        delta = float(Config.delta)

        score_map: dict[int, float] = {}
        match_map: dict[int, float] | None = None
        if Config.coord_weight != 0.0:
            match_map = {}

        for tid, w in zip(query_term_ids, query_term_weights):
            idf_val = float(self.corpus.idf_pow_array[tid])
            if idf_val <= 0.0:
                continue

            row = self.corpus.tf_matrix.getrow(tid)  # CSR row: indices are docids, data are tfs
            if row.nnz == 0:
                continue

            doc_ids = row.indices.astype(np.int64, copy=False)
            tf_vals = row.data.astype(np.float64, copy=False)

            norms = self.corpus.norm_array[doc_ids]
            tf_norm = (tf_vals * (k1 + 1.0)) / (tf_vals + k1 * norms + eps)
            add = (float(w) * idf_val) * (tf_norm + delta)

            # Iterate only over nnz postings (fast in practice; avoids dense materialization).
            for d, inc in zip(doc_ids.tolist(), add.tolist()):
                score_map[d] = score_map.get(d, 0.0) + float(inc)
                if match_map is not None:
                    match_map[d] = match_map.get(d, 0.0) + 1.0

        if not score_map:
            idx = np.arange(N, dtype=np.int64)
            scores = np.zeros(N, dtype=np.float64)
            return (idx[:top_k], scores[:top_k]) if top_k is not None else (idx, scores)

        cand_docs = np.fromiter(score_map.keys(), dtype=np.int64)
        cand_scores = np.fromiter(score_map.values(), dtype=np.float64)

        if match_map is not None:
            qn = float(len(query_term_ids))
            match_counts = np.fromiter((match_map.get(int(d), 0.0) for d in cand_docs), dtype=np.float64)
            coord = match_counts / (qn + Config.coord_k)
            cand_scores *= (1.0 + Config.coord_weight * coord)

        if top_k is None:
            order = np.argsort(-cand_scores).astype(np.int64)
            ranked_cand = cand_docs[order]
            ranked_scores = cand_scores[order]

            cand_set = set(map(int, ranked_cand.tolist()))
            rest = np.fromiter((i for i in range(N) if i not in cand_set), dtype=np.int64)
            out_idx = np.concatenate([ranked_cand, rest])
            out_scores = np.concatenate([ranked_scores, np.zeros(len(rest), dtype=np.float64)])
            return out_idx, out_scores

        k = int(top_k)
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        if len(cand_docs) > k:
            part = np.argpartition(-cand_scores, k - 1)[:k]
            part_scores = cand_scores[part]
            order = np.argsort(-part_scores).astype(np.int64)
            top_docs = cand_docs[part][order]
            top_scores = part_scores[order]
        else:
            order = np.argsort(-cand_scores).astype(np.int64)
            top_docs = cand_docs[order]
            top_scores = cand_scores[order]

        # Deterministic padding with zero-score docs if needed (preserve seed semantics)
        if len(top_docs) < k:
            need = k - len(top_docs)
            cand_set = set(map(int, top_docs.tolist()))
            pad = []
            i = 0
            while len(pad) < need and i < N:
                if i not in cand_set:
                    pad.append(i)
                i += 1
            if pad:
                top_docs = np.concatenate([top_docs, np.asarray(pad, dtype=np.int64)])
                top_scores = np.concatenate([top_scores, np.zeros(len(pad), dtype=np.float64)])

        return top_docs.astype(np.int64), top_scores.astype(np.float64)

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
    "DocumentRepr",
    "QueryRepr",
    "idf",
    "retrieval_score",
    "score_document",
]
