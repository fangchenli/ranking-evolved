"""
Freeform lexical retrieval seed — maximum freedom for discovering a new retrieval method.

Core idea: document representation + query representation + scoring method.
The evaluator requires: BM25, Corpus, tokenize, LuceneTokenizer; BM25 must have rank() and score().
Everything else is evolvable. Default behavior: Lucene BM25 (same as current seed).
"""

from __future__ import annotations

import math
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

NUM_QUERY_WORKERS = 32
MIN_QUERIES_FOR_PARALLEL = 10


# -----------------------------------------------------------------------------
# Config — EVOLVE: add parameters for your retrieval method
# -----------------------------------------------------------------------------

class Config:
    # Keep the same general "concave evidence + coverage" family, but add a *second channel*
    # for very rare "keyword-like" matches. This helps theorem/proof/technical QA corpora where
    # exact rare identifiers are decisive (BRIGHT theoremqa, pony), while keeping robustness.
    epsilon: float = 1e-9

    # Evidence: wt * log1p(tf/base), then log1p(total evidence).
    tf_log_base: float = 1.0

    # Soft-AND: reward covering more of the *informative* query mass.
    coverage_gamma: float = 0.25

    # Query-side clarity gate: clarity=(idf/(idf+1))^p in [0,1]
    q_clarity_power: float = 0.6

    # Sublinear query repetition weighting: count**p (keeps emphasis w/o verbosity blowups)
    qtf_power: float = 0.5

    # Mild length prior: downweight extremely long docs gently (helps precision w/o killing recall).
    dl_alpha: float = 0.15

    # New: add a small "rare-term channel" that acts like a lexical key-match prior.
    # Use a high-idf hinge so it only fires for truly discriminative terms.
    rare_idf_pivot: float = 4.0
    rare_boost: float = 0.25

    # Compatibility leftovers (Corpus references b/k1; keep but don't use in scoring)
    k1: float = 0.9
    b: float = 0.4
    dl_p: float = 0.75


# -----------------------------------------------------------------------------
# IDF — EVOLVE: fundamental term importance (e.g. rarity, discriminativity)
# -----------------------------------------------------------------------------

def idf(df: float | NDArray[np.float64], N: int) -> float | NDArray[np.float64]:
    """
    Smoothed "surprisal" IDF:
    - Interprets df/N as an empirical occurrence probability p(t in doc).
    - Uses -log(p) with add-one style smoothing to avoid infinities.
    This tends to behave better than classic BM25 IDF on very spiky corpora.
    """
    df = np.asarray(df, dtype=np.float64)
    p = (df + 1.0) / (N + 2.0)
    return -np.log(p)


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
        self.terms = terms
        self.term_weights = term_weights or {t: 1.0 for t in terms}

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> QueryRepr:
        """
        Unique query constraints + sublinear repetition weights.

        Rationale: repetition sometimes encodes emphasis, but linear qtf is brittle
        on verbose queries. Use count**p with p≈0.5.
        """
        if not tokens:
            return cls(terms=[], term_weights={})
        c = Counter(tokens)
        terms = list(c.keys())
        w = {t: float(cnt) ** Config.qtf_power for t, cnt in c.items()}
        return cls(terms=terms, term_weights=w)


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
    Two-channel concave evidence:

    Channel A (robust): clarity-gated surprisal evidence with log1p TF utility + IDF-mass coverage.
    Channel B (key-match): extra boost for *very rare* query terms that appear at least once.

    Intuition for Channel B:
    In theorem/proof/technical QA, a single exact match on a rare symbol/name often matters more
    than repeated matches on medium-common vocabulary. We add a bounded hinge on IDF so it only
    activates for truly rare terms, helping BRIGHT theoremqa/pony without destabilizing BEIR.
    """
    if not query_repr.terms:
        return 0.0

    eps = Config.epsilon
    base = Config.tf_log_base

    sum_evidence = 0.0
    cov_num = 0.0
    cov_den = 0.0
    rare_hits = 0.0

    for term in query_repr.terms:
        df = float(corpus_df.get(term, 1.0))
        term_idf = float(idf(df, N))
        if term_idf <= 0.0:
            continue

        rarity = term_idf / (term_idf + 1.0)
        clarity = rarity ** Config.q_clarity_power

        wq = float(query_repr.term_weights.get(term, 1.0))
        wt = wq * term_idf * clarity
        cov_den += wt

        tf = float(doc_tf.get(term, 0.0))
        if tf <= 0.0:
            continue

        cov_num += wt
        sum_evidence += wt * math.log1p(tf / (base + eps))

        # Rare-term key-match: count a hit if idf above pivot (hinge, not a hard filter).
        if term_idf > Config.rare_idf_pivot:
            rare_hits += (term_idf - Config.rare_idf_pivot) / (term_idf + eps)

    if sum_evidence <= 0.0:
        return 0.0

    score = math.log1p(sum_evidence)

    if cov_den > 0.0 and Config.coverage_gamma != 0.0:
        score *= 1.0 + Config.coverage_gamma * (cov_num / (cov_den + eps))

    # Add the bounded rare-hit channel (kept small and length-penalized with the main score).
    if Config.rare_boost != 0.0 and rare_hits > 0.0:
        score *= 1.0 + Config.rare_boost * math.log1p(rare_hits)

    length_ratio = (doc_length + 1.0) / (avgdl + 1.0)
    dl_damp = 1.0 + Config.dl_alpha * math.log1p(length_ratio)
    return score / (dl_damp + eps)


def score_document(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Entry point used by BM25.score(). EVOLVE: change pipeline if needed."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0
    doc_tf = corpus.get_term_frequencies(doc_idx)
    doc_length = float(corpus.doc_lengths[doc_idx])
    return retrieval_score(q, doc_tf, doc_length, corpus.N, corpus.avgdl, corpus.document_frequency)


# -----------------------------------------------------------------------------
# Tokenization (fixed for evaluator)
# -----------------------------------------------------------------------------

_TOKENIZER: _BaseLuceneTokenizer | None = None

def _get_tokenizer() -> _BaseLuceneTokenizer:
    global _TOKENIZER
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
        self.documents = documents
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.document_count = self.N
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        self._doc_tf_dicts: list[Counter[str]] = [Counter(doc) for doc in documents]

        for doc_idx, doc in enumerate(documents):
            term_counts = Counter(doc)
            seen = set()
            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_matrix_lil)
        self.idf_array = np.asarray(idf(self._df, self.N), dtype=np.float64)

        # Must match retrieval_score normalization (dl^p).
        b = Config.b
        p_len = Config.dl_p
        dl = np.power(self.doc_lengths, p_len, dtype=np.float64)
        adl = float(max(self.avgdl, 1.0)) ** p_len
        self.norm_array = 1.0 - b + b * (dl / max(adl, 1.0))
        self.document_frequency = Counter(
            {term: max(1, int(self._df[tid])) for term, tid in self._vocab.items()}
        )
        self._posting_lists: dict[int, NDArray[np.int64]] = {
            tid: np.array(doc_ids, dtype=np.int64)
            for tid, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index
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
        return self._doc_tf_dicts[doc_idx]

    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        tid = self._vocab.get(term)
        return self._posting_lists.get(tid, np.array([], dtype=np.int64)) if tid is not None else np.array([], dtype=np.int64)

    def get_term_id(self, term: str) -> int | None:
        return self._vocab.get(term)

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self._id_to_idx[i] for i in ids if i in self._id_to_idx]

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx

    @property
    def term_frequency(self) -> list[Counter[str]]:
        return self._doc_tf_dicts

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

        eps = Config.epsilon
        base = Config.tf_log_base

        sum_evidence = np.zeros(len(candidate_docs), dtype=np.float64)
        cov_num = np.zeros(len(candidate_docs), dtype=np.float64)
        cov_den = 0.0
        rare_hits = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            idf_val = float(self.corpus.idf_array[term_id])
            if idf_val <= 0.0:
                continue

            rarity = idf_val / (idf_val + 1.0)
            clarity = rarity ** Config.q_clarity_power

            wq = float(query_term_weights[i]) if query_term_weights is not None else 1.0
            wt = wq * idf_val * clarity
            cov_den += wt

            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0.0).astype(np.float64)
            cov_num += wt * present

            sum_evidence += wt * np.log1p(tf_row / (base + eps))

            if Config.rare_boost != 0.0 and idf_val > Config.rare_idf_pivot:
                rare_hits += present * ((idf_val - Config.rare_idf_pivot) / (idf_val + eps))

        scores = np.log1p(np.maximum(sum_evidence, 0.0))

        if cov_den > 0.0 and Config.coverage_gamma != 0.0:
            scores *= 1.0 + Config.coverage_gamma * (cov_num / (cov_den + eps))

        if Config.rare_boost != 0.0:
            scores *= 1.0 + Config.rare_boost * np.log1p(np.maximum(rare_hits, 0.0))

        length_ratio = (self.corpus.doc_lengths[candidate_docs] + 1.0) / (self.corpus.avgdl + 1.0)
        dl_damp = 1.0 + Config.dl_alpha * np.log1p(length_ratio)
        return scores / (dl_damp + eps)

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not query:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)
        term_counts = Counter(query)
        query_term_ids = []
        query_term_weights = []
        for term, cnt in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                query_term_weights.append(float(cnt) ** Config.qtf_power)
        if not query_term_ids:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)
        qtf = np.array(query_term_weights, dtype=np.float64)
        candidate_set: set[int] = set()
        for tid in query_term_ids:
            candidate_set.update(self.corpus._posting_lists.get(tid, np.array([], dtype=np.int64)).tolist())
        candidate_docs = np.array(sorted(candidate_set), dtype=np.int64)
        candidate_scores = self._score_candidates_vectorized(query_term_ids, candidate_docs, qtf)
        all_scores = np.zeros(self.corpus.N, dtype=np.float64)
        all_scores[candidate_docs] = candidate_scores
        sorted_indices = np.argsort(-all_scores).astype(np.int64)
        sorted_scores = all_scores[sorted_indices]
        if top_k is not None:
            sorted_indices, sorted_scores = sorted_indices[:top_k], sorted_scores[:top_k]
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
    "DocumentRepr",
    "QueryRepr",
    "idf",
    "retrieval_score",
    "score_document",
]
