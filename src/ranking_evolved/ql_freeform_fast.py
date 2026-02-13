"""
Freeform Query Likelihood seed — maximum freedom for discovering a new probabilistic retrieval method.

Core idea: document representation + query representation + probabilistic scoring method.
The evaluator requires: QL, Corpus, tokenize, LuceneTokenizer; QL must have rank() and score().
Everything else is evolvable. Default behavior: Dirichlet smoothing (matches Pyserini LMDirichletSimilarity).
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
    mu: float = 2000.0  # Dirichlet smoothing parameter
    epsilon: float = 1e-9


# -----------------------------------------------------------------------------
# Collection Language Model — EVOLVE: how to compute P(w | C)
# -----------------------------------------------------------------------------

def collection_probability(term: str, corpus_term_freq: Counter[str], total_tokens: int) -> float:
    """
    Collection probability P(w | C) = total frequency / total tokens.
    EVOLVE: try other collection models (e.g., weighted by document importance, IDF-based, etc.).
    """
    if term not in corpus_term_freq:
        return Config.epsilon
    return corpus_term_freq[term] / max(total_tokens, 1)


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
        """EVOLVE: query expansion, term weighting, dedup, etc."""
        return cls(terms=tokens, term_weights={t: 1.0 for t in tokens})


# -----------------------------------------------------------------------------
# Probabilistic retrieval score — EVOLVE: the core relevance formula
# -----------------------------------------------------------------------------

def retrieval_score(
    query_repr: QueryRepr,
    doc_tf: Counter[str],
    doc_length: float,
    corpus_term_freq: Counter[str],
    total_tokens: int,
) -> float:
    """
    Score one document for one query using Query Likelihood with Dirichlet smoothing.

    Formula (Lucene/Pyserini variant):
        Score(D, Q) = Σ_{w in Q} max(0, log(1 + c(w,D)/(μ*P(w|C))) + log(μ/(L_D+μ)))

    This matches Pyserini's LMDirichletSimilarity which clamps per-term scores to 0.
    This differs from academic QL which allows negative per-term scores.

    Default behavior: Matches Pyserini's LMDirichletSimilarity with μ=2000.
    EVOLVE: design a probabilistic formulation with deep, fundamental, intuitive justification.
    Try other smoothing methods (Jelinek-Mercer, absolute discounting), document priors,
    term dependencies, multi-field models, etc.
    """
    mu, eps = Config.mu, Config.epsilon
    score = 0.0

    for term in query_repr.terms:
        # c(w, D): term count in document
        term_count = float(doc_tf.get(term, 0))

        # P(w | C): collection probability
        p_collection = collection_probability(term, corpus_term_freq, total_tokens)

        # Lucene/Pyserini formula: log(1 + freq/(μ*P(w|C))) + log(μ/(L_D+μ))
        # Clamp per-term score to 0 (Lucene's behavior for negative term scores)
        numerator = 1.0 + term_count / (mu * p_collection + eps)
        denominator = (doc_length + mu) / mu
        per_term_score = math.log(numerator / denominator + eps)

        # Apply query term weight and clamp to 0
        w = query_repr.term_weights.get(term, 1.0)
        score += w * max(per_term_score, 0.0)

    return score


def score_document(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Entry point used by QL.score(). EVOLVE: change pipeline if needed."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0
    doc_tf = corpus.get_term_frequencies(doc_idx)
    doc_length = float(corpus.doc_lengths[doc_idx])
    return retrieval_score(q, doc_tf, doc_length, corpus.corpus_term_freq, corpus.total_tokens)


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
        # MEMORY OPTIMIZATION: Don't store documents - only needed during construction
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

        # Collection statistics for Query Likelihood
        self.corpus_term_freq = Counter()  # Total frequency of each term in collection
        self.total_tokens = 0  # Sum of all doc lengths

        # Build sparse TF matrix and inverted index
        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        # MEMORY OPTIMIZATION: Don't precompute _doc_tf_dicts - reconstruct on-demand from tf_matrix

        for doc_idx, doc in enumerate(documents):
            self.total_tokens += len(doc)
            term_counts = Counter(doc)
            seen = set()
            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                self.corpus_term_freq[term] += count  # Accumulate collection frequencies
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_matrix_lil)

        # Collection probability array for vectorized scoring
        self._collection_prob = np.zeros(self.vocab_size, dtype=np.float64)
        for term, tid in self._vocab.items():
            self._collection_prob[tid] = collection_probability(
                term, self.corpus_term_freq, self.total_tokens
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
        # MEMORY OPTIMIZATION: Reconstruct on-demand if needed (rarely used)
        return [self.get_term_frequencies(i) for i in range(self.N)]

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

    @property
    def term_doc_matrix(self) -> None:
        return None


# -----------------------------------------------------------------------------
# QL (interface fixed for evaluator)
# -----------------------------------------------------------------------------

class QL:
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

        mu, eps = Config.mu, Config.epsilon
        doc_lengths = self.corpus.doc_lengths[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            # Get collection probability for this term
            p_collection = self.corpus._collection_prob[term_id]

            # Get term frequencies for candidates
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().flatten()

            # Lucene/Pyserini formula: log(1 + freq/(μ*P(w|C))) + log(μ/(L_D+μ))
            numerator = 1.0 + tf_row / (mu * p_collection + eps)
            denominator = (doc_lengths + mu) / mu
            per_term_scores = np.log(numerator / denominator + eps)

            # Apply query term weight and clamp to 0
            w = query_term_weights[i] if query_term_weights is not None else 1.0
            scores += w * np.maximum(per_term_scores, 0.0)

        return scores

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
        for term, count in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                query_term_weights.append(float(count))

        if not query_term_ids:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)

        qtf = np.array(query_term_weights, dtype=np.float64)

        # For large corpora, use NumPy operations instead of Python sets to avoid memory overhead
        posting_lists = []
        for tid in query_term_ids:
            pl = self.corpus._posting_lists.get(tid, np.array([], dtype=np.int64))
            if len(pl) > 0:
                posting_lists.append(pl)

        if not posting_lists:
            candidate_docs = np.array([], dtype=np.int64)
        elif len(posting_lists) == 1:
            candidate_docs = posting_lists[0]  # Already sorted in posting list
        else:
            # np.unique sorts and deduplicates - more memory efficient than Python set for large arrays
            candidate_docs = np.unique(np.concatenate(posting_lists))
        candidate_scores = self._score_candidates_vectorized(query_term_ids, candidate_docs, qtf)

        # CRITICAL: QL scores are negative log probabilities, so non-candidates must have very negative score
        # Otherwise documents without query terms (score=0.0) would rank HIGHER than relevant documents (negative scores)
        all_scores = np.full(self.corpus.N, -1e10, dtype=np.float64)
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
    "QL",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "LUCENE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "Config",
    "DocumentRepr",
    "QueryRepr",
    "collection_probability",
    "retrieval_score",
    "score_document",
]
