"""
Lexical retrieval as PRIMITIVES + STRUCTURE.
- Primitives: atomic scoring pieces (IDF, TF, saturation, length norm, aggregation).
- Structure: how they are combined (term score → doc score → ranking).
This seed is one structure (BM25-like); evolution can invent new primitives and new structure.
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


# ----- EVOLVE: Parameters -----

class EvolvedParameters:
    """Numeric parameters. Evolve values or add new ones. Defaults match Pyserini."""
    k1: float = 0.9
    b: float = 0.4
    k3: float = 8.0
    delta: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    # Light document-level priors
    gamma: float = 0.22  # slightly lower; will be combined with a *rarity-aware* coordination below

    # Frequent-term recall rescue (bounded)
    common_strength: float = 0.25
    common_pivot: float = 2.5

    # New: rarity-aware coordination (soft AND weighted by query-term rarity)
    coord_beta: float = 0.55

    epsilon: float = 1e-9
    max_idf: float = float("inf")
    min_idf: float = 0.0


# ----- EVOLVE: Primitives (atoms). Add new ones or change formulas. -----

class ScoringPrimitives:
    """IDF, TF, saturation, length norm, aggregation. Invent new primitives or new formulas."""

    @staticmethod
    def idf_balanced(df: float, N: int) -> float:
        """
        Bounded rarity for coordination/priors: log1p((1-p)/p) where p=df/N.
        Helps avoid ultra-rare terms fully dominating rarity-mass coverage.
        """
        p = df / (N + EvolvedParameters.epsilon)
        return math.log1p((1.0 - p) / (p + EvolvedParameters.epsilon))

    @staticmethod
    def idf_balanced_vectorized(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        p = df / (N + EvolvedParameters.epsilon)
        return np.log1p((1.0 - p) / (p + EvolvedParameters.epsilon))

    @staticmethod
    def commonness_rescue(idf: float, strength: float, pivot: float) -> float:
        """
        Boost frequent terms a bit (low idf), bounded and smooth.
        When idf << pivot => multiplier ~ 1 + strength
        When idf >> pivot => multiplier ~ 1
        """
        # 1 + strength * pivot/(pivot+idf)
        return 1.0 + strength * (pivot / (pivot + idf + EvolvedParameters.epsilon))

    @staticmethod
    def commonness_rescue_vectorized(
        idf: NDArray[np.float64], strength: float, pivot: float
    ) -> NDArray[np.float64]:
        return 1.0 + strength * (pivot / (pivot + idf + EvolvedParameters.epsilon))

    @staticmethod
    def coord_rarity_aware(
        matched_rarity: float, total_rarity: float, beta: float
    ) -> float:
        """
        New primitive: coordination based on rarity-mass coverage instead of term-count coverage.
        Intuition: matching rare query terms should matter more than matching generic ones.
        Returns (matched_rarity/total_rarity)^beta with smoothing.
        """
        t = max(total_rarity, EvolvedParameters.epsilon)
        m = max(0.0, matched_rarity)
        return (m / t) ** max(0.0, beta)

    @staticmethod
    def idf_classic(df: float, N: int) -> float:
        return math.log((N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))

    @staticmethod
    def idf_lucene(df: float, N: int) -> float:
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))

    @staticmethod
    def idf_lucene_vectorized(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log(1.0 + (N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))

    @staticmethod
    def idf_atire(df: float, N: int) -> float:
        return math.log(N / (df + EvolvedParameters.epsilon))

    @staticmethod
    def idf_bm25plus(df: float, N: int) -> float:
        return math.log((N + 1) / (df + EvolvedParameters.epsilon))

    @staticmethod
    def idf_smooth(df: float, N: int) -> float:
        return math.log((N + 0.5) / (df + 0.5))

    @staticmethod
    def tf_raw(tf: float) -> float:
        return tf

    @staticmethod
    def tf_log(tf: float) -> float:
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    @staticmethod
    def tf_double_log(tf: float) -> float:
        if tf <= 0:
            return 0.0
        return 1.0 + math.log(1.0 + math.log(tf + 1))

    @staticmethod
    def tf_boolean(tf: float) -> float:
        return 1.0 if tf > 0 else 0.0

    @staticmethod
    def tf_augmented(tf: float, max_tf: float) -> float:
        return 0.5 + 0.5 * (tf / max_tf) if max_tf > 0 else 0.5

    @staticmethod
    def saturate(x: float, k: float) -> float:
        return x / (x + k + EvolvedParameters.epsilon)

    @staticmethod
    def saturate_bm25(tf: float, k1: float, norm: float) -> float:
        denom = tf + k1 * norm + EvolvedParameters.epsilon
        return (tf * (k1 + 1)) / denom

    @staticmethod
    def saturate_lucene(tf: float, k1: float, norm: float) -> float:
        denom = tf + k1 * norm + EvolvedParameters.epsilon
        return tf / denom

    @staticmethod
    def saturate_lucene_vectorized(
        tf: NDArray[np.float64], k1: float, norm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        denom = tf + k1 * norm + EvolvedParameters.epsilon
        return tf / denom

    @staticmethod
    def saturate_bm25l(tf: float, k1: float, norm: float, delta: float) -> float:
        c = tf / (norm + EvolvedParameters.epsilon)
        c_delta = c + delta
        return ((k1 + 1) * c_delta) / (k1 + c_delta + EvolvedParameters.epsilon)

    @staticmethod
    def saturate_bm25plus(tf: float, k1: float, norm: float, delta: float) -> float:
        base = (tf * (k1 + 1)) / (tf + k1 * norm + EvolvedParameters.epsilon)
        return base + delta if tf > 0 else base

    @staticmethod
    def saturate_log(tf: float, k1: float, norm: float) -> float:
        bm25_sat = (tf * (k1 + 1)) / (tf + k1 * norm + EvolvedParameters.epsilon)
        return math.log(1.0 + bm25_sat)

    @staticmethod
    def length_norm_bm25(dl: float, avgdl: float, b: float) -> float:
        return 1.0 - b + b * (dl / max(avgdl, 1.0))

    @staticmethod
    def length_norm_bm25_vectorized(
        dl: NDArray[np.float64], avgdl: float, b: float
    ) -> NDArray[np.float64]:
        return 1.0 - b + b * (dl / max(avgdl, 1.0))

    @staticmethod
    def length_norm_pivot(dl: float, pivot: float, b: float) -> float:
        return 1.0 - b + b * (dl / max(pivot, 1.0))

    @staticmethod
    def length_norm_log(dl: float, avgdl: float, b: float) -> float:
        ratio = dl / max(avgdl, 1.0)
        return 1.0 + b * math.log(ratio) if ratio > 0 else 1.0

    @staticmethod
    def multiply(*args: float) -> float:
        result = 1.0
        for x in args:
            result *= x
        return result

    @staticmethod
    def add(*args: float) -> float:
        return sum(args)

    @staticmethod
    def weighted_sum(values: list[float], weights: list[float]) -> float:
        return sum(v * w for v, w in zip(values, weights, strict=False))

    @staticmethod
    def geometric_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        product = 1.0
        for v in values:
            if v <= 0:
                return 0.0
            product *= v
        return product ** (1.0 / len(values))

    @staticmethod
    def harmonic_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        reciprocal_sum = sum(1.0 / (v + EvolvedParameters.epsilon) for v in values)
        return len(values) / reciprocal_sum if reciprocal_sum > 0 else 0.0

    @staticmethod
    def soft_max(values: list[float], temperature: float = 1.0) -> float:
        if not values:
            return 0.0
        max_val = max(values)
        exp_sum = sum(math.exp((v - max_val) / temperature) for v in values)
        return max_val + temperature * math.log(exp_sum)

    @staticmethod
    def query_weight_uniform(qtf: float, k3: float) -> float:
        return 1.0

    @staticmethod
    def query_weight_frequency(qtf: float, k3: float) -> float:
        return qtf

    @staticmethod
    def query_weight_saturated(qtf: float, k3: float) -> float:
        return ((k3 + 1) * qtf) / (k3 + qtf + EvolvedParameters.epsilon)

    @staticmethod
    def coverage_bonus(matched_terms: int, total_query_terms: int) -> float:
        if total_query_terms <= 0:
            return 0.0
        coverage = matched_terms / total_query_terms
        return coverage * coverage

    @staticmethod
    def rarity_boost(idf: float, threshold: float = 3.0) -> float:
        return 1.0 + (idf - threshold) * 0.1 if idf > threshold else 1.0


# ----- EVOLVE: Term score (IDF × TF, or your formula) -----

class TermScorer:
    """One term's contribution. Evolve the formula; invent new combinations or new math."""

    @staticmethod
    def score(tf: float, df: float, N: int, dl: float, avgdl: float) -> float:
        if tf <= 0:
            return 0.0
        k1, b = EvolvedParameters.k1, EvolvedParameters.b
        idf = ScoringPrimitives.idf_lucene(df, N)
        idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))
        norm = ScoringPrimitives.length_norm_bm25(dl, avgdl, b)
        tf_comp = ScoringPrimitives.saturate_lucene(tf, k1, norm)

        rescue = ScoringPrimitives.commonness_rescue(
            idf, EvolvedParameters.common_strength, EvolvedParameters.common_pivot
        )
        return idf * tf_comp * rescue


# ----- EVOLVE: Doc score (aggregation of term scores) -----

class DocumentScorer:
    """Aggregate term scores into document score. Evolve aggregation or add new terms."""

    @staticmethod
    def score(
        term_scores: list[float],
        query_weights: list[float],
        matched_count: int,
        total_query_terms: int,
        matched_rarity: float = 0.0,
        total_rarity: float = 0.0,
    ) -> float:
        if not term_scores:
            return 0.0
        base = ScoringPrimitives.weighted_sum(term_scores, query_weights)

        # Additive coverage bonus keeps recall strong.
        if EvolvedParameters.gamma > 0:
            base += EvolvedParameters.gamma * ScoringPrimitives.coverage_bonus(
                matched_count, total_query_terms
            )

        # New: rarity-aware coordination multiplier improves early precision (nDCG@10)
        # while being less harsh than count-based coordination on long/noisy queries.
        if EvolvedParameters.coord_beta > 0 and total_rarity > 0:
            base *= ScoringPrimitives.coord_rarity_aware(
                matched_rarity, total_rarity, EvolvedParameters.coord_beta
            )

        return base


# ----- EVOLVE: Query handling -----

class QueryProcessor:
    """Turn raw query into (terms, weights). Evolve weighting or dedup strategy."""

    @staticmethod
    def process(query: list[str]) -> tuple[list[str], list[float]]:
        if not query:
            return [], []
        # Deduplicate terms and use saturated qtf weighting (classic BM25 query term factor idea)
        counts = Counter(query)
        terms = list(counts.keys())
        weights = [
            ScoringPrimitives.query_weight_saturated(float(counts[t]), EvolvedParameters.k3)
            for t in terms
        ]
        return terms, weights


# ----- EVOLVE: Full pipeline (or replace with new structure) -----

def score_kernel(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Orchestrate term/doc scoring. Evolve pipeline or replace with a different structure."""
    if not query:
        return 0.0
    query_terms, query_weights = QueryProcessor.process(query)
    if not query_terms:
        return 0.0

    doc_tf = corpus.get_term_frequencies(doc_idx)
    dl = corpus.doc_lengths[doc_idx]
    avgdl = corpus.avgdl
    N = corpus.N

    term_scores: list[float] = []
    used_weights: list[float] = []
    matched_count = 0

    # rarity-mass coverage tracking: use balanced rarity (less spiky than lucene idf)
    total_rarity = 0.0
    matched_rarity = 0.0
    for term in query_terms:
        df = corpus.get_df(term)
        total_rarity += max(0.0, ScoringPrimitives.idf_balanced(float(df), N))

    for term, w in zip(query_terms, query_weights, strict=False):
        tf = doc_tf.get(term, 0)
        if tf > 0:
            matched_count += 1
            df = corpus.get_df(term)
            term_scores.append(TermScorer.score(tf, df, N, dl, avgdl))
            used_weights.append(w)
            matched_rarity += max(0.0, ScoringPrimitives.idf_balanced(float(df), N))

    return DocumentScorer.score(
        term_scores,
        used_weights,
        matched_count,
        len(query_terms),
        matched_rarity=matched_rarity,
        total_rarity=total_rarity,
    )


# ----- Tokenization (fixed; do not evolve) -----

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


# ----- Corpus (fixed structure; evaluator expects this interface) -----

class Corpus:
    """Preprocessed collection; inverted index + sparse matrix. Interface must stay stable."""

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
        term_idx = 0
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = term_idx
                    term_idx += 1
        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        self._doc_tf_dicts: list[Counter[str]] = [Counter(doc) for doc in documents]

        for doc_idx, doc in enumerate(documents):
            term_counts = Counter(doc)
            seen_terms = set()
            for term, count in term_counts.items():
                term_id = self._vocab[term]
                tf_matrix_lil[term_id, doc_idx] = count
                if term_id not in seen_terms:
                    self._inverted_index[term_id].append(doc_idx)
                    self._df[term_id] += 1
                    seen_terms.add(term_id)

        self.tf_matrix = csr_matrix(tf_matrix_lil)
        self._posting_lists: dict[int, NDArray[np.int64]] = {
            term_id: np.array(doc_ids, dtype=np.int64)
            for term_id, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index

        self.idf_array = ScoringPrimitives.idf_lucene_vectorized(self._df, self.N)
        self.norm_array = ScoringPrimitives.length_norm_bm25_vectorized(
            self.doc_lengths, self.avgdl, EvolvedParameters.b
        )
        self.document_frequency = Counter(
            {term: int(self._df[term_id]) for term, term_id in self._vocab.items()}
        )
        self.document_length = self.doc_lengths

    def __len__(self) -> int:
        return self.N

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> Corpus:
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)

    def get_df(self, term: str) -> int:
        term_id = self._vocab.get(term)
        if term_id is None:
            return 1
        return max(1, int(self._df[term_id]))

    def get_tf(self, doc_idx: int, term: str) -> int:
        term_id = self._vocab.get(term)
        if term_id is None:
            return 0
        return int(self.tf_matrix[term_id, doc_idx])

    def get_term_frequencies(self, doc_idx: int) -> Counter[str]:
        return self._doc_tf_dicts[doc_idx]

    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        term_id = self._vocab.get(term)
        if term_id is None:
            return np.array([], dtype=np.int64)
        return self._posting_lists.get(term_id, np.array([], dtype=np.int64))

    def get_term_id(self, term: str) -> int | None:
        return self._vocab.get(term)

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self._id_to_idx[doc_id] for doc_id in ids if doc_id in self._id_to_idx]

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

    @property
    def term_doc_matrix(self) -> None:
        return None

    @property
    def term_frequency(self) -> list[Counter[str]]:
        return self._doc_tf_dicts


# ----- BM25 API (interface fixed for evaluator) -----

class BM25:
    """Scorer: uses score_kernel for single-doc; vectorized path for batch (same formula)."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: list[str], index: int) -> float:
        return score_kernel(query, index, self.corpus)

    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
        query_term_weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        norms = self.corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        matched = np.zeros(len(candidate_docs), dtype=np.float64)

        k1 = EvolvedParameters.k1
        for i, term_id in enumerate(query_term_ids):
            idf = self.corpus.idf_array[term_id]
            if idf <= 0:
                continue
            idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))

            rescue = ScoringPrimitives.commonness_rescue(
                float(idf), EvolvedParameters.common_strength, EvolvedParameters.common_pivot
            )

            weight = query_term_weights[i] if query_term_weights is not None else 1.0
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            matched += (tf_row > 0).astype(np.float64)

            tf_saturated = ScoringPrimitives.saturate_lucene_vectorized(tf_row, k1, norms)
            scores += weight * idf * rescue * tf_saturated

        qn = float(len(query_term_ids))
        if EvolvedParameters.gamma > 0 and qn > 0:
            coverage = matched / qn
            scores += EvolvedParameters.gamma * (coverage * coverage)

        # Rarity-aware coordination (match score_kernel/DocumentScorer structure).
        if EvolvedParameters.coord_beta > 0 and qn > 0:
            # balanced rarity per query term (bounded)
            df_q = self.corpus._df[np.array(query_term_ids, dtype=np.int64)]
            rarity_q = ScoringPrimitives.idf_balanced_vectorized(df_q, self.corpus.N)
            rarity_q = np.maximum(rarity_q, 0.0)
            total_rarity = float(np.sum(rarity_q))
            if total_rarity > 0:
                # matched_rarity: sum rarity of query terms present in doc
                matched_rarity = np.zeros(len(candidate_docs), dtype=np.float64)
                for i, term_id in enumerate(query_term_ids):
                    tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
                    matched_rarity += (tf_row > 0).astype(np.float64) * float(rarity_q[i])

                scores *= np.power(
                    np.maximum(matched_rarity / total_rarity, 0.0),
                    EvolvedParameters.coord_beta,
                )

        return scores

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not query:
            indices = np.arange(self.corpus.N, dtype=np.int64)
            scores = np.zeros(self.corpus.N, dtype=np.float64)
            return indices, scores

        # Keep rank() consistent with QueryProcessor: dedup + saturated qtf
        term_counts = Counter(query)
        query_term_ids: list[int] = []
        query_term_weights: list[float] = []
        for term, count in term_counts.items():
            term_id = self.corpus.get_term_id(term)
            if term_id is not None:
                query_term_ids.append(term_id)
                query_term_weights.append(
                    ScoringPrimitives.query_weight_saturated(float(count), EvolvedParameters.k3)
                )

        if not query_term_ids:
            indices = np.arange(self.corpus.N, dtype=np.int64)
            scores = np.zeros(self.corpus.N, dtype=np.float64)
            return indices, scores

        qtf_weights = np.array(query_term_weights, dtype=np.float64)

        candidate_set: set[int] = set()
        for term_id in query_term_ids:
            posting_list = self.corpus._posting_lists.get(term_id, np.array([], dtype=np.int64))
            candidate_set.update(posting_list.tolist())

        candidate_docs = np.array(sorted(candidate_set), dtype=np.int64)
        candidate_scores = self._score_candidates_vectorized(
            query_term_ids, candidate_docs, qtf_weights
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
            return [self.rank(query, top_k) for query in queries]
        with ThreadPoolExecutor(max_workers=NUM_QUERY_WORKERS) as executor:
            return list(executor.map(lambda q: self.rank(q, top_k), queries))


__all__ = [
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "LUCENE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "EvolvedParameters",
    "ScoringPrimitives",
    "TermScorer",
    "DocumentScorer",
    "QueryProcessor",
    "score_kernel",
]
