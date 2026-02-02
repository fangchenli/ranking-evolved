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

    # TF blending weight (kept)
    alpha: float = 0.6

    # Replace harsh coordination with bounded rarity-mass soft-AND (recall-safe)
    coord_beta: float = 0.55
    coord_floor: float = 0.35

    # Additive priors (recall-friendly)
    cov_gamma: float = 0.10       # small additive term-coverage bump
    idf_match_gamma: float = 0.08 # additive matched-IDF-share bump

    # Keep mild rare-term shaping but do it via IDF choice rather than tf multiplier
    gamma: float = 0.0

    # Keep experimental knobs defined (prevents runtime failure if referenced)
    idf_pivot: float = 3.0
    cov_power: float = 2.0

    epsilon: float = 1e-9
    max_idf: float = float("inf")
    min_idf: float = 0.0


# ----- EVOLVE: Primitives (atoms). Add new ones or change formulas. -----

class ScoringPrimitives:
    """IDF, TF, saturation, length norm, aggregation. Invent new primitives or new formulas."""

    @staticmethod
    def matched_idf_share(matched_idf: float, total_idf: float) -> float:
        """Bounded query-IDF mass coverage in [0,1]. Used as additive recall-friendly prior."""
        t = max(total_idf, EvolvedParameters.epsilon)
        return max(0.0, min(1.0, matched_idf / t))

    @staticmethod
    def matched_idf_share_vectorized(
        matched_idf: NDArray[np.float64], total_idf: float
    ) -> NDArray[np.float64]:
        t = max(float(total_idf), EvolvedParameters.epsilon)
        return np.clip(matched_idf / t, 0.0, 1.0)

    @staticmethod
    def coord_rarity_aware(
        matched_rarity: float, total_rarity: float, beta: float, floor: float
    ) -> float:
        """
        Bounded rarity-mass soft-AND multiplier:
          floor + (1-floor) * (matched_rarity/total_rarity)^beta
        """
        t = max(total_rarity, EvolvedParameters.epsilon)
        frac = max(0.0, matched_rarity) / t
        f = float(max(0.0, min(1.0, floor)))
        return f + (1.0 - f) * (frac ** max(0.0, beta))

    @staticmethod
    def coord_rarity_aware_vectorized(
        matched_rarity: NDArray[np.float64],
        total_rarity: float,
        beta: float,
        floor: float,
    ) -> NDArray[np.float64]:
        t = max(float(total_rarity), EvolvedParameters.epsilon)
        frac = np.maximum(matched_rarity, 0.0) / t
        f = float(max(0.0, min(1.0, floor)))
        return f + (1.0 - f) * np.power(frac, max(0.0, float(beta)))

    @staticmethod
    def idf_classic(df: float, N: int) -> float:
        return math.log((N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))

    @staticmethod
    def idf_balanced(df: float, N: int) -> float:
        """
        Bounded, two-sided IDF: reduces ultra-rare spikes while still penalizing frequent terms.
        """
        p = df / (N + EvolvedParameters.epsilon)
        return math.log1p((1.0 - p) / (p + EvolvedParameters.epsilon))

    @staticmethod
    def idf_balanced_vectorized(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        p = df / (N + EvolvedParameters.epsilon)
        return np.log1p((1.0 - p) / (p + EvolvedParameters.epsilon))

    @staticmethod
    def tf_salience(tf: float, dl: float, avgdl: float) -> float:
        """
        Length-aware TF salience: downweights inflated TF in long docs.
        """
        denom = tf + 0.5 * (dl / max(avgdl, 1.0)) + 1.0
        return tf / (denom + EvolvedParameters.epsilon)

    @staticmethod
    def tf_salience_vectorized(
        tf: NDArray[np.float64], dl: NDArray[np.float64], avgdl: float
    ) -> NDArray[np.float64]:
        denom = tf + 0.5 * (dl / max(avgdl, 1.0)) + 1.0
        return tf / (denom + EvolvedParameters.epsilon)

    @staticmethod
    def coord_factor(matched_terms: int, total_query_terms: int, beta: float) -> float:
        """
        Soft coordination factor: (matched/total)^beta. Multiplies score.
        """
        if total_query_terms <= 0 or matched_terms <= 0:
            return 0.0
        return (matched_terms / total_query_terms) ** max(beta, 0.0)

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
    def idf_mixed_compressed(df: float, N: int) -> float:
        """
        New primitive: blend two IDFs then compress extremes.
        Motivation: very rare terms can dominate nDCG@10; compression tends to improve
        ranking robustness across heterogeneous BEIR/BRIGHT corpora while preserving recall.
        """
        a = EvolvedParameters.alpha
        idf_a = ScoringPrimitives.idf_lucene(df, N)
        idf_b = ScoringPrimitives.idf_atire(df, N)
        mixed = a * idf_a + (1.0 - a) * idf_b
        # soft compression of very large idf values (keeps monotonicity)
        p = EvolvedParameters.idf_pivot
        beta = EvolvedParameters.beta
        return mixed / (1.0 + beta * max(0.0, mixed - p))

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
    def saturate_tflog_bm25(tf: float, k1: float, norm: float) -> float:
        """
        New primitive: apply log-TF before BM25-style saturation.
        Helps corpora with bursty term repetition (e.g., forum/stack traces) without
        killing signals for single occurrences.
        """
        if tf <= 0:
            return 0.0
        t = 1.0 + math.log(tf)
        denom = t + k1 * norm + EvolvedParameters.epsilon
        return (t * (k1 + 1.0)) / denom

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
        # more tunable than fixed square; tends to help nDCG@10 by preferring fuller matches
        return coverage ** max(1.0, EvolvedParameters.cov_power)

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

        # Term evidence IDF: Lucene is a strong default across BEIR/BRIGHT
        idf = ScoringPrimitives.idf_lucene(df, N)
        idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))

        norm = ScoringPrimitives.length_norm_bm25(dl, avgdl, b)

        # Blend: classic Lucene BM25-like sat + length-aware salience
        tf_sat = ScoringPrimitives.saturate_lucene(tf, k1, norm)
        tf_sal = ScoringPrimitives.tf_salience(tf, dl, avgdl)
        tf_comp = (1.0 - EvolvedParameters.alpha) * tf_sat + EvolvedParameters.alpha * tf_sal

        # Mild rare-term shaping (kept small)
        if EvolvedParameters.gamma > 0:
            tf_comp *= (1.0 + EvolvedParameters.gamma * math.tanh(idf))

        return idf * tf_comp


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
        matched_idf: float = 0.0,
        total_idf: float = 0.0,
    ) -> float:
        if not term_scores:
            return 0.0

        base = ScoringPrimitives.weighted_sum(term_scores, query_weights)

        # Recall-friendly additive priors
        if EvolvedParameters.cov_gamma > 0 and total_query_terms > 0:
            c = float(matched_count) / max(1.0, float(total_query_terms))
            base += EvolvedParameters.cov_gamma * (c * c)

        if EvolvedParameters.idf_match_gamma > 0 and total_idf > 0:
            base += EvolvedParameters.idf_match_gamma * ScoringPrimitives.matched_idf_share(
                matched_idf, total_idf
            )

        # Precision-friendly bounded soft-AND (doesn't zero out partial matches)
        if EvolvedParameters.coord_beta > 0 and total_rarity > 0:
            base *= ScoringPrimitives.coord_rarity_aware(
                matched_rarity,
                total_rarity,
                EvolvedParameters.coord_beta,
                EvolvedParameters.coord_floor,
            )

        return base


# ----- EVOLVE: Query handling -----

class QueryProcessor:
    """Turn raw query into (terms, weights). Evolve weighting or dedup strategy."""

    @staticmethod
    def process(query: list[str]) -> tuple[list[str], list[float]]:
        if not query:
            return [], []
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

    # For bounded coordination + priors
    total_rarity = 0.0
    matched_rarity = 0.0
    total_idf = 0.0
    matched_idf = 0.0

    for term in query_terms:
        df = corpus.get_df(term)
        total_rarity += max(0.0, ScoringPrimitives.idf_balanced(float(df), N))
        total_idf += max(0.0, ScoringPrimitives.idf_lucene(float(df), N))

    for term, w in zip(query_terms, query_weights, strict=False):
        tf = doc_tf.get(term, 0)
        if tf > 0:
            matched_count += 1
            df = corpus.get_df(term)
            term_scores.append(TermScorer.score(tf, df, N, dl, avgdl))
            used_weights.append(w)

            matched_rarity += max(0.0, ScoringPrimitives.idf_balanced(float(df), N))
            matched_idf += max(0.0, ScoringPrimitives.idf_lucene(float(df), N))

    return DocumentScorer.score(
        term_scores,
        used_weights,
        matched_count,
        len(query_terms),
        matched_rarity=matched_rarity,
        total_rarity=total_rarity,
        matched_idf=matched_idf,
        total_idf=total_idf,
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

        # Match TermScorer: lucene IDF for term evidence (balanced used only for coordination)
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
        dls = self.corpus.doc_lengths[candidate_docs]
        avgdl = self.corpus.avgdl

        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        matched = np.zeros(len(candidate_docs), dtype=np.float64)

        # For bounded coordination + IDF-share prior
        matched_rarity = np.zeros(len(candidate_docs), dtype=np.float64)
        matched_idf = np.zeros(len(candidate_docs), dtype=np.float64)

        k1 = EvolvedParameters.k1

        qids = np.array(query_term_ids, dtype=np.int64)
        df_q = self.corpus._df[qids] if len(qids) else np.array([], dtype=np.float64)

        rarity_q = (
            ScoringPrimitives.idf_balanced_vectorized(df_q, self.corpus.N)
            if df_q.size
            else np.array([], dtype=np.float64)
        )
        rarity_q = np.maximum(rarity_q, 0.0)
        total_rarity = float(np.sum(rarity_q)) if rarity_q.size else 0.0

        total_idf = float(np.sum(np.maximum(self.corpus.idf_array[qids], 0.0))) if len(qids) else 0.0

        for i, term_id in enumerate(query_term_ids):
            idf = float(self.corpus.idf_array[term_id])
            if idf <= 0:
                continue
            idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))

            weight = query_term_weights[i] if query_term_weights is not None else 1.0

            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0).astype(np.float64)
            matched += present
            if rarity_q.size:
                matched_rarity += present * float(rarity_q[i])
            matched_idf += present * max(0.0, idf)

            tf_sat = ScoringPrimitives.saturate_lucene_vectorized(tf_row, k1, norms)
            tf_sal = ScoringPrimitives.tf_salience_vectorized(tf_row, dls, avgdl)
            tf_comp = (1.0 - EvolvedParameters.alpha) * tf_sat + EvolvedParameters.alpha * tf_sal

            scores += weight * idf * tf_comp

        # Additive priors
        qn = float(len(query_term_ids))
        if EvolvedParameters.cov_gamma > 0 and qn > 0:
            c = matched / qn
            scores += EvolvedParameters.cov_gamma * (c * c)

        if EvolvedParameters.idf_match_gamma > 0 and total_idf > 0:
            share = ScoringPrimitives.matched_idf_share_vectorized(matched_idf, total_idf)
            scores += EvolvedParameters.idf_match_gamma * share

        # Bounded rarity-aware coordination
        if EvolvedParameters.coord_beta > 0 and total_rarity > 0:
            scores *= ScoringPrimitives.coord_rarity_aware_vectorized(
                matched_rarity,
                total_rarity,
                EvolvedParameters.coord_beta,
                EvolvedParameters.coord_floor,
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
        query_terms, query_weights = QueryProcessor.process(query)
        query_term_ids = []
        query_term_weights = []
        for term, w in zip(query_terms, query_weights, strict=False):
            term_id = self.corpus.get_term_id(term)
            if term_id is not None:
                query_term_ids.append(term_id)
                query_term_weights.append(float(w))
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
