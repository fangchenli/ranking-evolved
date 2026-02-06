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
    mu: float = 1750.0
    epsilon: float = 1e-9

    # Tempered background LM p_t(w) ∝ p(w)^tau
    collection_temper: float = 0.85

    # NEW: mix token-LM with df/N "presence LM" in the *background* (robust to bursty long docs).
    # This is a structural fix for StackOverflow/web-like corpora where tf can be dominated by a few docs.
    collection_df_alpha: float = 0.10  # 0 disables

    # Query term burstiness saturation (qtf^alpha)
    query_tf_power: float = 0.6

    # Document length prior (log-normal-ish); keep small to avoid recall loss
    length_prior_strength: float = 0.06

    # EDR gate: token-vs-document spread mismatch
    edr_strength: float = 0.45
    edr_clip: float = 2.5

    # Residual-IDF query weighting (df/N vs token LM)
    residual_idf_strength: float = 0.9

    # Two-stage background: (1-γ) p_col + γ * Uniform(V).
    uniform_bg_mass: float = 0.03  # 0 disables

    # Soft-AND coverage: reward covering more query terms (without hard booleaning).
    and_strength: float = 0.14  # 0 disables
    and_saturation: float = 3.0

    # Lightweight missing-term anti-evidence (scaled Dirichlet tf=0 term) inside candidates.
    missing_strength: float = 0.07  # keep small to protect recall@100

    # TF burstiness normalization (per-term exponent) to reduce domination by very common terms.
    burstiness_strength: float = 0.30  # 0 disables; keep modest for recall

    # Add back a tiny amount of negative evidence for "weak hits".
    neg_strength: float = 0.12  # 0 disables


# -----------------------------------------------------------------------------
# Collection Language Model — EVOLVE: how to compute P(w | C)
# -----------------------------------------------------------------------------

def collection_probability(term: str, corpus_term_freq: Counter[str], total_tokens: int) -> float:
    """
    Collection probability P(w | C).

    Base: tf_C(w) / |C|.
    EVOLVE (here): use a tempered background model to reduce dominance of very frequent terms:
        p_t(w) ∝ p(w)^tau, tau in (0,1]
    which increases relative mass of rarer terms (information gain) while staying a proper LM
    after renormalization. We precompute this normalization inside Corpus for speed; here we
    provide a safe fallback if called directly.
    """
    tf = corpus_term_freq.get(term, 0)
    if tf <= 0 or total_tokens <= 0:
        return Config.epsilon
    p = tf / float(total_tokens)
    tau = float(getattr(Config, "collection_temper", 1.0))
    p_t = p ** tau
    # Fallback approximate renorm: keep scale comparable; exact renorm done in Corpus.
    return max(p_t, Config.epsilon)


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
        Keep representation simple, but normalize very long queries by soft-booleaning:
        repeated terms are handled later via qtf^alpha; here we just keep tokens.
        """
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
    corpus: Corpus | None = None,
) -> float:
    """
    Dirichlet QL with two *information-diagnostic* modifiers:

    1) EDR gate (as before, but fallback uses a milder, monotone specificity proxy).
    2) Residual-IDF query weighting: boost query terms that are common as tokens when present
       yet not widely spread across documents.

    The residual notion is: token commonness p_col(w) vs doc spread p_doc(w)=df/N.
    In the vectorized path we can compute p_doc exactly and apply it per term id.
    """
    # Make score() match rank(): use the *same* precomputed collection LM + diagnostics when possible.
    mu, eps = Config.mu, Config.epsilon
    alpha = float(getattr(Config, "query_tf_power", 1.0))
    neg_s = float(getattr(Config, "neg_strength", 0.0))

    score = 0.0
    qtf = Counter(query_repr.terms)

    for term, c_q in qtf.items():
        tid = corpus.get_term_id(term) if corpus is not None else None
        if corpus is not None and tid is not None:
            p_collection = float(corpus._collection_prob[tid])
            gate = float(corpus._edr_gate[tid])
            ridf_w = float(corpus._ridf_qweight[tid])
            beta = float(corpus._tf_beta[tid]) if hasattr(corpus, "_tf_beta") else 1.0
        else:
            p_collection = collection_probability(term, corpus_term_freq, total_tokens)
            gate, ridf_w, beta = 1.0, 1.0, 1.0

        tf = float(doc_tf.get(term, 0.0))
        tf_eff = tf**beta if beta != 1.0 else tf

        numerator = 1.0 + tf_eff / (mu * p_collection + eps)
        denominator = (doc_length + mu) / mu
        per_term = math.log(numerator / denominator + eps)

        # Leak negative evidence in the same way as rank()
        if neg_s > 0.0 and per_term < 0.0:
            per_term *= neg_s

        w0 = float(query_repr.term_weights.get(term, 1.0))
        w = (w0 * ridf_w) ** alpha

        # Rank() uses w * per_term (with leaked negatives), not positive-only.
        score += w * gate * per_term

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
    return retrieval_score(q, doc_tf, doc_length, corpus.corpus_term_freq, corpus.total_tokens, corpus=corpus)


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

        # Build vocabulary
        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        # Collection statistics for Query Likelihood
        self.corpus_term_freq = Counter()
        self.total_tokens = 0

        # Build sparse TF matrix and inverted index
        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        self._doc_tf_dicts: list[Counter[str]] = [Counter(doc) for doc in documents]

        for doc_idx, doc in enumerate(documents):
            self.total_tokens += len(doc)
            term_counts = Counter(doc)
            seen = set()
            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                self.corpus_term_freq[term] += count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_matrix_lil)

        # Collection LM (tempered) with an optional df/N "presence LM" mix and a tiny uniform mixture.
        # Motivation: token-LM can be dominated by a few huge/bursty docs; df/N is stabler across corpora.
        tau = float(getattr(Config, "collection_temper", 1.0))
        base_p = np.zeros(self.vocab_size, dtype=np.float64)
        for term, tid in self._vocab.items():
            tf = float(self.corpus_term_freq.get(term, 0))
            base_p[tid] = tf / max(float(self.total_tokens), 1.0)

        if tau != 1.0:
            tmp = np.power(np.maximum(base_p, Config.epsilon), tau)
            z = float(np.sum(tmp))
            p_tf = np.maximum(tmp / max(z, Config.epsilon), Config.epsilon)
        else:
            p_tf = np.maximum(base_p, Config.epsilon)

        if self.N > 0:
            p_df = np.maximum(self._df / float(self.N), Config.epsilon)
        else:
            p_df = np.full(self.vocab_size, Config.epsilon, dtype=np.float64)

        mix = float(getattr(Config, "collection_df_alpha", 0.0))
        if mix > 0.0:
            p_col = np.maximum((1.0 - mix) * p_tf + mix * p_df, Config.epsilon)
            p_col = p_col / max(float(np.sum(p_col)), Config.epsilon)
        else:
            p_col = p_tf

        gamma = float(getattr(Config, "uniform_bg_mass", 0.0))
        if gamma > 0.0 and self.vocab_size > 0:
            p_uni = 1.0 / float(self.vocab_size)
            self._collection_prob = np.maximum((1.0 - gamma) * p_col + gamma * p_uni, Config.epsilon)
        else:
            self._collection_prob = p_col

        # Precompute per-term diagnostics using BOTH token LM and document-spread LM.
        lam = float(getattr(Config, "edr_strength", 0.0))
        clipc = float(getattr(Config, "edr_clip", 3.0))
        ridf_s = float(getattr(Config, "residual_idf_strength", 0.0))
        burst_s = float(getattr(Config, "burstiness_strength", 0.0))

        p_doc = p_df  # df-based LM

        # EDR gate: 1 + λ * clip(log(p_doc / p_col), [-c,c])
        if lam > 0.0 and self.N > 0:
            ratio = np.log(np.maximum(p_doc / np.maximum(self._collection_prob, Config.epsilon), Config.epsilon))
            ratio = np.clip(ratio, -clipc, clipc)
            self._edr_gate = 1.0 + lam * ratio
        else:
            self._edr_gate = np.ones(self.vocab_size, dtype=np.float64)

        # Residual-IDF query weight per term id: 1 + s * max(0, log(p_doc / p_col)).
        if ridf_s > 0.0 and self.N > 0:
            ridf = np.log(np.maximum(p_doc / np.maximum(self._collection_prob, Config.epsilon), Config.epsilon))
            ridf = np.maximum(ridf, 0.0)
            self._ridf_qweight = 1.0 + ridf_s * np.minimum(ridf, clipc) / max(clipc, Config.epsilon)
        else:
            self._ridf_qweight = np.ones(self.vocab_size, dtype=np.float64)

        # NEW: Per-term TF exponent beta(w) in [1-burst_s, 1], derived from normalized IDF.
        # Common terms saturate more: tf -> tf^beta(w).
        if burst_s > 0.0 and self.N > 0:
            idf = np.log((float(self.N) + 1.0) / (self._df + 1.0))
            idf01 = idf / max(float(np.max(idf)), Config.epsilon)
            self._tf_beta = 1.0 - burst_s * (1.0 - idf01)
        else:
            self._tf_beta = np.ones(self.vocab_size, dtype=np.float64)

        # Length prior
        s = float(getattr(Config, "length_prior_strength", 0.0))
        if s > 0:
            logL = np.log(np.maximum(self.doc_lengths, 1.0))
            m = math.log(max(self.avgdl, 1.0))
            self._length_prior = -s * np.square(logL - m)
        else:
            self._length_prior = np.zeros(self.N, dtype=np.float64)

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
        alpha = float(getattr(Config, "query_tf_power", 1.0))
        and_strength = float(getattr(Config, "and_strength", 0.0))
        and_sat = float(getattr(Config, "and_saturation", 3.0))
        miss_s = float(getattr(Config, "missing_strength", 0.0))
        neg_s = float(getattr(Config, "neg_strength", 0.0))

        doc_lengths = self.corpus.doc_lengths[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        and_acc = np.zeros(len(candidate_docs), dtype=np.float64) if and_strength > 0.0 else None

        for i, term_id in enumerate(query_term_ids):
            p_collection = self.corpus._collection_prob[term_id]
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()

            # TF burstiness saturation: tf -> tf^beta(w) (common terms saturate more).
            beta = self.corpus._tf_beta[term_id]
            tf_eff = np.power(tf_row, beta) if beta != 1.0 else tf_row

            numerator = 1.0 + tf_eff / (mu * p_collection + eps)
            denominator = (doc_lengths + mu) / mu
            per_term = np.log(numerator / denominator + eps)

            # Apply EDR gate (query-independent)
            per_term *= self.corpus._edr_gate[term_id]

            w0 = query_term_weights[i] if query_term_weights is not None else 1.0
            w = (w0 * self.corpus._ridf_qweight[term_id]) ** alpha

            # Keep most of the classic "only reward positive LLR", but leak a small fraction
            # of negative evidence for weak hits to improve early precision.
            if neg_s > 0.0:
                per_term = np.where(per_term >= 0.0, per_term, neg_s * per_term)

            present = np.maximum(per_term, 0.0)
            contrib = w * present
            scores += w * per_term  # includes leaked negatives if enabled

            # Missing-term anti-evidence (scaled tf=0 Dirichlet contribution) within candidates.
            if miss_s > 0.0:
                miss = (tf_row <= 0.0).astype(np.float64)
                if np.any(miss):
                    base0 = np.log((mu * p_collection + eps) / (doc_lengths + mu + eps) + eps)  # < 0
                    scores += miss_s * w * miss * base0

            # Soft-AND: saturating coverage reward (encourages matching more query terms)
            if and_acc is not None:
                and_acc += np.tanh(contrib / max(and_sat, eps))

        if and_acc is not None and len(query_term_ids) > 0:
            scores += and_strength * (and_acc / float(len(query_term_ids)))

        scores += self.corpus._length_prior[candidate_docs]
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
        candidate_set: set[int] = set()
        for tid in query_term_ids:
            candidate_set.update(self.corpus._posting_lists.get(tid, np.array([], dtype=np.int64)).tolist())

        candidate_docs = np.array(sorted(candidate_set), dtype=np.int64)
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
