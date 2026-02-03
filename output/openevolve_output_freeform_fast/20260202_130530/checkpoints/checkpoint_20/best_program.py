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
    # TF saturation (BM25-like)
    k1: float = 0.9

    # Pivoted length normalization strength
    b: float = 0.35

    # Mix of token length and unique-term length (focus prior)
    focus_mix: float = 0.65

    # Rarity shaping
    idf_power: float = 1.12

    # Soft-AND / coordination as a *reward* (never < 1)
    coord_alpha: float = 0.25
    coord_beta: float = 0.75

    # Query TF dampening: repeated query tokens often come from artifacts/noise
    use_log_qtf: bool = True

    # Novel signal: "informativeness-weighted coverage".
    # Intuition: matching rare/discriminative terms should count more toward coordination
    # than matching ubiquitous terms (improves early precision/nDCG without killing recall).
    cov_idf_power: float = 1.0

    # Soft phrase/proximity reward. Only uses adjacent query term pairs (cheap).
    prox_window: int = 8
    prox_alpha: float = 0.12

    epsilon: float = 1e-9


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
    def __init__(self, term_frequencies: Counter[str], length: float, positions: dict[str, list[int]]):
        self.term_frequencies = term_frequencies
        self.length = length
        self.positions = positions  # for lightweight proximity/phrase evidence

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> DocumentRepr:
        """Store positions for proximity reward (kept minimal: only per-term position lists)."""
        tf = Counter(tokens)
        pos: dict[str, list[int]] = {}
        for i, t in enumerate(tokens):
            pos.setdefault(t, []).append(i)
        return cls(term_frequencies=tf, length=float(len(tokens)), positions=pos)


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
        Use unique query terms + sublinear query-TF.

        Rationale: coordination becomes meaningful only over distinct terms, and
        (1+log qtf) reduces the impact of repetition artifacts (esp. long/noisy queries),
        usually improving nDCG@10 without harming recall.
        """
        c = Counter(tokens)
        terms = list(c.keys())
        if Config.use_log_qtf:
            tw = {t: 1.0 + math.log(float(q)) for t, q in c.items()}
        else:
            tw = {t: float(q) for t, q in c.items()}
        return cls(terms=terms, term_weights=tw)


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
    doc_pos: dict[str, list[int]] | None = None,
) -> float:
    """
    Two coupled ideas:

    (1) IDF-weighted coverage: coordination should be about *information covered*,
        not just #tokens matched. This tends to lift docs that hit the rare "spine"
        of the query (better nDCG@10) while leaving partial matches competitive (recall).

    (2) Cheap proximity reward: if adjacent query terms appear within a small window,
        add a bounded boost. This helps theorem/stackoverflow-ish "phrasey" queries.
    """
    k1, b, eps = Config.k1, Config.b, Config.epsilon

    doc_uniq = float(len(doc_tf))
    mix = Config.focus_mix
    eff_len = (1.0 - mix) * doc_length + mix * doc_uniq
    avg_eff = max(avgdl, 1.0)
    norm = 1.0 - b + b * (eff_len / (avg_eff + eps)) if avg_eff > 0 else 1.0

    score = 0.0
    cov_num = 0.0
    cov_den = 0.0

    # Term evidence + accumulate IDF mass for coverage
    for term in query_repr.terms:
        df = float(corpus_df.get(term, 1))
        base = float(idf(df, N))
        term_idf = float(max(base, 0.0) ** Config.idf_power)
        if term_idf <= 0.0:
            continue

        cov_den += term_idf ** Config.cov_idf_power

        tf = float(doc_tf.get(term, 0))
        if tf <= 0.0:
            continue

        cov_num += term_idf ** Config.cov_idf_power

        tf_part = tf / (tf + k1 * norm + eps)
        score += query_repr.term_weights.get(term, 1.0) * term_idf * tf_part

    if score <= 0.0:
        return 0.0

    # IDF-weighted coordination (bounded reward >= 1)
    if cov_den > 0.0:
        coverage = cov_num / (cov_den + eps)
        score *= (1.0 + Config.coord_alpha * coverage) ** Config.coord_beta

    # Proximity / soft-phrase reward from adjacent query term pairs
    if doc_pos is not None and len(query_repr.terms) > 1 and Config.prox_alpha > 0:
        hits = 0.0
        pairs = 0.0
        w = int(max(1, Config.prox_window))
        for a, bterm in zip(query_repr.terms, query_repr.terms[1:]):
            pa = doc_pos.get(a)
            pb = doc_pos.get(bterm)
            if not pa or not pb:
                pairs += 1.0
                continue
            # two-pointer min distance
            i = j = 0
            best = 1_000_000
            while i < len(pa) and j < len(pb):
                da = pa[i] - pb[j]
                ad = da if da >= 0 else -da
                if ad < best:
                    best = ad
                    if best == 0:
                        break
                if da < 0:
                    i += 1
                else:
                    j += 1
            if best <= w:
                hits += 1.0
            pairs += 1.0
        if pairs > 0.0:
            score *= (1.0 + Config.prox_alpha * (hits / pairs))

    return score


def score_document(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Entry point used by BM25.score()."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0
    d = corpus.doc_repr[doc_idx]
    return retrieval_score(q, d.term_frequencies, d.length, corpus.N, corpus.avgdl, corpus.document_frequency, d.positions)


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

        # Build richer doc repr once (tf + positions) for proximity scoring in score().
        self.doc_repr = [DocumentRepr.from_tokens(d) for d in documents]

        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        self.doc_uniq = np.array([len(set(d)) for d in documents], dtype=np.float64)
        self.avguq = float(np.mean(self.doc_uniq)) if self.N > 0 else 1.0

        self._vocab: dict[str, int] = {}
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        self._doc_tf_dicts: list[Counter[str]] = [d.term_frequencies for d in self.doc_repr]

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

        base_idf = np.asarray(idf(self._df, self.N), dtype=np.float64)
        self.idf_array = np.power(np.maximum(base_idf, 0.0), Config.idf_power)

        b = Config.b
        mix = Config.focus_mix
        eff_len = (1.0 - mix) * self.doc_lengths + mix * self.doc_uniq
        avg_eff = (1.0 - mix) * max(self.avgdl, 1.0) + mix * max(self.avguq, 1.0)
        self.norm_array = 1.0 - b + b * (eff_len / max(avg_eff, 1.0))

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
        """
        Vectorized core score for rank(); matches retrieval_score *except* proximity
        (which is applied only in score(), not rank()) to keep ranking fast.
        """
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        k1, eps = Config.k1, Config.epsilon
        norms = self.corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        cov_num = np.zeros(len(candidate_docs), dtype=np.float64)
        cov_den = 0.0

        for i, term_id in enumerate(query_term_ids):
            idf_val = float(self.corpus.idf_array[term_id])
            if idf_val <= 0.0:
                continue

            cov_den += idf_val ** Config.cov_idf_power

            w = query_term_weights[i] if query_term_weights is not None else 1.0
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0).astype(np.float64)
            cov_num += present * (idf_val ** Config.cov_idf_power)

            tf_part = tf_row / (tf_row + k1 * norms + eps)
            scores += w * idf_val * tf_part

        if cov_den > 0.0:
            coverage = cov_num / (cov_den + eps)
            scores *= (1.0 + Config.coord_alpha * coverage) ** Config.coord_beta

        return scores

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not query:
            return np.arange(self.corpus.N, dtype=np.int64), np.zeros(self.corpus.N, dtype=np.float64)

        # Mirror QueryRepr: distinct terms + optional log-qtf
        term_counts = Counter(query)
        query_term_ids = []
        query_term_weights = []
        for term, count in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                if Config.use_log_qtf:
                    query_term_weights.append(1.0 + math.log(float(count)))
                else:
                    query_term_weights.append(float(count))

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
