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
    # Core: concave surprisal evidence + informative-coverage boost + mild verbosity prior.
    k1: float = 0.9   # kept for backwards compatibility (Corpus.norm_array)
    b: float = 0.4    # kept for backwards compatibility (Corpus.norm_array)
    epsilon: float = 1e-9

    tf_log_base: float = 1.0
    dl_alpha: float = 0.15
    q_clarity_power: float = 0.6
    coverage_gamma: float = 0.25
    qtf_power: float = 0.5

    # "Facet prior": reallocates query mass toward discriminative constraints (query-only).
    facet_mix: float = 0.12
    facet_power: float = 1.6

    # Soft-AND / coordination pressure based on informative mass covered.
    coord_beta: float = 0.08

    # --- Robust lexical matching (secondary channel) ---
    prefix_len: int = 5
    prefix_weight: float = 0.18

    # NEW: character n-gram channel for tokenization mismatch (URLs, code, hyphenation).
    # We keep it tiny + cheap: only query ngrams, treated as pseudo-terms.
    ngram_n: int = 4
    ngram_max_per_token: int = 2
    ngram_weight: float = 0.10

    # --- Rare-key presence (bounded multiplier) ---
    rare_idf_pivot: float = 4.5
    rare_boost: float = 0.12


# -----------------------------------------------------------------------------
# IDF — EVOLVE: fundamental term importance (e.g. rarity, discriminativity)
# -----------------------------------------------------------------------------

def idf(df: float | NDArray[np.float64], N: int) -> float | NDArray[np.float64]:
    """
    Surprisal IDF: log1p(N/df).

    Interprets df/N as an occurrence probability; matching a term yields self-information.
    This is typically smoother and more robust cross-domain than BM25-odds IDF.
    """
    df = np.maximum(df, 1.0)
    return np.log1p(N / df)


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
        Represent the query as unique lexical constraints but keep sublinear repetition.

        Motivation: repeated tokens can encode emphasis, but linear qtf tends to
        over-weight verbosity/noisy tokenization.
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
    Concave surprisal evidence + clarity gating + coverage, plus:
      (1) query-internal "facet prior" (reallocates weight toward high-IDF constraints)
      (2) saturating coordination bonus based on informative mass covered
      (3) bounded rare-key presence multiplier

    Facet prior story:
      A query is often a mixture of a few decisive facets and many background hints.
      We approximate this by transforming per-term IDF inside the query:
        idf_used = (1-m)*idf + m*(idf^p / mean(idf^p))
      This sharpens ranking (nDCG@10) while keeping the evidence still additive/recall-friendly.
    """
    if not query_repr.terms:
        return 0.0

    eps = Config.epsilon
    base = Config.tf_log_base

    # Precompute query-level scale for facet prior (query-only; stable across documents).
    idf_pows = []
    for term in query_repr.terms:
        tidf = float(idf(float(corpus_df.get(term, 1.0)), N))
        if tidf > 0.0:
            idf_pows.append(tidf ** Config.facet_power)
    mean_idf_pow = (sum(idf_pows) / len(idf_pows)) if idf_pows else 1.0
    mean_idf_pow = max(mean_idf_pow, eps)
    mix = max(0.0, min(1.0, Config.facet_mix))

    sum_evidence = 0.0
    cov_num = 0.0
    cov_den = 0.0
    rare_hits = 0.0

    for term in query_repr.terms:
        df = float(corpus_df.get(term, 1.0))
        term_idf = float(idf(df, N))
        if term_idf <= 0.0:
            continue

        # bounded query clarity in [0,1]
        rarity = term_idf / (term_idf + 1.0)
        clarity = rarity ** Config.q_clarity_power

        # facet-reweighted idf (query-dependent only)
        facet = (term_idf ** Config.facet_power) / mean_idf_pow
        idf_used = (1.0 - mix) * term_idf + mix * facet

        wq = float(query_repr.term_weights.get(term, 1.0))

        # coverage uses plain IDF mass (keeps recall stable across corpora)
        cov_wt = wq * term_idf * clarity
        cov_den += cov_wt

        tf = float(doc_tf.get(term, 0.0))
        if tf <= 0.0:
            continue

        cov_num += cov_wt
        sum_evidence += (wq * clarity * idf_used) * math.log1p(tf / (base + eps))

        if Config.rare_boost != 0.0 and term_idf > Config.rare_idf_pivot:
            rare_hits += (term_idf - Config.rare_idf_pivot) / (term_idf + eps)

    if sum_evidence <= 0.0:
        return 0.0

    score = math.log1p(sum_evidence)

    # soft-AND via informative coverage
    if cov_den > 0.0 and Config.coverage_gamma != 0.0:
        coverage = cov_num / (cov_den + eps)
        score *= 1.0 + Config.coverage_gamma * coverage
        if Config.coord_beta != 0.0:
            # saturating coordination: emphasizes "more constraints satisfied" without hard AND
            score *= 1.0 + Config.coord_beta * (1.0 - math.exp(-3.0 * coverage))

    if Config.rare_boost != 0.0 and rare_hits > 0.0:
        score *= 1.0 + Config.rare_boost * math.log1p(rare_hits)

    length_ratio = (doc_length + 1.0) / (avgdl + 1.0)
    dl_damp = 1.0 + Config.dl_alpha * math.log1p(length_ratio)
    return score / (dl_damp + eps)


def score_document(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Entry point used by BM25.score()."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0

    doc_tf = corpus.get_term_frequencies(doc_idx)
    doc_length = float(corpus.doc_lengths[doc_idx])

    # Primary channel (exact tokens).
    s = retrieval_score(q, doc_tf, doc_length, corpus.N, corpus.avgdl, corpus.document_frequency)

    # Secondary channel (prefixes) for robust lexical matching.
    if Config.prefix_weight != 0.0 and Config.prefix_len > 0:
        pfx = max(1, int(Config.prefix_len))
        ptoks = [t[:pfx] for t in query if len(t) >= pfx]
        if ptoks:
            pq = QueryRepr.from_tokens(["P:" + t for t in ptoks])
            pdoc_tf = corpus.prefix_doc_tf_dicts[doc_idx]
            s += Config.prefix_weight * retrieval_score(
                pq, pdoc_tf, doc_length, corpus.N, corpus.avgdl, corpus.document_frequency
            )

    # NEW: tiny character n-gram channel (query-only extraction) to survive tokenization gaps.
    # This is still lexical and uses the same retrieval_score machinery via pseudo-terms.
    if Config.ngram_weight != 0.0 and Config.ngram_n > 1:
        n = int(Config.ngram_n)
        cap = max(1, int(Config.ngram_max_per_token))
        grams: list[str] = []
        for t in query:
            if len(t) < n:
                continue
            step = max(1, (len(t) - n) // cap)  # take a couple spaced grams, not all
            for j in range(0, len(t) - n + 1, step):
                grams.append("G:" + t[j : j + n])
                if len(grams) >= cap * max(1, len(query)):
                    break
        if grams:
            gq = QueryRepr.from_tokens(grams)
            s += Config.ngram_weight * retrieval_score(
                gq,
                corpus.ngram_doc_tf_dicts[doc_idx],
                doc_length,
                corpus.N,
                corpus.avgdl,
                corpus.document_frequency,
            )

    return s


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
        # MEMORY OPTIMIZATION: Don\'t store documents - only needed during construction
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.document_count = self.N
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        # Secondary "prefix lexicon" view (cheap robustness to morphology/identifiers).
        pfx = max(1, int(Config.prefix_len))
        docs_prefix = [[t[:pfx] for t in doc if len(t) >= pfx] for doc in documents]
        self.prefix_doc_tf_dicts: list[Counter[str]] = [Counter(d) for d in docs_prefix]

        # NEW: tiny character n-gram view (disjoint channel) for tokenization mismatch.
        n = int(Config.ngram_n) if getattr(Config, "ngram_n", 0) else 0
        cap = max(1, int(Config.ngram_max_per_token)) if getattr(Config, "ngram_max_per_token", 0) else 1
        if n > 1:
            docs_ngrams: list[list[str]] = []
            for doc in documents:
                gs: list[str] = []
                for t in doc:
                    if len(t) < n:
                        continue
                    step = max(1, (len(t) - n) // cap)
                    for j in range(0, len(t) - n + 1, step):
                        gs.append("G:" + t[j : j + n])
                        if len(gs) >= cap * max(1, len(doc)):
                            break
                docs_ngrams.append(gs)
            self.ngram_doc_tf_dicts: list[Counter[str]] = [Counter(d) for d in docs_ngrams]
        else:
            docs_ngrams = [[] for _ in documents]
            self.ngram_doc_tf_dicts = [Counter() for _ in documents]

        # Joint vocabulary over tokens + tagged prefixes + tagged ngrams.
        self._vocab: dict[str, int] = {}
        for doc, pdoc, gdoc in zip(documents, docs_prefix, docs_ngrams):
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
            for p in pdoc:
                key = "P:" + p
                if key not in self._vocab:
                    self._vocab[key] = len(self._vocab)
            for g in gdoc:
                if g not in self._vocab:
                    self._vocab[g] = len(self._vocab)
        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        # MEMORY OPTIMIZATION: Don\'t precompute _doc_tf_dicts - reconstruct on-demand from tf_matrix

        for doc_idx, (doc, pdoc, gdoc) in enumerate(zip(documents, docs_prefix, docs_ngrams)):
            term_counts = Counter(doc)
            pref_counts = Counter("P:" + p for p in pdoc)
            gram_counts = Counter(gdoc)
            seen = set()

            for term, count in term_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

            for term, count in pref_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

            for term, count in gram_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

        self.tf_matrix = csr_matrix(tf_matrix_lil)
        self.idf_array = np.asarray(idf(self._df, self.N), dtype=np.float64)
        b = Config.b
        self.norm_array = 1.0 - b + b * (self.doc_lengths / max(self.avgdl, 1.0))

        # Expose df for all channels (prefix + ngram keys included).
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

        # Query-level facet prior scale (computed from idf_array; query-only).
        idf_vals = np.array([float(self.corpus.idf_array[t]) for t in query_term_ids], dtype=np.float64)
        idf_vals = np.maximum(idf_vals, 0.0)
        idf_pow = np.power(idf_vals, Config.facet_power, dtype=np.float64)
        mean_idf_pow = float(np.mean(idf_pow)) if idf_pow.size > 0 else 1.0
        mean_idf_pow = max(mean_idf_pow, eps)
        mix = max(0.0, min(1.0, Config.facet_mix))

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

            facet = (idf_val ** Config.facet_power) / mean_idf_pow
            idf_used = (1.0 - mix) * idf_val + mix * facet

            wq = float(query_term_weights[i]) if query_term_weights is not None else 1.0

            cov_wt = wq * idf_val * clarity
            cov_den += cov_wt

            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0.0).astype(np.float64)

            cov_num += cov_wt * present
            sum_evidence += (wq * clarity * idf_used) * np.log1p(tf_row / (base + eps))

            if Config.rare_boost != 0.0 and idf_val > Config.rare_idf_pivot:
                rare_hits += present * ((idf_val - Config.rare_idf_pivot) / (idf_val + eps))

        scores = np.log1p(np.maximum(sum_evidence, 0.0))

        if cov_den > 0.0 and Config.coverage_gamma != 0.0:
            coverage = cov_num / (cov_den + eps)
            scores *= 1.0 + Config.coverage_gamma * coverage
            if Config.coord_beta != 0.0:
                scores *= 1.0 + Config.coord_beta * (1.0 - np.exp(-3.0 * coverage))

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

        # Combined query: tokens + (optional) tagged prefixes + (optional) tagged ngrams.
        term_counts = Counter(query)
        query_term_ids: list[int] = []
        query_term_weights: list[float] = []

        for term, count in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                query_term_weights.append(float(count) ** Config.qtf_power)

        if Config.prefix_weight != 0.0 and Config.prefix_len > 0:
            pfx = max(1, int(Config.prefix_len))
            pcounts = Counter(t[:pfx] for t in query if len(t) >= pfx)
            for p, c in pcounts.items():
                tid = self.corpus.get_term_id("P:" + p)
                if tid is not None:
                    query_term_ids.append(tid)
                    query_term_weights.append(Config.prefix_weight * (float(c) ** Config.qtf_power))

        if Config.ngram_weight != 0.0 and Config.ngram_n > 1:
            n = int(Config.ngram_n)
            cap = max(1, int(Config.ngram_max_per_token))
            gcounts: Counter[str] = Counter()
            for t in query:
                if len(t) < n:
                    continue
                step = max(1, (len(t) - n) // cap)
                for j in range(0, len(t) - n + 1, step):
                    gcounts["G:" + t[j : j + n]] += 1
            for g, c in gcounts.items():
                tid = self.corpus.get_term_id(g)
                if tid is not None:
                    query_term_ids.append(tid)
                    query_term_weights.append(Config.ngram_weight * (float(c) ** Config.qtf_power))

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
