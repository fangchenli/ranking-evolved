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
    epsilon: float = 1e-9
    tf_log_base: float = 1.0
    coverage_gamma: float = 0.25
    qtf_power: float = 0.5
    q_clarity_power: float = 0.6
    dl_alpha: float = 0.15

    # Calibrated coordination boost: rewards satisfying more distinct query constraints.
    # Useful to suppress noisy 1-hit matches introduced by prefix/micro channels.
    coord_gamma: float = 0.20
    coord_mass_tau: float = 2.5

    rare_idf_pivot: float = 4.2
    anchor_boost: float = 0.14

    spec_beta: float = 0.10
    spec_cap: float = 3.0
    spec_len_floor: float = 25.0

    # Keep disabled by default (was unstable across sets).
    gini_alpha: float = 0.0

    residual_idf_tau: float = 1.25

    micro_len: int = 3
    micro_min_token_len: int = 2
    micro_weight: float = 0.12
    micro_gate_pivot: float = 2.2
    micro_gate_k: float = 1.0

    # Prefix channel (cheap morphological/identifier robustness).
    prefix_len: int = 5
    prefix_weight: float = 0.10

    # Optional bigram channel: purely lexical phrase/proximity specificity.
    # Often improves nDCG@10 on SciDocs/FiQA/ArguAna with small weight.
    bigram_weight: float = 0.08
    bigram_clarity_power: float = 0.90

    k1: float = 0.9
    b: float = 0.4


# -----------------------------------------------------------------------------
# IDF — EVOLVE: fundamental term importance (e.g. rarity, discriminativity)
# -----------------------------------------------------------------------------

def idf(df: float | NDArray[np.float64], N: int) -> float | NDArray[np.float64]:
    """
    Smoothed surprisal IDF: -log p(t in doc) with add-one smoothing.
    More stable cross-domain than BM25-odds and avoids extreme spikes.
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
        # Treat query as unique constraints; keep a sublinear repetition signal as weight.
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
    Concave surprisal evidence + clarity gate + IDF-mass coverage,
    plus bounded priors that mainly affect early precision:

    (1) Rare-term *anchor* (single best rare hit; recall-safe, robust to verbose queries).
    (2) Specificity/aboutness gain via positive PMI (bounded).
    (3) Cohesion prior via Gini concentration of matched query evidence m_t = wt*tf.
        Higher concentration => doc likely focuses on a decisive subset of query constraints.
    """
    if not query_repr.terms:
        return 0.0

    eps = Config.epsilon
    base = Config.tf_log_base
    tau = float(getattr(Config, "residual_idf_tau", 0.0))

    sum_evidence = 0.0
    cov_num = 0.0
    cov_den = 0.0

    # Coordination: number of distinct query constraints matched (within this channel).
    matched = 0.0
    uq = float(len(query_repr.terms))

    # Rare anchor: strongest hinge among matched high-idf terms.
    anchor = 0.0
    pivot = float(getattr(Config, "rare_idf_pivot", 0.0))

    use_spec = float(getattr(Config, "spec_beta", 0.0)) != 0.0
    spec_sum = 0.0
    spec_cap = float(getattr(Config, "spec_cap", 3.0))
    dl_eff = max(float(doc_length), float(getattr(Config, "spec_len_floor", 0.0)))

    # Gini concentration accumulators over matched query terms.
    use_gini = float(getattr(Config, "gini_alpha", 0.0)) != 0.0
    m_sum = 0.0
    m_sq_sum = 0.0
    k_match = 0.0

    for term in query_repr.terms:
        df = float(corpus_df.get(term, 1.0))
        tidf = float(idf(df, N))
        if tidf <= 0.0:
            continue

        rarity = tidf / (tidf + 1.0)
        clarity = rarity ** Config.q_clarity_power

        # Common query tokens are weak constraints: apply a smooth reliability gate.
        residual = tidf / (tidf + tau) if tau > 0.0 else 1.0

        wq = float(query_repr.term_weights.get(term, 1.0))
        wt = wq * tidf * clarity * residual
        cov_den += wt

        tf = float(doc_tf.get(term, 0.0))
        if tf <= 0.0:
            continue

        matched += 1.0
        cov_num += wt
        sum_evidence += wt * math.log1p(tf / (base + eps))

        if getattr(Config, "anchor_boost", 0.0) != 0.0 and tidf > pivot:
            hinge = (tidf - pivot) / (tidf + eps)  # in (0,1)
            if hinge > anchor:
                anchor = hinge

        if use_spec:
            p_td = tf / (dl_eff + eps)
            p_t = df / (float(N) + eps)
            g = math.log((p_td + eps) / (p_t + eps))
            if g > 0.0:
                spec_sum += wt * min(g, spec_cap)

        if use_gini:
            mt = wt * tf
            if mt > 0.0:
                k_match += 1.0
                m_sum += mt
                m_sq_sum += mt * mt

    if sum_evidence <= 0.0:
        return 0.0

    score = math.log1p(sum_evidence)

    if cov_den > 0.0 and Config.coverage_gamma != 0.0:
        score *= 1.0 + Config.coverage_gamma * (cov_num / (cov_den + eps))

    if use_spec and spec_sum > 0.0 and cov_den > 0.0:
        score *= 1.0 + float(Config.spec_beta) * (spec_sum / (cov_den + eps))

    # Coordination multiplier: behaves like a soft-AND, but attenuates when query already
    # has high informative mass (cov_den large). Recall-safe (only affects matching docs).
    if uq > 0.0 and getattr(Config, "coord_gamma", 0.0) != 0.0 and matched > 0.0 and cov_den > 0.0:
        q_mass = math.log1p(max(cov_den, 0.0))
        cal = 1.0 / (1.0 + q_mass / (float(getattr(Config, "coord_mass_tau", 2.5)) + eps))
        score *= 1.0 + float(Config.coord_gamma) * cal * (matched / (uq + eps))

    if getattr(Config, "anchor_boost", 0.0) != 0.0 and anchor > 0.0:
        score *= 1.0 + float(Config.anchor_boost) * math.log1p(anchor)

    if use_gini and k_match >= 2.0 and m_sum > 0.0:
        # Proxy Gini via Herfindahl: H = sum (p_i^2) where p_i = m_i/sum m.
        # Map to [0,1]: conc = (H - 1/K) / (1 - 1/K). High when concentrated.
        H = m_sq_sum / ((m_sum * m_sum) + eps)
        invk = 1.0 / (k_match + eps)
        conc = (H - invk) / (1.0 - invk + eps)
        conc = max(0.0, min(1.0, conc))
        score *= 1.0 + float(Config.gini_alpha) * conc

    length_ratio = (doc_length + 1.0) / (avgdl + 1.0)
    dl_damp = 1.0 + Config.dl_alpha * math.log1p(length_ratio)
    return score / (dl_damp + eps)


def score_document(query: list[str], doc_idx: int, corpus: Corpus) -> float:
    """Entry point used by BM25.score(). Adds prefix + gated micro-token channel."""
    if not query:
        return 0.0
    q = QueryRepr.from_tokens(query)
    if not q.terms:
        return 0.0

    doc_tf = corpus.get_term_frequencies(doc_idx)
    doc_length = float(corpus.doc_lengths[doc_idx])

    s = retrieval_score(q, doc_tf, doc_length, corpus.N, corpus.avgdl, corpus.document_frequency)

    # Prefix channel: robust to morphology / brittle tokenization, but keep weight small to stay precise.
    pw = float(getattr(Config, "prefix_weight", 0.0))
    if pw != 0.0 and getattr(Config, "prefix_len", 0) > 0:
        pfx = max(1, int(getattr(Config, "prefix_len", 5)))
        ptoks = [t[:pfx] for t in query if len(t) >= pfx]
        if ptoks:
            pq = QueryRepr.from_tokens(["P:" + t for t in ptoks])
            s += pw * retrieval_score(
                pq,
                corpus.prefix_doc_tf_dicts[doc_idx],
                doc_length,
                corpus.N,
                corpus.avgdl,
                corpus.document_frequency,
            )

    # Bigram channel: cheap phrase/proximity specificity.
    bw = float(getattr(Config, "bigram_weight", 0.0))
    if bw != 0.0 and len(query) >= 2:
        qb = ["B:" + query[i] + " " + query[i + 1] for i in range(len(query) - 1)]
        if qb:
            bq = QueryRepr.from_tokens(qb)
            s += bw * retrieval_score(
                bq,
                corpus.bigram_doc_tf_dicts[doc_idx],
                doc_length,
                corpus.N,
                corpus.avgdl,
                corpus.document_frequency,
            )

    # Micro channel: char n-grams help mostly on identifier-ish queries; gate by avg query IDF.
    mw = float(getattr(Config, "micro_weight", 0.0))
    if mw != 0.0 and getattr(Config, "micro_len", 0) > 0:
        idf_sum = 0.0
        idf_k = 0.0
        for t in q.terms:
            df = float(corpus.document_frequency.get(t, 1.0))
            idf_sum += float(idf(df, corpus.N))
            idf_k += 1.0
        avg_idf = idf_sum / (idf_k + Config.epsilon) if idf_k > 0.0 else 0.0

        pivot = float(getattr(Config, "micro_gate_pivot", 2.2))
        k = float(getattr(Config, "micro_gate_k", 1.0))
        gate = 1.0 / (1.0 + math.exp(-(avg_idf - pivot) / (k + Config.epsilon)))

        if gate > 0.01:
            m = max(2, int(Config.micro_len))
            min_tok = max(1, int(getattr(Config, "micro_min_token_len", 2)))
            mtoks: list[str] = []
            for t in query:
                if len(t) < min_tok:
                    continue
                if len(t) <= m:
                    mtoks.append("M:" + t)
                else:
                    for i in range(0, len(t) - m + 1):
                        mtoks.append("M:" + t[i : i + m])
            if mtoks:
                mq = QueryRepr.from_tokens(mtoks)
                s += (mw * gate) * retrieval_score(
                    mq,
                    corpus.micro_doc_tf_dicts[doc_idx],
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
        self.documents = documents
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.N = len(documents)
        self.document_count = self.N
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl

        # Prefix lexicon view (tagged) + bigram view + micro-token view.
        pfx = max(1, int(getattr(Config, "prefix_len", 5)))
        docs_prefix = [[t[:pfx] for t in doc if len(t) >= pfx] for doc in documents]
        self.prefix_doc_tf_dicts: list[Counter[str]] = [Counter(d) for d in docs_prefix]

        docs_bigram: list[list[str]] = []
        for doc in documents:
            if len(doc) < 2:
                docs_bigram.append([])
            else:
                docs_bigram.append(["B:" + doc[i] + " " + doc[i + 1] for i in range(len(doc) - 1)])
        self.bigram_doc_tf_dicts: list[Counter[str]] = [Counter(d) for d in docs_bigram]

        m = max(2, int(getattr(Config, "micro_len", 3)))
        min_tok = max(1, int(getattr(Config, "micro_min_token_len", 2)))
        micro_docs: list[list[str]] = []
        for doc in documents:
            grams: list[str] = []
            for t in doc:
                if len(t) < min_tok:
                    continue
                if len(t) <= m:
                    grams.append("M:" + t)
                else:
                    for i in range(0, len(t) - m + 1):
                        grams.append("M:" + t[i : i + m])
            micro_docs.append(grams)
        self.micro_doc_tf_dicts: list[Counter[str]] = [Counter(g) for g in micro_docs]

        # Joint vocabulary: base tokens + tagged prefixes + bigrams + micro tokens.
        self._vocab: dict[str, int] = {}
        for doc, pdoc, bdoc, mdoc in zip(documents, docs_prefix, docs_bigram, micro_docs):
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
            for p in pdoc:
                key = "P:" + p
                if key not in self._vocab:
                    self._vocab[key] = len(self._vocab)
            for bg in bdoc:
                if bg not in self._vocab:
                    self._vocab[bg] = len(self._vocab)
            for g in mdoc:
                if g not in self._vocab:
                    self._vocab[g] = len(self._vocab)

        self.vocab_size = len(self._vocab)

        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        self._doc_tf_dicts: list[Counter[str]] = [Counter(doc) for doc in documents]

        for doc_idx, (doc, pdoc, bdoc, mdoc) in enumerate(zip(documents, docs_prefix, docs_bigram, micro_docs)):
            term_counts = Counter(doc)
            pref_counts = Counter("P:" + p for p in pdoc)
            bigr_counts = Counter(bdoc)
            micro_counts = Counter(mdoc)
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

            for term, count in bigr_counts.items():
                tid = self._vocab[term]
                tf_matrix_lil[tid, doc_idx] = count
                if tid not in seen:
                    self._inverted_index[tid].append(doc_idx)
                    self._df[tid] += 1
                    seen.add(tid)

            for term, count in micro_counts.items():
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
        tau = float(getattr(Config, "residual_idf_tau", 0.0))

        sum_evidence = np.zeros(len(candidate_docs), dtype=np.float64)
        cov_num = np.zeros(len(candidate_docs), dtype=np.float64)
        cov_den = 0.0

        # Rare anchor (max hinge) per doc.
        anchor = np.zeros(len(candidate_docs), dtype=np.float64)
        pivot = float(getattr(Config, "rare_idf_pivot", 0.0))

        use_spec = float(getattr(Config, "spec_beta", 0.0)) != 0.0
        spec_sum = np.zeros(len(candidate_docs), dtype=np.float64)
        spec_cap = float(getattr(Config, "spec_cap", 3.0))
        dl_eff = np.maximum(
            self.corpus.doc_lengths[candidate_docs],
            float(getattr(Config, "spec_len_floor", 0.0)),
        )

        # Gini proxy via Herfindahl concentration over m_t = wt*tf.
        use_gini = float(getattr(Config, "gini_alpha", 0.0)) != 0.0
        m_sum = np.zeros(len(candidate_docs), dtype=np.float64)
        m_sq_sum = np.zeros(len(candidate_docs), dtype=np.float64)
        k_match = np.zeros(len(candidate_docs), dtype=np.float64)

        for i, term_id in enumerate(query_term_ids):
            idf_val = float(self.corpus.idf_array[term_id])
            if idf_val <= 0.0:
                continue

            rarity = idf_val / (idf_val + 1.0)
            clarity = rarity ** Config.q_clarity_power

            residual = idf_val / (idf_val + tau) if tau > 0.0 else 1.0
            wq = float(query_term_weights[i]) if query_term_weights is not None else 1.0
            wt = wq * idf_val * clarity * residual
            cov_den += wt

            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0.0).astype(np.float64)
            cov_num += wt * present
            sum_evidence += wt * np.log1p(tf_row / (base + eps))

            if getattr(Config, "anchor_boost", 0.0) != 0.0 and idf_val > pivot:
                hinge = (idf_val - pivot) / (idf_val + eps)
                anchor = np.maximum(anchor, present * hinge)

            if use_spec:
                df_val = float(self.corpus._df[term_id])
                p_td = tf_row / (dl_eff + eps)
                p_t = df_val / (float(self.corpus.N) + eps)
                g = np.log((p_td + eps) / (p_t + eps))
                g = np.minimum(g, spec_cap)
                spec_sum += wt * np.maximum(g, 0.0)

            if use_gini:
                mt = wt * tf_row
                mt_pos = np.maximum(mt, 0.0)
                k_match += (mt_pos > 0.0).astype(np.float64)
                m_sum += mt_pos
                m_sq_sum += mt_pos * mt_pos

        scores = np.log1p(np.maximum(sum_evidence, 0.0))

        if cov_den > 0.0 and Config.coverage_gamma != 0.0:
            scores *= 1.0 + Config.coverage_gamma * (cov_num / (cov_den + eps))

        if use_spec and cov_den > 0.0:
            scores *= 1.0 + float(Config.spec_beta) * (spec_sum / (cov_den + eps))

        if getattr(Config, "anchor_boost", 0.0) != 0.0:
            scores *= 1.0 + float(Config.anchor_boost) * np.log1p(np.maximum(anchor, 0.0))

        if use_gini:
            mask = (k_match >= 2.0) & (m_sum > 0.0)
            if np.any(mask):
                H = m_sq_sum / ((m_sum * m_sum) + eps)
                invk = 1.0 / (k_match + eps)
                conc = (H - invk) / (1.0 - invk + eps)
                conc = np.clip(conc, 0.0, 1.0)
                scores *= 1.0 + float(Config.gini_alpha) * (conc * mask.astype(np.float64))

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

        # Combined query over base tokens + prefixes + (gated) micro (char n-gram) tokens.
        term_counts = Counter(query)
        query_term_ids: list[int] = []
        query_term_weights: list[float] = []

        for term, count in term_counts.items():
            tid = self.corpus.get_term_id(term)
            if tid is not None:
                query_term_ids.append(tid)
                query_term_weights.append(float(count) ** Config.qtf_power)

        pw = float(getattr(Config, "prefix_weight", 0.0))
        if pw != 0.0 and getattr(Config, "prefix_len", 0) > 0:
            pfx = max(1, int(getattr(Config, "prefix_len", 5)))
            pcounts = Counter(t[:pfx] for t in query if len(t) >= pfx)
            for p, c in pcounts.items():
                tid = self.corpus.get_term_id("P:" + p)
                if tid is not None:
                    query_term_ids.append(tid)
                    query_term_weights.append(pw * (float(c) ** Config.qtf_power))

        bw = float(getattr(Config, "bigram_weight", 0.0))
        if bw != 0.0 and len(query) >= 2:
            bcounts = Counter("B:" + query[i] + " " + query[i + 1] for i in range(len(query) - 1))
            for bg, c in bcounts.items():
                tid = self.corpus.get_term_id(bg)
                if tid is not None:
                    query_term_ids.append(tid)
                    query_term_weights.append(bw * (float(c) ** Config.qtf_power))

        mw = float(getattr(Config, "micro_weight", 0.0))
        if mw != 0.0 and getattr(Config, "micro_len", 0) > 0 and term_counts:
            idf_sum = 0.0
            idf_k = 0.0
            for term in term_counts.keys():
                df = float(self.corpus.document_frequency.get(term, 1.0))
                idf_sum += float(idf(df, self.corpus.N))
                idf_k += 1.0
            avg_idf = idf_sum / (idf_k + Config.epsilon) if idf_k > 0.0 else 0.0
            pivot = float(getattr(Config, "micro_gate_pivot", 2.2))
            k = float(getattr(Config, "micro_gate_k", 1.0))
            gate = 1.0 / (1.0 + math.exp(-(avg_idf - pivot) / (k + Config.epsilon)))

            if gate > 0.01:
                m = max(2, int(Config.micro_len))
                min_tok = max(1, int(getattr(Config, "micro_min_token_len", 2)))
                grams: list[str] = []
                for t in query:
                    if len(t) < min_tok:
                        continue
                    if len(t) <= m:
                        grams.append("M:" + t)
                    else:
                        for i in range(0, len(t) - m + 1):
                            grams.append("M:" + t[i : i + m])
                if grams:
                    gcounts = Counter(grams)
                    for g, c in gcounts.items():
                        tid = self.corpus.get_term_id(g)
                        if tid is not None:
                            query_term_ids.append(tid)
                            query_term_weights.append((mw * gate) * (float(c) ** Config.qtf_power))

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
