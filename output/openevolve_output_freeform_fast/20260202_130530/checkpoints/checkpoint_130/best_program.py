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

    # --- NEW: discriminativity shaping ---
    # Convert raw IDF into a *lift* over the collection's average IDF:
    #   idf_lift = idf / mean_idf
    # This makes "important" mean "more discriminative than average" and reduces
    # dataset-to-dataset drift where absolute idf scale differs.
    idf_lift_power: float = 0.45  # 0 disables lift; small power keeps it gentle

    # --- NEW: query DF dropout (only for long/noisy queries) ---
    # For long queries, extremely common tokens behave like glue and increase false positives.
    # We drop terms with df/N above threshold, but only when query length >= q_drop_min_len.
    q_drop_min_len: int = 8
    q_drop_df_ratio: float = 0.22

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

    # Co-occurrence synergy (kept small; primarily helps top ranks)
    pair_boost: float = 0.06
    pair_power: float = 1.0

    # High-df terms are "glue words"; softly downweight instead of hard stopwording
    common_df_cut: float = 0.12   # fraction of corpus considered "common"
    common_penalty: float = 0.35  # max downweight for very common terms

    # Query specificity gating for AND-like effects (coordination + synergy).
    spec_floor: float = 0.55
    spec_power: float = 1.20

    # Query entropy gate (complements "peaky vs balanced").
    entropy_floor: float = 0.35
    entropy_power: float = 0.9

    # NEW: "anchor-first" mixing. Many tasks have 1–2 intent-defining rare terms.
    # We blend a pure-anchor score with the full evidence score:
    #   final = (1-w)*full + w*anchor
    # where w is high when the query is peaky (max-idf dominates sum-idf).
    # This often improves nDCG@10 by preventing broad modifiers from outranking
    # the document that best matches the anchor, while recall@100 is kept by full.
    anchor_mix_alpha: float = 0.35   # maximum mixture weight
    anchor_mix_power: float = 1.6    # sharpness vs peakiness
    anchor_residual: float = 0.55    # which terms count as "anchor-like" (as fraction of max_idf)

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
) -> float:
    """
    Two-channel lexical scoring:

    1) full_score: additive saturated evidence + gated "soft AND" (coord + pair)
    2) anchor_score: same evidence but only for "anchor-like" query terms
       (terms whose shaped-idf is close to the query max-idf).

    Final score is a query-dependent mixture. For peaky queries we trust anchors more
    (improves nDCG@10 by reducing modifier-driven false positives), while full_score
    maintains recall@100.
    """
    k1, b, eps = Config.k1, Config.b, Config.epsilon

    doc_uniq = float(len(doc_tf))
    mix = Config.focus_mix
    eff_len = (1.0 - mix) * doc_length + mix * doc_uniq
    avg_eff = max(avgdl, 1.0)
    norm = 1.0 - b + b * (eff_len / (avg_eff + eps)) if avg_eff > 0 else 1.0

    full_score = 0.0
    anchor_score = 0.0
    matched = 0.0

    m_idf: list[float] = []
    m_w: list[float] = []
    m_tfpart: list[float] = []
    m_is_anchor: list[bool] = []

    common_thr = Config.common_df_cut * float(N)

    # Query gates + anchor set (doc-independent).
    # NEW: for long queries, drop ultra-common tokens (df/N above threshold).
    # NEW: apply an IDF "lift" normalization vs mean IDF to stabilize importance across corpora.
    q_idfs: list[float] = []
    q_terms: list[str] = []

    q_idf_sum = 0.0
    q_idf_max = 0.0

    # Approx mean idf: compute from corpus_df on the fly if corpus doesn't provide it.
    # (Corpus path will provide mean_idf; this fallback keeps function standalone.)
    mean_idf = 1.0
    if hasattr(corpus_df, "_mean_idf_hint"):
        mean_idf = float(getattr(corpus_df, "_mean_idf_hint"))
    # If no hint, keep mean_idf=1.0 (lift becomes near-no-op).

    long_query = len(query_repr.terms) >= Config.q_drop_min_len
    for term in query_repr.terms:
        df = float(corpus_df.get(term, 1))

        if long_query and Config.q_drop_df_ratio > 0.0 and N > 0:
            if (df / float(N)) >= Config.q_drop_df_ratio:
                continue

        q_idf = float(max(float(idf(df, N)), 0.0) ** Config.idf_power)

        if df >= common_thr:
            frac = min(1.0, (df - common_thr) / (float(N) - common_thr + eps))
            q_idf *= (1.0 - Config.common_penalty * frac)

        if Config.idf_lift_power > 0.0:
            lift = q_idf / (mean_idf + eps)
            q_idf *= float(max(lift, 0.0) ** Config.idf_lift_power)

        q_terms.append(term)
        q_idfs.append(q_idf)
        q_idf_sum += q_idf
        if q_idf > q_idf_max:
            q_idf_max = q_idf

    qn = float(len(q_terms))

    spec = (q_idf_max / (q_idf_sum + eps)) if qn > 0.0 else 0.0
    spec_gate = max(Config.spec_floor, (1.0 - spec) ** Config.spec_power)

    ent_gate = 1.0
    if qn > 1.0 and q_idf_sum > 0.0:
        H = 0.0
        for v in q_idfs:
            p = v / (q_idf_sum + eps)
            if p > 0.0:
                H -= p * math.log(p + eps)
        Hn = H / (math.log(qn + eps) + eps)
        ent_gate = max(Config.entropy_floor, Hn ** Config.entropy_power)

    and_gate = spec_gate * ent_gate

    # Anchor mixture weight: larger when query is peaky (spec large).
    # Use a smooth monotone mapping of spec to [0, anchor_mix_alpha].
    w_anchor = Config.anchor_mix_alpha * (spec ** Config.anchor_mix_power)

    # Anchor threshold in shaped-idf space.
    anchor_thr = Config.anchor_residual * q_idf_max

    for term, q_idf in zip(q_terms, q_idfs):
        tf = float(doc_tf.get(term, 0))
        if tf <= 0.0:
            continue
        matched += 1.0

        term_idf = q_idf
        if term_idf <= 0.0:
            continue

        tf_part = tf / (tf + k1 * norm + eps)
        wq = float(query_repr.term_weights.get(term, 1.0))

        add = wq * term_idf * tf_part
        full_score += add

        is_anchor = term_idf >= anchor_thr
        if is_anchor:
            anchor_score += add

        m_idf.append(term_idf)
        m_w.append(wq)
        m_tfpart.append(tf_part)
        m_is_anchor.append(is_anchor)

    if full_score <= 0.0:
        return 0.0

    if qn > 1.0:
        coverage = matched / (qn + eps)
        full_score *= (1.0 + (Config.coord_alpha * and_gate) * coverage) ** Config.coord_beta

    # Pair synergy stays on full_score (anchors are meant to be conservative).
    m = len(m_idf)
    if m >= 2 and Config.pair_boost > 0.0:
        max_idf = max(m_idf) if m_idf else 0.0
        r = [max(0.0, v - 0.5 * max_idf) for v in m_idf]

        pair = 0.0
        for i in range(m):
            ri = r[i]
            if ri <= 0.0:
                continue
            for j in range(i + 1, m):
                rj = r[j]
                if rj <= 0.0:
                    continue
                gate = (m_tfpart[i] * m_tfpart[j]) ** 0.5
                pair += (ri * rj) ** Config.pair_power * (m_w[i] * m_w[j]) ** 0.5 * gate

        full_score *= (1.0 + (Config.pair_boost * and_gate) * pair)

    # Mixture. Ensure anchor_score doesn't go to 0 for multi-term matches that lack the max term:
    # keep a tiny floor fraction of full_score in the anchor channel.
    anchor_score = max(anchor_score, 0.15 * full_score)

    return (1.0 - w_anchor) * full_score + w_anchor * anchor_score


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

        # Unique-term length as a proxy for topical breadth / verbosity.
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

        # Discriminativity: classic BM25 idf, then sharpened (idf^power).
        base_idf = np.asarray(idf(self._df, self.N), dtype=np.float64)
        self.idf_array = np.power(np.maximum(base_idf, 0.0), Config.idf_power)

        # NEW: collection mean IDF (after base BM25 idf, before power is already applied above).
        # Used for "idf lift" normalization: terms matter insofar as they are more
        # discriminative than the average term in this corpus.
        self.mean_idf = float(np.mean(np.maximum(self.idf_array, 0.0))) if self.vocab_size > 0 else 1.0
        if self.mean_idf <= 0.0:
            self.mean_idf = 1.0

        # Pivoted normalization on a mix of token length and unique-term length.
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
        """Vectorized scoring for rank(); must match retrieval_score formula."""
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        k1, eps = Config.k1, Config.epsilon
        norms = self.corpus.norm_array[candidate_docs]
        mean_idf = float(getattr(self.corpus, "mean_idf", 1.0))
        if mean_idf <= 0.0:
            mean_idf = 1.0
        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        matched = np.zeros(len(candidate_docs), dtype=np.float64)

        idfs: list[float] = []
        ws: list[float] = []
        tfparts: list[NDArray[np.float64]] = []
        presents: list[NDArray[np.float64]] = []

        common_thr = Config.common_df_cut * float(self.corpus.N)

        # Query gate stats (must match retrieval_score()).
        q_idf_sum = 0.0
        q_idf_max = 0.0

        long_query = len(query_term_ids) >= Config.q_drop_min_len

        for i, term_id in enumerate(query_term_ids):
            df = float(self.corpus._df[term_id])

            if long_query and Config.q_drop_df_ratio > 0.0 and self.corpus.N > 0:
                if (df / float(self.corpus.N)) >= Config.q_drop_df_ratio:
                    continue

            idf_val = float(self.corpus.idf_array[term_id])
            if idf_val <= 0.0:
                continue

            if df >= common_thr:
                frac = min(1.0, (df - common_thr) / (float(self.corpus.N) - common_thr + eps))
                idf_val *= (1.0 - Config.common_penalty * frac)

            if Config.idf_lift_power > 0.0:
                lift = idf_val / (mean_idf + eps)
                idf_val *= float(max(lift, 0.0) ** Config.idf_lift_power)

            q_idf_sum += idf_val
            if idf_val > q_idf_max:
                q_idf_max = idf_val

            w = float(query_term_weights[i] if query_term_weights is not None else 1.0)
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().ravel()
            present = (tf_row > 0).astype(np.float64)
            matched += present
            tf_part = tf_row / (tf_row + k1 * norms + eps)

            scores += w * idf_val * tf_part

            idfs.append(idf_val)
            ws.append(w)
            tfparts.append(tf_part)
            presents.append(present)

        qn = float(len(query_term_ids))
        spec = (q_idf_max / (q_idf_sum + eps)) if qn > 0.0 else 0.0
        spec_gate = max(Config.spec_floor, (1.0 - spec) ** Config.spec_power)

        ent_gate = 1.0
        if qn > 1.0 and q_idf_sum > 0.0 and len(idfs) == len(query_term_ids):
            p = np.maximum(0.0, np.array(idfs, dtype=np.float64)) / (q_idf_sum + eps)
            H = -float(np.sum(np.where(p > 0.0, p * np.log(p + eps), 0.0)))
            Hn = H / (math.log(qn + eps) + eps)
            ent_gate = max(Config.entropy_floor, float(Hn ** Config.entropy_power))

        and_gate = spec_gate * ent_gate

        if qn > 1.0:
            coverage = matched / (qn + eps)
            scores *= (1.0 + (Config.coord_alpha * and_gate) * coverage) ** Config.coord_beta

        # Pair synergy on full score.
        m = len(idfs)
        if m >= 2 and Config.pair_boost > 0.0:
            idfs_arr = np.array(idfs, dtype=np.float64)
            max_idf = float(np.max(idfs_arr))
            r = np.maximum(0.0, idfs_arr - 0.5 * max_idf)

            pair = np.zeros(len(candidate_docs), dtype=np.float64)
            for i in range(m):
                if r[i] <= 0.0:
                    continue
                for j in range(i + 1, m):
                    if r[j] <= 0.0:
                        continue
                    gate = np.sqrt(tfparts[i] * tfparts[j])
                    pair += (r[i] * r[j]) ** Config.pair_power * math.sqrt(ws[i] * ws[j]) * gate * (presents[i] * presents[j])

            scores *= (1.0 + (Config.pair_boost * and_gate) * pair)

        # Anchor-first mixing (must match retrieval_score()).
        if q_idf_max > 0.0 and Config.anchor_mix_alpha > 0.0:
            w_anchor = Config.anchor_mix_alpha * (spec ** Config.anchor_mix_power)
            anchor_thr = Config.anchor_residual * q_idf_max

            # anchor_scores: sum only for anchor-like terms, but computed from already-built pieces.
            anchor_scores = np.zeros(len(candidate_docs), dtype=np.float64)
            for idf_val, w, tfp in zip(idfs, ws, tfparts):
                if idf_val >= anchor_thr:
                    anchor_scores += w * idf_val * tfp

            anchor_scores = np.maximum(anchor_scores, 0.15 * scores)
            scores = (1.0 - w_anchor) * scores + w_anchor * anchor_scores

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
