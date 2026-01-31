"""
BM25 Pyserini-Compatible Implementation - Seed Program for Evolution.

This file reproduces Pyserini/Lucene's BM25 behavior exactly, designed as
a starting point for AlphaEvolve to optimize.

Key Features:
- Uses Pyserini's actual Lucene tokenizer (requires Java 21)
- Matches Pyserini's BM25 scoring exactly
- All scoring components are exposed as EVOLUTION TARGETs

Pyserini Configuration (defaults):
    - k1=0.9, b=0.4 (Lucene defaults)
    - IDF: log(1 + (N - df + 0.5) / (df + 0.5))
    - TF: tf / (tf + k1 * norm)  [Lucene formula, no (k1+1) multiplier]
    - Length norm: 1 - b + b * (dl / avgdl)
    - Tokenization: Lucene DefaultEnglishAnalyzer (Porter stemming + stopwords)

Usage:
    from ranking_evolved.bm25_pyserini import BM25, Corpus, tokenize, LuceneTokenizer

    tokenizer = LuceneTokenizer()  # Uses Pyserini's Lucene analyzer
    corpus = Corpus([tokenizer(doc) for doc in documents])
    bm25 = BM25(corpus)
    indices, scores = bm25.rank(tokenizer(query))
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ranking_evolved.bm25 import (
    LUCENE_STOPWORDS,
)

# Import our pure-Python LuceneTokenizer for fallback (with Porter stemming)
from ranking_evolved.bm25 import (
    LuceneTokenizer as PurePythonLuceneTokenizer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# EVOLUTION TARGET 1: BM25 Parameters
# =============================================================================


class Parameters:
    """
    BM25 parameters - EVOLUTION TARGET.

    These match Pyserini/Lucene defaults. AlphaEvolve can modify these
    to discover optimal combinations for specific domains.

    Alternatives to explore:
        k1: 0.5-2.0 range, controls TF saturation speed
        b: 0.0-1.0 range, controls length normalization strength
        k3: 0-1000 range, controls query TF saturation (if used)
    """

    k1: float = 0.9  # TF saturation (Pyserini default)
    b: float = 0.4  # Length normalization (Pyserini default)
    k3: float = 8.0  # Query TF saturation


# =============================================================================
# EVOLUTION TARGET 2: IDF Formula
# =============================================================================


class IDFFormula:
    """
    IDF computation - EVOLUTION TARGET.

    Default: Lucene formula log(1 + (N - df + 0.5) / (df + 0.5))

    Alternative formulas to explore:
        - Robertson: log((N - df + 0.5) / (df + 0.5))  [can be negative]
        - ATIRE: log(N / df)
        - BM25L: log((N + 1) / (df + 0.5))
        - BM25+: log((N + 1) / df)
        - Probabilistic: log((N - df) / df)
    """

    @staticmethod
    def compute(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        """
        Compute IDF values - EVOLUTION TARGET.

        Args:
            df: Document frequencies for each term
            N: Total number of documents

        Returns:
            IDF values (same shape as df)
        """
        # ===== EVOLVE THIS FORMULA =====
        # Lucene/Pyserini IDF (always non-negative)
        return np.log(1 + (N - df + 0.5) / (df + 0.5))


# =============================================================================
# EVOLUTION TARGET 3: TF Saturation Formula
# =============================================================================


class TFFormula:
    """
    Term frequency saturation - EVOLUTION TARGET.

    Default (Lucene): tf / (tf + k1 * norm)

    Note: Lucene's BM25 does NOT use the (k1+1) multiplier in the numerator.
    This matches Pyserini's scoring exactly.

    Alternative formulas to explore:
        - Robertson: (tf * (k1 + 1)) / (tf + k1 * norm)  [original paper]
        - Linear: tf / norm
        - Log: log(1 + tf) / norm
        - BM25+: classic + delta (adds bonus)
        - PL2: tf * log(1 + c * avgdl / dl)
    """

    @staticmethod
    def compute(tf: float, k1: float, norm: float) -> float:
        """
        Compute saturated TF - EVOLUTION TARGET.

        Args:
            tf: Raw term frequency in document
            k1: Saturation parameter
            norm: Length normalization factor

        Returns:
            Saturated TF score
        """
        # ===== EVOLVE THIS FORMULA =====
        # Lucene's formula (no (k1+1) multiplier)
        return tf / (tf + k1 * norm)


# =============================================================================
# EVOLUTION TARGET 4: Length Normalization
# =============================================================================


class LengthNorm:
    """
    Document length normalization - EVOLUTION TARGET.

    Default: 1 - b + b * (dl / avgdl)

    Alternative formulas to explore:
        - None: 1.0 (no normalization)
        - Pivoted: (1 - s) + s * (dl / avgdl)
        - Log: 1 / log(e + dl)
        - Sqrt: 1 / sqrt(dl)
    """

    @staticmethod
    def compute(doc_len: int, avgdl: float, b: float) -> float:
        """
        Compute length normalization factor - EVOLUTION TARGET.

        Args:
            doc_len: Document length (number of terms)
            avgdl: Average document length in corpus
            b: Normalization strength (0=none, 1=full)

        Returns:
            Normalization factor (>0)
        """
        # ===== EVOLVE THIS FORMULA =====
        return 1 - b + b * (doc_len / avgdl)


# =============================================================================
# EVOLUTION TARGET 5: Query Term Weighting
# =============================================================================


class QueryWeighting:
    """
    Query term weighting mode - EVOLUTION TARGET.

    Controls how repeated query terms are handled.

    Modes:
        - "unique": Each unique term contributes once (bag-of-words)
        - "count": Weight by term frequency in query
        - "saturated": Apply BM25-style saturation: (k3+1)*qtf/(k3+qtf)
    """

    mode: str = "unique"

    @staticmethod
    def get_weights(
        query: list[str],
        k3: float,
        mode: str = "unique",
    ) -> tuple[list[str], NDArray[np.float64]]:
        """
        Compute query term weights - EVOLUTION TARGET.

        Args:
            query: List of query terms (may have duplicates)
            k3: Saturation parameter for "saturated" mode
            mode: Weighting mode

        Returns:
            Tuple of (unique_terms, weights)
        """
        if not query:
            return [], np.array([], dtype=np.float64)

        # ===== EVOLVE THIS LOGIC =====
        term_counts = Counter(query)
        unique_terms = list(term_counts.keys())

        if mode == "unique":
            weights = np.ones(len(unique_terms), dtype=np.float64)
        elif mode == "count":
            weights = np.array([term_counts[t] for t in unique_terms], dtype=np.float64)
        elif mode == "saturated":
            qtf = np.array([term_counts[t] for t in unique_terms], dtype=np.float64)
            weights = ((k3 + 1) * qtf) / (k3 + qtf)
        else:
            weights = np.ones(len(unique_terms), dtype=np.float64)

        return unique_terms, weights


# =============================================================================
# EVOLUTION TARGET 6: Score Aggregation
# =============================================================================


class ScoreAggregation:
    """
    Score aggregation method - EVOLUTION TARGET.

    Controls how individual term scores are combined.

    Modes:
        - "sum": Simple sum of term scores (standard BM25)
        - "weighted_sum": Sum weighted by query term importance
        - "max": Maximum term score
        - "mean": Average of term scores
    """

    mode: str = "sum"

    @staticmethod
    def aggregate(
        term_scores: NDArray[np.float64],
        query_weights: NDArray[np.float64] | None = None,
        mode: str = "sum",
    ) -> float:
        """
        Aggregate term scores - EVOLUTION TARGET.

        Args:
            term_scores: Individual term IDF*TF scores
            query_weights: Optional query term weights
            mode: Aggregation mode

        Returns:
            Final document score
        """
        if len(term_scores) == 0:
            return 0.0

        # ===== EVOLVE THIS LOGIC =====
        if mode == "sum":
            return float(np.sum(term_scores))
        elif mode == "weighted_sum" and query_weights is not None:
            return float(np.sum(term_scores * query_weights))
        elif mode == "max":
            return float(np.max(term_scores))
        elif mode == "mean":
            return float(np.mean(term_scores))
        else:
            return float(np.sum(term_scores))


# =============================================================================
# Tokenization - Uses Pyserini's Lucene Analyzer
# =============================================================================


def _get_pyserini_tokenizer() -> Callable[[str], list[str]] | None:
    """Try to get Pyserini's Lucene tokenizer."""
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        analyzer = Analyzer(get_lucene_analyzer())
        return analyzer.analyze
    except Exception:
        return None


# Lucene English stopwords (fallback)
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _fallback_tokenize(text: str) -> list[str]:
    """Simple tokenizer (lowercase + alphanumeric only)."""
    return [t for t in text.lower().split() if t.isalnum()]


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for compatibility."""
    return _fallback_tokenize(text)


class LuceneTokenizer:
    """
    Lucene-compatible tokenizer.

    Uses Pyserini's actual Lucene DefaultEnglishAnalyzer when available,
    which applies:
    - Tokenization on non-letter boundaries
    - Lowercasing
    - Porter stemming
    - English stopword removal (33 words)

    Falls back to our pure-Python LuceneTokenizer (with Porter stemming
    and the official 33-word Lucene stoplist) if Pyserini/Java unavailable.
    """

    def __init__(self):
        self._pyserini_tokenize = _get_pyserini_tokenizer()
        self._fallback_tokenizer = None
        if self._pyserini_tokenize is None:
            # Use our pure-Python LuceneTokenizer with official 33-word stoplist
            self._fallback_tokenizer = PurePythonLuceneTokenizer(stopwords=LUCENE_STOPWORDS)

    def __call__(self, text: str) -> list[str]:
        """Tokenize text using Lucene analyzer."""
        if self._pyserini_tokenize is not None:
            return self._pyserini_tokenize(text)

        # Fallback: pure-Python LuceneTokenizer with Porter stemming + 33-word stoplist
        return self._fallback_tokenizer(text)


# =============================================================================
# Corpus
# =============================================================================


class Corpus:
    """
    Pre-processed document collection.

    Stores tokenized documents and computes corpus statistics needed for BM25.
    """

    def __init__(
        self,
        documents: list[list[str]],
        ids: list[str] | None = None,
    ):
        """
        Initialize corpus.

        Args:
            documents: List of tokenized documents (each is list of terms)
            ids: Optional document IDs
        """
        self.documents = documents
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}

        # Corpus statistics
        self.N = len(documents)
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0

        # Build vocabulary and document frequencies
        self._vocab: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._doc_term_freqs: list[dict[str, int]] = []

        for doc in documents:
            term_counts = Counter(doc)
            self._doc_term_freqs.append(term_counts)
            for term in term_counts:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
                    self._df[term] = 0
                self._df[term] += 1

    def __len__(self) -> int:
        return self.N

    def get_df(self, term: str) -> int:
        """Get document frequency for a term."""
        return self._df.get(term, 0)

    def get_tf(self, doc_idx: int, term: str) -> int:
        """Get term frequency in a specific document."""
        return self._doc_term_freqs[doc_idx].get(term, 0)

    @property
    def map_id_to_idx(self) -> dict[str, int]:
        """Mapping from document ID to index."""
        return self._id_to_idx

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in corpus."""
        return len(self._vocab)

    @property
    def idf_array(self) -> NDArray[np.float64]:
        """IDF values as numpy array (for pre-computation)."""
        idf = np.zeros(self.vocabulary_size, dtype=np.float64)
        for term, idx in self._vocab.items():
            df = np.array([self._df[term]], dtype=np.float64)
            idf[idx] = IDFFormula.compute(df, self.N)[0]
        return idf

    @property
    def term_doc_matrix(self) -> None:
        """Placeholder for compatibility (not used in this implementation)."""
        return None

    def id_to_idx(self, ids: list[str]) -> list[int]:
        """Convert document IDs to indices."""
        return [self._id_to_idx[doc_id] for doc_id in ids if doc_id in self._id_to_idx]


# =============================================================================
# BM25 Scorer
# =============================================================================


class BM25:
    """
    BM25 Scorer matching Pyserini/Lucene behavior.

    The core scoring logic is in score_document() which combines:
    - IDF computation (IDFFormula)
    - TF saturation (TFFormula)
    - Length normalization (LengthNorm)
    - Query term weighting (QueryWeighting)
    - Score aggregation (ScoreAggregation)

    All components are EVOLUTION TARGETs that AlphaEvolve can modify.
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float | None = None,
        b: float | None = None,
        k3: float | None = None,
    ):
        """
        Initialize BM25 scorer.

        Args:
            corpus: Pre-processed Corpus instance
            k1: TF saturation parameter (default: 0.9)
            b: Length normalization parameter (default: 0.4)
            k3: Query TF saturation parameter (default: 8.0)
        """
        self.corpus = corpus
        self.k1 = k1 if k1 is not None else Parameters.k1
        self.b = b if b is not None else Parameters.b
        self.k3 = k3 if k3 is not None else Parameters.k3

        # Pre-compute IDF for all terms
        self._idf_cache: dict[str, float] = {}
        for term in corpus._vocab:
            df = np.array([corpus.get_df(term)], dtype=np.float64)
            idf = IDFFormula.compute(df, corpus.N)
            self._idf_cache[term] = float(idf[0])

    def score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """
        Score a single document - EVOLUTION TARGET.

        This is the core BM25 scoring function. It combines:
        1. IDF for each query term
        2. TF saturation for document term frequency
        3. Length normalization based on document length
        4. Query term frequency (qtf) - terms appearing multiple times contribute more

        Args:
            query_terms: Query terms (may contain duplicates for qtf weighting)
            doc_idx: Index of document to score

        Returns:
            BM25 relevance score
        """
        # Get document length and compute normalization
        doc_len = int(self.corpus.doc_lengths[doc_idx])
        norm = LengthNorm.compute(doc_len, self.corpus.avgdl, self.b)

        # ===== EVOLVE THIS SCORING LOGIC =====
        score = 0.0
        for term in query_terms:
            # Get IDF
            idf = self._idf_cache.get(term, 0.0)
            if idf == 0:
                continue

            # Get TF
            tf = self.corpus.get_tf(doc_idx, term)
            if tf == 0:
                continue

            # Compute term contribution: IDF × saturated_TF
            tf_score = TFFormula.compute(tf, self.k1, norm)
            score += idf * tf_score

        return score

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Rank all documents by relevance to query.

        Args:
            query: Tokenized query (list of terms, may contain duplicates)
            top_k: Optional limit on results

        Returns:
            Tuple of (sorted_indices, sorted_scores) in descending order
        """
        # Pyserini/Lucene uses query term frequency (qtf) - pass full query with duplicates
        # This naturally multiplies each term's contribution by its frequency in the query
        # Score all documents
        scores = np.array(
            [self.score_document(query, i) for i in range(self.corpus.N)], dtype=np.float64
        )

        # Sort by score descending
        sorted_indices = np.argsort(-scores).astype(np.int64)
        sorted_scores = scores[sorted_indices]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def score(self, query: list[str], doc_idx: int) -> float:
        """Score a single document (convenience method)."""
        # Use full query to include query term frequency (matches Pyserini)
        return self.score_document(query, doc_idx)
