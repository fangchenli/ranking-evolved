"""
Classic BM25 implementation for OpenEvolve optimization.

This file contains vanilla Robertson BM25 as a starting point for evolution.
All formulas are standard/classic BM25 without any modifications.

Parameters (Robertson Classic):
    k1 = 1.5 (TF saturation)
    b = 0.75 (length normalization)
    k3 = 8.0 (query TF saturation)

Evolution targets:
1. ClassicParameters - k1, b, k3 parameters
2. ClassicStopwords - stopword list and token filtering
3. ClassicStemmer - stemming rules
4. ClassicTokenizer - full tokenization pipeline
5. ClassicIDF - IDF formula
6. ClassicTF - TF saturation formula
7. ClassicLengthNorm - document length normalization
8. ClassicQueryWeighting - query term handling (unique/weighted/saturated)
9. ClassicScoreAggregation - how term scores are combined
10. BM25.score_kernel() - main scoring function

Run with:
    export OPENAI_API_KEY="your-key"
    uv run openevolve-run src/ranking_evolved/bm25_classic.py evaluator_bright.py --config openevolve_config.yaml
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ranking_evolved.bm25 import (
    LUCENE_STOPWORDS,
    Corpus,
    LuceneTokenizer,
    PorterStemmer,
    lucene_tokenize,
    tokenize,
)

# =============================================================================
# EVOLUTION TARGET 1: Parameters
# =============================================================================


class ClassicParameters:
    """
    Classic BM25 parameters - PRIMARY OPENEVOLVE TARGET.

    These parameters control the behavior of BM25 scoring.
    Modify these values to discover optimal parameter combinations.

    Robertson Classic defaults:
        k1 = 1.5  (TF saturation - higher = less saturation)
        b = 0.75  (Length normalization - 0 = no normalization, 1 = full)
        k3 = 8.0  (Query TF saturation)
    """

    # ===== EVOLVE THESE VALUES =====
    k1: float = 1.5
    b: float = 0.75
    k3: float = 8.0


# =============================================================================
# EVOLUTION TARGET 2: Stopwords
# =============================================================================


class ClassicStopwords:
    """
    Classic stopword handling - OPENEVOLVE TARGET.

    Modify the stopword list or add/remove words to improve ranking.
    Stopwords are filtered out during tokenization.
    """

    # ===== EVOLVE THIS SET =====
    # Start with Lucene English stopwords (33 words)
    words: frozenset[str] = LUCENE_STOPWORDS

    # Additional stopwords to add (domain-specific)
    extra_stopwords: frozenset[str] = frozenset()

    # Words to exclude from stopword filtering (keep even if in stopwords)
    keep_words: frozenset[str] = frozenset()

    @classmethod
    def get_stopwords(cls) -> frozenset[str]:
        """Get the effective stopword set."""
        base = cls.words | cls.extra_stopwords
        return base - cls.keep_words

    @classmethod
    def is_stopword(cls, token: str) -> bool:
        """Check if a token is a stopword."""
        return token in cls.get_stopwords()


# =============================================================================
# EVOLUTION TARGET 3: Stemmer
# =============================================================================


class ClassicStemmer:
    """
    Classic stemming rules - OPENEVOLVE TARGET.

    Modify stemming behavior to improve term matching.
    The default uses Porter Stemmer, but rules can be customized.
    """

    # ===== EVOLVE THESE SETTINGS =====
    enabled: bool = True  # Set to False to disable stemming entirely

    # Custom stem overrides (token -> stem)
    # These take precedence over Porter Stemmer
    custom_stems: dict[str, str] = {}

    # Suffixes to strip (applied before Porter)
    strip_suffixes: tuple[str, ...] = ()

    _stemmer: PorterStemmer | None = None

    @classmethod
    def get_stemmer(cls) -> PorterStemmer:
        """Get or create the Porter stemmer instance."""
        if cls._stemmer is None:
            cls._stemmer = PorterStemmer()
        return cls._stemmer

    @classmethod
    def stem(cls, token: str) -> str:
        """
        Stem a single token.

        Args:
            token: Input token (lowercase).

        Returns:
            Stemmed token.
        """
        if not cls.enabled:
            return token

        # Check custom overrides first
        if token in cls.custom_stems:
            return cls.custom_stems[token]

        # Apply suffix stripping
        for suffix in cls.strip_suffixes:
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break

        # Apply Porter stemmer
        return cls.get_stemmer().stem(token)


# =============================================================================
# EVOLUTION TARGET 4: Full Tokenizer
# =============================================================================


class ClassicTokenizer:
    """
    Classic tokenization pipeline - OPENEVOLVE TARGET.

    Full control over how text is converted to tokens.
    Modify any step: splitting, normalization, filtering, stemming.
    """

    # ===== EVOLVE THESE SETTINGS =====

    # Minimum token length (tokens shorter than this are filtered)
    min_token_length: int = 1

    # Maximum token length (tokens longer than this are truncated)
    max_token_length: int = 50

    # Whether to lowercase tokens
    lowercase: bool = True

    # Whether to filter numeric-only tokens
    filter_numbers: bool = False

    # Whether to keep hyphenated words as single tokens
    keep_hyphens: bool = False

    # Token split pattern (regex)
    # Default: split on non-alphanumeric
    split_pattern: str = r"[^a-z0-9]+"

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Tokenize text using classic rules.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        if not text:
            return []

        # ===== EVOLVE THIS PIPELINE =====

        # Step 1: Lowercase
        if cls.lowercase:
            text = text.lower()

        # Step 2: Handle hyphens
        if cls.keep_hyphens:
            text = text.replace("-", "_HYPHEN_")

        # Step 3: Split into tokens
        tokens = re.split(cls.split_pattern, text)

        if cls.keep_hyphens:
            tokens = [t.replace("_HYPHEN_", "-") for t in tokens]

        # Step 4: Filter and process
        result = []
        stopwords = ClassicStopwords.get_stopwords()

        for token in tokens:
            # Skip empty
            if not token:
                continue

            # Length filtering
            if len(token) < cls.min_token_length:
                continue
            if len(token) > cls.max_token_length:
                token = token[: cls.max_token_length]

            # Filter numbers
            if cls.filter_numbers and token.isdigit():
                continue

            # Stopword filtering
            if token in stopwords:
                continue

            # Stemming
            token = ClassicStemmer.stem(token)

            # Skip if empty after stemming
            if token:
                result.append(token)

        return result


# =============================================================================
# EVOLUTION TARGET 5: IDF Strategy
# =============================================================================


class ClassicIDF:
    """
    Classic IDF strategy - PRIMARY OPENEVOLVE TARGET.

    Robertson's classic IDF formula:
        log((N - df + 0.5) / (df + 0.5))

    Note: This can produce negative values for very common terms.
    Modify this to discover improved IDF formulations.
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        """
        Compute IDF values for terms.

        Args:
            df: Document frequency array for each term.
            N: Total number of documents in corpus.

        Returns:
            IDF values array.
        """
        # ===== EVOLVE THIS FORMULA =====
        # Classic Robertson IDF (can be negative for common terms)
        idf = np.log((N - df + 0.5) / (df + 0.5))
        return idf


# =============================================================================
# EVOLUTION TARGET 6: TF Strategy
# =============================================================================


class ClassicTF:
    """
    Classic TF strategy - PRIMARY OPENEVOLVE TARGET.

    Robertson's classic TF saturation formula:
        (tf * (k1 + 1)) / (tf + k1 * norm)

    Modify this to discover improved TF saturation curves.
    """

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute TF saturation values.

        Args:
            tf: Raw term frequency values.
            k1: Saturation parameter.
            norm: Document length normalization (1 - b + b * dl/avgdl).

        Returns:
            Saturated TF values.
        """
        # ===== EVOLVE THIS FORMULA =====
        # Classic BM25 TF saturation
        return (tf * (k1 + 1)) / (tf + k1 * norm)


# =============================================================================
# EVOLUTION TARGET 7: Length Normalization
# =============================================================================


class ClassicLengthNorm:
    """
    Classic document length normalization - OPENEVOLVE TARGET.

    Controls how document length affects scoring.
    Standard BM25: norm = 1 - b + b * (dl / avgdl)
    """

    @staticmethod
    def compute(
        dl: NDArray[np.float64],
        avgdl: float,
        b: float,
    ) -> NDArray[np.float64]:
        """
        Compute document length normalization.

        Args:
            dl: Document lengths array.
            avgdl: Average document length.
            b: Length normalization parameter.

        Returns:
            Normalization factors array.
        """
        # ===== EVOLVE THIS FORMULA =====
        # Standard BM25 length normalization
        return 1.0 - b + b * (dl / max(avgdl, 1.0))


# =============================================================================
# EVOLUTION TARGET 8: Query Term Weighting
# =============================================================================


class ClassicQueryWeighting:
    """
    Classic query term handling - OPENEVOLVE TARGET.

    Controls how query terms are weighted in scoring.
    Classic BM25 uses bag-of-words (unique terms only).
    """

    # ===== EVOLVE THESE SETTINGS =====

    # Mode: "unique", "sum_all", "saturated"
    mode: str = "unique"

    @staticmethod
    def compute_weights(
        query: list[str],
        k3: float,
        mode: str | None = None,
    ) -> tuple[list[str], NDArray[np.float64]]:
        """
        Compute query term weights.

        Args:
            query: List of query terms (may contain duplicates).
            k3: Saturation parameter for saturated mode.
            mode: Override mode (default uses class setting).

        Returns:
            Tuple of (unique_terms, weights).
        """
        if mode is None:
            mode = ClassicQueryWeighting.mode

        if not query:
            return [], np.array([], dtype=np.float64)

        # ===== EVOLVE THIS LOGIC =====

        if mode == "unique":
            # Each unique term has weight 1 (bag-of-words)
            unique_terms = list(dict.fromkeys(query))
            weights = np.ones(len(unique_terms), dtype=np.float64)

        elif mode == "sum_all":
            # Weight = term count in query
            term_counts = Counter(query)
            unique_terms = list(term_counts.keys())
            weights = np.array([term_counts[t] for t in unique_terms], dtype=np.float64)

        elif mode == "saturated":
            # Weight = ((k3 + 1) * qtf) / (k3 + qtf)
            term_counts = Counter(query)
            unique_terms = list(term_counts.keys())
            qtf = np.array([term_counts[t] for t in unique_terms], dtype=np.float64)
            weights = ((k3 + 1) * qtf) / (k3 + qtf)

        else:
            # Fallback to unique
            unique_terms = list(dict.fromkeys(query))
            weights = np.ones(len(unique_terms), dtype=np.float64)

        return unique_terms, weights


# =============================================================================
# EVOLUTION TARGET 9: Score Aggregation
# =============================================================================


class ClassicScoreAggregation:
    """
    Classic score aggregation - OPENEVOLVE TARGET.

    Controls how individual term scores are combined into a final score.
    Standard BM25: sum of term scores
    """

    # ===== EVOLVE THESE SETTINGS =====

    # Aggregation mode: "sum", "weighted_sum", "max", "mean"
    mode: str = "sum"

    @staticmethod
    def aggregate(
        term_scores: NDArray[np.float64],
        query_weights: NDArray[np.float64] | None = None,
        mode: str | None = None,
    ) -> float:
        """
        Aggregate term scores into a final document score.

        Args:
            term_scores: Array of per-term scores.
            query_weights: Optional query term weights.
            mode: Override mode (default uses class setting).

        Returns:
            Final aggregated score.
        """
        if mode is None:
            mode = ClassicScoreAggregation.mode

        if len(term_scores) == 0:
            return 0.0

        # ===== EVOLVE THIS LOGIC =====

        if mode == "sum":
            return float(np.sum(term_scores))

        elif mode == "weighted_sum":
            if query_weights is not None:
                return float(np.sum(term_scores * query_weights))
            return float(np.sum(term_scores))

        elif mode == "max":
            return float(np.max(term_scores))

        elif mode == "mean":
            return float(np.mean(term_scores))

        else:
            return float(np.sum(term_scores))


# =============================================================================
# EVOLUTION TARGET 10: BM25 Scorer
# =============================================================================


class BM25:
    """
    BM25 ranking with all classic components.

    The score_kernel and rank methods use all evolution targets.

    Args:
        corpus: Pre-processed Corpus instance.
        k1: TF saturation parameter (default from ClassicParameters).
        b: Length normalization parameter (default from ClassicParameters).
        k3: Query TF saturation parameter (default from ClassicParameters).
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float | None = None,
        b: float | None = None,
        k3: float | None = None,
    ):
        self.corpus = corpus

        # Use classic parameters as defaults
        self.k1 = k1 if k1 is not None else ClassicParameters.k1
        self.b = b if b is not None else ClassicParameters.b
        self.k3 = k3 if k3 is not None else ClassicParameters.k3

        # Initialize classic components
        self._idf_strategy = ClassicIDF()
        self._tf_strategy = ClassicTF()

        # Compute IDF with classic strategy
        df = corpus.df_array
        N = corpus.document_count
        self._idf_array = self._idf_strategy.compute(df, N)
        self._idf_dict = {
            term: float(self._idf_array[idx]) for term, idx in corpus.vocabulary.items()
        }

        # Compute document normalization with classic strategy
        dl = corpus.document_length
        avgdl = corpus.average_document_length or 1.0
        self._doc_norm = ClassicLengthNorm.compute(dl, avgdl, self.b)

    @staticmethod
    def score_kernel(
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
        k1: float,
        k3: float = 8.0,
    ) -> float:
        """
        Compute BM25 score for a single document - PRIMARY EVOLUTION TARGET.

        Args:
            query: List of query terms (tokenized).
            norm: Document length normalization factor.
            frequencies: Term frequency counter for the document.
            idf: IDF values dictionary (term -> idf).
            k1: TF saturation parameter.
            k3: Query TF saturation parameter.

        Returns:
            BM25 relevance score (float).
        """
        if not query:
            return 0.0

        # Get query term weights using classic weighting
        unique_terms, query_weights = ClassicQueryWeighting.compute_weights(query, k3)

        if not unique_terms:
            return 0.0

        # Get term frequencies
        tf = np.array(
            [frequencies.get(term, 0) for term in unique_terms],
            dtype=np.float64,
        )

        if np.all(tf == 0):
            return 0.0

        # Get IDF values
        idf_values = np.array(
            [idf.get(term, 0.0) for term in unique_terms],
            dtype=np.float64,
        )

        # ===== EVOLVE THIS SCORING FORMULA =====
        # Classic BM25: IDF * TF_saturation
        denom = tf + k1 * norm
        tf_raw = (tf * (k1 + 1.0)) / np.maximum(denom, 1e-9)
        term_scores = idf_values * tf_raw

        # Use classic aggregation
        return ClassicScoreAggregation.aggregate(term_scores, query_weights)

    def score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        return self.score_kernel(
            query,
            float(self._doc_norm[index]),
            self.corpus.term_frequency[index],
            self._idf_dict,
            self.k1,
            self.k3,
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rank all documents by relevance to query (vectorized for speed).

        Uses all classic components for scoring.
        """
        if not query:
            n = len(self.corpus)
            return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

        # Get query term weights using classic weighting
        unique_terms, query_weights = ClassicQueryWeighting.compute_weights(query, self.k3)

        # Get vocabulary indices for query terms
        vocab = self.corpus.vocabulary
        valid_indices = []
        valid_weights = []
        for i, term in enumerate(unique_terms):
            if term in vocab:
                valid_indices.append(vocab[term])
                valid_weights.append(query_weights[i])

        if not valid_indices:
            n = len(self.corpus)
            return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

        term_indices = valid_indices
        query_weights = np.array(valid_weights, dtype=np.float64)

        # Get IDF values for query terms
        idf_values = self._idf_array[term_indices]

        # Get TF matrix slice for query terms (sparse -> dense for these rows)
        tf_matrix = self.corpus.term_doc_matrix[term_indices, :].toarray()

        # Vectorized BM25 scoring
        k1 = self.k1
        norm = self._doc_norm

        # ===== EVOLVE THIS SCORING FORMULA =====
        # Classic BM25: IDF * TF_saturation
        denom = tf_matrix + k1 * norm
        tf_raw = (tf_matrix * (k1 + 1.0)) / np.maximum(denom, 1e-9)
        term_scores = idf_values[:, np.newaxis] * tf_raw

        # Apply query weights based on aggregation mode
        if ClassicScoreAggregation.mode == "weighted_sum":
            term_scores = term_scores * query_weights[:, np.newaxis]

        # Aggregate scores
        if ClassicScoreAggregation.mode == "max":
            scores = np.max(term_scores, axis=0)
        elif ClassicScoreAggregation.mode == "mean":
            scores = np.mean(term_scores, axis=0)
        else:  # sum or weighted_sum
            scores = np.sum(term_scores, axis=0)

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores


# =============================================================================
# Classic tokenize function (uses ClassicTokenizer)
# =============================================================================


def classic_tokenize(text: str) -> list[str]:
    """
    Tokenize text using classic tokenization pipeline.

    This is an alternative to the standard tokenize function that uses
    all classic tokenization components.
    """
    return ClassicTokenizer.tokenize(text)


# =============================================================================
# Module exports (required by evaluator_bright.py)
# =============================================================================

__all__ = [
    # Core classes
    "BM25",
    "Corpus",
    # Tokenization (standard imports)
    "tokenize",
    "LuceneTokenizer",
    "lucene_tokenize",
    "PorterStemmer",
    "LUCENE_STOPWORDS",
    # Classic components
    "ClassicParameters",
    "ClassicStopwords",
    "ClassicStemmer",
    "ClassicTokenizer",
    "ClassicIDF",
    "ClassicTF",
    "ClassicLengthNorm",
    "ClassicQueryWeighting",
    "ClassicScoreAggregation",
    "classic_tokenize",
]
