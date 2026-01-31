"""
Expanded BM25 implementation for OpenEvolve optimization.

This file contains ALL evolution targets - every aspect of the BM25 system
that OpenEvolve can modify to discover better ranking formulas.

Evolution targets:
1. EvolvedParameters - k1, b, k3 parameters
2. EvolvedStopwords - stopword list and token filtering
3. EvolvedStemmer - stemming rules
4. EvolvedTokenizer - full tokenization pipeline
5. EvolvedIDF - IDF formula and bounds
6. EvolvedTF - TF saturation formula
7. EvolvedLengthNorm - document length normalization
8. EvolvedQueryWeighting - query term handling (unique/weighted/saturated)
9. EvolvedScoreAggregation - how term scores are combined
10. BM25.score_kernel() - main scoring function

Run with:
    export OPENAI_API_KEY="your-key"
    uv run openevolve-run src/ranking_evolved/bm25_evolved.py evaluator_bright.py --config openevolve_config.yaml
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


class EvolvedParameters:
    """
    Evolved BM25 parameters - PRIMARY OPENEVOLVE TARGET.

    These parameters control the behavior of BM25 scoring.
    Modify these values to discover optimal parameter combinations.

    Current best values from hyperparameter search:
        k1 = 0.9  (TF saturation - lower = more saturation)
        b = 0.4   (Length normalization - 0 = no normalization, 1 = full)
        k3 = 2.0  (Query TF saturation for saturated mode)
    """

    # ===== EVOLVE THESE VALUES =====
    k1: float = 0.9
    b: float = 0.4
    k3: float = 2.0

    # IDF bounds
    max_idf: float = 8.0
    min_idf: float = 0.0

    # Score aggregation weight (for weighted sum mode)
    idf_weight: float = 1.0
    tf_weight: float = 1.0


# =============================================================================
# EVOLUTION TARGET 2: Stopwords
# =============================================================================


class EvolvedStopwords:
    """
    Evolved stopword handling - OPENEVOLVE TARGET.

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


class EvolvedStemmer:
    """
    Evolved stemming rules - OPENEVOLVE TARGET.

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


class EvolvedTokenizer:
    """
    Evolved tokenization pipeline - OPENEVOLVE TARGET.

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
        Tokenize text using evolved rules.

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
        stopwords = EvolvedStopwords.get_stopwords()

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
            token = EvolvedStemmer.stem(token)

            # Skip if empty after stemming
            if token:
                result.append(token)

        return result


# =============================================================================
# EVOLUTION TARGET 5: IDF Strategy
# =============================================================================


class EvolvedIDF:
    """
    Evolved IDF strategy - PRIMARY OPENEVOLVE TARGET.

    Current best formula:
        clip(log((N + 0.5) / (df + 0.5)), 0, max_idf)

    Modify this to discover improved IDF formulations.
    """

    def __init__(self, max_idf: float | None = None, min_idf: float | None = None):
        self.max_idf = max_idf if max_idf is not None else EvolvedParameters.max_idf
        self.min_idf = min_idf if min_idf is not None else EvolvedParameters.min_idf

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
        # Standard Robertson IDF with smoothing
        idf = np.log((N + 0.5) / (df + 0.5) + 1)
        return np.clip(idf, self.min_idf, self.max_idf)


# =============================================================================
# EVOLUTION TARGET 6: TF Strategy
# =============================================================================


class EvolvedTF:
    """
    Evolved TF strategy - PRIMARY OPENEVOLVE TARGET.

    Current best formula:
        tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm)
        tf_sat = tf / (tf + k1 + 0.5)
        result = log(1 + tf_raw * tf_sat)

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
        return np.log1p((tf * (k1 + 1)) / (tf + k1 * norm + 0.5))


# =============================================================================
# EVOLUTION TARGET 7: Length Normalization
# =============================================================================


class EvolvedLengthNorm:
    """
    Evolved document length normalization - OPENEVOLVE TARGET.

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


class EvolvedQueryWeighting:
    """
    Evolved query term handling - OPENEVOLVE TARGET.

    Controls how query terms are weighted in scoring.
    Options: unique (dedupe), sum_all (count repeats), saturated (k3 saturation)
    """

    # ===== EVOLVE THESE SETTINGS =====

    # Mode: "unique", "sum_all", "saturated"
    mode: str = "sum_all"

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
            mode = EvolvedQueryWeighting.mode

        if not query:
            return [], np.array([], dtype=np.float64)

        # ===== EVOLVE THIS LOGIC =====

        if mode == "unique":
            # Each unique term has weight 1
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


class EvolvedScoreAggregation:
    """
    Evolved score aggregation - OPENEVOLVE TARGET.

    Controls how individual term scores are combined into a final score.
    Standard BM25: sum of term scores
    """

    # ===== EVOLVE THESE SETTINGS =====

    # Aggregation mode: "sum", "weighted_sum", "max", "mean"
    mode: str = "weighted_sum"

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
            mode = EvolvedScoreAggregation.mode

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
    BM25 ranking with all evolved components.

    The score_kernel and rank methods use all evolution targets.

    Args:
        corpus: Pre-processed Corpus instance.
        k1: TF saturation parameter (default from EvolvedParameters).
        b: Length normalization parameter (default from EvolvedParameters).
        k3: Query TF saturation parameter (default from EvolvedParameters).
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float | None = None,
        b: float | None = None,
        k3: float | None = None,
    ):
        self.corpus = corpus

        # Use evolved parameters as defaults
        self.k1 = k1 if k1 is not None else EvolvedParameters.k1
        self.b = b if b is not None else EvolvedParameters.b
        self.k3 = k3 if k3 is not None else EvolvedParameters.k3

        # Initialize evolved components
        self._idf_strategy = EvolvedIDF()
        self._tf_strategy = EvolvedTF()

        # Compute IDF with evolved strategy
        df = corpus.df_array
        N = corpus.document_count
        self._idf_array = self._idf_strategy.compute(df, N)
        self._idf_dict = {
            term: float(self._idf_array[idx]) for term, idx in corpus.vocabulary.items()
        }

        # Compute document normalization with evolved strategy
        dl = corpus.document_length
        avgdl = corpus.average_document_length or 1.0
        self._doc_norm = EvolvedLengthNorm.compute(dl, avgdl, self.b)

    @staticmethod
    def score_kernel(
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
        k1: float,
        k3: float = 2.0,
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

        # Get query term weights using evolved weighting
        unique_terms, query_weights = EvolvedQueryWeighting.compute_weights(query, k3)

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
        denom = tf + k1 * norm
        tf_raw = (tf * (k1 + 1.0)) / np.maximum(denom, 1e-9)
        tf_sat = tf / (tf + k1 + 0.5)
        term_scores = idf_values * np.log1p(tf_raw * tf_sat)

        # Use evolved aggregation
        return EvolvedScoreAggregation.aggregate(term_scores, query_weights)

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

        Uses all evolved components for scoring.
        """
        if not query:
            n = len(self.corpus)
            return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

        # Get query term weights using evolved weighting
        unique_terms, query_weights = EvolvedQueryWeighting.compute_weights(query, self.k3)

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
        denom = tf_matrix + k1 * norm
        tf_raw = (tf_matrix * (k1 + 1.0)) / np.maximum(denom, 1e-9)
        tf_sat = tf_matrix / (tf_matrix + k1 + 0.5)
        term_scores = idf_values[:, np.newaxis] * np.log1p(tf_raw * tf_sat)

        # Apply query weights based on aggregation mode
        if EvolvedScoreAggregation.mode == "weighted_sum":
            term_scores = term_scores * query_weights[:, np.newaxis]

        # Aggregate scores
        if EvolvedScoreAggregation.mode == "max":
            scores = np.max(term_scores, axis=0)
        elif EvolvedScoreAggregation.mode == "mean":
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
# Evolved tokenize function (uses EvolvedTokenizer)
# =============================================================================


def evolved_tokenize(text: str) -> list[str]:
    """
    Tokenize text using evolved tokenization pipeline.

    This is an alternative to the standard tokenize function that uses
    all evolved tokenization components.
    """
    return EvolvedTokenizer.tokenize(text)


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
    # Evolved components
    "EvolvedParameters",
    "EvolvedStopwords",
    "EvolvedStemmer",
    "EvolvedTokenizer",
    "EvolvedIDF",
    "EvolvedTF",
    "EvolvedLengthNorm",
    "EvolvedQueryWeighting",
    "EvolvedScoreAggregation",
    "evolved_tokenize",
]
