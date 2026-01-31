"""
Freeform Ranking Seed Program - Maximum Structural Freedom for AlphaEvolve.

This seed program gives AlphaEvolve maximum freedom to explore novel ranking
approaches beyond traditional BM25. It provides a flexible multi-signal
architecture where almost everything can be evolved.

=============================================================================
DESIGN PHILOSOPHY:
=============================================================================

This is NOT a BM25 implementation. It's a flexible framework where:
- Multiple scoring signals can be computed
- Signals can be combined in arbitrary ways
- New signals can be added
- The entire scoring strategy can be restructured
- Even the representation of documents/queries can be evolved

The goal is to see if AlphaEvolve can discover something fundamentally different
from BM25 that works better for the target domain.

=============================================================================
EVOLUTION TARGETS (almost everything!):
=============================================================================

1. GLOBAL CONFIG (Config) - All parameters and settings
2. FEATURE EXTRACTORS (FeatureExtractors) - Extract signals from query/doc pairs
3. SIGNAL DEFINITIONS (Signals) - Define what signals to compute
4. SIGNAL COMBINER (SignalCombiner) - How to combine signals
5. DOCUMENT REPRESENTATION (DocumentRepr) - How to represent documents
6. QUERY REPRESENTATION (QueryRepr) - How to represent queries
7. SCORING ENGINE (ScoringEngine) - The main scoring logic
8. MAIN KERNEL (score_document) - Entry point for scoring

=============================================================================

Run evaluation with:
    uv run python evaluator_bright.py

For AlphaEvolve:
    uv run openevolve-run src/ranking_evolved/bm25_freeform.py evaluator_bright.py --config openevolve_config.yaml
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from ranking_evolved.bm25 import ENGLISH_STOPWORDS, LUCENE_STOPWORDS

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# EVOLUTION TARGET 1: Global Configuration
# =============================================================================


class Config:
    """
    Global configuration - EVOLUTION TARGET.

    Contains ALL tunable parameters. AlphaEvolve can add new parameters,
    remove unused ones, or completely restructure the configuration.
    """

    # ===== EVOLVE THESE VALUES =====

    # Traditional BM25 parameters (may or may not be used)
    # Defaults match Pyserini/Lucene for equivalence
    k1: float = 0.9  # TF saturation (Pyserini default)
    b: float = 0.4  # Length normalization (Pyserini default)
    k3: float = 8.0

    # Signal weights (for multi-signal combination)
    weight_lexical: float = 1.0  # Weight for lexical matching signal
    weight_coverage: float = 0.0  # Weight for query coverage signal
    weight_density: float = 0.0  # Weight for term density signal
    weight_position: float = 0.0  # Weight for term position signal
    weight_length: float = 0.0  # Weight for document length signal
    weight_rarity: float = 0.0  # Weight for rare term bonus
    weight_custom: float = 0.0  # Weight for custom signal

    # Combination strategy
    combination_mode: str = "linear"  # Options: linear, multiplicative, max, learned

    # Normalization
    normalize_signals: bool = False

    # Bounds and constraints
    max_score: float = 100.0
    min_score: float = 0.0
    epsilon: float = 1e-9

    # Feature extraction settings
    use_positions: bool = False  # Track term positions (expensive)
    use_bigrams: bool = False  # Include bigram matching
    use_field_weights: bool = False  # Weight different fields differently

    # ===== ADD NEW PARAMETERS HERE =====


# =============================================================================
# EVOLUTION TARGET 2: Feature Extractors
# =============================================================================


class FeatureExtractors:
    """
    Low-level feature extraction functions - EVOLUTION TARGET.

    These extract raw features from documents and queries that can be
    used by the signal computation functions.
    """

    @staticmethod
    def term_frequency(doc_tf: Counter[str], term: str) -> float:
        """Get term frequency in document."""
        return float(doc_tf.get(term, 0))

    @staticmethod
    def document_frequency(corpus_df: Counter[str], term: str) -> float:
        """Get document frequency of term in corpus."""
        return float(corpus_df.get(term, 1))

    @staticmethod
    def inverse_document_frequency(df: float, N: int) -> float:
        """Compute IDF from document frequency."""
        # ===== EVOLVE THIS FORMULA =====
        # Lucene/Pyserini IDF (always non-negative)
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    @staticmethod
    def term_density(tf: float, doc_length: float) -> float:
        """Compute term density (tf / doc_length)."""
        return tf / (doc_length + Config.epsilon)

    @staticmethod
    def relative_length(doc_length: float, avg_length: float) -> float:
        """Compute relative document length."""
        return doc_length / (avg_length + Config.epsilon)

    @staticmethod
    def query_coverage(matched_terms: int, total_terms: int) -> float:
        """Compute what fraction of query terms appear in document."""
        if total_terms <= 0:
            return 0.0
        return matched_terms / total_terms

    @staticmethod
    def rarity_score(idf: float, threshold: float = 3.0) -> float:
        """Score based on term rarity."""
        return max(0.0, idf - threshold)

    # ===== ADD NEW FEATURE EXTRACTORS HERE =====

    @staticmethod
    def term_burstiness(tf: float, doc_length: float, avg_tf: float) -> float:
        """
        Measure if term is "bursty" (appears more than expected).
        High burstiness might indicate topical relevance.
        """
        expected = avg_tf * (doc_length / 100.0)  # Simple expectation
        return tf / (expected + Config.epsilon) - 1.0 if expected > 0 else 0.0


# =============================================================================
# EVOLUTION TARGET 3: Signal Definitions
# =============================================================================


class Signals:
    """
    Compute individual scoring signals - EVOLUTION TARGET.

    Each signal captures a different aspect of relevance.
    AlphaEvolve can modify signals, add new ones, or remove unused ones.
    """

    @staticmethod
    def lexical_signal(
        query_terms: list[str],
        doc_tf: Counter[str],
        corpus_df: Counter[str],
        N: int,
        doc_length: float,
        avg_length: float,
    ) -> float:
        """
        Classic lexical matching signal (BM25-like) - EVOLUTION TARGET.

        This is the core relevance signal based on term matching.
        """
        # ===== EVOLVE THIS SIGNAL =====

        k1 = Config.k1
        b = Config.b

        score = 0.0
        for term in query_terms:
            tf = FeatureExtractors.term_frequency(doc_tf, term)
            if tf <= 0:
                continue

            df = FeatureExtractors.document_frequency(corpus_df, term)
            idf = FeatureExtractors.inverse_document_frequency(df, N)

            # Length normalization (Lucene formula, matches pyserini exactly)
            norm = 1.0 - b + b * (doc_length / avg_length) if avg_length > 0 else 1.0

            # TF saturation (Lucene formula: no k1+1 multiplier, matches pyserini exactly)
            tf_component = tf / (tf + k1 * norm) if (tf + k1 * norm) > 0 else 0.0

            score += idf * tf_component

        return score

    @staticmethod
    def coverage_signal(
        query_terms: list[str],
        doc_tf: Counter[str],
    ) -> float:
        """
        Query coverage signal - EVOLUTION TARGET.

        Rewards documents that match more query terms.
        """
        # ===== EVOLVE THIS SIGNAL =====

        if not query_terms:
            return 0.0

        matched = sum(1 for term in query_terms if doc_tf.get(term, 0) > 0)
        coverage = matched / len(query_terms)

        # Non-linear scaling (rewards full coverage)
        return coverage**2

    @staticmethod
    def density_signal(
        query_terms: list[str],
        doc_tf: Counter[str],
        doc_length: float,
    ) -> float:
        """
        Term density signal - EVOLUTION TARGET.

        Rewards documents where query terms are dense.
        """
        # ===== EVOLVE THIS SIGNAL =====

        if doc_length <= 0:
            return 0.0

        total_tf = sum(doc_tf.get(term, 0) for term in query_terms)
        density = total_tf / doc_length

        # Saturate to prevent extreme values
        return math.log(1.0 + density * 10)

    @staticmethod
    def length_signal(
        doc_length: float,
        avg_length: float,
    ) -> float:
        """
        Document length signal - EVOLUTION TARGET.

        Can be used to prefer shorter or longer documents.
        """
        # ===== EVOLVE THIS SIGNAL =====

        ratio = doc_length / (avg_length + Config.epsilon)

        # Penalize very long documents
        if ratio > 2.0:
            return -math.log(ratio)

        return 0.0

    @staticmethod
    def rarity_signal(
        query_terms: list[str],
        doc_tf: Counter[str],
        corpus_df: Counter[str],
        N: int,
    ) -> float:
        """
        Rare term bonus signal - EVOLUTION TARGET.

        Extra reward for matching very rare terms.
        """
        # ===== EVOLVE THIS SIGNAL =====

        score = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf <= 0:
                continue

            df = corpus_df.get(term, 1)
            idf = FeatureExtractors.inverse_document_frequency(df, N)

            # Only boost very rare terms
            if idf > 4.0:
                score += (idf - 4.0) * math.log(1 + tf)

        return score

    @staticmethod
    def custom_signal(
        query_terms: list[str],
        doc_tf: Counter[str],
        corpus_df: Counter[str],
        N: int,
        doc_length: float,
        avg_length: float,
    ) -> float:
        """
        Custom signal - PRIMARY EVOLUTION TARGET.

        This is a blank slate for AlphaEvolve to create entirely new
        relevance signals that don't fit existing categories.
        """
        # ===== EVOLVE: CREATE SOMETHING NEW HERE =====

        # Placeholder: return 0 (no contribution by default)
        return 0.0

    # ===== ADD MORE SIGNALS HERE =====


# =============================================================================
# EVOLUTION TARGET 4: Signal Combiner
# =============================================================================


class SignalCombiner:
    """
    Combines multiple signals into a final score - EVOLUTION TARGET.

    AlphaEvolve can change the combination strategy entirely.
    """

    @staticmethod
    def combine(signals: dict[str, float]) -> float:
        """
        Combine signals into final score - EVOLUTION TARGET.

        Args:
            signals: Dictionary of signal_name -> signal_value

        Returns:
            Combined score
        """
        # ===== EVOLVE THIS COMBINATION STRATEGY =====

        mode = Config.combination_mode

        if mode == "linear":
            # Weighted linear combination
            return SignalCombiner._linear_combination(signals)

        elif mode == "multiplicative":
            # Product of (1 + signal) factors
            return SignalCombiner._multiplicative_combination(signals)

        elif mode == "max":
            # Maximum signal value
            return SignalCombiner._max_combination(signals)

        elif mode == "learned":
            # Non-linear learned combination
            return SignalCombiner._learned_combination(signals)

        else:
            return SignalCombiner._linear_combination(signals)

    @staticmethod
    def _linear_combination(signals: dict[str, float]) -> float:
        """Weighted linear sum of signals."""
        weights = {
            "lexical": Config.weight_lexical,
            "coverage": Config.weight_coverage,
            "density": Config.weight_density,
            "length": Config.weight_length,
            "rarity": Config.weight_rarity,
            "custom": Config.weight_custom,
        }

        score = 0.0
        for name, value in signals.items():
            weight = weights.get(name, 0.0)
            score += weight * value

        return score

    @staticmethod
    def _multiplicative_combination(signals: dict[str, float]) -> float:
        """Product of (1 + weighted_signal) factors."""
        weights = {
            "lexical": Config.weight_lexical,
            "coverage": Config.weight_coverage,
            "density": Config.weight_density,
            "length": Config.weight_length,
            "rarity": Config.weight_rarity,
            "custom": Config.weight_custom,
        }

        product = 1.0
        for name, value in signals.items():
            weight = weights.get(name, 0.0)
            if weight > 0:
                product *= 1.0 + weight * value

        return product - 1.0  # Subtract 1 to start from 0

    @staticmethod
    def _max_combination(signals: dict[str, float]) -> float:
        """Take maximum weighted signal."""
        weights = {
            "lexical": Config.weight_lexical,
            "coverage": Config.weight_coverage,
            "density": Config.weight_density,
            "length": Config.weight_length,
            "rarity": Config.weight_rarity,
            "custom": Config.weight_custom,
        }

        weighted_signals = [weights.get(name, 0.0) * value for name, value in signals.items()]

        return max(weighted_signals) if weighted_signals else 0.0

    @staticmethod
    def _learned_combination(signals: dict[str, float]) -> float:
        """
        Non-linear combination - EVOLUTION TARGET.

        This is where AlphaEvolve can discover complex interactions
        between signals.
        """
        # ===== EVOLVE THIS FUNCTION =====

        lex = signals.get("lexical", 0.0)
        cov = signals.get("coverage", 0.0)
        _den = signals.get("density", 0.0)  # noqa: F841 - reserved for evolution
        rar = signals.get("rarity", 0.0)

        # Example: non-linear interaction
        base = lex * Config.weight_lexical

        # Boost if coverage is high
        if cov > 0.8:
            base *= 1.2

        # Add rarity bonus
        base += rar * Config.weight_rarity

        return base


# =============================================================================
# EVOLUTION TARGET 5: Document Representation
# =============================================================================


class DocumentRepr:
    """
    Document representation - EVOLUTION TARGET.

    How we represent a document for scoring. AlphaEvolve can add new
    fields or change how documents are represented.
    """

    def __init__(self, term_frequencies: Counter[str], length: float):
        # Core representation
        self.term_frequencies = term_frequencies
        self.length = length

    # ===== EVOLVE: ADD NEW REPRESENTATIONS HERE =====
    # Examples:
    # - bigram_frequencies: Counter[str] = field(default_factory=Counter)
    # - term_positions: dict[str, list[int]] = field(default_factory=dict)
    # - embedding: NDArray[np.float64] = None

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> DocumentRepr:
        """Create representation from tokens."""
        return cls(
            term_frequencies=Counter(tokens),
            length=float(len(tokens)),
        )


# =============================================================================
# EVOLUTION TARGET 6: Query Representation
# =============================================================================


class QueryRepr:
    """
    Query representation - EVOLUTION TARGET.

    How we represent a query for scoring. AlphaEvolve can add new
    fields or change how queries are processed.
    """

    def __init__(self, terms: list[str], term_weights: dict[str, float]):
        # Core representation
        self.terms = terms
        self.term_weights = term_weights

    # ===== EVOLVE: ADD NEW REPRESENTATIONS HERE =====
    # Examples:
    # - expanded_terms: list[str] = field(default_factory=list)
    # - intent: str = "unknown"
    # - embedding: NDArray[np.float64] = None

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> QueryRepr:
        """Create representation from tokens."""
        # ===== EVOLVE THIS PROCESSING =====

        # Pyserini-style: keep all tokens (including duplicates) for query term frequency
        # This matches how pyserini handles queries - each occurrence contributes
        all_terms = tokens

        # Uniform weights by default (each occurrence contributes equally)
        term_weights = {term: 1.0 for term in all_terms}

        return cls(
            terms=all_terms,
            term_weights=term_weights,
        )


# =============================================================================
# EVOLUTION TARGET 7: Scoring Engine
# =============================================================================


class ScoringEngine:
    """
    Main scoring engine - EVOLUTION TARGET.

    This orchestrates the entire scoring process. AlphaEvolve can
    restructure the pipeline, add preprocessing, postprocessing, etc.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: QueryRepr, doc_idx: int) -> float:
        """
        Score a document for a query - EVOLUTION TARGET.

        Args:
            query: Query representation
            doc_idx: Document index

        Returns:
            Relevance score
        """
        # ===== EVOLVE THIS ENTIRE METHOD =====

        # Step 1: Get document data
        doc_tf = self.corpus.term_frequency[doc_idx]
        doc_length = self.corpus.document_length[doc_idx]
        avg_length = self.corpus.average_document_length
        N = self.corpus.document_count

        # Step 2: Compute signals
        signals = {}

        # Lexical signal (always computed)
        signals["lexical"] = Signals.lexical_signal(
            query.terms,
            doc_tf,
            self.corpus.document_frequency,
            N,
            doc_length,
            avg_length,
        )

        # Optional signals (controlled by weights)
        if Config.weight_coverage > 0:
            signals["coverage"] = Signals.coverage_signal(query.terms, doc_tf)

        if Config.weight_density > 0:
            signals["density"] = Signals.density_signal(
                query.terms,
                doc_tf,
                doc_length,
            )

        if Config.weight_length > 0:
            signals["length"] = Signals.length_signal(doc_length, avg_length)

        if Config.weight_rarity > 0:
            signals["rarity"] = Signals.rarity_signal(
                query.terms,
                doc_tf,
                self.corpus.document_frequency,
                N,
            )

        if Config.weight_custom > 0:
            signals["custom"] = Signals.custom_signal(
                query.terms,
                doc_tf,
                self.corpus.document_frequency,
                N,
                doc_length,
                avg_length,
            )

        # Step 3: Combine signals
        score = SignalCombiner.combine(signals)

        # Step 4: Apply bounds
        score = max(Config.min_score, min(score, Config.max_score))

        return score


# =============================================================================
# EVOLUTION TARGET 8: Main Scoring Function
# =============================================================================


def score_document(
    query: list[str],
    doc_idx: int,
    corpus: Corpus,
) -> float:
    """
    Main entry point for document scoring - EVOLUTION TARGET.

    This is the top-level function that can be completely restructured.

    Args:
        query: Raw query tokens
        doc_idx: Document index to score
        corpus: Corpus with statistics

    Returns:
        Relevance score
    """
    # ===== EVOLVE THIS ENTIRE FUNCTION =====

    if not query:
        return 0.0

    # Create query representation
    query_repr = QueryRepr.from_tokens(query)

    if not query_repr.terms:
        return 0.0

    # Use scoring engine
    engine = ScoringEngine(corpus)

    return engine.score(query_repr, doc_idx)


# =============================================================================
# Tokenization
# =============================================================================
# LUCENE_STOPWORDS and ENGLISH_STOPWORDS are imported from ranking_evolved.bm25


def tokenize(text: str) -> list[str]:
    """Simple tokenization."""
    return re.findall(r"\w+", text.lower())


# =============================================================================
# Lucene Tokenizer (for Pyserini compatibility)
# =============================================================================


def _get_pyserini_tokenizer() -> Callable[[str], list[str]] | None:
    """Try to get Pyserini's Lucene tokenizer."""
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        analyzer = Analyzer(get_lucene_analyzer())
        return analyzer.analyze
    except Exception:
        return None


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _fallback_tokenize(text: str) -> list[str]:
    """Simple tokenizer (lowercase + alphanumeric only)."""
    return [t for t in text.lower().split() if t.isalnum()]


class LuceneTokenizer:
    """
    Lucene-compatible tokenizer.

    Uses Pyserini's actual Lucene DefaultEnglishAnalyzer when available,
    which applies:
    - Tokenization on non-letter boundaries
    - Lowercasing
    - Porter stemming
    - English stopword removal

    Falls back to a simple approximation if Pyserini/Java unavailable.
    """

    def __init__(self):
        self._pyserini_tokenize = _get_pyserini_tokenizer()
        if self._pyserini_tokenize is None:
            import warnings

            warnings.warn(
                "Pyserini not available. Using fallback tokenizer. "
                "For exact Pyserini reproduction, install pyserini and Java 21.",
                stacklevel=2,
            )

    def __call__(self, text: str) -> list[str]:
        """Tokenize text using Lucene analyzer."""
        if self._pyserini_tokenize is not None:
            return self._pyserini_tokenize(text)

        # Fallback: simple tokenization without stemming
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return [t for t in tokens if t not in LUCENE_STOPWORDS and len(t) > 1]


# =============================================================================
# Corpus
# =============================================================================


class Corpus:
    """Corpus with pre-computed statistics."""

    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        self.documents = documents
        self.document_count = len(documents)
        self.ids = ids

    def __len__(self) -> int:
        return self.document_count

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> Corpus:
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)

    @cached_property
    def term_frequency(self) -> list[Counter[str]]:
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter[str]:
        return Counter(term for doc in self.documents for term in set(doc))

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        return {term: idx for idx, term in enumerate(self.document_frequency.keys())}

    @cached_property
    def document_length(self) -> NDArray[np.float64]:
        return np.array([len(doc) for doc in self.documents], dtype=np.float64)

    @cached_property
    def average_document_length(self) -> float:
        # Match pyserini: default to 1.0 if empty corpus
        return float(np.mean(self.document_length)) if self.document_count > 0 else 1.0

    @cached_property
    def map_id_to_idx(self) -> dict[str, int]:
        return {id_: idx for idx, id_ in enumerate(self.ids)} if self.ids else {}

    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self.map_id_to_idx[id_] for id_ in ids]

    @cached_property
    def vocabulary_size(self) -> int:
        """Number of unique terms in corpus (for evaluator compatibility)."""
        return len(self.vocabulary)

    @cached_property
    def idf_array(self) -> NDArray[np.float64]:
        """IDF values as numpy array (for evaluator compatibility)."""
        idf = np.zeros(self.vocabulary_size, dtype=np.float64)
        for term, idx in self.vocabulary.items():
            df = self.document_frequency[term]
            # Lucene IDF: log(1 + (N - df + 0.5) / (df + 0.5))
            idf[idx] = math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))
        return idf

    @property
    def term_doc_matrix(self) -> None:
        """Placeholder for evaluator compatibility."""
        return None


# =============================================================================
# BM25 Interface (for compatibility with evaluator)
# =============================================================================


class BM25:
    """BM25-compatible interface using freeform scoring."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: list[str], index: int) -> float:
        """Score a single document."""
        return score_document(query, index, self.corpus)

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank all documents by relevance."""
        scores = np.array(
            [self.score(query, idx) for idx in range(len(self.corpus))],
            dtype=np.float64,
        )

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def batch_rank(
        self,
        queries: list[list[str]],
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Interface classes
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "LUCENE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    # Evolution targets
    "Config",
    "FeatureExtractors",
    "Signals",
    "SignalCombiner",
    "DocumentRepr",
    "QueryRepr",
    "ScoringEngine",
    "score_document",
]
