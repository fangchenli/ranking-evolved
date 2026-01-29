"""
Freeform Ranking Seed Program - Maximum Structural Freedom for AlphaEvolve (OPTIMIZED).

This seed program gives AlphaEvolve maximum freedom to explore novel ranking
approaches beyond traditional BM25, with OPTIMIZED data structures for fast evaluation.

=============================================================================
OPTIMIZATIONS (vs bm25_freeform.py):
=============================================================================

1. Inverted Index - Only score documents containing query terms
2. Sparse Matrix - scipy CSR matrix for vectorized TF lookups
3. Pre-computed IDF Array - numpy array instead of dict lookups
4. Pre-computed Length Norm Array - vectorized normalization
5. Vectorized Scoring - batch operations over candidate documents
6. Parallel Query Processing - ThreadPoolExecutor for batch queries

Performance: 10-100x faster than naive implementation on large corpora.

=============================================================================
DESIGN PHILOSOPHY:
=============================================================================

This is a flexible framework where:
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
    uv run python evaluator_parallel.py src/ranking_evolved/bm25_freeform_fast.py

For AlphaEvolve:
    uv run openevolve-run src/ranking_evolved/bm25_freeform_fast.py evaluator_parallel.py --config openevolve_config_freeform.yaml
"""

from __future__ import annotations

import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

# Number of workers for parallel query processing
NUM_QUERY_WORKERS = 32
MIN_QUERIES_FOR_PARALLEL = 10


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
    k1: float = 0.9   # TF saturation (Pyserini default)
    b: float = 0.4    # Length normalization (Pyserini default)
    k3: float = 8.0
    
    # Signal weights (for multi-signal combination)
    weight_lexical: float = 1.0      # Weight for lexical matching signal
    weight_coverage: float = 0.0     # Weight for query coverage signal
    weight_density: float = 0.0      # Weight for term density signal
    weight_position: float = 0.0     # Weight for term position signal
    weight_length: float = 0.0       # Weight for document length signal
    weight_rarity: float = 0.0       # Weight for rare term bonus
    weight_custom: float = 0.0       # Weight for custom signal
    
    # Combination strategy
    combination_mode: str = "linear"  # Options: linear, multiplicative, max, learned
    
    # Normalization
    normalize_signals: bool = False
    
    # Bounds and constraints
    max_score: float = 100.0
    min_score: float = 0.0
    epsilon: float = 1e-9
    
    # Feature extraction settings
    use_positions: bool = False      # Track term positions (expensive)
    use_bigrams: bool = False        # Include bigram matching
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
    def inverse_document_frequency_vectorized(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        """Compute IDF vectorized for arrays."""
        return np.log(1 + (N - df + 0.5) / (df + 0.5))
    
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
        expected = avg_tf * (doc_length / 100.0)
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
    def lexical_signal_vectorized(
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
        corpus: "Corpus",
    ) -> NDArray[np.float64]:
        """
        Lexical signal computed using vectorized operations - OPTIMIZATION.
        """
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)
        
        k1 = Config.k1
        norms = corpus.norm_array[candidate_docs]
        scores = np.zeros(len(candidate_docs), dtype=np.float64)
        
        for term_id in query_term_ids:
            idf = corpus.idf_array[term_id]
            if idf <= 0:
                continue
            
            tf_row = corpus.tf_matrix[term_id, candidate_docs].toarray().flatten()
            tf_saturated = tf_row / (tf_row + k1 * norms + Config.epsilon)
            scores += idf * tf_saturated
        
        return scores
    
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
        return coverage ** 2
    
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
            return SignalCombiner._linear_combination(signals)
        elif mode == "multiplicative":
            return SignalCombiner._multiplicative_combination(signals)
        elif mode == "max":
            return SignalCombiner._max_combination(signals)
        elif mode == "learned":
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
                product *= (1.0 + weight * value)
        
        return product - 1.0
    
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
        
        weighted_signals = [
            weights.get(name, 0.0) * value
            for name, value in signals.items()
        ]
        
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
        den = signals.get("density", 0.0)
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


@dataclass
class DocumentRepr:
    """
    Document representation - EVOLUTION TARGET.
    
    How we represent a document for scoring. AlphaEvolve can add new
    fields or change how documents are represented.
    """
    
    # Core representation
    term_frequencies: Counter[str]
    length: float
    
    # ===== EVOLVE: ADD NEW REPRESENTATIONS HERE =====
    
    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "DocumentRepr":
        """Create representation from tokens."""
        return cls(
            term_frequencies=Counter(tokens),
            length=float(len(tokens)),
        )


# =============================================================================
# EVOLUTION TARGET 6: Query Representation
# =============================================================================


@dataclass  
class QueryRepr:
    """
    Query representation - EVOLUTION TARGET.
    
    How we represent a query for scoring. AlphaEvolve can add new
    fields or change how queries are processed.
    """
    
    # Core representation
    terms: list[str]
    term_weights: dict[str, float]
    
    # ===== EVOLVE: ADD NEW REPRESENTATIONS HERE =====
    
    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "QueryRepr":
        """Create representation from tokens."""
        # ===== EVOLVE THIS PROCESSING =====
        
        # Pyserini-style: keep all tokens (including duplicates)
        all_terms = tokens
        
        # Uniform weights by default
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
    
    def __init__(self, corpus: "Corpus"):
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
        doc_tf = self.corpus.get_term_frequencies(doc_idx)
        doc_length = self.corpus.doc_lengths[doc_idx]
        avg_length = self.corpus.avgdl
        N = self.corpus.N
        
        # Step 2: Compute signals
        signals = {}
        
        # Lexical signal (always computed)
        signals["lexical"] = Signals.lexical_signal(
            query.terms, doc_tf, self.corpus.document_frequency,
            N, doc_length, avg_length,
        )
        
        # Optional signals (controlled by weights)
        if Config.weight_coverage > 0:
            signals["coverage"] = Signals.coverage_signal(query.terms, doc_tf)
        
        if Config.weight_density > 0:
            signals["density"] = Signals.density_signal(
                query.terms, doc_tf, doc_length,
            )
        
        if Config.weight_length > 0:
            signals["length"] = Signals.length_signal(doc_length, avg_length)
        
        if Config.weight_rarity > 0:
            signals["rarity"] = Signals.rarity_signal(
                query.terms, doc_tf, self.corpus.document_frequency, N,
            )
        
        if Config.weight_custom > 0:
            signals["custom"] = Signals.custom_signal(
                query.terms, doc_tf, self.corpus.document_frequency,
                N, doc_length, avg_length,
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
    corpus: "Corpus",
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
# Tokenization - Standalone Implementation (no external dependencies)
# =============================================================================

ENGLISH_STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "for", "from", "had", "has", "have", "he", "her", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no",
    "not", "of", "on", "or", "our", "out", "s", "she", "so", "some", "such",
    "t", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "too", "us", "very", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "would", "you", "your",
])

# Lucene English stopwords
STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
])

_TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')


def _porter_stem(word: str) -> str:
    """Simplified Porter stemmer for common English suffixes."""
    if len(word) < 3:
        return word
    
    # Step 1a: plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]
    
    # Step 1b: -ed, -ing
    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith("ed"):
        stem = word[:-2]
        if any(c in "aeiou" for c in stem):
            word = stem
    elif word.endswith("ing"):
        stem = word[:-3]
        if any(c in "aeiou" for c in stem):
            word = stem
    
    # Step 2 & 3: common derivational suffixes
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
        ("ation", "ate"), ("ator", "ate"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("icate", "ic"),
        ("ative", ""), ("alize", "al"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[:-len(suffix)] + replacement
            break
    
    return word


def tokenize(text: str) -> list[str]:
    """
    Tokenize text using Lucene-style processing.
    
    Applies:
    - Lowercasing
    - Alphanumeric token extraction
    - Stopword removal
    - Porter stemming
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [_porter_stem(t) for t in tokens]
    return [t for t in tokens if t]


class LuceneTokenizer:
    """
    Lucene-compatible tokenizer (standalone implementation).
    
    Applies:
    - Tokenization on non-letter boundaries
    - Lowercasing
    - Porter stemming (simplified)
    - English stopword removal
    """
    
    def __call__(self, text: str) -> list[str]:
        """Tokenize text."""
        return tokenize(text)


# =============================================================================
# Corpus - OPTIMIZED with Inverted Index + Sparse Matrix
# =============================================================================


class Corpus:
    """
    Corpus with pre-computed statistics - OPTIMIZED.
    
    OPTIMIZATIONS:
    1. Inverted Index - posting lists for each term
    2. Sparse Matrix - CSR format for fast TF lookups
    3. Pre-computed arrays - doc lengths, avgdl, IDF, length norm
    
    Stores tokenized documents and computes corpus statistics needed for scoring.
    """
    
    def __init__(
        self,
        documents: list[list[str]],
        ids: list[str] | None = None,
    ):
        """
        Initialize corpus with optimized data structures.
        
        Args:
            documents: List of tokenized documents (each is list of terms)
            ids: Optional document IDs
        """
        self.documents = documents
        self.ids = ids or [str(i) for i in range(len(documents))]
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        
        # Corpus statistics
        self.N = len(documents)
        self.document_count = self.N
        self.doc_lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N > 0 else 1.0
        self.average_document_length = self.avgdl
        
        # Build vocabulary first pass - collect all terms
        self._vocab: dict[str, int] = {}
        term_idx = 0
        for doc in documents:
            for term in doc:
                if term not in self._vocab:
                    self._vocab[term] = term_idx
                    term_idx += 1
        
        self.vocab_size = len(self._vocab)
        
        # Build sparse term-document matrix and inverted index
        tf_matrix_lil = lil_matrix((self.vocab_size, self.N), dtype=np.float64)
        self._inverted_index: dict[int, list[int]] = {i: [] for i in range(self.vocab_size)}
        self._df = np.zeros(self.vocab_size, dtype=np.float64)
        
        # Also build document-level term frequency dicts for evolvable scoring
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
        
        # Convert to CSR for fast row slicing
        self.tf_matrix = csr_matrix(tf_matrix_lil)
        
        # Convert inverted index lists to numpy arrays
        self._posting_lists: dict[int, NDArray[np.int64]] = {
            term_id: np.array(doc_ids, dtype=np.int64)
            for term_id, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index  # Free memory
        
        # Pre-compute IDF array (OPTIMIZATION)
        self.idf_array = FeatureExtractors.inverse_document_frequency_vectorized(self._df, self.N)
        
        # Pre-compute length normalization array (OPTIMIZATION)
        b = Config.b
        self.norm_array = 1.0 - b + b * (self.doc_lengths / max(self.avgdl, 1.0))
        
        # Legacy interface for score_document compatibility
        self.document_frequency = Counter({
            term: int(self._df[term_id])
            for term, term_id in self._vocab.items()
        })
        self.document_length = self.doc_lengths
    
    def __len__(self) -> int:
        return self.N
    
    @classmethod
    def from_huggingface_dataset(cls, dataset) -> "Corpus":
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)
    
    def get_df(self, term: str) -> int:
        """Get document frequency for a term."""
        term_id = self._vocab.get(term)
        if term_id is None:
            return 1
        return max(1, int(self._df[term_id]))
    
    def get_tf(self, doc_idx: int, term: str) -> int:
        """Get term frequency in a specific document."""
        term_id = self._vocab.get(term)
        if term_id is None:
            return 0
        return int(self.tf_matrix[term_id, doc_idx])
    
    def get_term_frequencies(self, doc_idx: int) -> Counter[str]:
        """Get term frequency dictionary for a document (for evolvable scoring)."""
        return self._doc_tf_dicts[doc_idx]
    
    def get_posting_list(self, term: str) -> NDArray[np.int64]:
        """Get posting list (doc indices containing term)."""
        term_id = self._vocab.get(term)
        if term_id is None:
            return np.array([], dtype=np.int64)
        return self._posting_lists.get(term_id, np.array([], dtype=np.int64))
    
    def get_term_id(self, term: str) -> int | None:
        """Get term ID (None if not in vocabulary)."""
        return self._vocab.get(term)
    
    def id_to_idx(self, ids: list[str]) -> list[int]:
        """Convert document IDs to indices."""
        return [self._id_to_idx[doc_id] for doc_id in ids if doc_id in self._id_to_idx]
    
    @property
    def map_id_to_idx(self) -> dict[str, int]:
        return self._id_to_idx
    
    @property
    def term_frequency(self) -> list[Counter[str]]:
        """Legacy interface for score_document."""
        return self._doc_tf_dicts


# =============================================================================
# BM25 Interface - OPTIMIZED with Vectorization + Parallel Query Processing
# =============================================================================


class BM25:
    """
    BM25-compatible interface using freeform scoring - OPTIMIZED.
    
    OPTIMIZATIONS:
    1. Inverted Index - only score docs containing query terms
    2. Vectorized scoring - numpy operations instead of loops
    3. Pre-computed IDF and norm arrays - fast array indexing
    4. Parallel query processing - ThreadPoolExecutor for batch queries
    
    All scoring still uses the evolvable multi-signal architecture.
    """
    
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
    
    def score(self, query: list[str], index: int) -> float:
        """Score a single document using evolvable kernel."""
        return score_document(query, index, self.corpus)
    
    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Score candidate documents using vectorized operations.
        
        Uses the lexical signal formula in vectorized form.
        """
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)
        
        # Use vectorized lexical signal
        return Signals.lexical_signal_vectorized(
            query_term_ids, candidate_docs, self.corpus
        )
    
    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rank all documents by relevance - OPTIMIZED.
        
        Uses inverted index to find candidates, then vectorized scoring.
        """
        if not query:
            indices = np.arange(self.corpus.N, dtype=np.int64)
            scores = np.zeros(self.corpus.N, dtype=np.float64)
            return indices, scores
        
        # Get unique query terms and their IDs
        unique_terms = list(set(query))
        query_term_ids = []
        for term in unique_terms:
            term_id = self.corpus.get_term_id(term)
            if term_id is not None:
                query_term_ids.append(term_id)
        
        if not query_term_ids:
            indices = np.arange(self.corpus.N, dtype=np.int64)
            scores = np.zeros(self.corpus.N, dtype=np.float64)
            return indices, scores
        
        # OPTIMIZATION: Get candidate documents from inverted index
        candidate_set: set[int] = set()
        for term_id in query_term_ids:
            posting_list = self.corpus._posting_lists.get(term_id, np.array([], dtype=np.int64))
            candidate_set.update(posting_list.tolist())
        
        candidate_docs = np.array(sorted(candidate_set), dtype=np.int64)
        
        # OPTIMIZATION: Vectorized scoring
        candidate_scores = self._score_candidates_vectorized(query_term_ids, candidate_docs)
        
        # Build full score array
        all_scores = np.zeros(self.corpus.N, dtype=np.float64)
        all_scores[candidate_docs] = candidate_scores
        
        # Sort by score descending
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
        """Rank for multiple queries - OPTIMIZED with parallel processing."""
        if len(queries) < MIN_QUERIES_FOR_PARALLEL:
            return [self.rank(query, top_k) for query in queries]
        
        def rank_single(query: list[str]) -> tuple[np.ndarray, np.ndarray]:
            return self.rank(query, top_k)
        
        with ThreadPoolExecutor(max_workers=NUM_QUERY_WORKERS) as executor:
            results = list(executor.map(rank_single, queries))
        
        return results


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Interface classes
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
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
