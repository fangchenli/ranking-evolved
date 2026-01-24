"""
Composable BM25 Seed Program - Building Blocks for AlphaEvolve.

This seed program provides a library of scoring primitives that AlphaEvolve
can combine in novel ways. It's more flexible than classic BM25 but still
provides structure to guide evolution.

=============================================================================
DESIGN PHILOSOPHY:
=============================================================================

Instead of evolving a fixed formula, AlphaEvolve can:
1. Choose which primitives to use
2. Decide how to combine them
3. Add new computations using the primitives
4. Change the overall scoring strategy

The primitives are well-tested building blocks. The composition is evolved.

=============================================================================
EVOLUTION TARGETS:
=============================================================================

1. PARAMETERS (EvolvedParameters) - Numeric constants
2. PRIMITIVES (ScoringPrimitives) - Building blocks (add new ones!)
3. TERM SCORER (TermScorer.score) - How to score a single term
4. DOCUMENT SCORER (DocumentScorer.score) - How to combine term scores
5. QUERY PROCESSOR (QueryProcessor.process) - How to handle queries
6. MAIN KERNEL (score_kernel) - The overall scoring strategy

=============================================================================

Run evaluation with:
    uv run python evaluator_bright.py

For AlphaEvolve:
    uv run openevolve-run src/ranking_evolved/bm25_composable.py evaluator_bright.py --config openevolve_config.yaml
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# EVOLUTION TARGET 1: Parameters
# =============================================================================


class EvolvedParameters:
    """
    Numeric parameters - EVOLUTION TARGET.
    
    AlphaEvolve can tune these values or add new parameters.
    
    Defaults match Pyserini/Lucene for equivalence.
    """
    
    # ===== EVOLVE THESE VALUES =====
    
    # Classic BM25 parameters (Pyserini/Lucene defaults)
    k1: float = 0.9          # TF saturation (Pyserini default)
    b: float = 0.4           # Length normalization (Pyserini default)
    k3: float = 8.0          # Query TF saturation
    
    # Extended parameters (for new primitives)
    delta: float = 0.5       # Bonus for matching terms (BM25+/BM25L style)
    alpha: float = 1.0       # IDF weight
    beta: float = 1.0        # TF weight
    gamma: float = 0.0       # Coverage bonus weight
    epsilon: float = 1e-9    # Numerical stability
    
    # Bounds
    max_idf: float = 10.0    # Cap extreme IDF values
    min_idf: float = 0.0     # Floor for IDF


# =============================================================================
# EVOLUTION TARGET 2: Scoring Primitives (Building Blocks)
# =============================================================================


class ScoringPrimitives:
    """
    Library of scoring primitives - EVOLUTION TARGET.
    
    These are building blocks that can be combined in the scoring functions.
    AlphaEvolve can:
    - Use existing primitives in new ways
    - Modify primitive implementations
    - Add entirely new primitives
    """
    
    # ----- IDF Primitives -----
    
    @staticmethod
    def idf_classic(df: float, N: int) -> float:
        """Classic Robertson IDF: log((N - df + 0.5) / (df + 0.5))"""
        return math.log((N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))
    
    @staticmethod
    def idf_lucene(df: float, N: int) -> float:
        """Lucene IDF (non-negative): log(1 + (N - df + 0.5) / (df + 0.5))"""
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))
    
    @staticmethod
    def idf_atire(df: float, N: int) -> float:
        """ATIRE IDF: log(N / df)"""
        return math.log(N / (df + EvolvedParameters.epsilon))
    
    @staticmethod
    def idf_bm25plus(df: float, N: int) -> float:
        """BM25+ IDF: log((N + 1) / df)"""
        return math.log((N + 1) / (df + EvolvedParameters.epsilon))
    
    @staticmethod
    def idf_smooth(df: float, N: int) -> float:
        """Smoothed IDF: log((N + 0.5) / (df + 0.5))"""
        return math.log((N + 0.5) / (df + 0.5))
    
    # ----- TF Primitives -----
    
    @staticmethod
    def tf_raw(tf: float) -> float:
        """Raw term frequency (no transformation)."""
        return tf
    
    @staticmethod
    def tf_log(tf: float) -> float:
        """Logarithmic TF: 1 + log(tf) for tf > 0."""
        return 1.0 + math.log(tf) if tf > 0 else 0.0
    
    @staticmethod
    def tf_double_log(tf: float) -> float:
        """Double log TF: 1 + log(1 + log(tf)) for tf > 0."""
        if tf <= 0:
            return 0.0
        return 1.0 + math.log(1.0 + math.log(tf + 1))
    
    @staticmethod
    def tf_boolean(tf: float) -> float:
        """Boolean TF: 1 if present, 0 otherwise."""
        return 1.0 if tf > 0 else 0.0
    
    @staticmethod
    def tf_augmented(tf: float, max_tf: float) -> float:
        """Augmented TF: 0.5 + 0.5 * (tf / max_tf)."""
        if max_tf <= 0:
            return 0.5
        return 0.5 + 0.5 * (tf / max_tf)
    
    # ----- Saturation Primitives -----
    
    @staticmethod
    def saturate(x: float, k: float) -> float:
        """Basic saturation: x / (x + k)."""
        return x / (x + k + EvolvedParameters.epsilon)
    
    @staticmethod
    def saturate_bm25(tf: float, k1: float, norm: float) -> float:
        """BM25-style saturation: (tf * (k1 + 1)) / (tf + k1 * norm)."""
        denom = tf + k1 * norm + EvolvedParameters.epsilon
        return (tf * (k1 + 1)) / denom
    
    @staticmethod
    def saturate_lucene(tf: float, k1: float, norm: float) -> float:
        """Lucene/Pyserini saturation: tf / (tf + k1 * norm) [no k1+1 multiplier]."""
        denom = tf + k1 * norm + EvolvedParameters.epsilon
        return tf / denom
    
    @staticmethod
    def saturate_bm25l(tf: float, k1: float, norm: float, delta: float) -> float:
        """BM25L saturation with length-corrected TF."""
        c = tf / (norm + EvolvedParameters.epsilon)
        c_delta = c + delta
        return ((k1 + 1) * c_delta) / (k1 + c_delta + EvolvedParameters.epsilon)
    
    @staticmethod
    def saturate_bm25plus(tf: float, k1: float, norm: float, delta: float) -> float:
        """BM25+ saturation with lower-bound bonus."""
        base = (tf * (k1 + 1)) / (tf + k1 * norm + EvolvedParameters.epsilon)
        return base + delta if tf > 0 else base
    
    @staticmethod
    def saturate_log(tf: float, k1: float, norm: float) -> float:
        """Log-damped saturation: log(1 + BM25_saturation)."""
        bm25_sat = (tf * (k1 + 1)) / (tf + k1 * norm + EvolvedParameters.epsilon)
        return math.log(1.0 + bm25_sat)
    
    # ----- Normalization Primitives -----
    
    @staticmethod
    def length_norm_bm25(dl: float, avgdl: float, b: float) -> float:
        """Classic BM25 length normalization: 1 - b + b * (dl / avgdl)."""
        return 1.0 - b + b * (dl / max(avgdl, 1.0))
    
    @staticmethod
    def length_norm_pivot(dl: float, pivot: float, b: float) -> float:
        """Pivoted length normalization: 1 - b + b * (dl / pivot)."""
        return 1.0 - b + b * (dl / max(pivot, 1.0))
    
    @staticmethod
    def length_norm_log(dl: float, avgdl: float, b: float) -> float:
        """Log-based length normalization."""
        ratio = dl / max(avgdl, 1.0)
        return 1.0 + b * math.log(ratio) if ratio > 0 else 1.0
    
    # ----- Combination Primitives -----
    
    @staticmethod
    def multiply(*args: float) -> float:
        """Multiply all arguments."""
        result = 1.0
        for x in args:
            result *= x
        return result
    
    @staticmethod
    def add(*args: float) -> float:
        """Add all arguments."""
        return sum(args)
    
    @staticmethod
    def weighted_sum(values: list[float], weights: list[float]) -> float:
        """Weighted sum of values."""
        return sum(v * w for v, w in zip(values, weights))
    
    @staticmethod
    def geometric_mean(values: list[float]) -> float:
        """Geometric mean of values."""
        if not values:
            return 0.0
        product = 1.0
        for v in values:
            if v <= 0:
                return 0.0
            product *= v
        return product ** (1.0 / len(values))
    
    @staticmethod
    def harmonic_mean(values: list[float]) -> float:
        """Harmonic mean of values."""
        if not values:
            return 0.0
        reciprocal_sum = sum(1.0 / (v + EvolvedParameters.epsilon) for v in values)
        return len(values) / reciprocal_sum if reciprocal_sum > 0 else 0.0
    
    @staticmethod
    def soft_max(values: list[float], temperature: float = 1.0) -> float:
        """Soft maximum (log-sum-exp)."""
        if not values:
            return 0.0
        max_val = max(values)
        exp_sum = sum(math.exp((v - max_val) / temperature) for v in values)
        return max_val + temperature * math.log(exp_sum)
    
    # ----- Query Primitives -----
    
    @staticmethod
    def query_weight_uniform(qtf: float, k3: float) -> float:
        """Uniform query term weight (bag-of-words)."""
        return 1.0
    
    @staticmethod
    def query_weight_frequency(qtf: float, k3: float) -> float:
        """Weight by query term frequency."""
        return qtf
    
    @staticmethod
    def query_weight_saturated(qtf: float, k3: float) -> float:
        """BM25-style query term saturation: ((k3 + 1) * qtf) / (k3 + qtf)."""
        return ((k3 + 1) * qtf) / (k3 + qtf + EvolvedParameters.epsilon)
    
    # ===== ADD NEW PRIMITIVES HERE =====
    
    @staticmethod
    def coverage_bonus(matched_terms: int, total_query_terms: int) -> float:
        """Bonus for covering more query terms."""
        if total_query_terms <= 0:
            return 0.0
        coverage = matched_terms / total_query_terms
        return coverage * coverage  # Quadratic reward for full coverage
    
    @staticmethod
    def rarity_boost(idf: float, threshold: float = 3.0) -> float:
        """Extra boost for very rare terms."""
        if idf > threshold:
            return 1.0 + (idf - threshold) * 0.1
        return 1.0


# =============================================================================
# EVOLUTION TARGET 3: Term Scorer
# =============================================================================


class TermScorer:
    """
    Computes score contribution for a single term - EVOLUTION TARGET.
    
    This class decides how to combine IDF and TF for a single term.
    AlphaEvolve can completely restructure this logic.
    """
    
    @staticmethod
    def score(
        tf: float,
        df: float,
        N: int,
        dl: float,
        avgdl: float,
    ) -> float:
        """
        Compute score for a single term - EVOLUTION TARGET.
        
        Args:
            tf: Term frequency in document
            df: Document frequency of term
            N: Total documents in corpus
            dl: Document length
            avgdl: Average document length
            
        Returns:
            Term score contribution
        """
        if tf <= 0:
            return 0.0
        
        # ===== EVOLVE THIS SCORING LOGIC =====
        
        # Get parameters
        k1 = EvolvedParameters.k1
        b = EvolvedParameters.b
        
        # Step 1: Compute IDF (choose/combine primitives)
        # Use Lucene IDF to match Pyserini
        idf = ScoringPrimitives.idf_lucene(df, N)
        
        # Step 2: Apply IDF bounds
        idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))
        
        # Step 3: Compute length normalization
        norm = ScoringPrimitives.length_norm_bm25(dl, avgdl, b)
        
        # Step 4: Compute TF saturation (choose/combine primitives)
        # Use Lucene TF saturation to match Pyserini (no k1+1 multiplier)
        tf_component = ScoringPrimitives.saturate_lucene(tf, k1, norm)
        
        # Step 5: Combine IDF and TF (choose combination strategy)
        term_score = ScoringPrimitives.multiply(idf, tf_component)
        
        return term_score


# =============================================================================
# EVOLUTION TARGET 4: Document Scorer
# =============================================================================


class DocumentScorer:
    """
    Combines term scores into a document score - EVOLUTION TARGET.
    
    This class decides the aggregation strategy.
    AlphaEvolve can change how term scores are combined.
    """
    
    @staticmethod
    def score(
        term_scores: list[float],
        query_weights: list[float],
        matched_count: int,
        total_query_terms: int,
    ) -> float:
        """
        Aggregate term scores into document score - EVOLUTION TARGET.
        
        Args:
            term_scores: Individual term score contributions
            query_weights: Weights for each query term
            matched_count: Number of query terms that matched
            total_query_terms: Total number of unique query terms
            
        Returns:
            Final document score
        """
        if not term_scores:
            return 0.0
        
        # ===== EVOLVE THIS AGGREGATION LOGIC =====
        
        # Strategy 1: Weighted sum (classic)
        base_score = ScoringPrimitives.weighted_sum(term_scores, query_weights)
        
        # Strategy 2: Optional coverage bonus
        gamma = EvolvedParameters.gamma
        if gamma > 0:
            coverage = ScoringPrimitives.coverage_bonus(matched_count, total_query_terms)
            base_score += gamma * coverage
        
        return base_score


# =============================================================================
# EVOLUTION TARGET 5: Query Processor
# =============================================================================


class QueryProcessor:
    """
    Processes query terms and computes weights - EVOLUTION TARGET.
    
    AlphaEvolve can change how queries are interpreted.
    
    Default behavior matches Pyserini: pass full query with duplicates.
    """
    
    @staticmethod
    def process(query: list[str]) -> tuple[list[str], list[float]]:
        """
        Process query terms - EVOLUTION TARGET.
        
        Args:
            query: Raw query terms (may have duplicates)
            
        Returns:
            Tuple of (query_terms, weights)
            
        Note: To match Pyserini, we return the full query (with duplicates)
        and weights of 1.0 for each term. The score_kernel will iterate
        over all terms, naturally handling query term frequency.
        """
        if not query:
            return [], []
        
        # ===== EVOLVE THIS QUERY PROCESSING =====
        
        # Pyserini-style: pass full query with duplicates
        # Each term occurrence contributes separately
        # Return uniform weights (1.0) for each term
        weights = [1.0] * len(query)
        
        return query, weights


# =============================================================================
# EVOLUTION TARGET 6: Main Scoring Kernel
# =============================================================================


def score_kernel(
    query: list[str],
    doc_idx: int,
    corpus: "Corpus",
) -> float:
    """
    Main scoring function - PRIMARY EVOLUTION TARGET.
    
    This orchestrates all components. AlphaEvolve can restructure
    the entire scoring pipeline here.
    
    Default behavior matches Pyserini: iterate over all query terms
    (including duplicates) and sum contributions.
    
    Args:
        query: Tokenized query terms (may contain duplicates)
        doc_idx: Document index to score
        corpus: Corpus with pre-computed statistics
        
    Returns:
        Relevance score for the document
    """
    if not query:
        return 0.0
    
    # ===== EVOLVE THIS ENTIRE PIPELINE =====
    
    # Step 1: Process query (Pyserini-style: get full query with duplicates)
    query_terms, query_weights = QueryProcessor.process(query)
    
    if not query_terms:
        return 0.0
    
    # Step 2: Get document statistics
    doc_tf = corpus.term_frequency[doc_idx]
    dl = corpus.document_length[doc_idx]
    avgdl = corpus.average_document_length
    N = corpus.document_count
    
    # Step 3: Score each query term occurrence (Pyserini-style)
    # Iterate over all terms including duplicates
    score = 0.0
    matched_count = 0
    
    for term in query_terms:
        tf = doc_tf.get(term, 0)
        
        if tf > 0:
            matched_count += 1
            df = corpus.document_frequency.get(term, 1)
            
            # Use TermScorer to get contribution for this term occurrence
            term_score = TermScorer.score(tf, df, N, dl, avgdl)
            score += term_score
    
    # Early exit if no matches
    if matched_count == 0:
        return 0.0
    
    # Step 4: Return sum (Pyserini-style: simple sum over all term occurrences)
    # Note: DocumentScorer could be used for more complex aggregation, but
    # for Pyserini equivalence, we just return the sum directly
    return score


# =============================================================================
# Tokenization (not evolved)
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

# Lucene English stopwords (fallback, matches bm25_pyserini.py)
STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
])

_TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')


def _get_pyserini_tokenizer() -> Callable[[str], list[str]] | None:
    """Try to get Pyserini's Lucene tokenizer."""
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        analyzer = Analyzer(get_lucene_analyzer())
        return analyzer.analyze
    except Exception:
        return None


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
    - English stopword removal
    
    Falls back to a simple approximation if Pyserini/Java unavailable.
    """
    
    def __init__(self):
        self._pyserini_tokenize = _get_pyserini_tokenizer()
        if self._pyserini_tokenize is None:
            import warnings
            warnings.warn(
                "Pyserini not available. Using fallback tokenizer. "
                "For exact Pyserini reproduction, install pyserini and Java 21."
            )
    
    def __call__(self, text: str) -> list[str]:
        """Tokenize text using Lucene analyzer."""
        if self._pyserini_tokenize is not None:
            return self._pyserini_tokenize(text)
        
        # Fallback: simple tokenization without stemming
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# =============================================================================
# Corpus (infrastructure, not evolved)
# =============================================================================


class Corpus:
    """Pre-computed corpus statistics."""
    
    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        self.documents = documents
        self.document_count = len(documents)
        self.ids = ids
    
    def __len__(self) -> int:
        return self.document_count
    
    @classmethod
    def from_huggingface_dataset(cls, dataset) -> "Corpus":
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
        return float(np.mean(self.document_length)) if self.document_count > 0 else 0.0
    
    @cached_property
    def map_id_to_idx(self) -> dict[str, int]:
        return {id_: idx for idx, id_ in enumerate(self.ids)} if self.ids else {}
    
    def id_to_idx(self, ids: list[str]) -> list[int]:
        return [self.map_id_to_idx[id_] for id_ in ids]


# =============================================================================
# BM25 Scorer (uses the evolved kernel)
# =============================================================================


class BM25:
    """BM25 scorer using composable primitives."""
    
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
    
    def score(self, query: list[str], index: int) -> float:
        """Score a single document."""
        return score_kernel(query, index, self.corpus)
    
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
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "ENGLISH_STOPWORDS",
    # Evolution targets
    "EvolvedParameters",
    "ScoringPrimitives",
    "TermScorer",
    "DocumentScorer",
    "QueryProcessor",
    "score_kernel",
]
