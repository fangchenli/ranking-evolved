"""
Composable BM25 Fast Seed Program - Building Blocks for AlphaEvolve (OPTIMIZED).

This seed program provides a library of scoring primitives that AlphaEvolve
can combine in novel ways, with OPTIMIZED data structures for fast evaluation.

=============================================================================
OPTIMIZATIONS (vs bm25_composable.py):
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
    uv run python evaluator_parallel.py src/ranking_evolved/bm25_composable_fast.py

For AlphaEvolve:
    uv run openevolve-run src/ranking_evolved/bm25_composable_fast.py evaluator_parallel.py --config openevolve_config_composable.yaml
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
)
from ranking_evolved.bm25 import (
    LuceneTokenizer as _BaseLuceneTokenizer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

# Number of workers for parallel query processing
NUM_QUERY_WORKERS = 32
MIN_QUERIES_FOR_PARALLEL = 10


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
    k1: float = 0.9  # TF saturation (Pyserini default)
    b: float = 0.4  # Length normalization (Pyserini default)
    k3: float = 8.0  # Query TF saturation

    # Extended parameters (for new primitives)
    delta: float = 0.5  # Bonus for matching terms (BM25+/BM25L style)
    alpha: float = 1.0  # IDF weight
    beta: float = 1.0  # TF weight
    gamma: float = 0.0  # Coverage bonus weight
    epsilon: float = 1e-9  # Numerical stability

    # Bounds
    max_idf: float = 10.0  # Cap extreme IDF values
    min_idf: float = 0.0  # Floor for IDF


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
    def idf_lucene_vectorized(df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        """Lucene IDF vectorized for arrays."""
        return np.log(1.0 + (N - df + 0.5) / (df + 0.5 + EvolvedParameters.epsilon))

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
    def saturate_lucene_vectorized(
        tf: NDArray[np.float64], k1: float, norm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Lucene/Pyserini saturation - vectorized."""
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
    def length_norm_bm25_vectorized(
        dl: NDArray[np.float64], avgdl: float, b: float
    ) -> NDArray[np.float64]:
        """Classic BM25 length normalization - vectorized."""
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
        return sum(v * w for v, w in zip(values, weights, strict=False))

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
    corpus: Corpus,
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
    doc_tf = corpus.get_term_frequencies(doc_idx)
    dl = corpus.doc_lengths[doc_idx]
    avgdl = corpus.avgdl
    N = corpus.N

    # Step 3: Score each query term occurrence (Pyserini-style)
    # Iterate over all terms including duplicates
    score = 0.0
    matched_count = 0

    for term in query_terms:
        tf = doc_tf.get(term, 0)

        if tf > 0:
            matched_count += 1
            df = corpus.get_df(term)

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
# Tokenization - Uses proper Porter stemmer from bm25.py
# =============================================================================
# LUCENE_STOPWORDS, ENGLISH_STOPWORDS, and LuceneTokenizer imported from ranking_evolved.bm25

# Global tokenizer instance (lazy initialization)
_TOKENIZER: _BaseLuceneTokenizer | None = None


def _get_tokenizer() -> _BaseLuceneTokenizer:
    """Get or create the shared tokenizer instance."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _BaseLuceneTokenizer()
    return _TOKENIZER


def tokenize(text: str) -> list[str]:
    """
    Tokenize text using Lucene-style processing.

    Uses the proper PorterStemmer from bm25.py that matches Pyserini/Lucene exactly.

    Applies:
    - Lowercasing
    - Alphanumeric token extraction
    - Stopword removal (33 Lucene stopwords)
    - Porter stemming (full algorithm)
    """
    return _get_tokenizer()(text)


class LuceneTokenizer:
    """
    Lucene-compatible tokenizer.

    Uses the proper PorterStemmer from bm25.py that matches Pyserini/Lucene exactly.

    Applies:
    - Tokenization on non-letter boundaries
    - Lowercasing
    - Porter stemming (full algorithm matching Lucene)
    - English stopword removal (33 words)
    """

    def __init__(self):
        self._tokenizer = _BaseLuceneTokenizer()

    def __call__(self, text: str) -> list[str]:
        """Tokenize text."""
        return self._tokenizer(text)


# =============================================================================
# Corpus - OPTIMIZED with Inverted Index + Sparse Matrix
# =============================================================================


class Corpus:
    """
    Pre-processed document collection with OPTIMIZED data structures.

    OPTIMIZATIONS:
    1. Inverted Index - posting lists for each term
    2. Sparse Matrix - CSR format for fast TF lookups
    3. Pre-computed arrays - doc lengths, avgdl, IDF, length norm

    Stores tokenized documents and computes corpus statistics needed for BM25.
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

        # Also build document-level term frequency dicts for evolvable score_kernel
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

        # Convert to CSR for fast row slicing and arithmetic
        self.tf_matrix = csr_matrix(tf_matrix_lil)

        # Convert inverted index lists to numpy arrays
        self._posting_lists: dict[int, NDArray[np.int64]] = {
            term_id: np.array(doc_ids, dtype=np.int64)
            for term_id, doc_ids in self._inverted_index.items()
            if doc_ids
        }
        del self._inverted_index  # Free memory

        # Pre-compute IDF array (OPTIMIZATION)
        self.idf_array = ScoringPrimitives.idf_lucene_vectorized(self._df, self.N)

        # Pre-compute length normalization array (OPTIMIZATION)
        self.norm_array = ScoringPrimitives.length_norm_bm25_vectorized(
            self.doc_lengths, self.avgdl, EvolvedParameters.b
        )

        # Legacy interface for score_kernel compatibility
        self.document_frequency = Counter(
            {term: int(self._df[term_id]) for term, term_id in self._vocab.items()}
        )
        self.document_length = self.doc_lengths

    def __len__(self) -> int:
        return self.N

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> Corpus:
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)

    def get_df(self, term: str) -> int:
        """Get document frequency for a term."""
        term_id = self._vocab.get(term)
        if term_id is None:
            return 1  # Return 1 to avoid division by zero
        return max(1, int(self._df[term_id]))

    def get_tf(self, doc_idx: int, term: str) -> int:
        """Get term frequency in a specific document."""
        term_id = self._vocab.get(term)
        if term_id is None:
            return 0
        return int(self.tf_matrix[term_id, doc_idx])

    def get_term_frequencies(self, doc_idx: int) -> Counter[str]:
        """Get term frequency dictionary for a document (for evolvable kernel)."""
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
    def vocabulary_size(self) -> int:
        """Number of unique terms in corpus (for evaluator compatibility)."""
        return self.vocab_size

    @property
    def term_doc_matrix(self) -> None:
        """Placeholder for evaluator compatibility."""
        return None

    @property
    def term_frequency(self) -> list[Counter[str]]:
        """Legacy interface for score_kernel."""
        return self._doc_tf_dicts


# =============================================================================
# BM25 Scorer - OPTIMIZED with Vectorization + Parallel Query Processing
# =============================================================================


class BM25:
    """
    BM25 scorer using composable primitives - OPTIMIZED.

    OPTIMIZATIONS:
    1. Inverted Index - only score docs containing query terms
    2. Vectorized scoring - numpy operations instead of loops
    3. Pre-computed IDF and norm arrays - fast array indexing
    4. Parallel query processing - ThreadPoolExecutor for batch queries

    All scoring components still use the evolvable primitives.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def score(self, query: list[str], index: int) -> float:
        """Score a single document using evolvable kernel."""
        return score_kernel(query, index, self.corpus)

    def _score_candidates_vectorized(
        self,
        query_term_ids: list[int],
        candidate_docs: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Score candidate documents using vectorized operations.

        This is the OPTIMIZED scoring path using numpy/scipy while
        still using the same formula as the evolvable primitives.
        """
        if len(candidate_docs) == 0:
            return np.array([], dtype=np.float64)

        # Get pre-computed values for candidates
        norms = self.corpus.norm_array[candidate_docs]

        # Initialize scores
        scores = np.zeros(len(candidate_docs), dtype=np.float64)

        # Get parameters
        k1 = EvolvedParameters.k1

        # Score each query term (vectorized over documents)
        for term_id in query_term_ids:
            idf = self.corpus.idf_array[term_id]
            if idf <= 0:
                continue

            # Clamp IDF to bounds
            idf = max(EvolvedParameters.min_idf, min(idf, EvolvedParameters.max_idf))

            # Get TF for all candidates at once
            tf_row = self.corpus.tf_matrix[term_id, candidate_docs].toarray().flatten()

            # Vectorized Lucene TF formula - matches saturate_lucene
            tf_saturated = ScoringPrimitives.saturate_lucene_vectorized(tf_row, k1, norms)

            # Add term contribution
            scores += idf * tf_saturated

        return scores

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
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "LUCENE_STOPWORDS",
    "ENGLISH_STOPWORDS",
    # Evolution targets
    "EvolvedParameters",
    "ScoringPrimitives",
    "TermScorer",
    "DocumentScorer",
    "QueryProcessor",
    "score_kernel",
]
