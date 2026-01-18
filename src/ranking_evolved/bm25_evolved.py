"""
Minimal BM25 implementation for OpenEvolve optimization.

This file contains ONLY the evolution targets - the core scoring components
that OpenEvolve should modify. All infrastructure (tokenizers, corpus, etc.)
is imported from the main bm25.py module.

Evolution targets:
- EvolvedIDF.compute() - IDF formula
- EvolvedTF.compute() - TF saturation formula
- BM25.score_kernel() - Main scoring function

Run with:
    export OPENAI_API_KEY="your-key"
    uv run openevolve-run src/ranking_evolved/bm25_evolved.py evaluator_bright.py --config openevolve_config.yaml
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import infrastructure from main module
from ranking_evolved.bm25 import (
    ENGLISH_STOPWORDS,
    Corpus,
    LuceneTokenizer,
    PorterStemmer,
    lucene_tokenize,
    tokenize,
)

# =============================================================================
# EVOLUTION TARGET 1: IDF Strategy
# =============================================================================


class EvolvedIDF:
    """
    Evolved IDF strategy - PRIMARY OPENEVOLVE TARGET.

    Current best formula:
        clip(log((N + 0.5) / (df + 0.5)), 0, max_idf)

    Modify this to discover improved IDF formulations.
    """

    def __init__(self, max_idf: float = 8.0):
        self.max_idf = max_idf

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
        idf = np.log((N + 0.5) / (df + 0.5))
        return np.clip(idf, 0.0, self.max_idf)


# =============================================================================
# EVOLUTION TARGET 2: TF Strategy
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
        tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm + 1e-9)
        tf_sat = tf / (tf + k1 + 0.5)
        return np.log1p(tf_raw * tf_sat)


# =============================================================================
# EVOLUTION TARGET 3: BM25 Scorer
# =============================================================================


class BM25:
    """
    BM25 ranking with evolved IDF and TF components.

    The score_kernel method is the main evolution target.

    Args:
        corpus: Pre-processed Corpus instance.
        k1: TF saturation parameter (default: 0.9).
        b: Length normalization parameter (default: 0.4).
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float = 0.9,
        b: float = 0.4,
    ):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Rebuild corpus with evolved IDF
        self._idf_strategy = EvolvedIDF()
        self._tf_strategy = EvolvedTF()

        # Recompute IDF with evolved strategy
        df = corpus.df_array
        N = corpus.document_count
        self._idf_array = self._idf_strategy.compute(df, N)
        self._idf_dict = {
            term: float(self._idf_array[idx]) for term, idx in corpus.vocabulary.items()
        }

        # Pre-compute document normalization
        dl = corpus.document_length
        avgdl = corpus.average_document_length or 1.0
        self._doc_norm = 1.0 - b + b * (dl / avgdl)

    @staticmethod
    def score_kernel(
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
        k1: float,
    ) -> float:
        """
        Compute BM25 score for a single document - PRIMARY EVOLUTION TARGET.

        Args:
            query: List of query terms (tokenized).
            norm: Document length normalization factor.
            frequencies: Term frequency counter for the document.
            idf: IDF values dictionary (term -> idf).
            k1: TF saturation parameter.

        Returns:
            BM25 relevance score (float).
        """
        if not query:
            return 0.0

        # Use unique query terms
        unique_terms = list(dict.fromkeys(query))
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

        return float(np.sum(term_scores))

    def score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        return self.score_kernel(
            query,
            float(self._doc_norm[index]),
            self.corpus.term_frequency[index],
            self._idf_dict,
            self.k1,
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank all documents by relevance to query (vectorized for speed)."""
        if not query:
            n = len(self.corpus)
            return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

        # Get unique query terms that exist in vocabulary
        unique_terms = list(dict.fromkeys(query))
        vocab = self.corpus.vocabulary
        term_indices = [vocab[t] for t in unique_terms if t in vocab]

        if not term_indices:
            n = len(self.corpus)
            return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.float64)

        # Get IDF values for query terms
        idf_values = self._idf_array[term_indices]

        # Get TF matrix slice for query terms (sparse -> dense for these rows)
        tf_matrix = self.corpus.term_doc_matrix[term_indices, :].toarray()  # (n_terms, n_docs)

        # Vectorized BM25 scoring
        # tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm)
        # tf_sat = tf / (tf + k1 + 0.5)
        # score = sum(idf * log1p(tf_raw * tf_sat))
        k1 = self.k1
        norm = self._doc_norm  # (n_docs,)

        denom = tf_matrix + k1 * norm  # broadcasting: (n_terms, n_docs)
        tf_raw = (tf_matrix * (k1 + 1.0)) / np.maximum(denom, 1e-9)
        tf_sat = tf_matrix / (tf_matrix + k1 + 0.5)
        term_scores = idf_values[:, np.newaxis] * np.log1p(tf_raw * tf_sat)  # (n_terms, n_docs)

        scores = np.sum(term_scores, axis=0)  # (n_docs,)

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores


# =============================================================================
# Module exports (required by evaluator_bright.py)
# =============================================================================

__all__ = [
    "BM25",
    "Corpus",
    "tokenize",
    "LuceneTokenizer",
    "lucene_tokenize",
    "PorterStemmer",
    "ENGLISH_STOPWORDS",
    "EvolvedIDF",
    "EvolvedTF",
]
