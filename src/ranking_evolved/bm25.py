"""
BM25 Ranking Implementation - Single-file module for OpenEvolve optimization.

This module provides a modular BM25 implementation with swappable IDF and TF
strategies, optimized for evolution via OpenEvolve. All components are contained
in this single file to satisfy OpenEvolve's requirements.

Usage:
    from ranking_evolved.bm25 import BM25, Corpus, tokenize

    corpus = Corpus.from_huggingface_dataset(dataset)
    bm25 = BM25(corpus)
    indices, scores = bm25.rank(tokenize("search query"))

For OpenEvolve: The `score_kernel` function and IDF/TF strategy classes are
the primary evolution targets.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Query Term Counting Mode
# =============================================================================


class QueryTermMode(Enum):
    """
    How to handle repeated terms in the query.

    - UNIQUE: Each unique term contributes once (bag-of-words). Default for most use cases.
    - SUM_ALL: Sum scores for all occurrences (Pyserini/Lucene style).
    - SATURATED: Apply BM25-style saturation to query term frequency.
    """

    UNIQUE = "unique"  # Bag-of-words: deduplicate query terms
    SUM_ALL = "sum_all"  # Pyserini-style: sum over all occurrences
    SATURATED = "saturated"  # Apply (k3+1)*qtf/(k3+qtf) saturation


# =============================================================================
# Tokenization
# =============================================================================

# Lucene English stopwords (from org.apache.lucene.analysis.en.EnglishAnalyzer)
# This is the complete set from Lucene's EnglishAnalyzer default stopwords
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "s",
        "she",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase terms.

    This is a simple whitespace + alphanumeric tokenizer suitable for
    English text. Can be evolved or replaced for domain-specific needs.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"\w+", text.lower())


class LuceneTokenizer:
    """
    Pure Python implementation of Lucene's DefaultEnglishAnalyzer.

    Replicates the tokenization pipeline used by Pyserini/Anserini:
    1. StandardTokenizer - Unicode-aware word segmentation
    2. EnglishPossessiveFilter - Remove 's and ' suffixes
    3. LowerCaseFilter - Convert to lowercase
    4. StopFilter - Remove common English stopwords
    5. PorterStemFilter - Apply Porter stemming algorithm

    This eliminates the need for Java/Pyserini while producing compatible output.

    Example:
        tokenizer = LuceneTokenizer()
        tokens = tokenizer("The quick brown fox's running")
        # Returns: ['quick', 'brown', 'fox', 'run']
    """

    def __init__(
        self,
        stopwords: frozenset[str] | None = None,
        stem: bool = True,
    ):
        """
        Initialize the tokenizer.

        Args:
            stopwords: Set of stopwords to remove. Defaults to ENGLISH_STOPWORDS.
            stem: Whether to apply Porter stemming. Defaults to True.
        """
        self.stopwords = stopwords if stopwords is not None else ENGLISH_STOPWORDS
        self.stem = stem
        self._stemmer = PorterStemmer() if stem else None

    def __call__(self, text: str) -> list[str]:
        """Tokenize text using the Lucene pipeline."""
        # Step 1: StandardTokenizer - extract alphanumeric sequences
        # Also handles contractions and possessives
        tokens = re.findall(r"\w+", text)

        result = []
        for token in tokens:
            # Step 2: EnglishPossessiveFilter - strip possessive endings
            if token.endswith("'s") or token.endswith("'s"):
                token = token[:-2]
            elif token.endswith("'") or token.endswith("'"):
                token = token[:-1]

            # Skip empty tokens
            if not token:
                continue

            # Step 3: LowerCaseFilter
            token = token.lower()

            # Step 4: StopFilter
            if token in self.stopwords:
                continue

            # Step 5: PorterStemFilter
            if self._stemmer is not None:
                token = self._stemmer.stem(token)

            result.append(token)

        return result


class PorterStemmer:
    """
    Porter Stemmer implementation for English.

    Based on the Porter Stemming Algorithm (1980) by Martin Porter.
    Reference: https://tartarus.org/martin/PorterStemmer/

    This is a pure Python implementation that produces the same output as
    Lucene's PorterStemFilter for English text.
    """

    def __init__(self):
        self._vowels = frozenset("aeiou")

    def _is_consonant(self, word: str, i: int) -> bool:
        """Check if character at position i is a consonant."""
        if word[i] in self._vowels:
            return False
        if word[i] == "y":
            return i == 0 or not self._is_consonant(word, i - 1)
        return True

    def _measure(self, word: str) -> int:
        """
        Calculate the measure m of a word.

        m = number of VC (vowel-consonant) sequences in the word.
        """
        n = 0
        i = 0
        length = len(word)

        # Skip initial consonants
        while i < length and self._is_consonant(word, i):
            i += 1

        while i < length:
            # Skip vowels
            while i < length and not self._is_consonant(word, i):
                i += 1
            if i >= length:
                break

            # Count this VC sequence
            n += 1

            # Skip consonants
            while i < length and self._is_consonant(word, i):
                i += 1

        return n

    def _has_vowel(self, word: str) -> bool:
        """Check if word contains a vowel."""
        return any(not self._is_consonant(word, i) for i in range(len(word)))

    def _ends_double_consonant(self, word: str) -> bool:
        """Check if word ends with a double consonant."""
        return len(word) >= 2 and word[-1] == word[-2] and self._is_consonant(word, len(word) - 1)

    def _ends_cvc(self, word: str) -> bool:
        """
        Check if word ends with consonant-vowel-consonant,
        where final consonant is not w, x, or y.
        """
        if len(word) < 3:
            return False
        return (
            self._is_consonant(word, len(word) - 1)
            and not self._is_consonant(word, len(word) - 2)
            and self._is_consonant(word, len(word) - 3)
            and word[-1] not in "wxy"
        )

    def _replace_suffix(
        self, word: str, suffix: str, replacement: str, m_threshold: int = 0
    ) -> str:
        """Replace suffix if word ends with it and measure > threshold."""
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if self._measure(stem) > m_threshold:
                return stem + replacement
        return word

    def stem(self, word: str) -> str:
        """Apply Porter stemming algorithm to word."""
        if len(word) <= 2:
            return word

        word = word.lower()

        # Step 1a: SSES -> SS, IES -> I, SS -> SS, S -> (remove)
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b: (m>0) EED -> EE, (*v*) ED -> , (*v*) ING ->
        if word.endswith("eed"):
            if self._measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                # Additional rules after ED removal
                if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                    word = word + "e"
                elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif self._measure(word) == 1 and self._ends_cvc(word):
                    word = word + "e"
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                # Additional rules after ING removal
                if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                    word = word + "e"
                elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif self._measure(word) == 1 and self._ends_cvc(word):
                    word = word + "e"

        # Step 1c: (*v*) Y -> I
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2: Suffix replacements with m > 0
        step2_suffixes = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ]
        for suffix, replacement in step2_suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 3: Suffix replacements with m > 0
        step3_suffixes = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]
        for suffix, replacement in step3_suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 4: Suffix removal with m > 1
        step4_suffixes = [
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ion",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        ]
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if suffix == "ion":
                    if stem and stem[-1] in "st" and self._measure(stem) > 1:
                        word = stem
                elif self._measure(stem) > 1:
                    word = stem
                break

        # Step 5a: (m>1) E -> , (m=1 and not *o) E ->
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._ends_cvc(stem)):
                word = stem

        # Step 5b: (m>1 and *d and *L) -> single letter
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]

        return word


def lucene_tokenize(text: str) -> list[str]:
    """
    Tokenize text using Lucene-compatible pipeline.

    This is a convenience function that creates a default LuceneTokenizer
    and applies it to the text. For repeated use, create a LuceneTokenizer
    instance directly for better performance.

    Args:
        text: Input text to tokenize.

    Returns:
        List of stemmed, lowercased tokens with stopwords removed.
    """
    return LuceneTokenizer()(text)


# =============================================================================
# IDF Strategies (Evolvable)
# =============================================================================


class IDFStrategy(Protocol):
    """Protocol for IDF computation strategies."""

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        """
        Compute IDF values for terms.

        Args:
            df: Document frequency array for each term.
            N: Total number of documents in corpus.

        Returns:
            IDF values array.
        """
        ...


class ClassicIDF:
    """
    Classic Robertson BM25 IDF.

    Formula: log((N - df + 0.5) / (df + 0.5))

    Note: Can produce negative values for terms in >50% of documents.
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log((N - df + 0.5) / (df + 0.5))


class LuceneIDF:
    """
    Lucene-style BM25 IDF (non-negative).

    Formula: log(1 + (N - df + 0.5) / (df + 0.5))
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log(1.0 + (N - df + 0.5) / (df + 0.5))


class ATIREIDF:
    """
    ATIRE-style IDF.

    Formula: log(N / df)
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log(N / np.maximum(df, 1e-9))


class BM25LIDF:
    """
    BM25L IDF (non-negative, for long document correction).

    Formula: log((N + 1) / (df + 0.5))
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log((N + 1) / (df + 0.5))


class BM25PlusIDF:
    """
    BM25+ IDF (non-negative).

    Formula: log((N + 1) / df)
    """

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        return np.log((N + 1) / np.maximum(df, 1e-9))


class ClippedIDF:
    """
    Clipped IDF to prevent extreme values.

    Formula: clip(log((N + 0.5) / (df + 0.5)), 0, max_idf)

    This variant was found effective in evolution experiments.
    """

    def __init__(self, max_idf: float = 8.0):
        self.max_idf = max_idf

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        idf = np.log((N + 0.5) / (df + 0.5))
        return np.clip(idf, 0.0, self.max_idf)


# Default IDF for evolution - this is the primary evolution target
class EvolvedIDF:
    """
    Evolved IDF strategy - primary OpenEvolve target.

    This IDF variant can be modified by OpenEvolve to discover
    improved formulations.

    Current best formula (from evolution experiments):
        clip(log((N + 0.5) / (df + 0.5)), 0, 8)
    """

    def __init__(self, max_idf: float = 8.0):
        self.max_idf = max_idf

    def compute(self, df: NDArray[np.float64], N: int) -> NDArray[np.float64]:
        # OpenEvolve can modify this formula
        idf = np.log((N + 0.5) / (df + 0.5))
        return np.clip(idf, 0.0, self.max_idf)


# =============================================================================
# TF Strategies (Evolvable)
# =============================================================================


class TFStrategy(Protocol):
    """Protocol for TF computation strategies."""

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
            norm: Document length normalization factor (1 - b + b * dl/avgdl).

        Returns:
            Saturated TF values.
        """
        ...


class ClassicTF:
    """
    Classic BM25 TF saturation.

    Formula: (tf * (k1 + 1)) / (tf + k1 * norm)
    """

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return (tf * (k1 + 1)) / (tf + k1 * norm + 1e-9)


class BM25LTF:
    """
    BM25L TF with delta boost for long documents.

    Formula:
        c = tf / norm
        tf_saturated = ((k1 + 1) * (c + delta)) / (k1 + c + delta)
    """

    def __init__(self, delta: float = 0.5):
        self.delta = delta

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        c = tf / (norm + 1e-9)
        c_delta = c + self.delta
        return ((k1 + 1) * c_delta) / (k1 + c_delta + 1e-9)


class BM25PlusTF:
    """
    BM25+ TF with lower-bound bonus.

    Formula: ((tf * (k1 + 1)) / (tf + k1 * norm)) + delta
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        base_tf = (tf * (k1 + 1)) / (tf + k1 * norm + 1e-9)
        # Only add delta where tf > 0
        return np.where(tf > 0, base_tf + self.delta, base_tf)


class ATIRETF:
    """
    ATIRE-style TF (mathematically equivalent to classic, different form).

    Formula: ((k1 + 1) * tf) / (k1 * norm + tf)
    """

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return ((k1 + 1) * tf) / (k1 * norm + tf + 1e-9)


# Default TF for evolution - this is the primary evolution target
class EvolvedTF:
    """
    Evolved TF strategy - primary OpenEvolve target.

    This TF variant can be modified by OpenEvolve to discover
    improved formulations.

    Current best formula (from evolution experiments):
        tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm)
        tf_sat = tf / (tf + k1 + 0.5)
        result = log(1 + tf_raw * tf_sat)
    """

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # OpenEvolve can modify this formula
        tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm + 1e-9)
        tf_sat = tf / (tf + k1 + 0.5)
        return np.log1p(tf_raw * tf_sat)


class DoubleLogTF:
    """
    TF_l∘δ∘p×IDF - Double-log TF saturation from Rousseau & Vazirgiannis (SIGIR 2013).

    Formula:
        c = tf / norm  (length-normalized tf)
        tf_component = (1 + log(1 + log(c))) + delta

    This applies double logarithmic scaling to model the non-linear gain
    of a term occurring multiple times in a document. The delta parameter
    ensures terms occurring at least once get a minimum boost.

    Reference: "Composition of TF normalizations: new insights on scoring
    functions for ad hoc IR" (SIGIR 2013)
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute(
        self,
        tf: NDArray[np.float64],
        k1: float,  # Not used in this formula, but kept for interface compatibility
        norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Length-normalized TF: c = tf / norm
        c = tf / (norm + 1e-9)

        # Double-log saturation with delta boost for matching terms
        # For tf > 0: (1 + log(1 + log(c))) + delta
        # For tf = 0: 0
        result = np.where(
            tf > 0,
            (1.0 + np.log1p(np.log1p(np.maximum(c, 1e-9)))) + self.delta,
            0.0,
        )
        return result


# =============================================================================
# Query TF Strategies (for query-side BM25)
# =============================================================================


class QueryTFStrategy(Protocol):
    """Protocol for query term frequency weighting strategies."""

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        """
        Compute query TF weights.

        Args:
            qtf: Query term frequency values.
            k3: Query TF saturation parameter.

        Returns:
            Query TF weights.
        """
        ...


class NoQueryTF:
    """
    No query TF weighting (bag-of-words style).

    All query terms weighted equally (binary presence).
    This is the default for most BM25 implementations.
    """

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        # Return 1.0 for all terms (no weighting)
        return np.ones_like(qtf)


class ClassicQueryTF:
    """
    Classic BM25 query TF saturation.

    Formula: (k3 + 1) * qtf / (k3 + qtf)

    This gives diminishing returns for repeated query terms.
    """

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        return ((k3 + 1) * qtf) / (k3 + qtf + 1e-9)


class LinearQueryTF:
    """
    Linear query TF weighting.

    Formula: qtf (raw frequency)

    Simple linear weighting - repeated terms get proportionally more weight.
    """

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        return qtf


class LogQueryTF:
    """
    Logarithmic query TF weighting.

    Formula: 1 + log(qtf) for qtf > 0, else 0

    Sublinear scaling for query term frequency.
    """

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        return np.where(qtf > 0, 1 + np.log(qtf), 0.0)


class EvolvedQueryTF:
    """
    Evolved query TF strategy - OpenEvolve target.

    Can be modified to discover optimal query term weighting.
    """

    def compute(
        self,
        qtf: NDArray[np.float64],
        k3: float,
    ) -> NDArray[np.float64]:
        # Default: classic saturation with evolved potential
        return ((k3 + 1) * qtf) / (k3 + qtf + 1e-9)


# =============================================================================
# Corpus
# =============================================================================


class Corpus:
    """
    A preprocessed collection of tokenized documents for BM25 ranking.

    This class pre-computes and caches corpus statistics needed for BM25
    scoring, including term frequencies, document frequencies, IDF values,
    and length normalization factors.

    Args:
        documents: List of tokenized documents (each document is a list of terms).
        ids: Optional list of document IDs corresponding to documents.

    Example:
        >>> docs = [["hello", "world"], ["hello", "there"]]
        >>> corpus = Corpus(docs, ids=["doc1", "doc2"])
        >>> len(corpus)
        2
    """

    def __init__(
        self,
        documents: list[list[str]],
        ids: list[str] | None = None,
        idf_strategy: IDFStrategy | None = None,
    ):
        self.documents = documents
        self.document_count = len(documents)
        self.ids = ids
        self._idf_strategy = idf_strategy or EvolvedIDF()

    def __len__(self) -> int:
        return self.document_count

    def __getitem__(self, index: int) -> list[str]:
        return self.documents[index]

    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.documents)

    @classmethod
    def from_huggingface_dataset(
        cls,
        dataset,
        idf_strategy: IDFStrategy | None = None,
    ) -> Corpus:
        """
        Create a Corpus from a HuggingFace dataset.

        Expects dataset rows to have 'id' and 'content' fields.

        Args:
            dataset: HuggingFace dataset with 'id' and 'content' columns.
            idf_strategy: Optional IDF strategy to use.

        Returns:
            Corpus instance.
        """
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids, idf_strategy)

    @cached_property
    def map_id_to_idx(self) -> dict[str, int]:
        """Mapping from document ID to index."""
        return {id_: idx for idx, id_ in enumerate(self.ids)} if self.ids else {}

    def id_to_idx(self, ids: list[str]) -> list[int]:
        """Convert document IDs to indices."""
        if not self.ids:
            raise ValueError("Corpus does not have document IDs.")
        return [self.map_id_to_idx[id_] for id_ in ids]

    @cached_property
    def term_frequency(self) -> list[Counter[str]]:
        """Term frequency counter for each document."""
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter[str]:
        """Document frequency of each term (number of documents containing term)."""
        return Counter(term for doc in self.documents for term in set(doc))

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        """Vocabulary mapping: term -> index."""
        return {term: idx for idx, term in enumerate(self.document_frequency.keys())}

    @cached_property
    def vocabulary_size(self) -> int:
        """Number of unique terms in corpus."""
        return len(self.vocabulary)

    @cached_property
    def document_length(self) -> NDArray[np.float64]:
        """Length of each document (number of terms)."""
        return np.array([len(doc) for doc in self.documents], dtype=np.float64)

    @cached_property
    def average_document_length(self) -> float:
        """Average document length in the corpus."""
        if self.document_count == 0:
            return 0.0
        return float(np.mean(self.document_length))

    @cached_property
    def df_array(self) -> NDArray[np.float64]:
        """Document frequency as numpy array (indexed by vocabulary)."""
        df = np.zeros(self.vocabulary_size, dtype=np.float64)
        for term, idx in self.vocabulary.items():
            df[idx] = self.document_frequency[term]
        return df

    @cached_property
    def idf_array(self) -> NDArray[np.float64]:
        """IDF values as numpy array (indexed by vocabulary)."""
        return self._idf_strategy.compute(self.df_array, self.document_count)

    @cached_property
    def inverse_document_frequency(self) -> dict[str, float]:
        """IDF values as dictionary (term -> idf)."""
        return {term: float(self.idf_array[idx]) for term, idx in self.vocabulary.items()}

    @cached_property
    def term_doc_matrix(self) -> sparse.csr_matrix:
        """
        Sparse term-document matrix (vocabulary_size x document_count).

        Each column is a document, each row is a term.
        Values are term frequencies.
        """
        rows = []
        cols = []
        data = []

        for doc_idx, tf_counter in enumerate(self.term_frequency):
            for term, freq in tf_counter.items():
                if term in self.vocabulary:
                    rows.append(self.vocabulary[term])
                    cols.append(doc_idx)
                    data.append(freq)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.vocabulary_size, self.document_count),
            dtype=np.float64,
        )


# =============================================================================
# BM25 Scorer
# =============================================================================


class BM25:
    """
    BM25 ranking implementation with modular IDF and TF strategies.

    This class computes BM25 scores for documents given a query. The IDF
    and TF computation strategies can be swapped to implement different
    BM25 variants (Classic, Lucene, BM25L, BM25+, ATIRE, etc.).

    Args:
        corpus: Pre-processed Corpus instance.
        k1: Term frequency saturation parameter (default: 1.5).
        b: Length normalization parameter (default: 0.75).
        tf_strategy: TF computation strategy (default: EvolvedTF).

    Example:
        >>> corpus = Corpus.from_huggingface_dataset(dataset)
        >>> bm25 = BM25(corpus, k1=1.5, b=0.75)
        >>> indices, scores = bm25.rank(tokenize("search query"), top_k=10)
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
        tf_strategy: TFStrategy | None = None,
    ):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tf_strategy = tf_strategy or EvolvedTF()

        # Pre-compute document normalization factors
        dl = self.corpus.document_length
        avgdl = self.corpus.average_document_length or 1.0
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
        Compute BM25 score for a single document given a query.

        This is the primary evolution target for OpenEvolve. The function
        computes the relevance score between a query and a document using
        the BM25 formula with evolved IDF and TF components.

        Args:
            query: List of query terms (tokenized).
            norm: Document length normalization factor (1 - b + b * dl/avgdl).
            frequencies: Term frequency counter for the document.
            idf: IDF values dictionary (term -> idf).
            k1: Term frequency saturation parameter.

        Returns:
            BM25 relevance score (float).
        """
        if not query:
            return 0.0

        # Use unique query terms (order-preserving)
        unique_terms = list(dict.fromkeys(query))

        if not unique_terms:
            return 0.0

        # Get term frequencies for query terms
        tf = np.array(
            [frequencies.get(term, 0) for term in unique_terms],
            dtype=np.float64,
        )

        # Early exit if no terms match
        if np.all(tf == 0):
            return 0.0

        # Get IDF values for query terms
        idf_values = np.array(
            [idf.get(term, 0.0) for term in unique_terms],
            dtype=np.float64,
        )

        # =================================================================
        # EVOLUTION TARGET: The scoring formula below can be modified
        # =================================================================

        # Evolved BM25 scoring formula
        # tf_raw: Standard BM25 TF saturation
        denom = tf + k1 * norm
        tf_raw = (tf * (k1 + 1.0)) / np.maximum(denom, 1e-9)

        # tf_sat: Additional saturation factor (evolved)
        tf_sat = tf / (tf + k1 + 0.5)

        # Combine with log damping (evolved)
        term_scores = idf_values * np.log1p(tf_raw * tf_sat)

        return float(np.sum(term_scores))

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25 score for a single document.

        Args:
            query: List of query terms (tokenized).
            index: Document index in corpus.

        Returns:
            BM25 relevance score.
        """
        return self.score_kernel(
            query,
            float(self._doc_norm[index]),
            self.corpus.term_frequency[index],
            self.corpus.inverse_document_frequency,
            self.k1,
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Rank all documents by relevance to query.

        Args:
            query: List of query terms (tokenized).
            top_k: Optional limit on number of results.

        Returns:
            Tuple of (sorted_indices, sorted_scores).
        """
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
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """
        Rank documents for multiple queries.

        Args:
            queries: List of tokenized queries.
            top_k: Optional limit on number of results per query.

        Returns:
            List of (sorted_indices, sorted_scores) tuples.
        """
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# Query-Side BM25 (with query term frequency weighting)
# =============================================================================


class BM25QuerySide:
    """
    BM25 with query-side term frequency weighting.

    This variant applies BM25-style saturation to query term frequencies,
    giving diminishing returns for repeated terms in the query. This is
    the full BM25 formula as originally described by Robertson et al.

    The scoring formula is:
        score(D, Q) = Σ IDF(t) × doc_tf_component × query_tf_component

    Where:
        - doc_tf_component = (tf * (k1 + 1)) / (tf + k1 * norm)
        - query_tf_component = (k3 + 1) * qtf / (k3 + qtf)

    Args:
        corpus: Pre-processed Corpus instance.
        k1: Document TF saturation parameter (default: 1.5).
        b: Length normalization parameter (default: 0.75).
        k3: Query TF saturation parameter (default: 8.0).
        query_tf_strategy: Strategy for query TF weighting.

    Example:
        >>> corpus = Corpus.from_huggingface_dataset(dataset)
        >>> bm25 = BM25QuerySide(corpus, k1=1.5, b=0.75, k3=8.0)
        >>> indices, scores = bm25.rank(tokenize("important important query"), top_k=10)
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
        k3: float = 8.0,
        query_tf_strategy: QueryTFStrategy | None = None,
    ):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.query_tf_strategy = query_tf_strategy or ClassicQueryTF()

        # Pre-compute document normalization factors
        dl = self.corpus.document_length
        avgdl = self.corpus.average_document_length or 1.0
        self._doc_norm = 1.0 - b + b * (dl / avgdl)

    @staticmethod
    def score_kernel_query_side(
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
        k1: float,
        k3: float,
    ) -> float:
        """
        Compute BM25 score with query-side term frequency weighting.

        This kernel includes query TF saturation, giving diminishing
        returns for repeated query terms.

        Args:
            query: List of query terms (tokenized, may have duplicates).
            norm: Document length normalization factor.
            frequencies: Term frequency counter for the document.
            idf: IDF values dictionary.
            k1: Document TF saturation parameter.
            k3: Query TF saturation parameter.

        Returns:
            BM25 relevance score with query weighting.
        """
        if not query:
            return 0.0

        # Count query term frequencies
        query_tf = Counter(query)
        unique_terms = list(query_tf.keys())

        if not unique_terms:
            return 0.0

        # Get document term frequencies
        doc_tf = np.array(
            [frequencies.get(term, 0) for term in unique_terms],
            dtype=np.float64,
        )

        # Early exit if no terms match
        if np.all(doc_tf == 0):
            return 0.0

        # Get query term frequencies
        qtf = np.array(
            [query_tf[term] for term in unique_terms],
            dtype=np.float64,
        )

        # Get IDF values
        idf_values = np.array(
            [idf.get(term, 0.0) for term in unique_terms],
            dtype=np.float64,
        )

        # =================================================================
        # Document-side TF saturation
        # =================================================================
        denom = doc_tf + k1 * norm
        doc_tf_component = (doc_tf * (k1 + 1.0)) / np.maximum(denom, 1e-9)

        # =================================================================
        # Query-side TF saturation
        # =================================================================
        query_tf_component = ((k3 + 1) * qtf) / (k3 + qtf + 1e-9)

        # =================================================================
        # Combined score: IDF × doc_TF × query_TF
        # =================================================================
        term_scores = idf_values * doc_tf_component * query_tf_component

        return float(np.sum(term_scores))

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25 score for a single document with query weighting.

        Args:
            query: List of query terms (tokenized).
            index: Document index in corpus.

        Returns:
            BM25 relevance score.
        """
        return self.score_kernel_query_side(
            query,
            float(self._doc_norm[index]),
            self.corpus.term_frequency[index],
            self.corpus.inverse_document_frequency,
            self.k1,
            self.k3,
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Rank all documents by relevance to query.

        Args:
            query: List of query terms (tokenized).
            top_k: Optional limit on number of results.

        Returns:
            Tuple of (sorted_indices, sorted_scores).
        """
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
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """
        Rank documents for multiple queries.

        Args:
            queries: List of tokenized queries.
            top_k: Optional limit on number of results per query.

        Returns:
            List of (sorted_indices, sorted_scores) tuples.
        """
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# BM25 with Pyserini-style Query Term Counting
# =============================================================================


class BM25PyseriniStyle:
    """
    BM25 with Pyserini/Lucene-style query term counting.

    This variant sums scores for ALL query term occurrences, NOT just unique terms.
    If "light" appears 4 times in the query, its IDF×TF contribution is added 4 times.

    This matches how Pyserini/Lucene computes BM25 scores, which differs from the
    typical "bag-of-words" interpretation where each unique term contributes once.

    Use this for:
    - Verifying behavior against Pyserini/Lucene
    - Understanding why Pyserini scores differ from bag-of-words implementations

    Note: For BRIGHT's long natural-language queries, this approach typically
    performs worse because query term repetition is incidental, not a relevance signal.

    Args:
        corpus: Pre-processed Corpus instance.
        k1: Term frequency saturation parameter (default: 1.2).
        b: Length normalization parameter (default: 0.75).
        tf_strategy: TF computation strategy (default: ClassicTF).

    Example:
        >>> corpus = Corpus.from_huggingface_dataset(dataset)
        >>> bm25 = BM25PyseriniStyle(corpus, k1=0.9, b=0.4)
        >>> indices, scores = bm25.rank(tokenize("light light heat"), top_k=10)
        # "light" contributes twice to the score
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float = 1.2,
        b: float = 0.75,
        tf_strategy: TFStrategy | None = None,
    ):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tf_strategy = tf_strategy or ClassicTF()

        # Pre-compute document normalization factors
        dl = self.corpus.document_length
        avgdl = self.corpus.average_document_length or 1.0
        self._doc_norm = 1.0 - b + b * (dl / avgdl)

    @staticmethod
    def score_kernel_pyserini_style(
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
        k1: float,
    ) -> float:
        """
        Compute BM25 score with Pyserini-style query term counting.

        This sums over ALL query term occurrences, not unique terms.

        Args:
            query: List of query terms (tokenized, may have duplicates).
            norm: Document length normalization factor.
            frequencies: Term frequency counter for the document.
            idf: IDF values dictionary.
            k1: Term frequency saturation parameter.

        Returns:
            BM25 relevance score (Pyserini-style).
        """
        if not query:
            return 0.0

        score = 0.0

        # Sum over ALL query term occurrences (including duplicates)
        for term in query:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            term_idf = idf.get(term, 0.0)
            if term_idf <= 0:
                continue

            # Standard BM25 TF saturation
            denom = tf + k1 * norm
            tf_component = (tf * (k1 + 1.0)) / max(denom, 1e-9)

            # Add to score (each occurrence adds separately)
            score += term_idf * tf_component

        return score

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25 score for a single document (Pyserini-style).

        Args:
            query: List of query terms (tokenized).
            index: Document index in corpus.

        Returns:
            BM25 relevance score.
        """
        return self.score_kernel_pyserini_style(
            query,
            float(self._doc_norm[index]),
            self.corpus.term_frequency[index],
            self.corpus.inverse_document_frequency,
            self.k1,
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Rank all documents by relevance to query.

        Args:
            query: List of query terms (tokenized).
            top_k: Optional limit on number of results.

        Returns:
            Tuple of (sorted_indices, sorted_scores).
        """
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
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """
        Rank documents for multiple queries.

        Args:
            queries: List of tokenized queries.
            top_k: Optional limit on number of results per query.

        Returns:
            List of (sorted_indices, sorted_scores) tuples.
        """
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# BM25-adpt (Adaptive Per-Term k1)
# =============================================================================


class BM25Adaptive:
    """
    BM25-adpt: Adaptive term frequency normalization.

    This variant computes a different k1 value for each term based on
    information gain, as proposed by Lv & Zhai (CIKM 2011).

    The optimal k1 for a term is found by minimizing:
        k1' = argmin_k1 Σ(G_r/G_1 - (k1+1)*r/(k1+r))²

    Where G_r is the information gain of a term occurring r times vs r-1 times.

    For terms where the optimal k1 is undefined (most low-df terms),
    we default to k1=0.001 following Trotman et al.

    Reference: "Adaptive term frequency normalization for BM25" (CIKM 2011)

    Args:
        corpus: Pre-processed Corpus instance.
        b: Length normalization parameter (default: 0.4).
        default_k1: Default k1 when optimal cannot be computed (default: 0.001).
    """

    def __init__(
        self,
        corpus: Corpus,
        b: float = 0.4,
        default_k1: float = 0.001,
    ):
        self.corpus = corpus
        self.b = b
        self.default_k1 = default_k1

        # Pre-compute document normalization factors
        dl = self.corpus.document_length
        avgdl = self.corpus.average_document_length or 1.0
        self._doc_norm = 1.0 - b + b * (dl / avgdl)

        # Pre-compute per-term optimal k1 and G1 values
        self._term_k1: dict[str, float] = {}
        self._term_g1: dict[str, float] = {}
        self._compute_adaptive_params()

    def _compute_adaptive_params(self) -> None:
        """
        Compute per-term k1 values based on information gain.

        Following Trotman et al.'s finding that ~90% of terms don't have
        a unique optimal k1, we use a simplified approach:
        - Use standard Robertson IDF as G1 (information gain of first occurrence)
        - Use default k1 for all terms (0.001 per Trotman et al.)

        The full Lv & Zhai optimization is complex and rarely improves results.
        """
        N = self.corpus.document_count
        df_array = self.corpus.df_array
        vocab = self.corpus.vocabulary

        # Use standard Robertson IDF as G1 (information gain of first occurrence)
        for term, term_idx in vocab.items():
            df = df_array[term_idx]
            # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5))
            # Clip to non-negative (like Lucene IDF)
            g1 = np.log((N - df + 0.5) / (df + 0.5))
            self._term_g1[term] = max(0.0, float(g1))
            # Default k1 for all terms (as per Trotman et al.)
            self._term_k1[term] = self.default_k1

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25-adpt score for a single document.

        Uses per-term adaptive k1 values instead of a global k1.
        """
        if not query:
            return 0.0

        unique_terms = list(dict.fromkeys(query))
        if not unique_terms:
            return 0.0

        norm = float(self._doc_norm[index])
        frequencies = self.corpus.term_frequency[index]

        score = 0.0
        for term in unique_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            # Get term-specific k1 and G1 (for IDF weighting)
            k1 = self._term_k1.get(term, self.default_k1)
            g1 = self._term_g1.get(term, 0.0)

            if g1 <= 0:
                continue

            # BM25-adpt TF component with adaptive k1
            tf_component = ((k1 + 1) * tf) / (tf + k1 * norm + 1e-9)

            # Use G1 as the IDF weight
            score += g1 * tf_component

        return score

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank all documents by relevance to query."""
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
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """Rank documents for multiple queries."""
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# Unified Configurable BM25
# =============================================================================


# Mapping from string names to IDF strategy classes
IDF_STRATEGIES: dict[str, type[IDFStrategy]] = {
    "classic": ClassicIDF,
    "lucene": LuceneIDF,
    "atire": ATIREIDF,
    "bm25l": BM25LIDF,
    "bm25+": BM25PlusIDF,
    "clipped": ClippedIDF,
    "evolved": EvolvedIDF,
}

# Mapping from string names to TF strategy classes
TF_STRATEGIES: dict[str, type[TFStrategy]] = {
    "classic": ClassicTF,
    "bm25l": BM25LTF,
    "bm25+": BM25PlusTF,
    "atire": ATIRETF,
    "evolved": EvolvedTF,
    "doublelog": DoubleLogTF,
}


class BM25Config:
    """
    Configuration for a unified BM25 scorer.

    This allows you to configure any combination of:
    - IDF strategy (how to weight rare vs common terms)
    - TF strategy (how to saturate term frequency)
    - Query term mode (how to handle repeated query terms)
    - Parameters (k1, b, k3, delta)

    Example:
        >>> config = BM25Config(
        ...     idf="lucene",
        ...     tf="classic",
        ...     query_mode="unique",
        ...     k1=0.9,
        ...     b=0.4,
        ... )
        >>> bm25 = BM25Unified(corpus, config)
    """

    def __init__(
        self,
        idf: str | IDFStrategy = "lucene",
        tf: str | TFStrategy = "classic",
        query_mode: str | QueryTermMode = "unique",
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 8.0,
        delta: float = 0.5,
    ):
        """
        Initialize BM25 configuration.

        Args:
            idf: IDF strategy name or instance.
                Options: "classic", "lucene", "atire", "bm25l", "bm25+", "clipped", "evolved"
            tf: TF strategy name or instance.
                Options: "classic", "bm25l", "bm25+", "atire", "evolved", "doublelog"
            query_mode: How to handle repeated query terms.
                Options: "unique" (bag-of-words), "sum_all" (Pyserini-style), "saturated"
            k1: TF saturation parameter (default: 1.2).
            b: Length normalization parameter (default: 0.75).
            k3: Query TF saturation parameter, used when query_mode="saturated" (default: 8.0).
            delta: Bonus parameter for BM25L/BM25+ TF strategies (default: 0.5).
        """
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.delta = delta

        # Parse IDF strategy
        if isinstance(idf, str):
            idf_lower = idf.lower()
            if idf_lower not in IDF_STRATEGIES:
                raise ValueError(
                    f"Unknown IDF strategy: {idf}. Options: {list(IDF_STRATEGIES.keys())}"
                )
            idf_cls = IDF_STRATEGIES[idf_lower]
            # Handle strategies that take parameters
            if idf_lower == "clipped":
                self.idf_strategy = idf_cls(max_idf=8.0)
            elif idf_lower == "evolved":
                self.idf_strategy = idf_cls(max_idf=8.0)
            else:
                self.idf_strategy = idf_cls()
        else:
            self.idf_strategy = idf

        # Parse TF strategy
        if isinstance(tf, str):
            tf_lower = tf.lower()
            if tf_lower not in TF_STRATEGIES:
                raise ValueError(
                    f"Unknown TF strategy: {tf}. Options: {list(TF_STRATEGIES.keys())}"
                )
            tf_cls = TF_STRATEGIES[tf_lower]
            # Handle strategies that take parameters
            if tf_lower == "bm25l":
                self.tf_strategy = tf_cls(delta=delta)
            elif tf_lower == "bm25+":
                self.tf_strategy = tf_cls(delta=delta)
            elif tf_lower == "doublelog":
                self.tf_strategy = tf_cls(delta=delta)
            else:
                self.tf_strategy = tf_cls()
        else:
            self.tf_strategy = tf

        # Parse query mode
        if isinstance(query_mode, str):
            query_mode_lower = query_mode.lower()
            mode_map = {
                "unique": QueryTermMode.UNIQUE,
                "sum_all": QueryTermMode.SUM_ALL,
                "saturated": QueryTermMode.SATURATED,
            }
            if query_mode_lower not in mode_map:
                raise ValueError(
                    f"Unknown query_mode: {query_mode}. Options: {list(mode_map.keys())}"
                )
            self.query_mode = mode_map[query_mode_lower]
        else:
            self.query_mode = query_mode

    def __repr__(self) -> str:
        return (
            f"BM25Config(idf={type(self.idf_strategy).__name__}, "
            f"tf={type(self.tf_strategy).__name__}, "
            f"query_mode={self.query_mode.value}, "
            f"k1={self.k1}, b={self.b})"
        )

    @classmethod
    def classic(cls, k1: float = 1.2, b: float = 0.75) -> BM25Config:
        """Classic Robertson BM25."""
        return cls(idf="classic", tf="classic", query_mode="unique", k1=k1, b=b)

    @classmethod
    def lucene(cls, k1: float = 0.9, b: float = 0.4) -> BM25Config:
        """Lucene-style BM25 (non-negative IDF)."""
        return cls(idf="lucene", tf="classic", query_mode="unique", k1=k1, b=b)

    @classmethod
    def atire(cls, k1: float = 1.2, b: float = 0.75) -> BM25Config:
        """ATIRE-style BM25."""
        return cls(idf="atire", tf="atire", query_mode="unique", k1=k1, b=b)

    @classmethod
    def bm25l(cls, k1: float = 1.2, b: float = 0.75, delta: float = 0.5) -> BM25Config:
        """BM25L (long document friendly)."""
        return cls(idf="bm25l", tf="bm25l", query_mode="unique", k1=k1, b=b, delta=delta)

    @classmethod
    def bm25_plus(cls, k1: float = 1.2, b: float = 0.75, delta: float = 1.0) -> BM25Config:
        """BM25+ (lower-bound bonus)."""
        return cls(idf="bm25+", tf="bm25+", query_mode="unique", k1=k1, b=b, delta=delta)

    @classmethod
    def pyserini(cls, k1: float = 0.9, b: float = 0.4) -> BM25Config:
        """Pyserini/Lucene-style (sums repeated query terms)."""
        return cls(idf="lucene", tf="classic", query_mode="sum_all", k1=k1, b=b)

    @classmethod
    def evolved(cls, k1: float = 1.5, b: float = 0.75) -> BM25Config:
        """Evolved BM25 (this project's best)."""
        return cls(idf="evolved", tf="evolved", query_mode="unique", k1=k1, b=b)

    @classmethod
    def doublelog(cls, k1: float = 1.2, b: float = 0.75, delta: float = 1.0) -> BM25Config:
        """
        TF_l∘δ∘p×IDF from Rousseau & Vazirgiannis (SIGIR 2013).

        Uses double-log TF saturation: 1 + log(1 + log(tf/norm)) + delta
        """
        return cls(idf="bm25+", tf="doublelog", query_mode="unique", k1=k1, b=b, delta=delta)


class BM25Unified:
    """
    Unified BM25 scorer with fully configurable IDF, TF, and query term handling.

    This class allows you to configure any BM25 variant by selecting:
    - IDF strategy: How to compute inverse document frequency
    - TF strategy: How to compute term frequency saturation
    - Query term mode: How to handle repeated terms in queries

    Example:
        >>> # Using preset configurations
        >>> bm25 = BM25Unified(corpus, BM25Config.lucene())
        >>> bm25 = BM25Unified(corpus, BM25Config.pyserini())
        >>> bm25 = BM25Unified(corpus, BM25Config.bm25l())

        >>> # Custom configuration
        >>> config = BM25Config(
        ...     idf="lucene",
        ...     tf="bm25+",
        ...     query_mode="unique",
        ...     k1=1.0,
        ...     b=0.5,
        ...     delta=1.0,
        ... )
        >>> bm25 = BM25Unified(corpus, config)

        >>> # Rank documents
        >>> indices, scores = bm25.rank(tokenize("search query"), top_k=10)
    """

    def __init__(
        self,
        corpus: Corpus,
        config: BM25Config | None = None,
    ):
        """
        Initialize unified BM25 scorer.

        Args:
            corpus: Pre-processed Corpus instance.
            config: BM25 configuration. Defaults to Lucene-style BM25.
        """
        self.config = config or BM25Config.lucene()

        # Rebuild corpus with the configured IDF strategy
        self.corpus = Corpus(
            corpus.documents,
            corpus.ids,
            idf_strategy=self.config.idf_strategy,
        )

        self.k1 = self.config.k1
        self.b = self.config.b
        self.k3 = self.config.k3
        self.tf_strategy = self.config.tf_strategy
        self.query_mode = self.config.query_mode

        # Pre-compute document normalization factors
        dl = self.corpus.document_length
        avgdl = self.corpus.average_document_length or 1.0
        self._doc_norm = 1.0 - self.b + self.b * (dl / avgdl)

    def __repr__(self) -> str:
        return f"BM25Unified({self.config})"

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25 score for a single document.

        Args:
            query: List of query terms (tokenized).
            index: Document index in corpus.

        Returns:
            BM25 relevance score.
        """
        if not query:
            return 0.0

        norm = float(self._doc_norm[index])
        frequencies = self.corpus.term_frequency[index]
        idf = self.corpus.inverse_document_frequency

        if self.query_mode == QueryTermMode.UNIQUE:
            return self._score_unique(query, norm, frequencies, idf)
        elif self.query_mode == QueryTermMode.SUM_ALL:
            return self._score_sum_all(query, norm, frequencies, idf)
        else:  # SATURATED
            return self._score_saturated(query, norm, frequencies, idf)

    def _score_unique(
        self,
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
    ) -> float:
        """Score using unique query terms (bag-of-words)."""
        unique_terms = list(dict.fromkeys(query))
        if not unique_terms:
            return 0.0

        tf = np.array([frequencies.get(term, 0) for term in unique_terms], dtype=np.float64)
        if np.all(tf == 0):
            return 0.0

        idf_values = np.array([idf.get(term, 0.0) for term in unique_terms], dtype=np.float64)

        # Compute TF component using configured strategy
        norm_array = np.full_like(tf, norm)
        tf_component = self.tf_strategy.compute(tf, self.k1, norm_array)

        return float(np.sum(idf_values * tf_component))

    def _score_sum_all(
        self,
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
    ) -> float:
        """Score summing over all query term occurrences (Pyserini-style)."""
        score = 0.0
        for term in query:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            term_idf = idf.get(term, 0.0)
            if term_idf <= 0:
                continue

            # Compute TF component
            tf_array = np.array([tf], dtype=np.float64)
            norm_array = np.array([norm], dtype=np.float64)
            tf_component = self.tf_strategy.compute(tf_array, self.k1, norm_array)[0]

            score += term_idf * tf_component

        return score

    def _score_saturated(
        self,
        query: list[str],
        norm: float,
        frequencies: Counter[str],
        idf: dict[str, float],
    ) -> float:
        """Score with saturated query term frequency weighting."""
        query_tf = Counter(query)
        unique_terms = list(query_tf.keys())
        if not unique_terms:
            return 0.0

        tf = np.array([frequencies.get(term, 0) for term in unique_terms], dtype=np.float64)
        if np.all(tf == 0):
            return 0.0

        idf_values = np.array([idf.get(term, 0.0) for term in unique_terms], dtype=np.float64)
        qtf = np.array([query_tf[term] for term in unique_terms], dtype=np.float64)

        # Compute TF component
        norm_array = np.full_like(tf, norm)
        tf_component = self.tf_strategy.compute(tf, self.k1, norm_array)

        # Compute query TF saturation: (k3 + 1) * qtf / (k3 + qtf)
        query_tf_component = ((self.k3 + 1) * qtf) / (self.k3 + qtf + 1e-9)

        return float(np.sum(idf_values * tf_component * query_tf_component))

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Rank all documents by relevance to query.

        Args:
            query: List of query terms (tokenized).
            top_k: Optional limit on number of results.

        Returns:
            Tuple of (sorted_indices, sorted_scores).
        """
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
    ) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """
        Rank documents for multiple queries.

        Args:
            queries: List of tokenized queries.
            top_k: Optional limit on number of results per query.

        Returns:
            List of (sorted_indices, sorted_scores) tuples.
        """
        return [self.rank(query, top_k) for query in queries]


# =============================================================================
# Convenience factory functions
# =============================================================================


def create_bm25_classic(corpus: Corpus, k1: float = 1.2, b: float = 0.75) -> BM25:
    """Create a classic BM25 scorer (Robertson et al. original)."""
    corpus_classic = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=ClassicIDF(),
    )
    return BM25(corpus_classic, k1=k1, b=b, tf_strategy=ClassicTF())


def create_bm25_lucene(corpus: Corpus, k1: float = 1.2, b: float = 0.75) -> BM25:
    """Create a Lucene-style BM25 scorer."""
    corpus_lucene = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=LuceneIDF(),
    )
    return BM25(corpus_lucene, k1=k1, b=b, tf_strategy=ClassicTF())


def create_bm25l(
    corpus: Corpus,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 0.5,
) -> BM25:
    """Create a BM25L scorer (long document correction)."""
    corpus_bm25l = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=BM25LIDF(),
    )
    return BM25(corpus_bm25l, k1=k1, b=b, tf_strategy=BM25LTF(delta=delta))


def create_bm25_plus(
    corpus: Corpus,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
) -> BM25:
    """Create a BM25+ scorer (lower-bound bonus)."""
    corpus_plus = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=BM25PlusIDF(),
    )
    return BM25(corpus_plus, k1=k1, b=b, tf_strategy=BM25PlusTF(delta=delta))


def create_bm25_atire(corpus: Corpus, k1: float = 1.2, b: float = 0.75) -> BM25:
    """Create an ATIRE-style BM25 scorer."""
    corpus_atire = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=ATIREIDF(),
    )
    return BM25(corpus_atire, k1=k1, b=b, tf_strategy=ATIRETF())


def create_bm25_query_side(
    corpus: Corpus,
    k1: float = 1.5,
    b: float = 0.75,
    k3: float = 8.0,
) -> BM25QuerySide:
    """
    Create a query-side BM25 scorer with query term frequency weighting.

    This is the full BM25 formula with query TF saturation. Use this when
    query terms may be repeated and you want diminishing returns for
    repeated terms.

    Args:
        corpus: Corpus instance.
        k1: Document TF saturation parameter (default: 1.5).
        b: Length normalization parameter (default: 0.75).
        k3: Query TF saturation parameter (default: 8.0).
            Higher k3 = more linear query TF weighting.
            Lower k3 = faster saturation.

    Returns:
        BM25QuerySide instance.
    """
    return BM25QuerySide(corpus, k1=k1, b=b, k3=k3)


def create_bm25_pyserini_style(
    corpus: Corpus,
    k1: float = 0.9,
    b: float = 0.4,
) -> BM25PyseriniStyle:
    """
    Create a BM25 scorer with Pyserini/Lucene-style query term counting.

    This sums over ALL query term occurrences (not unique terms).
    Use this for comparison with Pyserini benchmarks.

    Args:
        corpus: Corpus instance.
        k1: TF saturation parameter (default: 0.9, Pyserini default).
        b: Length normalization parameter (default: 0.4, Pyserini default).

    Returns:
        BM25PyseriniStyle instance.
    """
    corpus_lucene = Corpus(
        corpus.documents,
        corpus.ids,
        idf_strategy=LuceneIDF(),
    )
    return BM25PyseriniStyle(corpus_lucene, k1=k1, b=b)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Unified configurable BM25 (recommended)
    "BM25Unified",
    "BM25Config",
    "QueryTermMode",
    # Strategy mappings for introspection
    "IDF_STRATEGIES",
    "TF_STRATEGIES",
    # Legacy core classes (still supported)
    "BM25",
    "BM25QuerySide",
    "BM25PyseriniStyle",
    "Corpus",
    # Tokenization
    "tokenize",
    "LuceneTokenizer",
    "lucene_tokenize",
    "PorterStemmer",
    "ENGLISH_STOPWORDS",
    # IDF strategies
    "IDFStrategy",
    "ClassicIDF",
    "LuceneIDF",
    "ATIREIDF",
    "BM25LIDF",
    "BM25PlusIDF",
    "ClippedIDF",
    "EvolvedIDF",
    # Document TF strategies
    "TFStrategy",
    "ClassicTF",
    "BM25LTF",
    "BM25PlusTF",
    "ATIRETF",
    "EvolvedTF",
    "DoubleLogTF",
    # Adaptive BM25
    "BM25Adaptive",
    # Query TF strategies
    "QueryTFStrategy",
    "NoQueryTF",
    "ClassicQueryTF",
    "LinearQueryTF",
    "LogQueryTF",
    "EvolvedQueryTF",
    # Factory functions (legacy, prefer BM25Unified)
    "create_bm25_classic",
    "create_bm25_lucene",
    "create_bm25l",
    "create_bm25_plus",
    "create_bm25_atire",
    "create_bm25_query_side",
    "create_bm25_pyserini_style",
]
