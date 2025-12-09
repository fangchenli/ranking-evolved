from collections import Counter
from functools import cached_property
from typing import Iterator
import re

import numpy as np


def tokenize(text: str) -> list[str]:
    """Tokenizes the input text into a list of terms."""
    return re.findall(r"\w+", text.lower())


class Corpus:
    """
    A preprocessed collection of tokenized documents for use in ranking algorithms like BM25.

    Args:
        documents (List[List[str]]): List of tokenized documents. Each document is a list of terms.

    Attributes:
        documents (List[List[str]]): The raw tokenized documents.
        document_count (int): Total number of documents in the corpus.
        ids (List[str] | None): Optional list of document IDs corresponding to the documents.
    """

    def __init__(self, documents: list[list[str]], ids: list[str] | None = None):
        self.documents = documents
        self.document_count = len(documents)
        self.ids = ids

    def __len__(self) -> int:
        return self.document_count

    def __getitem__(self, index: int) -> list[str]:
        return self.documents[index]

    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.documents)

    @classmethod
    def from_huggingface_dataset(cls, dataset) -> "Corpus":
        ids = [doc["id"] for doc in dataset]
        documents = [tokenize(doc["content"]) for doc in dataset]
        return cls(documents, ids)

    @cached_property
    def map_id_to_idx(self) -> dict[str, int]:
        return {id: idx for idx, id in enumerate(self.ids)} if self.ids else {}

    @cached_property
    def term_frequency(self) -> list[Counter[str]]:
        """Term frequency for each document."""
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter[str]:
        """Document frequency of each term (i.e. in how many documents each term appears)."""
        return Counter(term for doc in self.documents for term in set(doc))

    @cached_property
    def document_length(self) -> np.ndarray:
        """Length of each document in the corpus."""
        return np.array([len(doc) for doc in self.documents])

    @cached_property
    def average_document_length(self) -> float:
        """Average number of terms per document."""
        return float(np.mean(self.document_length)) if len(self.document_length) else 0.0

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        """Vocabulary mapping: assigns each term a unique integer ID."""
        return {term: idx for idx, term in enumerate(self.document_frequency.keys())}

    @cached_property
    def inverse_document_frequency(self) -> dict[str, float]:
        """
        Classic BM25 IDF:
            idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        """
        df_values = np.array(list(self.document_frequency.values()), dtype=float)
        idf = np.log((self.document_count - df_values + 0.5) / (df_values + 0.5) + 1.0)
        return {term: float(idf_value) for term, idf_value in zip(self.document_frequency.keys(), idf)}

    def id_to_idx(self, ids: list[str]) -> list[int]:
        if not self.ids:
            raise ValueError("Corpus does not have document IDs.")
        return [self.map_id_to_idx[id] for id in ids]


class BM25:
    """
    Classic BM25 ranking class using a preprocessed Corpus.

    Args:
        corpus (Corpus): A corpus object containing document statistics.
        k1 (float): Term frequency saturation parameter.
        b (float): Length normalization parameter.
    """

    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        dl = self.corpus.document_length.astype(np.float32)
        avg_dl = float(self.corpus.average_document_length) or 1e-9
        self._doc_norm = 1.0 - b + b * (dl / avg_dl)

    @staticmethod
    def score_kernel(
        query: list[str],
        norm: float,
        frequencies: Counter,
        idf: dict[str, float],
        k1: float,
    ) -> float:
        """Classic BM25 score for a single document and query."""
        if not query:
            return 0.0
        tf = np.array([frequencies.get(term, 0) for term in query], dtype=float)
        if np.all(tf == 0):
            return 0.0

        idf_values = np.array([idf.get(term, 0.0) for term in query], dtype=float)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * norm
        scores = idf_values * (numerator / np.maximum(denominator, 1e-9))
        return float(np.sum(scores))

    def score(self, query: list[str], index: int) -> float:
        frequencies = self.corpus.term_frequency[index]
        return self.score_kernel(
            query,
            float(self._doc_norm[index]),
            frequencies,
            self.corpus.inverse_document_frequency,
            self.k1,
        )

    def rank(self, query: list[str], top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.array([self.score(query, idx) for idx in range(len(self.corpus))])
        sorted_indices = np.argsort(scores)[::-1]
        scores_sorted = scores[sorted_indices]
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            scores_sorted = scores_sorted[:top_k]
        return sorted_indices, scores_sorted
