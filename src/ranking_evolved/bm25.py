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
        """
        Term frequency for each document.

        Returns:
            list[Counter[str]]: A list of Counters, one per document, mapping terms to frequency.
        """
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter[str]:
        """
        Document frequency of each term (i.e. in how many documents each term appears).

        Returns:
            Counter[str]: Mapping from term to number of documents it appears in.
        """
        return Counter(term for doc in self.documents for term in set(doc))

    @cached_property
    def document_length(self) -> np.ndarray:
        """
        Length of each document in the corpus.

        Returns:
            np.ndarray: Array of document lengths.
        """
        return np.array([len(doc) for doc in self.documents])

    @cached_property
    def average_document_length(self) -> float:
        """
        Average number of terms per document.

        Returns:
            float: The average document length.
        """
        return np.mean(self.document_length)

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        """
        Vocabulary mapping: assigns each term a unique integer ID.

        Returns:
            dict[str, int]: Mapping from term to index.
        """
        return {term: idx for idx, term in enumerate(self.document_frequency.keys())}

    @cached_property
    def inverse_document_frequency(self) -> dict[str, float]:
        """
        Inverse document frequency (IDF) for each term.

        Formula:
            idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Returns:
            dict[str, float]: Mapping from term to IDF value.
        """
        df_values = np.array(list(self.document_frequency.values()))
        idf = np.log((self.document_count - df_values + 0.5) / (df_values + 0.5) + 1)
        return {
            term: idf_value
            for term, idf_value in zip(self.document_frequency.keys(), idf)
        }

    def id_to_idx(self, ids: list[str]) -> list[int]:
        if not self.ids:
            raise ValueError("Corpus does not have document IDs.")
        return [self.map_id_to_idx[id] for id in ids]


class BM25:
    """
    BM25 ranking class using a preprocessed Corpus.

    Args:
        corpus (Corpus): A corpus object containing document statistics.
        k1 (float): Term frequency saturation parameter.
        b (float): Length normalization parameter.
    """

    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus

    @staticmethod
    def score_kernel(
        query: list[str],
        document_length: int,
        frequencies: Counter,
        idf: dict[str, float],
        k1: float,
        b: float,
        avg_doc_length: float,
    ) -> float:
        """
        Computes BM25 score for a single document and query.

        Returns:
            float: Score of the document for the given query.
        """
        frequency_list = np.array([frequencies.get(term, 0) for term in query])
        idf_values = np.array([idf.get(term, 0) for term in query])
        numerator = frequency_list * (k1 + 1)
        denominator = frequency_list + k1 * (
            1 - b + b * document_length / avg_doc_length
        )
        scores = idf_values * (numerator / denominator)
        return np.sum(scores)

    def score(self, query: list[str], index: int) -> float:
        doc_len = self.corpus.document_length[index]

        frequencies = self.corpus.term_frequency[index]
        return self.score_kernel(
            query,
            doc_len,
            frequencies,
            self.corpus.inverse_document_frequency,
            self.k1,
            self.b,
            self.corpus.average_document_length,
        )

    def rank(self, query: list[str]) -> tuple[np.ndarray, np.ndarray]:
        scores = np.array([self.score(query, idx) for idx in range(len(self.corpus))])
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices, scores[sorted_indices]
