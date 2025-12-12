"""
BM25 implementation using the classic Okapi BM25 equations (as in gensim's OkapiBM25Model),
but exposed with the same API shape as the other in-repo implementations.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Iterator
import math
import re

import numpy as np


def tokenize(text: str) -> list[str]:
    """Lowercase and split on contiguous word characters."""
    return re.findall(r"\w+", text.lower())


class Corpus:
    """
    Tokenized corpus for BM25.
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
        ids = [row.get("id") for row in dataset]
        documents = [tokenize(row["content"]) for row in dataset]
        return cls(documents, ids)

    @cached_property
    def term_frequency(self) -> list[Counter[str]]:
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter[str]:
        return Counter(term for doc in self.documents for term in set(doc))

    @cached_property
    def document_length(self) -> np.ndarray:
        return np.array([len(doc) for doc in self.documents], dtype=np.float32)

    @cached_property
    def average_document_length(self) -> float:
        dl = self.document_length
        return float(dl.mean()) if len(dl) else 0.0

    # @cached_property
    # def inverse_document_frequency(self) -> dict[str, float]:
    #     """
    #     Classic BM25 IDF:
    #         idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
    #     """
    #     df = np.array(list(self.document_frequency.values()), dtype=np.float32)
    #     idf = np.log(np.maximum((self.document_count - df + 0.5) / (df + 0.5), 1e-9))
    #     return {t: float(v) for t, v in zip(self.document_frequency.keys(), idf)}

    def id_to_idx(self, ids: list[str]) -> list[int]:
        if not self.ids:
            raise ValueError("Corpus does not have document IDs.")
        mp = {id_: idx for idx, id_ in enumerate(self.ids)}
        return [mp[i] for i in ids]


class BM25:
    """Lucene-style BM25 scorer (BM25Similarity formulation)."""

    def __init__(self, corpus: Corpus, k1: float = 1.2, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        dl = corpus.document_length
        avg_dl = corpus.average_document_length or 1e-9
        self._norm = 1.0 - b + b * (dl / avg_dl)
        self._idf = self._precompute_idf(corpus.document_frequency, len(corpus))

    def _precompute_idf(self, dfs: Counter[str], num_docs: int) -> dict[str, float]:
        idfs: dict[str, float] = {}
        for term, freq in dfs.items():
            idf = math.log(num_docs + 1.0) - math.log(freq + 0.5)
            idfs[term] = idf
        return idfs

    def score(self, query: list[str], index: int) -> float:
        tf = self.corpus.term_frequency[index]
        norm = float(self._norm[index])
        k1 = self.k1
        scores = []
        for term in query:
            tf_ij = tf.get(term, 0)
            if tf_ij == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf_ij + k1 * norm
            scores.append(idf * (tf_ij / denom))
        return float(np.sum(scores)) if scores else 0.0

    def rank(self, query: list[str], top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.array([self.score(query, idx) for idx in range(len(self.corpus))], dtype=float)
        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[:top_k]
        return order, scores[order]
