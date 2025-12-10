from collections import Counter
from functools import cached_property
from typing import Iterator
import re

import numpy as np


# --- Tokenization -----------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase and split on contiguous word characters."""
    return re.findall(r"\w+", text.lower())


# --- Corpus -----------------------------------------------------------------


class Corpus:
    """
    Preprocessed collection of tokenized documents for BM25-style ranking.
    Interface compatible with previous versions.
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
        ids = [row["id"] for row in dataset]
        documents = [tokenize(row["content"]) for row in dataset]
        return cls(documents, ids)

    @cached_property
    def map_id_to_idx(self) -> dict[str, int]:
        return {id_: idx for idx, id_ in enumerate(self.ids)} if self.ids else {}

    @cached_property
    def term_frequency(self) -> list[Counter[str]]:
        """Term frequency per document."""
        return [Counter(doc) for doc in self.documents]

    @cached_property
    def document_frequency(self) -> Counter:
        """In how many documents each term appears."""
        df = Counter()
        for doc in self.documents:
            if doc:
                df.update(set(doc))
        return df

    @cached_property
    def document_length(self) -> np.ndarray:
        """Length of each document in tokens."""
        return np.fromiter((len(doc) for doc in self.documents), dtype=np.float32)

    @cached_property
    def average_document_length(self) -> float:
        dl = self.document_length
        return float(dl.mean()) if len(dl) else 0.0

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        """Mapping from term to integer index."""
        return {t: i for i, t in enumerate(self.document_frequency.keys())}

    @cached_property
    def inverse_document_frequency(self) -> dict[str, float]:
        """Softened BM25-style IDF tuned for BRIGHT."""
        N = self.document_count
        if N == 0:
            return {}

        df_vals = np.fromiter(self.document_frequency.values(), dtype=np.float32)

        num = N - df_vals + 0.5
        den = df_vals + 0.5
        base = np.log(np.maximum(num / np.maximum(den, 1e-9), 1e-9))
        idf = np.clip(base + 0.63, 0.07, 5.0)

        df_ratio = df_vals / max(N, 1.0)
        mid_mask = (df_ratio > 0.015) & (df_ratio < 0.20)
        idf[mid_mask] += 0.05

        terms = list(self.document_frequency.keys())
        return {t: float(v) for t, v in zip(terms, idf)}

    def id_to_idx(self, ids: list[str]) -> list[int]:
        if not self.ids:
            raise ValueError("Corpus does not have document IDs.")
        mp = self.map_id_to_idx
        return [mp[i] for i in ids]


# --- BM25 Ranker ------------------------------------------------------------


class BM25:
    """
    BM25 ranking class using a preprocessed Corpus.

    Interface identical to the baseline.

    BRIGHT tuning objectives:
    - Improve psychology ndcg/MAP/MRR via coverage-friendly scoring.
    - Keep strong domains (including math/code) stable.
    - Avoid making weak math splits worse by capping repetition gains.
    """

    def __init__(self, corpus: Corpus, k1: float = 1.31, b: float = 0.64):
        # Parameters chosen to stay aligned with historically strong configs.
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus = corpus

        dl = self.corpus.document_length.astype(np.float32)
        avg_dl = max(self.corpus.average_document_length, 1e-9)

        # Standard BM25 length normalization, slightly gentle (friendlier to
        # longer psych explanations but still protective for math).
        self._doc_norm = (1.0 - self.b) + self.b * (dl / avg_dl)

        # Tiny log-length based adjustment: nudges away from extreme lengths.
        # Effect is a near-1.0 multiplicative factor so regressions are unlikely.
        if len(dl):
            log_dl = np.log1p(np.maximum(dl, 0.0))
            m = float(log_dl.mean())
            s = float(log_dl.std() + 1e-9)
            z = (log_dl - m) / s
            # Bounded to roughly [-0.03, 0.03].
            self._len_adj = np.clip(z * 0.02, -0.03, 0.03).astype(np.float32)
        else:
            self._len_adj = np.zeros_like(dl, dtype=np.float32)

    # --- Query helpers -------------------------------------------------------

    @staticmethod
    def _unique_terms(query: list[str]) -> list[str]:
        """Order‑preserving de-duplication of query terms."""
        if not query:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for t in query:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def _coverage_boost(tf: np.ndarray, n_terms: int) -> float:
        """
        Small, bounded bonus for matching more distinct query terms.

        Helps multi-aspect psychology queries; effect is tiny so math splits
        (short queries, few aspects) remain stable.
        """
        if n_terms <= 0:
            return 1.0
        matched = (tf > 0).astype(np.float32)
        cov = matched.sum()
        if cov <= 0:
            return 1.0
        frac = cov / float(n_terms)
        # Slightly asymmetric: reward high coverage a bit more.
        return 1.0 + (0.11 if frac >= 0.75 else 0.07) * frac

    # --- Core scoring --------------------------------------------------------

    @staticmethod
    def _tf_component(tf: np.ndarray, norm: float, k1: float) -> np.ndarray:
        """BM25 TF with extra mild saturation."""
        denom = tf + k1 * norm
        base = (tf * (k1 + 1.0)) / np.maximum(denom, 1e-9)

        # Extra soft saturation leaning slightly toward coverage over repetition.
        tf_soft = tf / (tf + k1 + 0.75)
        return base * (0.60 + 0.40 * tf_soft)

    @staticmethod
    def score_kernel(
        query: list[str],
        norm: float,
        frequencies: Counter,
        idf: dict[str, float],
        k1: float,
        len_adj: float = 0.0,
    ) -> float:
        """
        Compute BM25-style score for a single document and query.

        Differences vs vanilla BM25:
        - Order‑preserving unique query terms.
        - Extra TF saturation and log1p damping to avoid repetition dominance.
        - Small coverage bonus for documents that match more distinct terms.
        - Tiny length-based multiplicative adjustment (near 1.0 factor).
        """
        terms = BM25._unique_terms(query)
        if not terms:
            return 0.0

        tf = np.fromiter((frequencies.get(t, 0) for t in terms), dtype=np.float32)
        if not tf.any():
            return 0.0

        idf_vals = np.fromiter((idf.get(t, 0.0) for t in terms), dtype=np.float32)

        tf_part = BM25._tf_component(tf, norm, k1) * BM25._coverage_boost(
            tf, len(terms)
        )
        score_vec = idf_vals * np.log1p(tf_part)
        score = float(score_vec.sum())

        if len_adj:
            score *= 1.0 + float(len_adj)

        return score

    # --- Public API ----------------------------------------------------------

    def score(self, query: list[str], index: int) -> float:
        tf = self.corpus.term_frequency[index]
        return self.score_kernel(
            query=query,
            norm=float(self._doc_norm[index]),
            frequencies=tf,
            idf=self.corpus.inverse_document_frequency,
            k1=self.k1,
            len_adj=float(self._len_adj[index]),
        )

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_docs = len(self.corpus)
        if n_docs == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        scores = np.empty(n_docs, dtype=np.float32)
        tf_list = self.corpus.term_frequency
        idf = self.corpus.inverse_document_frequency
        norms = self._doc_norm
        len_adj = self._len_adj
        k1 = self.k1

        for idx in range(n_docs):
            scores[idx] = self.score_kernel(
                query=query,
                norm=float(norms[idx]),
                frequencies=tf_list[idx],
                idf=idf,
                k1=k1,
                len_adj=float(len_adj[idx]),
            )

        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[:top_k]
        return order, scores[order]
