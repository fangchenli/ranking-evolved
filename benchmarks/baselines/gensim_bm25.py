"""
Gensim BM25 Baseline Wrappers.

This module provides wrappers around Gensim's BM25 models for use
as baselines in benchmarks. It implements the same interface as the
main BM25 class for seamless integration with the benchmark framework.

Requires: gensim >= 4.3.0

Available models (using proper dot product scoring):
- GensimOkapiBM25Baseline: Uses OkapiBM25Model (classic BM25 with IDF clamping)
- GensimAtireBM25Baseline: Uses AtireBM25Model (ATIRE variant)
- GensimLuceneBM25Baseline: Uses LuceneBM25Model (has formula bugs - for comparison)

Note: These wrappers use scipy sparse matrix dot product instead of Gensim's
SparseMatrixSimilarity (which incorrectly uses cosine similarity).

Known Gensim issues:
- LuceneBM25Model: Wrong IDF formula + missing (k1+1) in TF
- OkapiBM25Model: Clamps negative IDF to epsilon*avg_idf (over-weights common terms)

Usage:
    from benchmarks.baselines.gensim_bm25 import (
        GensimOkapiBM25Baseline,   # Recommended
        GensimAtireBM25Baseline,
    )

    baseline = GensimOkapiBM25Baseline.from_corpus(corpus, k1=1.5, b=0.75)
    indices, scores = baseline.rank(query_tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ranking_evolved.bm25 import Corpus


class GensimOkapiBM25Baseline:
    """
    Wrapper around Gensim's OkapiBM25Model for benchmarking.

    OkapiBM25Model uses the classic BM25 formula:
    - IDF: log((N - df + 0.5) / (df + 0.5)) with negative IDF clamped to epsilon*avg_idf
    - TF: tf * (k1 + 1) / (tf + k1 * norm)

    This wrapper uses proper dot product scoring (not cosine similarity).
    """

    def __init__(
        self,
        doc_matrix: sparse.csr_matrix,
        model,
        dictionary,
        corpus: Corpus,
    ):
        self._doc_matrix = doc_matrix
        self._model = model
        self._dictionary = dictionary
        self._corpus = corpus
        self._num_terms = len(dictionary)

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> GensimOkapiBM25Baseline:
        """Create a Gensim Okapi BM25 baseline from a Corpus."""
        try:
            from gensim.corpora import Dictionary
            from gensim.models import OkapiBM25Model
        except ImportError as e:
            raise ImportError(
                "Gensim is required for this baseline. "
                "Install with: uv add gensim --group benchmark"
            ) from e

        dictionary = Dictionary(corpus.documents)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus.documents]

        model = OkapiBM25Model(
            corpus=bow_corpus,
            dictionary=dictionary,
            k1=k1,
            b=b,
        )

        # Transform corpus to BM25 weighted vectors
        transformed = list(model[bow_corpus])

        # Build sparse matrix (documents x terms) with BM25 weights
        num_docs = len(corpus)
        num_terms = len(dictionary)

        rows, cols, data = [], [], []
        for doc_idx, doc_vec in enumerate(transformed):
            for term_id, weight in doc_vec:
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)

        doc_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_terms))

        return cls(doc_matrix, model, dictionary, corpus)

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank all documents by relevance to query using dot product."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = self._model[query_bow]

        # Build sparse query vector
        if not query_vec:
            # No known terms in query
            scores = np.zeros(self._doc_matrix.shape[0])
        else:
            q_cols = [term_id for term_id, _ in query_vec]
            q_data = [weight for _, weight in query_vec]
            q_rows = [0] * len(q_cols)
            query_sparse = sparse.csr_matrix((q_data, (q_rows, q_cols)), shape=(1, self._num_terms))

            # Dot product: query × doc_matrix.T
            scores = (query_sparse @ self._doc_matrix.T).toarray().flatten()

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices].astype(np.float64)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = dict(self._model[query_bow])
        doc_vec = self._doc_matrix[index].toarray().flatten()

        score = 0.0
        for term_id, q_weight in query_vec.items():
            score += q_weight * doc_vec[term_id]
        return score


class GensimAtireBM25Baseline:
    """
    Wrapper around Gensim's AtireBM25Model for benchmarking.

    AtireBM25Model uses the ATIRE BM25 formula:
    - IDF: log(N / df)
    - TF: tf * (k1 + 1) / (tf + k1 * norm)

    This wrapper uses proper dot product scoring (not cosine similarity).
    """

    def __init__(
        self,
        doc_matrix: sparse.csr_matrix,
        model,
        dictionary,
        corpus: Corpus,
    ):
        self._doc_matrix = doc_matrix
        self._model = model
        self._dictionary = dictionary
        self._corpus = corpus
        self._num_terms = len(dictionary)

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> GensimAtireBM25Baseline:
        """Create a Gensim ATIRE BM25 baseline from a Corpus."""
        try:
            from gensim.corpora import Dictionary
            from gensim.models import AtireBM25Model
        except ImportError as e:
            raise ImportError(
                "Gensim is required for this baseline. "
                "Install with: uv add gensim --group benchmark"
            ) from e

        dictionary = Dictionary(corpus.documents)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus.documents]

        model = AtireBM25Model(
            corpus=bow_corpus,
            dictionary=dictionary,
            k1=k1,
            b=b,
        )

        # Transform corpus to BM25 weighted vectors
        transformed = list(model[bow_corpus])

        # Build sparse matrix (documents x terms) with BM25 weights
        num_docs = len(corpus)
        num_terms = len(dictionary)

        rows, cols, data = [], [], []
        for doc_idx, doc_vec in enumerate(transformed):
            for term_id, weight in doc_vec:
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)

        doc_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_terms))

        return cls(doc_matrix, model, dictionary, corpus)

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank all documents by relevance to query using dot product."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = self._model[query_bow]

        # Build sparse query vector
        if not query_vec:
            scores = np.zeros(self._doc_matrix.shape[0])
        else:
            q_cols = [term_id for term_id, _ in query_vec]
            q_data = [weight for _, weight in query_vec]
            q_rows = [0] * len(q_cols)
            query_sparse = sparse.csr_matrix((q_data, (q_rows, q_cols)), shape=(1, self._num_terms))

            # Dot product: query × doc_matrix.T
            scores = (query_sparse @ self._doc_matrix.T).toarray().flatten()

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices].astype(np.float64)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = dict(self._model[query_bow])
        doc_vec = self._doc_matrix[index].toarray().flatten()

        score = 0.0
        for term_id, q_weight in query_vec.items():
            score += q_weight * doc_vec[term_id]
        return score


class GensimLuceneBM25Baseline:
    """
    Wrapper around Gensim's LuceneBM25Model for benchmarking.

    WARNING: LuceneBM25Model has implementation bugs:
    - Wrong IDF: uses log((N+1)/(df+0.5)) instead of log(1+(N-df+0.5)/(df+0.5))
    - Missing (k1+1) in TF: uses tf/(tf+k1*norm) instead of tf*(k1+1)/(tf+k1*norm)

    This wrapper is provided for comparison purposes only.
    """

    def __init__(
        self,
        doc_matrix: sparse.csr_matrix,
        model,
        dictionary,
        corpus: Corpus,
    ):
        self._doc_matrix = doc_matrix
        self._model = model
        self._dictionary = dictionary
        self._corpus = corpus
        self._num_terms = len(dictionary)

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> GensimLuceneBM25Baseline:
        """Create a Gensim Lucene BM25 baseline from a Corpus."""
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LuceneBM25Model
        except ImportError as e:
            raise ImportError(
                "Gensim is required for this baseline. "
                "Install with: uv add gensim --group benchmark"
            ) from e

        dictionary = Dictionary(corpus.documents)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus.documents]

        model = LuceneBM25Model(
            corpus=bow_corpus,
            dictionary=dictionary,
            k1=k1,
            b=b,
        )

        # Transform corpus to BM25 weighted vectors
        transformed = list(model[bow_corpus])

        # Build sparse matrix (documents x terms) with BM25 weights
        num_docs = len(corpus)
        num_terms = len(dictionary)

        rows, cols, data = [], [], []
        for doc_idx, doc_vec in enumerate(transformed):
            for term_id, weight in doc_vec:
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)

        doc_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, num_terms))

        return cls(doc_matrix, model, dictionary, corpus)

    def rank(
        self,
        query: list[str],
        top_k: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank all documents by relevance to query using dot product."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = self._model[query_bow]

        # Build sparse query vector
        if not query_vec:
            scores = np.zeros(self._doc_matrix.shape[0])
        else:
            q_cols = [term_id for term_id, _ in query_vec]
            q_data = [weight for _, weight in query_vec]
            q_rows = [0] * len(q_cols)
            query_sparse = sparse.csr_matrix((q_data, (q_rows, q_cols)), shape=(1, self._num_terms))

            # Dot product: query × doc_matrix.T
            scores = (query_sparse @ self._doc_matrix.T).toarray().flatten()

        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices].astype(np.float64)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        query_bow = self._dictionary.doc2bow(query)
        query_vec = dict(self._model[query_bow])
        doc_vec = self._doc_matrix[index].toarray().flatten()

        score = 0.0
        for term_id, q_weight in query_vec.items():
            score += q_weight * doc_vec[term_id]
        return score


# Legacy alias for backwards compatibility
GensimBM25Baseline = GensimLuceneBM25Baseline


def evaluate_gensim_on_bright(
    domain: str = "biology",
    k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    model: str = "okapi",
) -> dict:
    """
    Standalone evaluation of Gensim BM25 on BRIGHT.

    Args:
        domain: BRIGHT split to evaluate.
        k: Cutoff for @k metrics.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
        model: Which gensim model to use ("okapi", "atire", or "lucene").

    Returns:
        Dictionary of metrics.
    """
    from datasets import load_dataset

    from ranking_evolved.bm25 import Corpus, tokenize
    from ranking_evolved.metrics import (
        average_precision,
        mean_average_precision,
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )

    # Load data
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)

    # Build corpus
    corpus = Corpus.from_huggingface_dataset(documents)

    # Create baseline
    if model == "okapi":
        baseline = GensimOkapiBM25Baseline.from_corpus(corpus, k1=k1, b=b)
    elif model == "atire":
        baseline = GensimAtireBM25Baseline.from_corpus(corpus, k1=k1, b=b)
    else:
        baseline = GensimLuceneBM25Baseline.from_corpus(corpus, k1=k1, b=b)

    # Prepare queries
    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    # Evaluate
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    rr_scores = []
    ap_scores = []
    all_relevant = []
    all_retrieved = []

    for query_text, gold in zip(queries, gold_indices, strict=False):
        query_tokens = tokenize(query_text)
        ranked_indices, _ = baseline.rank(query_tokens)

        relevant = np.array(gold, dtype=np.int64)
        retrieved = np.array(ranked_indices, dtype=np.int64)

        all_relevant.append(relevant)
        all_retrieved.append(retrieved)

        precision_scores.append(precision_at_k(relevant, retrieved, k))
        recall_scores.append(recall_at_k(relevant, retrieved, k))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))
        rr_scores.append(reciprocal_rank(relevant, retrieved))
        ap_scores.append(average_precision(relevant, retrieved))

    return {
        "domain": domain,
        "model": model,
        "k": k,
        "k1": k1,
        "b": b,
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "precision_at_k": float(np.mean(precision_scores)),
        "recall_at_k": float(np.mean(recall_scores)),
        "map": mean_average_precision(all_relevant, all_retrieved),
        "mrr": mean_reciprocal_rank(all_relevant, all_retrieved),
        "num_queries": len(queries),
    }


if __name__ == "__main__":
    import json

    for model in ["okapi", "atire", "lucene"]:
        results = evaluate_gensim_on_bright(domain="biology", k=10, model=model)
        print(f"\n{model.upper()}:")
        print(json.dumps(results, indent=2))
