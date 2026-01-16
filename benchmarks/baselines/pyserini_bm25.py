"""
Pyserini BM25 Baseline Wrapper.

This module provides a wrapper around Pyserini's Lucene-based BM25 implementation
for use as a baseline in benchmarks. Pyserini provides high-quality, well-tested
BM25 scoring through the Anserini IR toolkit.

Requires:
    - pyserini >= 0.25.0
    - Java 21 (Pyserini's Lucene backend requires Java)

Environment setup (macOS with Homebrew):
    export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib

Note: Due to the Java dependency, this baseline is optional and may not be
available in all environments.

IMPORTANT: Tokenization Behavior
--------------------------------
Pyserini's LuceneIndexer uses DefaultEnglishAnalyzer which applies:
- Porter stemming (e.g., "respiration" -> "respir")
- Stopword removal (e.g., "is", "the", "of" are removed)

This means:
1. Documents are tokenized by Lucene during indexing
2. Queries are tokenized by Lucene during search
3. Pre-tokenized input will be RE-tokenized (double tokenization issue)

For fair comparison, use evaluate_pyserini_on_bright_raw() which passes
raw text to Pyserini, letting Lucene handle all tokenization consistently.

Cross-validation results (Biology domain, BRIGHT dataset):
- Pyserini (k1=0.9, b=0.4):  NDCG@10 = 0.1810
- Pyserini (k1=1.2, b=0.75): NDCG@10 = 0.0793
- Our BM25 + Lucene tok:     NDCG@10 = 0.2524 (k1=0.9, b=0.4)

The ~28% gap is due to differences in avgdl calculation, query parsing,
or other Lucene internals.

Usage:
    from benchmarks.baselines.pyserini_bm25 import (
        PyseriniBM25Baseline,
        evaluate_pyserini_on_bright_raw,
    )

    # For benchmarking with raw text (recommended)
    results = evaluate_pyserini_on_bright_raw(domain="biology", k=10)

    # For API-compatible wrapper (note: has double-tokenization issues)
    baseline = PyseriniBM25Baseline.from_corpus(corpus, k1=0.9, b=0.4)
    indices, scores = baseline.rank(query_tokens)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ranking_evolved.bm25 import Corpus


class PyseriniBM25Baseline:
    """
    Wrapper around Pyserini's Lucene BM25 for benchmarking.

    This class provides the same interface as the BM25 class to allow
    direct comparison in benchmarks. Pyserini uses Lucene's battle-tested
    BM25 implementation under the hood.

    Note: This baseline requires Java 21 to be installed and available
    in the system PATH.
    """

    def __init__(
        self,
        searcher,
        doc_ids: list[str],
        id_to_idx: dict[str, int],
        corpus: Corpus,
        index_dir: Path,
    ):
        """
        Initialize wrapper (use from_corpus() factory instead).

        Args:
            searcher: Pyserini LuceneSearcher instance.
            doc_ids: List of document IDs in corpus order.
            id_to_idx: Mapping from doc ID to corpus index.
            corpus: Original Corpus for reference.
            index_dir: Temporary index directory (for cleanup).
        """
        self._searcher = searcher
        self._doc_ids = doc_ids
        self._id_to_idx = id_to_idx
        self._corpus = corpus
        self._index_dir = index_dir
        self._num_docs = len(doc_ids)

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        k1: float = 0.9,
        b: float = 0.4,
    ) -> PyseriniBM25Baseline:
        """
        Create a Pyserini BM25 baseline from a Corpus.

        This method:
        1. Creates a temporary directory for the Lucene index
        2. Converts corpus documents to JSONL format
        3. Builds a Lucene index using Pyserini
        4. Creates a searcher with configured BM25 parameters

        Args:
            corpus: Pre-tokenized Corpus instance.
            k1: BM25 k1 parameter (default: 0.9, Pyserini's default).
            b: BM25 b parameter (default: 0.4, Pyserini's default).

        Returns:
            PyseriniBM25Baseline instance.
        """
        try:
            from pyserini.index.lucene import LuceneIndexer
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as e:
            raise ImportError(
                "Pyserini is required for this baseline. "
                "Install with: pip install pyserini\n"
                "Note: Pyserini requires Java 21 to be installed."
            ) from e

        # Create temporary directory for index
        index_dir = Path(tempfile.mkdtemp(prefix="pyserini_index_"))

        try:
            # Convert corpus to dict format for indexing
            doc_ids = corpus.ids if corpus.ids else [str(i) for i in range(len(corpus))]

            # Build index (append=False creates new index)
            indexer = LuceneIndexer(str(index_dir / "index"), append=False)

            # Add documents in batches
            batch_size = 1000
            for i in range(0, len(corpus), batch_size):
                batch = []
                for j in range(i, min(i + batch_size, len(corpus))):
                    doc_text = " ".join(corpus.documents[j])
                    batch.append(
                        {
                            "id": doc_ids[j],
                            "contents": doc_text,
                        }
                    )
                indexer.add_batch_dict(batch)
            indexer.close()

            # Create searcher
            searcher = LuceneSearcher(str(index_dir / "index"))
            searcher.set_bm25(k1, b)

            # Build ID mapping
            id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

            return cls(searcher, doc_ids, id_to_idx, corpus, index_dir)

        except Exception as e:
            # Clean up on failure
            shutil.rmtree(index_dir, ignore_errors=True)
            raise e

    def __del__(self):
        """Clean up temporary index directory."""
        if hasattr(self, "_index_dir") and self._index_dir.exists():
            shutil.rmtree(self._index_dir, ignore_errors=True)

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
        # Join tokens back to query string
        query_text = " ".join(query)

        # Search with Pyserini (returns up to k results)
        k = top_k if top_k is not None else self._num_docs
        hits = self._searcher.search(query_text, k=min(k, self._num_docs))

        # Convert hits to indices and scores
        if not hits:
            # No results - return all docs with zero scores
            indices = np.arange(self._num_docs, dtype=np.int64)
            scores = np.zeros(self._num_docs, dtype=np.float64)
            return indices, scores

        # Build result arrays
        result_indices = []
        result_scores = []
        seen_indices = set()

        for hit in hits:
            doc_id = hit.docid
            if doc_id in self._id_to_idx:
                idx = self._id_to_idx[doc_id]
                if idx not in seen_indices:
                    result_indices.append(idx)
                    result_scores.append(hit.score)
                    seen_indices.add(idx)

        # Add remaining documents with zero score if not using top_k
        if top_k is None:
            for idx in range(self._num_docs):
                if idx not in seen_indices:
                    result_indices.append(idx)
                    result_scores.append(0.0)

        sorted_indices = np.array(result_indices, dtype=np.int64)
        sorted_scores = np.array(result_scores, dtype=np.float64)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        return sorted_indices, sorted_scores

    def score(self, query: list[str], index: int) -> float:
        """
        Compute BM25 score for a single document.

        Note: This is inefficient for single document scoring.
        For batch scoring, use rank() instead.

        Args:
            query: List of query terms (tokenized).
            index: Document index in corpus.

        Returns:
            BM25 relevance score.
        """
        indices, scores = self.rank(query)
        pos = np.where(indices == index)[0]
        if len(pos) > 0:
            return float(scores[pos[0]])
        return 0.0


def check_java_available() -> bool:
    """Check if Java is available in the system PATH."""
    import subprocess

    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def evaluate_pyserini_on_bright(
    domain: str = "biology",
    k: int = 10,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict:
    """
    Standalone evaluation of Pyserini BM25 on BRIGHT.

    Args:
        domain: BRIGHT split to evaluate.
        k: Cutoff for @k metrics.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.

    Returns:
        Dictionary of metrics.
    """
    if not check_java_available():
        raise RuntimeError(
            "Java is not available. Pyserini requires Java 21. "
            "Please install Java and ensure it's in your PATH."
        )

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
    baseline = PyseriniBM25Baseline.from_corpus(corpus, k1=k1, b=b)

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


def evaluate_pyserini_on_bright_raw(
    domain: str = "biology",
    k: int = 10,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict:
    """
    Evaluate Pyserini BM25 on BRIGHT using RAW text (recommended).

    This function passes raw document text to Pyserini, letting Lucene's
    DefaultEnglishAnalyzer handle all tokenization consistently for both
    documents and queries. This avoids double-tokenization issues.

    Args:
        domain: BRIGHT split to evaluate.
        k: Cutoff for @k metrics.
        k1: BM25 k1 parameter (default 0.9, Pyserini's default).
        b: BM25 b parameter (default 0.4, Pyserini's default).

    Returns:
        Dictionary of metrics.
    """
    if not check_java_available():
        raise RuntimeError(
            "Java is not available. Pyserini requires Java 21. "
            "Please install Java and ensure it's in your PATH."
        )

    try:
        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as e:
        raise ImportError("Pyserini is required. Install with: uv sync --group benchmark") from e

    from datasets import load_dataset

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

    # Extract raw text and IDs
    raw_texts = []
    doc_ids = []
    for doc in documents:
        content = doc.get("content") or doc.get("text") or ""
        doc_id = doc.get("id") or doc.get("_id")
        raw_texts.append(content)
        doc_ids.append(doc_id)

    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]

    # Create temp directory for index
    index_dir = Path(tempfile.mkdtemp(prefix="pyserini_raw_"))

    try:
        # Build index with RAW text
        indexer = LuceneIndexer(str(index_dir / "index"), append=False)
        batch_size = 1000
        for i in range(0, len(doc_ids), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(doc_ids))):
                batch.append({"id": doc_ids[j], "contents": raw_texts[j]})
            indexer.add_batch_dict(batch)
        indexer.close()

        # Build ID mapping
        id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        # Create searcher
        searcher = LuceneSearcher(str(index_dir / "index"))
        searcher.set_bm25(k1, b)

        # Evaluate
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        rr_scores = []
        ap_scores = []
        all_relevant = []
        all_retrieved = []

        for query_text, gold_ids in zip(queries, gold_id_lists, strict=False):
            gold_indices = [id_to_idx[gid] for gid in gold_ids if gid in id_to_idx]

            # Search with RAW query text
            hits = searcher.search(query_text, k=len(doc_ids))

            # Convert to indices
            retrieved = []
            seen = set()
            for hit in hits:
                if hit.docid in id_to_idx:
                    idx = id_to_idx[hit.docid]
                    if idx not in seen:
                        retrieved.append(idx)
                        seen.add(idx)

            # Pad with remaining docs
            for idx in range(len(doc_ids)):
                if idx not in seen:
                    retrieved.append(idx)

            relevant = np.array(gold_indices, dtype=np.int64)
            retrieved = np.array(retrieved, dtype=np.int64)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

            precision_scores.append(precision_at_k(relevant, retrieved, k))
            recall_scores.append(recall_at_k(relevant, retrieved, k))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))
            rr_scores.append(reciprocal_rank(relevant, retrieved))
            ap_scores.append(average_precision(relevant, retrieved))

        return {
            "domain": domain,
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

    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    if check_java_available():
        print("Evaluating Pyserini BM25 on BRIGHT (biology domain)...")
        print("\n=== Raw Text Evaluation (recommended) ===")

        for k1, b in [(0.9, 0.4), (1.2, 0.75), (1.5, 0.75)]:
            results = evaluate_pyserini_on_bright_raw(domain="biology", k=10, k1=k1, b=b)
            print(
                f"k1={k1}, b={b}: NDCG@10={results['ndcg_at_k']:.4f}, "
                f"MAP={results['map']:.4f}, MRR={results['mrr']:.4f}"
            )
    else:
        print("Java not available. Pyserini baseline requires Java 21.")
        print("Setup instructions:")
        print("  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
        print("  export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib")
