"""
Reimplementation of Gensim BM25 to cross-verify our analysis.

We implement exactly what Gensim does, then compare with:
1. Actual Gensim output
2. Our correct BM25 implementation

This verifies our understanding of why Gensim underperforms.

Usage:
    uv run python -m benchmarks.gensim_reimplementation
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import BM25Config, BM25Unified, Corpus, tokenize
from ranking_evolved.metrics import mean_average_precision, mean_reciprocal_rank, ndcg_at_k


class GensimOkapiBM25Reimplemented:
    """
    Exact reimplementation of Gensim's OkapiBM25Model logic.

    This replicates Gensim's behavior to verify our analysis.
    """

    def __init__(
        self,
        documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.documents = documents
        self.N = len(documents)

        # Build vocabulary and document frequencies
        self.vocab: dict[str, int] = {}
        self.df: dict[int, int] = {}

        for doc in documents:
            seen = set()
            for term in doc:
                if term not in self.vocab:
                    term_id = len(self.vocab)
                    self.vocab[term] = term_id
                    self.df[term_id] = 0
                term_id = self.vocab[term]
                if term_id not in seen:
                    self.df[term_id] += 1
                    seen.add(term_id)

        self.id_to_term = {v: k for k, v in self.vocab.items()}

        # Compute document lengths and avgdl
        self.doc_lens = [len(doc) for doc in documents]
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1.0

        # Compute IDF values (Gensim OkapiBM25 style)
        self.idfs = self._compute_idfs()

        # Pre-compute document vectors (IDF × TF for each term)
        self.doc_vectors = self._compute_doc_vectors()

    def _compute_idfs(self) -> dict[int, float]:
        """Compute IDF values exactly as Gensim OkapiBM25Model does."""
        idfs = {}

        # First pass: compute raw IDFs
        for term_id, df in self.df.items():
            # Classic BM25 IDF: log((N - df + 0.5) / (df + 0.5))
            idf = np.log((self.N - df + 0.5) / (df + 0.5))
            idfs[term_id] = idf

        # Compute average IDF (only positive values)
        positive_idfs = [v for v in idfs.values() if v > 0]
        avg_idf = np.mean(positive_idfs) if positive_idfs else 0.0

        # Clamp negative IDFs to epsilon * avg_idf
        # NOTE: Gensim only clamps negative, not zero!
        for term_id in idfs:
            if idfs[term_id] < 0:
                idfs[term_id] = self.epsilon * avg_idf
            # Zero stays zero (this is the bug!)

        return idfs

    def _compute_doc_vectors(self) -> list[dict[int, float]]:
        """Compute BM25 weight vectors for all documents."""
        doc_vectors = []

        for doc_idx, doc in enumerate(self.documents):
            doc_len = self.doc_lens[doc_idx]
            norm = 1 - self.b + self.b * (doc_len / self.avgdl)

            # Count term frequencies
            tf_counter = Counter(doc)

            vec = {}
            for term, tf in tf_counter.items():
                term_id = self.vocab[term]
                idf = self.idfs.get(term_id, 0.0)

                # BM25 TF saturation: tf * (k1 + 1) / (tf + k1 * norm)
                tf_sat = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

                # Weight = IDF × TF
                weight = idf * tf_sat
                if weight != 0:  # Only store non-zero weights
                    vec[term_id] = weight

            doc_vectors.append(vec)

        return doc_vectors

    def get_query_vector(self, query: list[str]) -> dict[int, float]:
        """Compute BM25 weight vector for query (treated as a document)."""
        query_len = len(query)
        if query_len == 0:
            return {}

        # Gensim uses corpus avgdl for query normalization too!
        norm = 1 - self.b + self.b * (query_len / self.avgdl)

        tf_counter = Counter(query)
        vec = {}

        for term, tf in tf_counter.items():
            if term not in self.vocab:
                continue
            term_id = self.vocab[term]
            idf = self.idfs.get(term_id, 0.0)

            # BM25 TF saturation
            tf_sat = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

            weight = idf * tf_sat
            if weight != 0:
                vec[term_id] = weight

        return vec

    def score(self, query: list[str], doc_idx: int) -> float:
        """Compute score via dot product of query and document vectors."""
        query_vec = self.get_query_vector(query)
        doc_vec = self.doc_vectors[doc_idx]

        # Dot product
        score = 0.0
        for term_id, q_weight in query_vec.items():
            d_weight = doc_vec.get(term_id, 0.0)
            score += q_weight * d_weight

        return score

    def rank(self, query: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Rank all documents by query."""
        scores = np.array([self.score(query, i) for i in range(self.N)])
        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices]
        return sorted_indices, sorted_scores


class GensimLuceneBM25Reimplemented:
    """
    Exact reimplementation of Gensim's LuceneBM25Model logic.

    Key differences from Okapi:
    - Different IDF formula: log(N + 1) - log(df + 0.5)
    - Missing (k1 + 1) in TF numerator
    """

    def __init__(
        self,
        documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)

        # Build vocabulary and document frequencies
        self.vocab: dict[str, int] = {}
        self.df: dict[int, int] = {}

        for doc in documents:
            seen = set()
            for term in doc:
                if term not in self.vocab:
                    term_id = len(self.vocab)
                    self.vocab[term] = term_id
                    self.df[term_id] = 0
                term_id = self.vocab[term]
                if term_id not in seen:
                    self.df[term_id] += 1
                    seen.add(term_id)

        self.id_to_term = {v: k for k, v in self.vocab.items()}

        # Compute document lengths and avgdl
        self.doc_lens = [len(doc) for doc in documents]
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1.0

        # Compute IDF values (Gensim LuceneBM25 style)
        self.idfs = self._compute_idfs()

        # Pre-compute document vectors
        self.doc_vectors = self._compute_doc_vectors()

    def _compute_idfs(self) -> dict[int, float]:
        """Compute IDF values exactly as Gensim LuceneBM25Model does."""
        idfs = {}
        for term_id, df in self.df.items():
            # Gensim Lucene IDF: log(N + 1) - log(df + 0.5)
            idf = np.log(self.N + 1) - np.log(df + 0.5)
            idfs[term_id] = idf
        return idfs

    def _compute_doc_vectors(self) -> list[dict[int, float]]:
        """Compute BM25 weight vectors for all documents."""
        doc_vectors = []

        for doc_idx, doc in enumerate(self.documents):
            doc_len = self.doc_lens[doc_idx]
            norm = 1 - self.b + self.b * (doc_len / self.avgdl)

            tf_counter = Counter(doc)

            vec = {}
            for term, tf in tf_counter.items():
                term_id = self.vocab[term]
                idf = self.idfs.get(term_id, 0.0)

                # Gensim LuceneBM25 TF: tf / (tf + k1 * norm)
                # NOTE: Missing (k1 + 1) factor!
                tf_sat = tf / (tf + self.k1 * norm)

                weight = idf * tf_sat
                if weight != 0:
                    vec[term_id] = weight

            doc_vectors.append(vec)

        return doc_vectors

    def get_query_vector(self, query: list[str]) -> dict[int, float]:
        """Compute BM25 weight vector for query."""
        query_len = len(query)
        if query_len == 0:
            return {}

        norm = 1 - self.b + self.b * (query_len / self.avgdl)
        tf_counter = Counter(query)
        vec = {}

        for term, tf in tf_counter.items():
            if term not in self.vocab:
                continue
            term_id = self.vocab[term]
            idf = self.idfs.get(term_id, 0.0)

            # Missing (k1 + 1) factor
            tf_sat = tf / (tf + self.k1 * norm)

            weight = idf * tf_sat
            if weight != 0:
                vec[term_id] = weight

        return vec

    def score(self, query: list[str], doc_idx: int) -> float:
        """Compute score via dot product."""
        query_vec = self.get_query_vector(query)
        doc_vec = self.doc_vectors[doc_idx]

        score = 0.0
        for term_id, q_weight in query_vec.items():
            d_weight = doc_vec.get(term_id, 0.0)
            score += q_weight * d_weight

        return score

    def rank(self, query: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Rank all documents by query."""
        scores = np.array([self.score(query, i) for i in range(self.N)])
        sorted_indices = np.argsort(scores)[::-1].astype(np.int64)
        sorted_scores = scores[sorted_indices]
        return sorted_indices, sorted_scores


def verify_against_actual_gensim():
    """Verify our reimplementation matches actual Gensim."""
    print("=" * 70)
    print("VERIFICATION: Reimplementation vs Actual Gensim")
    print("=" * 70)

    try:
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model, OkapiBM25Model
    except ImportError:
        print("Gensim not installed. Skipping verification.")
        return

    # Test corpus
    docs = [
        ["the", "quick", "brown", "fox", "jumps"],
        ["the", "lazy", "dog", "sleeps", "all", "day"],
        ["quick", "brown", "bread", "with", "fox"],
        ["a", "rare", "term", "here", "only"],
    ]

    # Our reimplementation
    our_okapi = GensimOkapiBM25Reimplemented(docs, k1=1.5, b=0.75)
    our_lucene = GensimLuceneBM25Reimplemented(docs, k1=1.5, b=0.75)

    # Actual Gensim
    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    gensim_okapi = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75)
    gensim_lucene = LuceneBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75)

    # Compare IDF values
    print("\n--- OkapiBM25 IDF Comparison ---")
    print(f"{'Term':<15} {'Our IDF':>12} {'Gensim IDF':>12} {'Match':>8}")
    print("-" * 50)

    all_match = True
    for term, term_id in our_okapi.vocab.items():
        our_idf = our_okapi.idfs.get(term_id, 0)
        gensim_term_id = dictionary.token2id.get(term)
        gensim_idf = gensim_okapi.idfs.get(gensim_term_id, 0) if gensim_term_id is not None else 0

        match = "✓" if abs(our_idf - gensim_idf) < 1e-6 else "✗"
        if match == "✗":
            all_match = False
        print(f"{term:<15} {our_idf:>12.6f} {gensim_idf:>12.6f} {match:>8}")

    print(f"\nAll IDFs match: {all_match}")

    # Compare document vectors
    print("\n--- Document Vector Comparison (Doc 0) ---")
    our_vec = our_okapi.doc_vectors[0]
    gensim_vec = dict(gensim_okapi[bow_corpus[0]])

    print(f"{'Term':<15} {'Our Weight':>12} {'Gensim Weight':>14} {'Match':>8}")
    print("-" * 55)

    for term in docs[0]:
        our_term_id = our_okapi.vocab[term]
        gensim_term_id = dictionary.token2id[term]

        our_weight = our_vec.get(our_term_id, 0)
        gensim_weight = gensim_vec.get(gensim_term_id, 0)

        match = "✓" if abs(our_weight - gensim_weight) < 1e-6 else "✗"
        print(f"{term:<15} {our_weight:>12.6f} {gensim_weight:>14.6f} {match:>8}")

    # Compare query scores
    print("\n--- Query Score Comparison ---")
    query = ["fox", "lazy"]
    print(f"Query: {query}")

    our_query_vec = our_okapi.get_query_vector(query)
    gensim_query_vec = dict(gensim_okapi[dictionary.doc2bow(query)])

    print(f"\nQuery vectors:")
    print(f"  Our: {our_query_vec}")
    print(f"  Gensim: {gensim_query_vec}")

    print(f"\n{'Doc':>4} {'Our Score':>12} {'Gensim Score':>14} {'Match':>8}")
    print("-" * 45)

    for i in range(len(docs)):
        our_score = our_okapi.score(query, i)

        # Compute Gensim score via dot product
        doc_vec = dict(gensim_okapi[bow_corpus[i]])
        gensim_score = sum(gensim_query_vec.get(tid, 0) * doc_vec.get(tid, 0)
                          for tid in set(gensim_query_vec) | set(doc_vec))

        match = "✓" if abs(our_score - gensim_score) < 1e-6 else "✗"
        print(f"{i:>4} {our_score:>12.6f} {gensim_score:>14.6f} {match:>8}")


def evaluate_on_bright():
    """Evaluate all implementations on BRIGHT biology."""
    print("\n" + "=" * 70)
    print("EVALUATION: All Implementations on BRIGHT Biology")
    print("=" * 70)

    print("\nLoading data...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split="biology")
    examples = load_dataset("xlangai/BRIGHT", "examples", split="biology")

    # Build corpus
    corpus = Corpus.from_huggingface_dataset(documents)
    tokenized_docs = corpus.documents

    queries = [example["query"] for example in examples]
    gold_id_lists = [example["gold_ids"] for example in examples]
    gold_indices = [corpus.id_to_idx(ids) for ids in gold_id_lists]

    print(f"Corpus: {len(tokenized_docs)} documents, {len(queries)} queries")

    # Our correct BM25 - best configuration from README
    print("\nBuilding our BM25 (evolved TF, k1=0.9, b=0.4)...")
    config_evolved = BM25Config(idf="lucene", tf="evolved", query_mode="unique", k1=0.9, b=0.4)
    our_bm25_evolved = BM25Unified(corpus, config_evolved)

    # Also test classic TF for comparison
    print("Building our BM25 (classic TF, k1=1.5, b=0.75)...")
    config_classic = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=1.5, b=0.75)
    our_bm25_classic = BM25Unified(corpus, config_classic)

    # Reimplemented Gensim
    print("Building reimplemented Gensim Okapi...")
    reimpl_okapi = GensimOkapiBM25Reimplemented(tokenized_docs, k1=1.5, b=0.75)

    print("Building reimplemented Gensim Lucene...")
    reimpl_lucene = GensimLuceneBM25Reimplemented(tokenized_docs, k1=1.5, b=0.75)

    # Actual Gensim (if available)
    try:
        from benchmarks.baselines.gensim_bm25 import (
            GensimLuceneBM25Baseline,
            GensimOkapiBM25Baseline,
        )
        print("Building actual Gensim baselines...")
        actual_okapi = GensimOkapiBM25Baseline.from_corpus(corpus, k1=1.5, b=0.75)
        actual_lucene = GensimLuceneBM25Baseline.from_corpus(corpus, k1=1.5, b=0.75)
        has_gensim = True
    except ImportError:
        has_gensim = False
        print("Gensim not available, skipping actual Gensim evaluation")

    # Evaluate each
    implementations = [
        ("Our BM25 evolved (k1=0.9,b=0.4)", our_bm25_evolved),
        ("Our BM25 classic (k1=1.5,b=0.75)", our_bm25_classic),
        ("Reimpl Gensim Okapi", reimpl_okapi),
        ("Reimpl Gensim Lucene", reimpl_lucene),
    ]

    if has_gensim:
        implementations.extend([
            ("Actual Gensim Okapi", actual_okapi),
            ("Actual Gensim Lucene", actual_lucene),
        ])

    results = {}

    for name, impl in implementations:
        print(f"\nEvaluating {name}...")
        ndcg_scores = []
        all_relevant = []
        all_retrieved = []

        for i, (query_text, gold) in enumerate(zip(queries, gold_indices)):
            query_tokens = tokenize(query_text)
            ranked_indices, _ = impl.rank(query_tokens)

            relevant = np.array(gold, dtype=np.int64)
            retrieved = np.array(ranked_indices, dtype=np.int64)

            all_relevant.append(relevant)
            all_retrieved.append(retrieved)
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, 10))

            if i % 20 == 0:
                print(f"  Query {i}/{len(queries)}...")

        results[name] = {
            "ndcg_at_k": float(np.mean(ndcg_scores)),
            "map": mean_average_precision(all_relevant, all_retrieved),
            "mrr": mean_reciprocal_rank(all_relevant, all_retrieved),
        }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Implementation':<25} {'NDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 60)

    for name, metrics in sorted(results.items(), key=lambda x: -x[1]["ndcg_at_k"]):
        print(f"{name:<25} {metrics['ndcg_at_k']:>10.4f} {metrics['map']:>10.4f} {metrics['mrr']:>10.4f}")

    # Verify reimplementation matches actual
    if has_gensim:
        print("\n" + "=" * 70)
        print("VERIFICATION: Reimplementation matches Actual Gensim?")
        print("=" * 70)

        okapi_match = abs(results["Reimpl Gensim Okapi"]["ndcg_at_k"] -
                          results["Actual Gensim Okapi"]["ndcg_at_k"]) < 0.01
        lucene_match = abs(results["Reimpl Gensim Lucene"]["ndcg_at_k"] -
                           results["Actual Gensim Lucene"]["ndcg_at_k"]) < 0.01

        print(f"Okapi:  {'✓ MATCH' if okapi_match else '✗ MISMATCH'}")
        print(f"Lucene: {'✓ MATCH' if lucene_match else '✗ MISMATCH'}")


def main():
    verify_against_actual_gensim()
    evaluate_on_bright()


if __name__ == "__main__":
    main()
