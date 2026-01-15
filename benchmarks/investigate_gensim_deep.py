"""
Deep investigation of Gensim BM25 on actual BRIGHT data.

Compare document rankings between our implementation and Gensim
to see where they diverge.

Usage:
    uv run python -m benchmarks.investigate_gensim_deep
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset

from ranking_evolved.bm25 import BM25Config, BM25Unified, Corpus, tokenize


def compare_rankings():
    """Compare rankings between our implementation and Gensim on real data."""
    print("Loading BRIGHT biology dataset...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split="biology")
    examples = load_dataset("xlangai/BRIGHT", "examples", split="biology")

    print("Building corpus...")
    corpus = Corpus.from_huggingface_dataset(documents)

    # Our BM25 with classic TF (to match Gensim more closely)
    config = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=1.5, b=0.75)
    our_bm25 = BM25Unified(corpus, config)

    # Gensim BM25
    try:
        from benchmarks.baselines.gensim_bm25 import GensimOkapiBM25Baseline
        gensim_bm25 = GensimOkapiBM25Baseline.from_corpus(corpus, k1=1.5, b=0.75)
    except ImportError:
        print("Gensim not installed")
        return

    # Pick a few queries to analyze
    queries_to_analyze = [0, 1, 2]

    for q_idx in queries_to_analyze:
        example = examples[q_idx]
        query_text = example["query"]
        gold_ids = example["gold_ids"]
        gold_indices = set(corpus.id_to_idx(gold_ids))

        query_tokens = tokenize(query_text)

        print(f"\n{'='*70}")
        print(f"Query {q_idx}: {query_text[:100]}...")
        print(f"Query length: {len(query_tokens)} tokens")
        print(f"Unique tokens: {len(set(query_tokens))}")
        print(f"Gold documents: {len(gold_indices)}")

        # Get rankings
        our_indices, our_scores = our_bm25.rank(query_tokens)
        gensim_indices, gensim_scores = gensim_bm25.rank(query_tokens)

        # Compare top 10
        print(f"\n{'Rank':<6} | {'Our BM25':^20} | {'Gensim':^20} | {'Match?':^8}")
        print("-" * 60)

        our_top10 = set(our_indices[:10])
        gensim_top10 = set(gensim_indices[:10])

        for rank in range(10):
            our_idx = our_indices[rank]
            gensim_idx = gensim_indices[rank]

            our_gold = "✓" if our_idx in gold_indices else ""
            gensim_gold = "✓" if gensim_idx in gold_indices else ""

            match = "✓" if our_idx == gensim_idx else "✗"

            print(f"{rank+1:<6} | {our_idx:>8} {our_gold:<2} ({our_scores[rank]:>7.2f}) | {gensim_idx:>8} {gensim_gold:<2} ({gensim_scores[rank]:>7.2f}) | {match:^8}")

        # Count gold in top 10
        our_gold_in_top10 = len(our_top10 & gold_indices)
        gensim_gold_in_top10 = len(gensim_top10 & gold_indices)
        overlap = len(our_top10 & gensim_top10)

        print(f"\nGold docs in top 10: Ours={our_gold_in_top10}, Gensim={gensim_gold_in_top10}")
        print(f"Top 10 overlap: {overlap}/10")

        # Analyze score distribution
        print(f"\nScore statistics:")
        print(f"  Our scores:    min={our_scores[-1]:.4f}, max={our_scores[0]:.4f}, mean={our_scores.mean():.4f}")
        print(f"  Gensim scores: min={gensim_scores[-1]:.4f}, max={gensim_scores[0]:.4f}, mean={gensim_scores.mean():.4f}")


def analyze_term_contributions():
    """Analyze how individual terms contribute to scores."""
    print("\n" + "=" * 70)
    print("TERM CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Simple test case
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fox is a small wild animal with red fur",
        "The lazy dog sleeps all day long",
        "Quick brown bread with fox shaped cookies",
    ]

    tokenized_docs = [tokenize(d) for d in docs]
    corpus = Corpus(tokenized_docs, ids=[f"doc{i}" for i in range(len(docs))])

    # Our BM25
    config = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=1.5, b=0.75)
    our_bm25 = BM25Unified(corpus, config)

    # Gensim BM25
    try:
        from gensim.corpora import Dictionary
        from gensim.models import OkapiBM25Model
    except ImportError:
        print("Gensim not installed")
        return

    dictionary = Dictionary(tokenized_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    gensim_model = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75)

    query = ["fox", "lazy"]
    print(f"\nQuery: {query}")
    print(f"\nDocuments:")
    for i, doc in enumerate(docs):
        print(f"  Doc {i}: {doc}")

    print("\n--- Our BM25 ---")
    for i in range(len(docs)):
        score = our_bm25.score(query, i)
        print(f"Doc {i} score: {score:.4f}")

    print("\n--- Gensim BM25 (dot product) ---")
    query_bow = dictionary.doc2bow(query)
    query_vec = dict(gensim_model[query_bow])

    print(f"\nQuery vector: {dict(gensim_model[query_bow])}")
    for term_id, weight in query_vec.items():
        print(f"  '{dictionary[term_id]}': {weight:.4f}")

    for i, doc_bow in enumerate(bow_corpus):
        doc_vec = dict(gensim_model[doc_bow])

        score = 0.0
        details = []
        for term_id, q_weight in query_vec.items():
            d_weight = doc_vec.get(term_id, 0.0)
            contribution = q_weight * d_weight
            score += contribution
            if d_weight > 0:
                details.append(f"'{dictionary[term_id]}': {q_weight:.4f}×{d_weight:.4f}={contribution:.4f}")

        print(f"Doc {i} score: {score:.4f}  ({', '.join(details) if details else 'no matches'})")


def analyze_ranking_preservation():
    """Check if Gensim preserves ranking order or changes it."""
    print("\n" + "=" * 70)
    print("RANKING PRESERVATION ANALYSIS")
    print("=" * 70)

    print("Loading BRIGHT biology...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split="biology")
    examples = load_dataset("xlangai/BRIGHT", "examples", split="biology")

    corpus = Corpus.from_huggingface_dataset(documents)

    config = BM25Config(idf="lucene", tf="classic", query_mode="unique", k1=1.5, b=0.75)
    our_bm25 = BM25Unified(corpus, config)

    try:
        from benchmarks.baselines.gensim_bm25 import GensimOkapiBM25Baseline
        gensim_bm25 = GensimOkapiBM25Baseline.from_corpus(corpus, k1=1.5, b=0.75)
    except ImportError:
        print("Gensim not installed")
        return

    # Compute metrics for first 20 queries
    from ranking_evolved.metrics import ndcg_at_k

    our_ndcg = []
    gensim_ndcg = []
    kendall_tau = []

    for i in range(min(20, len(examples))):
        example = examples[i]
        query_tokens = tokenize(example["query"])
        gold_indices = np.array(corpus.id_to_idx(example["gold_ids"]), dtype=np.int64)

        our_indices, _ = our_bm25.rank(query_tokens)
        gensim_indices, _ = gensim_bm25.rank(query_tokens)

        our_ndcg.append(ndcg_at_k(gold_indices, our_indices, 10))
        gensim_ndcg.append(ndcg_at_k(gold_indices, gensim_indices, 10))

        # Compute rank correlation (Kendall tau) for top 100
        from scipy.stats import kendalltau
        our_top100 = our_indices[:100]
        gensim_top100 = gensim_indices[:100]

        # Find common documents
        common = set(our_top100) & set(gensim_top100)
        if len(common) > 10:
            our_ranks = {idx: r for r, idx in enumerate(our_top100) if idx in common}
            gensim_ranks = {idx: r for r, idx in enumerate(gensim_top100) if idx in common}
            common_list = list(common)
            our_r = [our_ranks[idx] for idx in common_list]
            gensim_r = [gensim_ranks[idx] for idx in common_list]
            tau, _ = kendalltau(our_r, gensim_r)
            kendall_tau.append(tau)

    print(f"\nResults over first 20 queries:")
    print(f"  Our NDCG@10 mean:    {np.mean(our_ndcg):.4f}")
    print(f"  Gensim NDCG@10 mean: {np.mean(gensim_ndcg):.4f}")
    print(f"  Kendall tau mean:    {np.mean(kendall_tau):.4f} (1.0 = identical ranking)")

    print(f"\nPer-query comparison:")
    print(f"{'Query':<8} | {'Our NDCG':>10} | {'Gensim NDCG':>12} | {'Tau':>8}")
    print("-" * 50)
    for i in range(min(10, len(our_ndcg))):
        tau = kendall_tau[i] if i < len(kendall_tau) else float('nan')
        print(f"{i:<8} | {our_ndcg[i]:>10.4f} | {gensim_ndcg[i]:>12.4f} | {tau:>8.4f}")


def main():
    analyze_term_contributions()
    analyze_ranking_preservation()
    compare_rankings()


if __name__ == "__main__":
    main()
