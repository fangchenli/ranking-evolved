"""
Investigate why Gensim BM25 underperforms.

The IDF² effect changes weights but shouldn't necessarily hurt ranking.
Let's dig deeper to understand the actual cause.

Usage:
    uv run python -m benchmarks.investigate_gensim_performance
"""

from __future__ import annotations

import numpy as np


def analyze_idf_squared_effect():
    """Analyze how IDF² affects ranking compared to IDF."""
    print("=" * 70)
    print("ANALYSIS: Does IDF² hurt ranking?")
    print("=" * 70)

    # Simulate a query with two terms: one rare, one common
    N = 1000

    # Term 1: rare (df=10), Term 2: common (df=500)
    df1, df2 = 10, 500

    # IDF values (Lucene style)
    idf1 = np.log(1 + (N - df1 + 0.5) / (df1 + 0.5))  # ~4.56
    idf2 = np.log(1 + (N - df2 + 0.5) / (df2 + 0.5))  # ~0.69

    print(f"\nQuery terms: term1 (rare, df={df1}), term2 (common, df={df2})")
    print(f"IDF values: idf1={idf1:.3f}, idf2={idf2:.3f}")
    print(f"IDF² values: idf1²={idf1**2:.3f}, idf2²={idf2**2:.3f}")
    print(f"\nRatio idf1/idf2 = {idf1 / idf2:.2f}")
    print(f"Ratio idf1²/idf2² = {(idf1**2) / (idf2**2):.2f}")

    print("\n" + "-" * 70)
    print("Document comparison:")
    print("-" * 70)

    # Three documents with different term distributions
    docs = [
        {"name": "Doc A", "tf1": 5, "tf2": 1, "desc": "matches rare term well"},
        {"name": "Doc B", "tf1": 1, "tf2": 10, "desc": "matches common term well"},
        {"name": "Doc C", "tf1": 3, "tf2": 3, "desc": "balanced"},
    ]

    # Simplified TF (ignore length normalization for clarity)
    k1 = 1.5

    print(
        f"\n{'Doc':<10} {'tf1':>5} {'tf2':>5} | {'Correct BM25':>14} | {'IDF² BM25':>12} | {'Winner':>10}"
    )
    print("-" * 70)

    correct_scores = []
    idf2_scores = []

    for doc in docs:
        tf1, tf2 = doc["tf1"], doc["tf2"]

        # Simplified TF saturation (norm=1)
        tf_sat1 = (tf1 * (k1 + 1)) / (tf1 + k1)
        tf_sat2 = (tf2 * (k1 + 1)) / (tf2 + k1)

        # Correct BM25: IDF × TF
        correct = idf1 * tf_sat1 + idf2 * tf_sat2

        # IDF² version: IDF² × TF
        idf2_score = (idf1**2) * tf_sat1 + (idf2**2) * tf_sat2

        correct_scores.append((doc["name"], correct))
        idf2_scores.append((doc["name"], idf2_score))

        print(f"{doc['name']:<10} {tf1:>5} {tf2:>5} | {correct:>14.3f} | {idf2_score:>12.3f} |")

    # Check if ranking order is preserved
    correct_order = [x[0] for x in sorted(correct_scores, key=lambda x: -x[1])]
    idf2_order = [x[0] for x in sorted(idf2_scores, key=lambda x: -x[1])]

    print(f"\nCorrect BM25 ranking: {' > '.join(correct_order)}")
    print(f"IDF² BM25 ranking:    {' > '.join(idf2_order)}")

    if correct_order == idf2_order:
        print("\n✓ Rankings are IDENTICAL - IDF² preserves ranking order!")
        print("  So IDF² alone doesn't explain Gensim's poor performance.")
    else:
        print("\n✗ Rankings DIFFER - IDF² changes the ranking order!")


def analyze_actual_gensim_behavior():
    """Examine what Gensim actually computes."""
    print("\n" + "=" * 70)
    print("ANALYSIS: What does Gensim actually compute?")
    print("=" * 70)

    try:
        from gensim.corpora import Dictionary
        from gensim.models import OkapiBM25Model
    except ImportError:
        print("\nGensim not installed. Skipping.")
        return

    # Simple corpus
    docs = [
        ["rare", "term", "here"],
        ["common", "common", "common", "word"],
        ["rare", "common"],
    ]

    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    model = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75)

    print("\nCorpus:")
    for i, doc in enumerate(docs):
        print(f"  Doc {i}: {doc}")

    print("\nDocument frequencies:")
    for term, term_id in dictionary.token2id.items():
        df = dictionary.dfs[term_id]
        print(f"  '{term}': df={df}")

    # Query: ["rare"]
    query = ["rare"]
    query_bow = dictionary.doc2bow(query)

    print(f"\nQuery: {query}")
    print(f"Query BOW: {query_bow}")

    # Get query vector
    query_vec = model[query_bow]
    print(f"Query BM25 vector: {query_vec}")

    # Get document vectors
    print("\nDocument BM25 vectors:")
    for i, doc_bow in enumerate(bow_corpus):
        doc_vec = model[doc_bow]
        print(f"  Doc {i}: {doc_vec}")

    # Compute scores via dot product
    print("\nScores (query · doc via dot product):")
    for i, doc_bow in enumerate(bow_corpus):
        doc_vec = dict(model[doc_bow])
        query_vec_dict = dict(query_vec)

        score = 0.0
        for term_id, q_weight in query_vec_dict.items():
            d_weight = doc_vec.get(term_id, 0.0)
            score += q_weight * d_weight
            if d_weight > 0:
                print(
                    f"    Doc {i}, term '{dictionary[term_id]}': q_weight={q_weight:.4f} × d_weight={d_weight:.4f} = {q_weight * d_weight:.4f}"
                )

        print(f"  Doc {i} total score: {score:.4f}")

    # Now compute correct BM25
    print("\nCorrect BM25 scores (IDF × TF_doc for query terms):")
    N = len(docs)
    avgdl = np.mean([len(doc) for doc in docs])
    k1, b = 1.5, 0.75

    for i, doc in enumerate(docs):
        doc_len = len(doc)
        norm = 1 - b + b * (doc_len / avgdl)

        score = 0.0
        for term in query:
            if term not in dictionary.token2id:
                continue
            term_id = dictionary.token2id[term]
            df = dictionary.dfs[term_id]
            tf = doc.count(term)

            # Correct IDF and TF
            idf = np.log((N - df + 0.5) / (df + 0.5))
            if idf < 0:
                idf = 0.25 * np.mean(
                    [
                        np.log((N - dictionary.dfs[t] + 0.5) / (dictionary.dfs[t] + 0.5))
                        for t in dictionary.token2id.values()
                    ]
                )
            tf_sat = (tf * (k1 + 1)) / (tf + k1 * norm)

            term_score = idf * tf_sat
            if tf > 0:
                print(
                    f"    Doc {i}, term '{term}': IDF={idf:.4f} × TF_sat={tf_sat:.4f} = {term_score:.4f}"
                )
            score += term_score

        print(f"  Doc {i} correct score: {score:.4f}")


def analyze_query_weighting():
    """The key insight: query terms also get BM25 weighted!"""
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Query term weighting in Gensim")
    print("=" * 70)

    print("""
In standard BM25:
    score(D, Q) = Σ IDF(t) × TF_doc(t, D)
                  t∈Q

In Gensim's vector-space model:
    query_vec[t] = IDF(t) × TF_query(t)  <- query gets BM25 weight too!
    doc_vec[t]   = IDF(t) × TF_doc(t)

    score = query_vec · doc_vec
          = Σ (IDF(t) × TF_q(t)) × (IDF(t) × TF_d(t))
          = Σ IDF(t)² × TF_q(t) × TF_d(t)

The TF_query component is the problem!

For a query like "rare" where the term appears once:
- Query length = 1
- Query avgdl ≈ corpus avgdl (used for normalization)
- norm_query = 1 - b + b × (1 / avgdl)  <- very small for short queries!

This means TF_query can be >> 1, massively boosting the query weight.
""")

    try:
        from gensim.corpora import Dictionary
        from gensim.models import OkapiBM25Model
    except ImportError:
        print("\nGensim not installed. Skipping demonstration.")
        return

    # Create corpus with known avgdl
    docs = [["word"] * 100 for _ in range(10)]  # avgdl = 100
    docs.append(["rare"])  # One doc with rare term

    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    model = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75)

    # Query with single term
    query = ["rare"]
    query_bow = dictionary.doc2bow(query)
    query_vec = model[query_bow]

    print("\nCorpus avgdl = 100 (10 docs with 100 words each)")
    print(f"Query: {query} (length=1)")
    print(f"\nQuery BM25 vector: {query_vec}")

    if query_vec:
        term_id, weight = query_vec[0]
        print(f"\nQuery weight for 'rare': {weight:.4f}")

        # What should it be?
        N = len(docs)
        df = dictionary.dfs[term_id]
        idf = np.log((N - df + 0.5) / (df + 0.5))

        # TF for query (tf=1, doc_len=1, avgdl=100)
        avgdl = 100
        k1, b = 1.5, 0.75
        norm = 1 - b + b * (1 / avgdl)  # Very small!
        tf_query = (1 * (k1 + 1)) / (1 + k1 * norm)

        expected = idf * tf_query
        print("\nBreakdown:")
        print(f"  IDF = {idf:.4f}")
        print(f"  norm = 1 - {b} + {b} × (1/{avgdl}) = {norm:.4f}")
        print(f"  TF_query = (1 × {k1 + 1}) / (1 + {k1} × {norm:.4f}) = {tf_query:.4f}")
        print(f"  Expected query weight = IDF × TF_query = {expected:.4f}")

        print(f"\n⚠️  TF_query = {tf_query:.2f} >> 1 because query is much shorter than avgdl!")
        print(f"   This inflates query term weights by ~{tf_query:.1f}x")


def main():
    analyze_idf_squared_effect()
    analyze_actual_gensim_behavior()
    analyze_query_weighting()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The main reasons Gensim BM25 underperforms:

1. **Query length normalization mismatch**: Gensim applies BM25 TF weighting
   to the query using corpus avgdl. Since queries are much shorter than
   documents, this creates TF_query >> 1, inflating query term weights.

2. **IDF² amplifies this problem**: When multiplied by the already-inflated
   query weight, rare terms get disproportionately over-weighted.

3. **The combination distorts ranking**: Documents matching rare query terms
   get boosted far more than they should, while documents matching common
   terms get relatively suppressed.

The fix would be to NOT apply TF weighting to query terms (just use IDF),
or use query-specific avgdl. But Gensim's vector-space design doesn't
support this easily.
""")


if __name__ == "__main__":
    main()
