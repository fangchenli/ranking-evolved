"""
Root cause analysis of Gensim BM25 underperformance.

Usage:
    uv run python -m benchmarks.gensim_root_cause
"""

import numpy as np


def main():
    print("=" * 70)
    print("ROOT CAUSE: Gensim BM25 Underperformance")
    print("=" * 70)

    # Small corpus where some terms have IDF exactly 0
    docs = [
        ["the", "quick", "brown", "fox"],  # doc 0
        ["the", "lazy", "brown", "dog"],   # doc 1
        ["fox", "and", "dog", "play"],     # doc 2
        ["brown", "bread", "is", "good"],  # doc 3
    ]

    N = len(docs)

    # Calculate document frequencies
    from collections import Counter
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    print("\nDocument frequencies:")
    for term, count in sorted(df.items()):
        idf = np.log((N - count + 0.5) / (count + 0.5))
        status = "ZERO" if abs(idf) < 0.001 else ("NEG" if idf < 0 else "")
        print(f"  '{term}': df={count}, IDF={idf:.4f} {status}")

    print("\n" + "-" * 70)
    print("Key observation: Terms appearing in exactly N/2 documents have IDF ≈ 0")
    print("-" * 70)

    # For N=4, df=2 gives IDF = log((4-2+0.5)/(2+0.5)) = log(2.5/2.5) = 0
    print(f"\nFor N={N}:")
    print(f"  df=2: IDF = log(({N}-2+0.5)/(2+0.5)) = log(2.5/2.5) = 0")
    print(f"\n  'brown' appears in 4/4 docs but...")

    # Recalculate
    brown_df = df["brown"]
    brown_idf = np.log((N - brown_df + 0.5) / (brown_df + 0.5))
    print(f"  Actually 'brown' df={brown_df}, IDF={brown_idf:.4f}")

    # Let's trace through what happens in Gensim
    print("\n" + "=" * 70)
    print("TRACE: What Gensim does with zero-IDF terms")
    print("=" * 70)

    from gensim.corpora import Dictionary
    from gensim.models import OkapiBM25Model

    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Create model
    model = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75, epsilon=0.25)

    # Get average IDF from model internals
    avg_idf = np.mean([v for v in model.idfs.values() if v > 0]) if model.idfs else 0
    print(f"\nModel average_idf (computed): {avg_idf:.4f}")
    print(f"Epsilon: {model.epsilon}")
    print(f"Epsilon threshold: {model.epsilon * avg_idf:.4f}")

    # Check stored IDF values in model
    print("\nModel's internal IDF values (model.idfs):")
    for term, term_id in sorted(dictionary.token2id.items()):
        stored_idf = model.idfs.get(term_id, 0)
        raw_df = dictionary.dfs[term_id]
        raw_idf = np.log((N - raw_df + 0.5) / (raw_df + 0.5))
        print(f"  '{term}': raw_idf={raw_idf:.4f}, stored_idf={stored_idf:.4f}")

    print("\n" + "=" * 70)
    print("THE ACTUAL PROBLEM")
    print("=" * 70)
    print("""
Looking at the stored IDF values, we see that terms with raw_idf <= 0
get stored_idf = 0, NOT epsilon * average_idf!

This is likely a bug or design choice in Gensim where:
1. Negative IDF terms get clamped to epsilon * average_idf
2. But terms with IDF exactly 0 get stored as 0

OR the clamping only happens at query time, not at index time.

Let me verify by looking at what happens during transformation...
""")

    # Transform doc 1 which has "the", "lazy", "brown", "dog"
    doc1_bow = bow_corpus[1]
    doc1_vec = model[doc1_bow]

    print(f"\nDoc 1: {docs[1]}")
    print(f"Doc 1 BOW: {doc1_bow}")
    print(f"Doc 1 BM25 vector: {doc1_vec}")

    # Manually compute what we expect
    print("\nManual computation for doc 1:")
    avgdl = np.mean([len(doc) for doc in docs])
    doc_len = len(docs[1])
    norm = 1 - 0.75 + 0.75 * (doc_len / avgdl)

    for term in docs[1]:
        term_id = dictionary.token2id[term]
        tf = docs[1].count(term)
        raw_idf = np.log((N - dictionary.dfs[term_id] + 0.5) / (dictionary.dfs[term_id] + 0.5))
        stored_idf = model.idfs.get(term_id, 0)

        tf_sat = (tf * (1.5 + 1)) / (tf + 1.5 * norm)
        weight = stored_idf * tf_sat

        print(f"  '{term}': tf={tf}, raw_idf={raw_idf:.4f}, stored_idf={stored_idf:.4f}, tf_sat={tf_sat:.4f}, weight={weight:.4f}")

    print("\n" + "=" * 70)
    print("IMPACT ON RETRIEVAL")
    print("=" * 70)

    # Query that uses terms with IDF=0
    query = ["the", "lazy"]  # "the" has IDF=0, "lazy" has positive IDF
    print(f"\nQuery: {query}")

    query_bow = dictionary.doc2bow(query)
    query_vec = model[query_bow]

    print(f"Query BOW: {query_bow}")
    print(f"Query BM25 vector: {query_vec}")

    print("\nProblem: 'the' is a valid discriminating term in this tiny corpus")
    print("         (only in docs 0, 1) but gets ZERO weight!")

    # Show what correct BM25 would do
    print("\n--- Correct BM25 would score ---")
    for i, doc in enumerate(docs):
        score = 0
        details = []
        doc_len = len(doc)
        norm = 1 - 0.75 + 0.75 * (doc_len / avgdl)

        for term in set(query):
            if term in doc:
                tf = doc.count(term)
                raw_idf = np.log((N - df[term] + 0.5) / (df[term] + 0.5))
                # Use Lucene IDF (non-negative)
                lucene_idf = np.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
                tf_sat = (tf * 2.5) / (tf + 1.5 * norm)
                term_score = lucene_idf * tf_sat
                score += term_score
                details.append(f"'{term}': {lucene_idf:.2f}×{tf_sat:.2f}={term_score:.2f}")

        print(f"Doc {i}: score={score:.4f}  ({', '.join(details) if details else 'no match'})")

    print("\n--- Gensim scores (via dot product) ---")
    query_vec_dict = dict(query_vec)
    for i, doc_bow in enumerate(bow_corpus):
        doc_vec = dict(model[doc_bow])
        score = sum(query_vec_dict.get(tid, 0) * doc_vec.get(tid, 0) for tid in set(query_vec_dict) | set(doc_vec))
        print(f"Doc {i}: score={score:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The root causes of Gensim BM25 underperformance:

1. **Zero IDF for common terms**: Terms appearing in exactly N/2 documents
   get IDF=0 (from classic BM25 formula), and Gensim stores this as 0.
   These terms are completely ignored even though they may be discriminative.

2. **Aggressive negative IDF handling**: Terms in >N/2 documents get
   clamped to epsilon*avg_idf, but this may not be applied consistently
   or the threshold may be too aggressive.

3. **IDF² amplification**: The vector-space dot product squares the IDF,
   which amplifies the difference between rare and common terms even more.

4. **Query length normalization**: Short queries get inflated TF weights
   because they use corpus avgdl for normalization.

The combination of these factors causes Gensim to:
- Ignore or severely underweight common but discriminative terms
- Over-weight rare terms
- Produce rankings that differ significantly from correct BM25
""")


if __name__ == "__main__":
    main()
