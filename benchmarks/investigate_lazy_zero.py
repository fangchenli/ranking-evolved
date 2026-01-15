"""
Why does 'lazy' have IDF=0 in Gensim?

Usage:
    uv run python -m benchmarks.investigate_lazy_zero
"""

import numpy as np


def main():
    docs = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["a", "fox", "is", "a", "small", "wild", "animal", "with", "red", "fur"],
        ["the", "lazy", "dog", "sleeps", "all", "day", "long"],
        ["quick", "brown", "bread", "with", "fox", "shaped", "cookies"],
    ]

    from gensim.corpora import Dictionary
    from gensim.models import OkapiBM25Model

    dictionary = Dictionary(docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    print("Vocabulary and document frequencies:")
    for term, term_id in sorted(dictionary.token2id.items(), key=lambda x: x[1]):
        df = dictionary.dfs[term_id]
        print(f"  {term_id}: '{term}' df={df}")

    N = len(docs)
    print(f"\nN = {N} documents")

    # Compute IDF for each term manually
    print("\nIDF values (manual calculation):")
    for term, term_id in sorted(dictionary.token2id.items(), key=lambda x: x[1]):
        df = dictionary.dfs[term_id]
        # Classic IDF
        classic_idf = np.log((N - df + 0.5) / (df + 0.5))
        print(
            f"  '{term}': classic_idf = log(({N} - {df} + 0.5) / ({df} + 0.5)) = {classic_idf:.4f}"
        )

    print("\nTerms with negative IDF (df > N/2):")
    for term, term_id in dictionary.token2id.items():
        df = dictionary.dfs[term_id]
        classic_idf = np.log((N - df + 0.5) / (df + 0.5))
        if classic_idf < 0:
            print(
                f"  '{term}': df={df}, IDF={classic_idf:.4f} (appears in {df}/{N} = {df / N * 100:.0f}% of docs)"
            )

    # Now create the model and look at what it computes
    model = OkapiBM25Model(corpus=bow_corpus, dictionary=dictionary, k1=1.5, b=0.75, epsilon=0.25)

    print("\n" + "=" * 60)
    print("Gensim OkapiBM25Model behavior:")
    print("=" * 60)

    # Transform a document
    doc0 = bow_corpus[0]  # Has "the", "lazy", etc.
    doc0_vec = model[doc0]

    print(f"\nDoc 0 original: {docs[0]}")
    print(f"Doc 0 BOW: {doc0}")
    print(f"Doc 0 BM25 vector: {doc0_vec}")

    print("\nLet's see which terms got zeroed out:")
    doc0_terms = {term_id for term_id, count in doc0}
    doc0_vec_terms = {term_id for term_id, weight in doc0_vec}

    missing = doc0_terms - doc0_vec_terms
    for term_id in missing:
        term = dictionary[term_id]
        df = dictionary.dfs[term_id]
        classic_idf = np.log((N - df + 0.5) / (df + 0.5))
        print(f"  Missing: '{term}' (df={df}, classic_idf={classic_idf:.4f})")

    # Check the epsilon threshold
    print("\n" + "=" * 60)
    print("Epsilon clamping analysis:")
    print("=" * 60)

    # Calculate average IDF
    idfs = []
    for term_id in dictionary.token2id.values():
        df = dictionary.dfs[term_id]
        idf = np.log((N - df + 0.5) / (df + 0.5))
        if idf > 0:
            idfs.append(idf)

    avg_idf = np.mean(idfs) if idfs else 0
    epsilon = 0.25
    threshold = epsilon * avg_idf

    print(f"Average IDF (positive only): {avg_idf:.4f}")
    print(f"Epsilon: {epsilon}")
    print(f"Clamping threshold: epsilon * avg_idf = {threshold:.4f}")

    print("\nTerms that get clamped to epsilon*avg_idf:")
    for term, term_id in dictionary.token2id.items():
        df = dictionary.dfs[term_id]
        classic_idf = np.log((N - df + 0.5) / (df + 0.5))
        if classic_idf < threshold:
            print(f"  '{term}': classic_idf={classic_idf:.4f} -> clamped to {threshold:.4f}")

    # BUT WAIT - let's check if Gensim even includes terms below threshold
    print("\n" + "=" * 60)
    print("The REAL issue:")
    print("=" * 60)
    print("""
Looking at the output, 'lazy' doesn't appear in the BM25 vector at all!

This is because Gensim's OkapiBM25Model filters out terms with IDF below
epsilon * average_idf during the __getitem__ transformation.

From Gensim source code (gensim/models/bm25model.py):

    if idf < 0:
        idf = self.epsilon * self.average_idf  # clamp to small positive

Then later when computing the weight:
    weight = idf * tf_component

But the key is: if a term's IDF is clamped, it might still be excluded
from the result if the weight is below some internal threshold.

Actually, looking at doc0_vec output: [(0, ...), (1, ...), ...]
The term 'the' (which appears twice) should be there...

Let me check 'the' specifically:
""")

    # Check 'the'
    the_id = dictionary.token2id.get("the")
    if the_id is not None:
        the_df = dictionary.dfs[the_id]
        the_idf = np.log((N - the_df + 0.5) / (the_df + 0.5))
        print(f"'the': term_id={the_id}, df={the_df}, classic_idf={the_idf:.4f}")
        print(f"'the' in doc0_vec: {the_id in {t[0] for t in doc0_vec}}")

    # Check 'lazy'
    lazy_id = dictionary.token2id.get("lazy")
    if lazy_id is not None:
        lazy_df = dictionary.dfs[lazy_id]
        lazy_idf = np.log((N - lazy_df + 0.5) / (lazy_df + 0.5))
        print(f"'lazy': term_id={lazy_id}, df={lazy_df}, classic_idf={lazy_idf:.4f}")
        print(f"'lazy' in doc0_vec: {lazy_id in {t[0] for t in doc0_vec}}")

    print("\n" + "=" * 60)
    print("ACTUAL WEIGHTS from Gensim:")
    print("=" * 60)
    for term_id, weight in doc0_vec:
        term = dictionary[term_id]
        df = dictionary.dfs[term_id]
        print(f"  '{term}': weight={weight:.4f}, df={df}")


if __name__ == "__main__":
    main()
