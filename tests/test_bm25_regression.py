import numpy as np

from ranking_evolved.bm25 import BM25, Corpus


def test_bm25_ordering_regression() -> None:
    """
    Regression check to guard BM25 kernel behavior:
    - Documents with repeated query terms should score higher than those with fewer matches.
    - Non-matching documents should rank last.
    """
    documents = [
        "foo foo foo bar".split(),  # heavy tf on foo
        "foo bar baz".split(),  # single foo/bar
        "baz qux".split(),  # no query terms
    ]
    corpus = Corpus(documents)
    bm25 = BM25(corpus, k1=1.5, b=0.75)

    query = ["foo", "bar"]
    ranked_indices, scores = bm25.rank(query)

    # Expected ordering: doc0 > doc1 > doc2
    assert list(ranked_indices) == [0, 1, 2]

    # Ensure score gaps are meaningful (avoid degenerate normalization).
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] - scores[1] > 0.05
    assert np.isclose(scores[2], 0.0)
