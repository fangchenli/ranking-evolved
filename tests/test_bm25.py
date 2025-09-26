from datasets import load_dataset
import numpy as np
import pytest

from ranking_evolved.bm25 import Corpus, BM25, tokenize
from ranking_evolved.metrics import ndcg_at_k


@pytest.fixture
def bright_biology():
    dataset = load_dataset("xlangai/BRIGHT", "documents", split="biology")
    examples = load_dataset("xlangai/BRIGHT", "examples", split="biology")

    corpus = Corpus.from_huggingface_dataset(dataset)
    queries = [tokenize(example["query"]) for example in examples]
    gold_idx = [example["gold_ids"] for example in examples]
    expected = [corpus.id_to_idx(ids) for ids in gold_idx]
    return corpus, queries, expected


@pytest.mark.parametrize(
    "documents, query, expected_ranking",
    [
        (
            [
                "information retrieval is the activity of obtaining information system resources".split(),
                "BM25 ranks documents based on their relevance to a query".split(),
                "Python is widely used for text processing and ranking algorithms".split(),
            ],
            "information retrieval system".split(),
            [0, 2, 1],
        ),
    ],
)
def test_bm25(documents, query, expected_ranking):
    corpus = Corpus(documents)
    bm25 = BM25(corpus)

    ranked_indices, _ = bm25.rank(query)
    assert np.array_equal(ranked_indices, expected_ranking), (
        f"Expected ranking {expected_ranking}, got {ranked_indices}"
    )


def test_bright_biology(bright_biology):
    corpus, queries, expected = bright_biology
    bm25 = BM25(corpus)

    ndcg_at_10_scores = []
    for query, gold in zip(queries, expected):
        ranked_indices, _ = bm25.rank(query)
        ndcg_at_10 = ndcg_at_k(np.array(gold), ranked_indices, k=10)
        ndcg_at_10_scores.append(ndcg_at_10)

    average_ndcg_at_10 = np.mean(ndcg_at_10_scores)

    expected = 0.08
    assert average_ndcg_at_10 > expected
