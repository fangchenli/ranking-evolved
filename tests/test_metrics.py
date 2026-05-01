"""Unit tests for tasks._shared.metrics with hand-computed values."""

import numpy as np
import pytest

from tasks._shared.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    relevant = np.array([0, 1])
    retrieved = np.array([0, 2, 1, 3])
    # Top-2: [0, 2] -> 1 hit -> P@2 = 1/2
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    # Top-4: [0, 2, 1, 3] -> 2 hits -> P@4 = 2/4
    assert precision_at_k(relevant, retrieved, 4) == 0.5
    assert precision_at_k(relevant, retrieved, 1) == 1.0
    assert precision_at_k(relevant, retrieved, 0) == 0.0


def test_recall_at_k():
    relevant = np.array([0, 1])  # 2 relevant
    retrieved = np.array([0, 2, 1, 3])
    # Top-2: [0, 2] -> 1 hit -> R@2 = 1/2
    assert recall_at_k(relevant, retrieved, 2) == 0.5
    # Top-4: 2 hits -> R@4 = 2/2 = 1.0
    assert recall_at_k(relevant, retrieved, 4) == 1.0
    assert recall_at_k(relevant, np.array([2, 3]), 2) == 0.0
    assert recall_at_k(np.array([]), retrieved, 2) == 0.0


def test_average_precision():
    # relevant = {0, 1}, retrieved = [0, 2, 1]: hit at rank 1 (P=1/1), hit at rank 3 (P=2/3)
    # AP = (1 + 2/3) / 2 = (5/3)/2 = 5/6
    relevant = np.array([0, 1])
    retrieved = np.array([0, 2, 1])
    ap = average_precision(relevant, retrieved)
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert ap == pytest.approx(expected)
    # No relevant in retrieved -> AP = 0
    assert average_precision(relevant, np.array([2, 3, 4])) == 0.0
    assert average_precision(np.array([]), retrieved) == 0.0


def test_ndcg_at_k_binary_relevance():
    # Standard: DCG@k = sum_{i=1}^{k} gain(i) / log2(i+1), IDCG = best possible
    # relevant=[0], retrieved=[0,1,2], k=3: gains=[1,0,0], DCG=1/log2(2)=1, IDCG=1 -> NDCG=1
    assert ndcg_at_k(np.array([0]), np.array([0, 1, 2]), 3) == pytest.approx(1.0)

    # relevant=[0,1], retrieved=[2,1,0], k=3: gains=[0,1,1]
    # DCG = 1/log2(3) + 1/log2(4) = 1/1.585 + 1/2 ≈ 1.131, IDCG = 1 + 1/log2(3) ≈ 1.631
    # NDCG ≈ 1.131/1.631 ≈ 0.693
    relevant = np.array([0, 1])
    retrieved = np.array([2, 1, 0])
    ndcg = ndcg_at_k(relevant, retrieved, 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert ndcg == pytest.approx(dcg / idcg)

    # relevant=[0,1], retrieved=[0,1,2], k=3: perfect order -> NDCG=1
    assert ndcg_at_k(relevant, np.array([0, 1, 2]), 3) == pytest.approx(1.0)

    # No relevant in retrieved -> NDCG = 0
    assert ndcg_at_k(relevant, np.array([2, 3, 4]), 3) == pytest.approx(0.0)

    # Empty relevant -> NDCG = 0 (avoid div by zero)
    assert ndcg_at_k(np.array([]), np.array([0, 1, 2]), 3) == 0.0


def test_reciprocal_rank():
    # First relevant at rank 1 -> RR = 1
    assert reciprocal_rank(np.array([0]), np.array([0, 1, 2])) == 1.0
    # First relevant at rank 2 -> RR = 1/2
    assert reciprocal_rank(np.array([0]), np.array([1, 0, 2])) == 0.5
    # No relevant -> RR = 0
    assert reciprocal_rank(np.array([0]), np.array([1, 2, 3])) == 0.0
