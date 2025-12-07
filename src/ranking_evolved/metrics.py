import numpy as np


def precision_at_k(relevant: np.ndarray, retrieved: np.ndarray, k: int) -> float:
    """
    Computes Precision@K.

    Args:
        relevant: 1D array of relevant document indices.
        retrieved: 1D array of ranked document indices.
        k: Top-k cutoff.

    Returns:
        Precision at rank k.
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = np.isin(retrieved_k, relevant).sum()
    return hits / k


def recall_at_k(relevant: np.ndarray, retrieved: np.ndarray, k: int) -> float:
    """
    Computes Recall@K.

    Args:
        relevant: 1D array of relevant document indices.
        retrieved: 1D array of ranked document indices.
        k: Top-k cutoff.

    Returns:
        Recall at rank k.
    """
    if relevant.size == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = np.isin(retrieved_k, relevant).sum()
    return hits / len(relevant)


def average_precision(relevant: np.ndarray, retrieved: np.ndarray) -> float:
    """
    Computes Average Precision (AP) for a single query.

    Args:
        relevant: 1D array of relevant document indices.
        retrieved: 1D array of ranked document indices.

    Returns:
        Average precision score.
    """
    if relevant.size == 0:
        return 0.0

    relevant_set = set(relevant.tolist())
    hits, sum_precisions = 0, 0.0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / i

    return sum_precisions / len(relevant_set)


def mean_average_precision(
    all_relevant: list[np.ndarray], all_retrieved: list[np.ndarray]
) -> float:
    """
    Computes Mean Average Precision (MAP) over multiple queries.

    Args:
        all_relevant: List of 1D arrays of relevant document indices.
        all_retrieved: List of 1D arrays of ranked document indices.

    Returns:
        Mean Average Precision score.
    """
    if not all_relevant:
        return 0.0

    ap_scores = [
        average_precision(rel, ret) for rel, ret in zip(all_relevant, all_retrieved)
    ]
    return float(np.mean(ap_scores))


def ndcg_at_k(relevant: np.ndarray, retrieved: np.ndarray, k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at rank K.

    Args:
        relevant: 1D array of relevant document indices.
        retrieved: 1D array of ranked document indices.
        k: Top-k cutoff.

    Returns:
        NDCG at rank k.
    """
    retrieved_at_k = retrieved[:k]
    gains = [1 if doc in relevant else 0 for doc in retrieved_at_k]
    discounts = [
        np.log2(i + 2) for i in range(len(gains))
    ]  # i + 2 because log2(1+1)=1 for i=0
    dcg = np.sum(np.array(gains) / np.array(discounts))

    # Compute ideal DCG
    ideal_gains = sorted(
        [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k)), reverse=True
    )
    idcg = np.sum(np.array(ideal_gains) / np.array(discounts))

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(relevant: np.ndarray, retrieved: np.ndarray) -> float:
    """
    Computes Reciprocal Rank (RR) for a single query.

    Args:
        relevant: 1D array of relevant document indices.
        retrieved: 1D array of ranked document indices.

    Returns:
        Reciprocal rank of the first relevant document (0.0 if none are retrieved).
    """
    relevant_set = set(relevant.tolist())
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    all_relevant: list[np.ndarray], all_retrieved: list[np.ndarray]
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) over multiple queries.

    Args:
        all_relevant: List of 1D arrays of relevant document indices.
        all_retrieved: List of 1D arrays of ranked document indices.

    Returns:
        Mean reciprocal rank score.
    """
    if not all_relevant:
        return 0.0

    rr_scores = [
        reciprocal_rank(rel, ret) for rel, ret in zip(all_relevant, all_retrieved)
    ]
    return float(np.mean(rr_scores))
