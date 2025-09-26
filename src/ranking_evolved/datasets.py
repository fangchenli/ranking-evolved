from collections import defaultdict

import ir_datasets


def doc_id_to_idx(dataset: ir_datasets.Dataset) -> dict[int, int]:
    """
    Maps document IDs to their index in the dataset.
    """
    return {int(doc.id): idx for idx, doc in enumerate(dataset.docs_iter())}


def relevance_query_to_docs(dataset: ir_datasets.Dataset) -> dict[int, list[int]]:
    """
    Maps a query to its relevant documents in the dataset.

    Args:
        query (str): The query string.
        dataset (ir_datasets.Dataset): The dataset to search.

    Returns:
        dict[str, list[str]]: A mapping from query IDs to lists of relevant document IDs.
    """
    relevance_map = defaultdict(list)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            relevance_map[int(qrel.query_id)].append(int(qrel.doc_id))
    return relevance_map
