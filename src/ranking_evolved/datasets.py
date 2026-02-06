"""
Dataset loaders for IR benchmarks.

Provides unified interfaces for loading:
- BRIGHT (12 domains)
- BEIR (17 datasets)
- TREC DL (DL19, DL20)

All loaders return EvalDataset objects with consistent interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Dataset Constants
# =============================================================================

BRIGHT_SPLITS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

# BEIR datasets - publicly available only (ordered by corpus size)
# Excluded: robust04, trec-news, bioasq (require special access/registration)
BEIR_DATASETS = [
    # Tiny (< 10K docs) - can parallelize heavily
    "nfcorpus",  # 3,633 docs
    "scifact",  # 5,183 docs
    "arguana",  # 8,674 docs
    # Small (10K-50K docs)
    "scidocs",  # 25,657 docs
    # Medium (50K-200K docs)
    "fiqa",  # 57,638 docs
    "trec-covid",  # 171,332 docs
    # Large (200K-600K docs)
    "webis-touche2020",  # 382,545 docs
    "cqadupstack",  # 457,199 docs (12 sub-forums)
    "quora",  # 522,931 docs
    # Huge (> 2M docs) - run SOLO to avoid OOM
    "nq",  # 2,681,468 docs
    "dbpedia-entity",  # 4,635,922 docs
    "hotpotqa",  # 5,233,329 docs
    "fever",  # 5,416,568 docs
    "climate-fever",  # 5,416,593 docs
]

# Datasets that require special access (not auto-downloadable)
BEIR_RESTRICTED = [
    "robust04",  # 528,155 docs - TREC license required
    "trec-news",  # 594,977 docs - TREC license required
    "bioasq",  # 14,914,602 docs - BioASQ registration required
]

TREC_DL_DATASETS = ["dl19", "dl20"]

# Dataset size estimates (in docs) for scheduling
DATASET_SIZES = {
    # BRIGHT
    "bright_biology": 57_000,
    "bright_earth_science": 121_000,
    "bright_economics": 50_000,
    "bright_psychology": 52_000,
    "bright_robotics": 62_000,
    "bright_stackoverflow": 107_000,
    "bright_sustainable_living": 61_000,
    "bright_pony": 8_000,
    "bright_leetcode": 414_000,
    "bright_aops": 188_000,
    "bright_theoremqa_theorems": 24_000,
    "bright_theoremqa_questions": 188_000,
    # BEIR
    "beir_scifact": 5_000,
    "beir_nfcorpus": 4_000,
    "beir_arguana": 9_000,
    "beir_scidocs": 26_000,
    "beir_fiqa": 58_000,
    "beir_webis-touche2020": 383_000,
    "beir_trec-covid": 171_000,
    "beir_quora": 523_000,
    "beir_cqadupstack": 457_000,
    "beir_hotpotqa": 5_233_000,
    "beir_nq": 2_681_000,
    "beir_fever": 5_417_000,
    "beir_climate-fever": 5_417_000,
    "beir_dbpedia-entity": 4_636_000,
    # TREC DL
    "trec_dl_dl19": 8_841_823,
    "trec_dl_dl20": 8_841_823,
}


# =============================================================================
# Unified Dataset Interface
# =============================================================================


@dataclass
class EvalDataset:
    """
    Unified dataset interface for evaluation.

    Attributes:
        name: Dataset identifier (e.g., "bright_biology", "beir_scifact")
        benchmark: Benchmark name ("bright", "beir", "trec_dl")
        corpus: List of document texts
        corpus_ids: List of document IDs
        queries: List of query texts
        query_ids: List of query IDs
        qrels: Query relevance judgments {query_id: {doc_id: relevance}}
    """

    name: str
    benchmark: str
    corpus: list[str]
    corpus_ids: list[str]
    queries: list[str]
    query_ids: list[str]
    qrels: dict[str, dict[str, int]]

    def get_relevant_docs(self, query_id: str) -> list[str]:
        """Get list of relevant doc IDs for a query."""
        if query_id not in self.qrels:
            return []
        return [doc_id for doc_id, rel in self.qrels[query_id].items() if rel > 0]

    @property
    def corpus_size(self) -> int:
        """Number of documents in corpus."""
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        """Number of queries."""
        return len(self.queries)


# =============================================================================
# BRIGHT Loader
# =============================================================================


class BRIGHTLoader:
    """Loader for BRIGHT benchmark datasets."""

    def __init__(self):
        self._cache: dict[str, EvalDataset] = {}

    def load(self, domain: str) -> EvalDataset:
        """
        Load a BRIGHT domain.

        Args:
            domain: Domain name (e.g., "biology", "earth_science")

        Returns:
            EvalDataset with corpus, queries, and relevance judgments
        """
        if domain in self._cache:
            return self._cache[domain]

        from datasets import load_dataset

        # Load documents
        documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
        corpus = [doc["content"] for doc in documents]
        corpus_ids = [doc["id"] for doc in documents]

        # Load examples (queries + gold IDs)
        examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
        queries = [ex["query"] for ex in examples]
        query_ids = [f"q{i}" for i in range(len(queries))]

        # Build qrels from gold_ids
        qrels: dict[str, dict[str, int]] = {}
        for i, ex in enumerate(examples):
            qid = query_ids[i]
            qrels[qid] = {doc_id: 1 for doc_id in ex["gold_ids"]}

        dataset = EvalDataset(
            name=f"bright_{domain}",
            benchmark="bright",
            corpus=corpus,
            corpus_ids=corpus_ids,
            queries=queries,
            query_ids=query_ids,
            qrels=qrels,
        )

        self._cache[domain] = dataset
        return dataset


# =============================================================================
# BEIR Loader
# =============================================================================


class BEIRLoader:
    """Loader for BEIR benchmark datasets."""

    def __init__(self, data_dir: str = "datasets/beir"):
        self.data_dir = data_dir
        self._cache: dict[str, EvalDataset] = {}

    def load(self, dataset_name: str) -> EvalDataset:
        """
        Load a BEIR dataset.

        Args:
            dataset_name: Dataset name (e.g., "scifact", "nfcorpus")

        Returns:
            EvalDataset with corpus, queries, and relevance judgments
        """
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        data_path = Path(self.data_dir) / dataset_name

        if not data_path.exists():
            # Download dataset
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            util.download_and_unzip(url, self.data_dir)

        # CQADupstack: 12 sub-forums (gaming, tex, android, ...), each with corpus.jsonl, queries.jsonl, qrels/
        if dataset_name == "cqadupstack" and data_path.is_dir():
            subdirs = [
                d
                for d in sorted(data_path.iterdir())
                if d.is_dir() and (d / "corpus.jsonl").is_file()
            ]
            if subdirs:
                corpus_ids = []
                corpus = []
                query_ids = []
                queries = []
                qrels: dict[str, dict[str, int]] = {}
                for subdir in subdirs:
                    prefix = subdir.name
                    try:
                        c_dict, q_dict, qrel_dict = GenericDataLoader(str(subdir)).load(split="test")
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"beir cqadupstack: skip subforum '{prefix}' ({subdir}): {e}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    for doc_id, doc in c_dict.items():
                        pid = f"{prefix}_{doc_id}"
                        corpus_ids.append(pid)
                        title = doc.get("title", "") or ""
                        text = doc.get("text", "") or ""
                        combined = f"{title} {text}".strip() if title else text
                        corpus.append(combined)
                    for qid in qrel_dict:
                        q_key = f"{prefix}_{qid}"
                        query_ids.append(q_key)
                        queries.append(q_dict[qid])
                        qrels[q_key] = {
                            f"{prefix}_{doc_id}": score
                            for doc_id, score in qrel_dict[qid].items()
                        }
                dataset = EvalDataset(
                    name=f"beir_{dataset_name}",
                    benchmark="beir",
                    corpus=corpus,
                    corpus_ids=corpus_ids,
                    queries=queries,
                    query_ids=query_ids,
                    qrels=qrels,
                )
                self._cache[dataset_name] = dataset
                return dataset
            # fallthrough: no subdirs with corpus.jsonl, try top-level

        # Single-folder BEIR layout (or flat cqadupstack)
        corpus_dict, queries_dict, qrels = GenericDataLoader(str(data_path)).load(split="test")

        # Convert to lists
        corpus_ids = list(corpus_dict.keys())
        corpus = []
        for doc_id in corpus_ids:
            doc = corpus_dict[doc_id]
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            combined = f"{title} {text}".strip() if title else text
            corpus.append(combined)

        query_ids = list(queries_dict.keys())
        queries = [queries_dict[qid] for qid in query_ids]

        dataset = EvalDataset(
            name=f"beir_{dataset_name}",
            benchmark="beir",
            corpus=corpus,
            corpus_ids=corpus_ids,
            queries=queries,
            query_ids=query_ids,
            qrels=qrels,
        )

        self._cache[dataset_name] = dataset
        return dataset


# =============================================================================
# TREC DL Loader
# =============================================================================


class TRECDLLoader:
    """Loader for TREC Deep Learning Track datasets (DL19, DL20)."""

    def __init__(self, data_dir: str = "datasets/trec_dl"):
        self.data_dir = data_dir
        self._cache: dict[str, EvalDataset] = {}
        self._msmarco_corpus: tuple[list[str], list[str]] | None = None

    def _load_msmarco_corpus(self) -> tuple[list[str], list[str]]:
        """Load MS MARCO passage corpus (shared by DL19/DL20)."""
        if self._msmarco_corpus is not None:
            return self._msmarco_corpus

        from datasets import load_dataset

        # Load MS MARCO corpus
        _corpus_ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")  # noqa: F841

        # MS MARCO has passages in a different format - extract all passages
        corpus = []
        corpus_ids = []

        # Note: This is a simplified version. Full MS MARCO has 8.8M passages.
        # For DL19/DL20, we use the passage corpus from ir-datasets if available.
        try:
            import ir_datasets

            ds = ir_datasets.load("msmarco-passage")
            for doc in ds.docs_iter():
                corpus_ids.append(doc.doc_id)
                corpus.append(doc.text)
        except ImportError:
            # Fallback: use HuggingFace datasets
            # This is slower but works without ir-datasets
            passages = load_dataset("ms_marco", "v1.1", split="train")
            seen_ids = set()
            for item in passages:
                for i, passage in enumerate(item.get("passages", {}).get("passage_text", [])):
                    pid = f"{item['query_id']}_{i}"
                    if pid not in seen_ids:
                        corpus.append(passage)
                        corpus_ids.append(pid)
                        seen_ids.add(pid)

        self._msmarco_corpus = (corpus, corpus_ids)
        return self._msmarco_corpus

    def load(self, dataset_name: str) -> EvalDataset:
        """
        Load a TREC DL dataset.

        Args:
            dataset_name: "dl19" or "dl20"

        Returns:
            EvalDataset with corpus, queries, and relevance judgments
        """
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        # Load shared corpus
        corpus, corpus_ids = self._load_msmarco_corpus()

        # Load queries and qrels for specific year
        try:
            import ir_datasets

            if dataset_name == "dl19":
                ds = ir_datasets.load("msmarco-passage/trec-dl-2019")
            elif dataset_name == "dl20":
                ds = ir_datasets.load("msmarco-passage/trec-dl-2020")
            else:
                raise ValueError(f"Unknown TREC DL dataset: {dataset_name}")

            queries = []
            query_ids = []
            for query in ds.queries_iter():
                query_ids.append(query.query_id)
                queries.append(query.text)

            qrels: dict[str, dict[str, int]] = {}
            for qrel in ds.qrels_iter():
                qid = qrel.query_id
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][qrel.doc_id] = qrel.relevance

        except ImportError as err:
            raise ImportError(
                "ir-datasets is required for TREC DL. Install with: pip install ir-datasets"
            ) from err

        dataset = EvalDataset(
            name=f"trec_dl_{dataset_name}",
            benchmark="trec_dl",
            corpus=corpus,
            corpus_ids=corpus_ids,
            queries=queries,
            query_ids=query_ids,
            qrels=qrels,
        )

        self._cache[dataset_name] = dataset
        return dataset


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "BRIGHT_SPLITS",
    "BEIR_DATASETS",
    "TREC_DL_DATASETS",
    "DATASET_SIZES",
    "EvalDataset",
    "BRIGHTLoader",
    "BEIRLoader",
    "TRECDLLoader",
]
