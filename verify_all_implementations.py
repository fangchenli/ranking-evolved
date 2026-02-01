"""
Verify all BM25 implementations produce the same results.

Tests on small datasets from BEIR and BRIGHT.
Compares: freeform_fast, constrained_fast, composable_fast, and bm25_pyserini
"""

import numpy as np
from datasets import load_dataset

# Import all implementations
import ranking_evolved.bm25_freeform_fast as freeform
import ranking_evolved.bm25_constrained_fast as constrained
import ranking_evolved.bm25_composable_fast as composable
import ranking_evolved.bm25_pyserini as pyserini


def load_beir_dataset(name: str, max_docs: int = 3000, max_queries: int = 50):
    """Load a BEIR dataset."""
    print(f"Loading BEIR {name}...")
    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")

    if len(corpus) > max_docs:
        corpus = corpus.select(range(max_docs))

    docs = [f"{d['title']} {d['text']}" for d in corpus]
    doc_ids = [d["_id"] for d in corpus]
    query_texts = [q["text"] for q in queries][:max_queries]

    return docs, doc_ids, query_texts, name


def load_bright_dataset(domain: str, max_docs: int = 3000, max_queries: int = 50):
    """Load a BRIGHT dataset."""
    print(f"Loading BRIGHT {domain}...")
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)

    if len(documents) > max_docs:
        documents = documents.select(range(max_docs))

    docs = [d["content"] for d in documents]
    doc_ids = [d["id"] for d in documents]
    query_texts = [ex["query"] for ex in examples][:max_queries]

    return docs, doc_ids, query_texts, f"bright-{domain}"


def run_implementation(module, docs: list[str], doc_ids: list[str], queries: list[str], top_k: int = 100):
    """Run one BM25 implementation."""
    tokenizer = module.LuceneTokenizer()
    doc_tokens = [tokenizer(d) for d in docs]
    query_tokens = [tokenizer(q) for q in queries]

    corpus = module.Corpus(doc_tokens, ids=doc_ids)
    bm25 = module.BM25(corpus)

    # Use batch_rank if available, otherwise run queries one by one
    if hasattr(bm25, "batch_rank"):
        return bm25.batch_rank(query_tokens, top_k=top_k)
    else:
        return [bm25.rank(q, top_k=top_k) for q in query_tokens]


def compare_results(name1: str, results1, name2: str, results2, top_k: int = 100):
    """Compare two result sets, return max score diff."""
    max_diff = 0.0

    for (_, scores1), (_, scores2) in zip(results1, results2):
        k = min(top_k, len(scores1), len(scores2))
        if k == 0:
            continue

        # Compare scores for documents at same positions
        diff = np.max(np.abs(scores1[:k] - scores2[:k]))
        max_diff = max(max_diff, diff)

    return max_diff


def verify_dataset(docs, doc_ids, queries, dataset_name):
    """Verify all implementations on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Documents: {len(docs)}, Queries: {len(queries)}")
    print(f"{'='*60}")

    top_k = 100

    implementations = [
        ("Pyserini", pyserini),
        ("Freeform", freeform),
        ("Constrained", constrained),
        ("Composable", composable),
    ]

    results = {}
    for name, module in implementations:
        print(f"Running {name}...")
        results[name] = run_implementation(module, docs, doc_ids, queries, top_k)

    # Compare fast implementations against each other (main check)
    print("\nFast implementations (should match exactly):")
    all_pass = True

    for n1, n2 in [("Freeform", "Constrained"), ("Freeform", "Composable"), ("Constrained", "Composable")]:
        max_diff = compare_results(n1, results[n1], n2, results[n2], top_k)
        status = "✓" if max_diff < 1e-6 else "✗"
        if max_diff >= 1e-6:
            all_pass = False
        print(f"  {n1:12} vs {n2:12}: max_diff={max_diff:.2e} {status}")

    # Compare against Pyserini (should now match with qtf handling)
    print("\nvs Pyserini (reference implementation):")
    for name in ["Freeform", "Constrained", "Composable"]:
        max_diff = compare_results("Pyserini", results["Pyserini"], name, results[name], top_k)
        status = "✓" if max_diff < 1e-6 else "✗"
        if max_diff >= 1e-6:
            all_pass = False
        print(f"  Pyserini vs {name:12}: max_diff={max_diff:.2e} {status}")

    return all_pass


def main():
    print("BM25 Implementation Verification")
    print("Comparing: Pyserini, Freeform, Constrained, Composable")

    datasets = []

    # BEIR datasets (small ones)
    for name in ["scifact", "nfcorpus"]:
        try:
            datasets.append(load_beir_dataset(name, max_docs=3000, max_queries=50))
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    # BRIGHT datasets
    for domain in ["biology", "earth_science"]:
        try:
            datasets.append(load_bright_dataset(domain, max_docs=3000, max_queries=50))
        except Exception as e:
            print(f"Failed to load {domain}: {e}")

    # Run verification
    all_pass = True
    for docs, doc_ids, queries, name in datasets:
        passed = verify_dataset(docs, doc_ids, queries, name)
        all_pass = all_pass and passed

    print(f"\n{'='*60}")
    print("FINAL RESULT:", "ALL PASS ✓" if all_pass else "SOME FAILED ✗")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
