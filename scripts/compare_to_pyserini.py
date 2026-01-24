#!/usr/bin/env python3
"""
Compare our BM25 implementations to Pyserini across multiple benchmarks.

This script evaluates BM25 implementations on different benchmarks (BRIGHT, BEIR, TREC DL)
and compares against Pyserini (where available).

Usage:
    python scripts/compare_to_pyserini.py --benchmark bright
    python scripts/compare_to_pyserini.py --benchmark beir
    python scripts/compare_to_pyserini.py --benchmark trec_dl
    python scripts/compare_to_pyserini.py --benchmark bright --implementation freeform
    python scripts/compare_to_pyserini.py --benchmark bright --implementation composable
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Configure Java BEFORE any imports that might use it
java_home = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
jvm_path = f"{java_home}/lib/server/libjvm.dylib"

# Set Java environment variables (must be before any Java-related imports)
os.environ.setdefault("JAVA_HOME", java_home)
os.environ.setdefault("JVM_PATH", jvm_path)
os.environ.setdefault("COURSIER_JAVA_HOME", java_home)
os.environ.setdefault("PATH", f"{java_home}/bin:{os.environ.get('PATH', '')}")

# Set default Java heap - can be overridden per dataset
os.environ.setdefault("JAVA_OPTS", "-Xmx8g -Xms2g -XX:+UseG1GC")

# Verify Java path exists
if not Path(jvm_path).exists():
    print(f"WARNING: Java library not found at {jvm_path}")
    print(f"Please ensure Java 21 is installed at {java_home}")
    print("Continuing anyway - some features may not work...")

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import after setting Java env
from ranking_evolved.bm25_pyserini import BM25, Corpus, LuceneTokenizer
from ranking_evolved.metrics import ndcg_at_k, recall_at_k


# =============================================================================
# Benchmark Configurations
# =============================================================================

BRIGHT_REFERENCE = {
    "biology": 18.9,
    "earth_science": 27.2,
    "economics": 14.9,
    "psychology": 12.5,
    "robotics": 13.6,
    "stackoverflow": 18.4,
    "sustainable_living": 15.0,
    "pony": 7.9,
    "leetcode": 24.4,
    "aops": 6.2,
    "theoremqa_theorems": 4.9,
    "theoremqa_questions": 10.4,
}

BEIR_REFERENCE = {
    "scifact": 66.5,
    "nfcorpus": 32.5,
    "arguana": 31.5,
    "scidocs": 15.8,
    "fiqa": 23.6,
    "webis-touche2020": 36.7,
    "trec-covid": 65.6,
    "quora": 78.9,
    "cqadupstack": 29.9,
    "robust04": 40.8,
    "trec-news": 39.8,
    "msmarco": 22.8,
    "hotpotqa": 60.3,
    "nq": 32.9,
    "fever": 75.3,
    "climate-fever": 21.3,
    "dbpedia-entity": 31.3,
    "bioasq": 46.5,
}

TREC_DL_REFERENCE = {
    "dl19": 50.6,
    "dl20": 47.8,
}

# Datasets that are known to cause Pyserini indexing issues (very large)
BEIR_SKIP_PYSERINI = {
    "quora",  # 522K docs - known to hang at close()
    "msmarco",  # 8.8M docs - too large
    "bioasq",  # 14.9M docs - too large
}


# =============================================================================
# Common Evaluation Functions
# =============================================================================

def evaluate_our_implementation_bright(domain: str, implementation: str = "pyserini") -> dict:
    """Evaluate BRIGHT domain using our BM25 implementation."""
    from datasets import load_dataset
    
    start_time = time.time()
    
    # Load data
    documents = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    
    doc_ids = [doc["id"] for doc in documents]
    doc_texts = [doc["content"] for doc in documents]
    
    # Import the appropriate implementation - always use Lucene tokenizer
    if implementation == "pyserini":
        from ranking_evolved.bm25_pyserini import BM25, Corpus, LuceneTokenizer
        tokenizer = LuceneTokenizer()
    elif implementation == "composable":
        from ranking_evolved.bm25_composable import BM25, Corpus, LuceneTokenizer
        tokenizer = LuceneTokenizer()
    elif implementation == "freeform":
        from ranking_evolved.bm25_freeform import BM25, Corpus, LuceneTokenizer
        tokenizer = LuceneTokenizer()
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    # Tokenize documents
    print(f"    Tokenizing {len(doc_texts)} documents...")
    doc_tokens = [tokenizer(text) for text in doc_texts]
    
    # Build corpus and BM25
    print(f"    Building corpus and BM25...")
    corpus = Corpus(doc_tokens, ids=doc_ids)
    bm25 = BM25(corpus)
    
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Evaluate
    print(f"    Evaluating {len(examples)} queries...")
    ndcg_scores = []
    for ex in examples:
        query = ex["query"]
        gold_ids = ex["gold_ids"]
        gold_indices = [id_to_idx[gid] for gid in gold_ids if gid in id_to_idx]
        
        if not gold_indices:
            continue
        
        query_tokens = tokenizer(query)
        ranked_indices, _ = bm25.rank(query_tokens)
        
        ndcg = ndcg_at_k(
            np.array(gold_indices, dtype=np.int64),
            ranked_indices,
            10
        )
        ndcg_scores.append(ndcg)
    
    elapsed = time.time() - start_time
    return {
        "ndcg@10": float(np.mean(ndcg_scores)) * 100,
        "queries": len(ndcg_scores),
        "corpus_size": len(doc_ids),
        "time_seconds": elapsed,
    }


def evaluate_our_implementation_beir(dataset_name: str) -> dict:
    """Evaluate BEIR dataset using our BM25 implementation."""
    from ranking_evolved.datasets.beir import BEIRLoader
    
    start_time = time.time()
    
    # Load data
    loader = BEIRLoader()
    dataset = loader.load(dataset_name)
    
    doc_ids = dataset.corpus_ids
    doc_texts = dataset.corpus
    
    # Tokenize
    tokenizer = LuceneTokenizer()
    doc_tokens = [tokenizer(text) for text in doc_texts]
    
    # Build corpus and BM25
    corpus = Corpus(doc_tokens, ids=doc_ids)
    bm25 = BM25(corpus)
    
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Evaluate
    ndcg_scores = []
    recall_scores = []
    recall_k = dataset.metadata.get("recall_k")
    
    total_queries = len(dataset.queries)
    eval_start_time = time.time()
    
    print(f"    Evaluating {total_queries:,} queries against {len(doc_ids):,} documents...")
    sys.stdout.flush()
    
    for query_idx, (qid, query) in enumerate(zip(dataset.query_ids, dataset.queries), 1):
        # Get relevant docs
        relevant_doc_ids = list(dataset.qrels.get(qid, {}).keys())
        relevant_indices = [id_to_idx[gid] for gid in relevant_doc_ids if gid in id_to_idx]
        
        if not relevant_indices:
            continue
        
        query_tokens = tokenizer(query)
        ranked_indices, _ = bm25.rank(query_tokens)
        
        # NDCG@10
        ndcg = ndcg_at_k(
            np.array(relevant_indices, dtype=np.int64),
            ranked_indices,
            10
        )
        ndcg_scores.append(ndcg)
        
        # Recall@K if applicable
        if recall_k:
            recall = recall_at_k(
                np.array(relevant_indices, dtype=np.int64),
                ranked_indices,
                recall_k
            )
            recall_scores.append(recall)
        
        # Progress indicator
        if query_idx % max(100, total_queries // 100) == 0 or query_idx == total_queries:
            elapsed = time.time() - eval_start_time
            rate = query_idx / elapsed if elapsed > 0 else 0
            remaining = (total_queries - query_idx) / rate if rate > 0 else 0
            print(f"      Query {query_idx:,}/{total_queries:,} ({query_idx*100//total_queries}%) | "
                  f"Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m", flush=True)
            sys.stdout.flush()
    
    elapsed = time.time() - start_time
    result = {
        "ndcg@10": float(np.mean(ndcg_scores)) * 100,
        "queries": len(ndcg_scores),
        "corpus_size": len(doc_ids),
        "time_seconds": elapsed,
    }
    
    if recall_scores:
        result[f"recall@{recall_k}"] = float(np.mean(recall_scores)) * 100
    
    return result


def evaluate_our_implementation_trec_dl(dataset_name: str) -> dict:
    """Evaluate TREC DL dataset using our BM25 implementation."""
    from ranking_evolved.datasets.trec_dl import TRECDLLoader
    
    start_time = time.time()
    
    # Load data
    loader = TRECDLLoader()
    dataset = loader.load(dataset_name)
    
    doc_ids = dataset.corpus_ids
    doc_texts = dataset.corpus
    
    # Tokenize
    tokenizer = LuceneTokenizer()
    doc_tokens = [tokenizer(text) for text in doc_texts]
    
    # Build corpus and BM25
    corpus = Corpus(doc_tokens, ids=doc_ids)
    bm25 = BM25(corpus)
    
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Evaluate
    ndcg_scores = []
    recall_scores = []
    recall_k = dataset.metadata.get("recall_k", 1000)
    
    for qid, query in zip(dataset.query_ids, dataset.queries):
        # Get relevant docs
        relevant_doc_ids = list(dataset.qrels.get(qid, {}).keys())
        relevant_indices = [id_to_idx[gid] for gid in relevant_doc_ids if gid in id_to_idx]
        
        if not relevant_indices:
            continue
        
        query_tokens = tokenizer(query)
        ranked_indices, _ = bm25.rank(query_tokens)
        
        # NDCG@10
        ndcg = ndcg_at_k(
            np.array(relevant_indices, dtype=np.int64),
            ranked_indices,
            10
        )
        ndcg_scores.append(ndcg)
        
        # Recall@1000
        recall = recall_at_k(
            np.array(relevant_indices, dtype=np.int64),
            ranked_indices,
            recall_k
        )
        recall_scores.append(recall)
    
    elapsed = time.time() - start_time
    return {
        "ndcg@10": float(np.mean(ndcg_scores)) * 100,
        f"recall@{recall_k}": float(np.mean(recall_scores)) * 100,
        "queries": len(ndcg_scores),
        "corpus_size": len(doc_ids),
        "time_seconds": elapsed,
    }


def load_previous_results(results_path: Path) -> dict:
    """Load previously saved results."""
    if not results_path.exists():
        return {}
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    return data.get("datasets", {})


# =============================================================================
# Benchmark-Specific Runners
# =============================================================================

def run_bright_comparison(implementation: str = "pyserini"):
    """Run BRIGHT benchmark comparison."""
    from datasets import load_dataset
    
    print("=" * 70)
    print("BRIGHT BENCHMARK COMPARISON")
    impl_name = f"bm25_{implementation}.py"
    print(f"Our Implementation ({impl_name}) vs Saved Pyserini Results")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Implementation: {impl_name}")
    print(f"Tokenizer: Lucene (Pyserini's default)")
    print()
    
    # Load previous Pyserini results
    results_path = Path("experiment_summaries/test/bright_comparison_results.json")
    print(f"Loading previous Pyserini results from: {results_path}")
    try:
        pyserini_results = {}
        if results_path.exists():
            with open(results_path, "r") as f:
                data = json.load(f)
            for domain, domain_data in data.get("datasets", {}).items():
                if "pyserini" in domain_data and "ndcg@10" in domain_data["pyserini"]:
                    pyserini_results[domain] = domain_data["pyserini"]
        print(f"✓ Loaded Pyserini results for {len(pyserini_results)} datasets")
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        pyserini_results = {}
    
    print()
    
    bright_splits = list(BRIGHT_REFERENCE.keys())
    results = {
        "timestamp": datetime.now().isoformat(),
        "implementation": impl_name,
        "tokenizer": "lucene",
        "datasets": {},
    }
    
    total_datasets = len(bright_splits)
    print(f"Evaluating {total_datasets} BRIGHT datasets...")
    print()
    
    start_time = time.time()
    
    for idx, domain in enumerate(bright_splits, 1):
        dataset_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{total_datasets}] Evaluating: {domain}")
        print(f"{'='*70}")
        
        results["datasets"][domain] = {
            "reference": BRIGHT_REFERENCE[domain],
        }
        
        # Load saved Pyserini result
        if domain in pyserini_results:
            results["datasets"][domain]["pyserini"] = pyserini_results[domain]
            pyserini_score = pyserini_results[domain]["ndcg@10"]
            print(f"  Pyserini (saved): {pyserini_score:.2f}%")
        else:
            print(f"  ⚠️  No saved Pyserini result for {domain}")
            results["datasets"][domain]["pyserini"] = {"error": "not found in saved results"}
            pyserini_score = 0
        
        # Our implementation
        print(f"  Running {impl_name}...")
        sys.stdout.flush()
        try:
            our_result = evaluate_our_implementation_bright(domain, implementation)
            results["datasets"][domain]["ours"] = our_result
            our_score = our_result["ndcg@10"]
            dataset_time = our_result["time_seconds"]
            print(f"    nDCG@10: {our_score:.2f}% ({dataset_time:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["datasets"][domain]["ours"] = {"error": str(e)}
            our_score = 0
            dataset_time = time.time() - dataset_start_time
        
        # Summary
        ref = BRIGHT_REFERENCE[domain]
        gap = our_score - pyserini_score if pyserini_score > 0 else 0
        print(f"  Summary: Ref={ref}%, Pyserini={pyserini_score:.1f}%, Ours={our_score:.1f}%")
        if pyserini_score > 0:
            status = "✓" if abs(gap) < 1.0 else "⚠️" if abs(gap) < 3.0 else "❌"
            print(f"  Gap: Ours-Pyserini={gap:+.1f}% {status}")
        
        # Progress estimate
        elapsed = time.time() - start_time
        avg_time_per_dataset = elapsed / idx
        remaining_datasets = total_datasets - idx
        estimated_remaining = avg_time_per_dataset * remaining_datasets
        
        print(f"  Progress: {idx}/{total_datasets} datasets | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"Est. remaining: {estimated_remaining/60:.1f}m")
    
    # Compute averages
    pyserini_scores = [
        r["pyserini"]["ndcg@10"] for r in results["datasets"].values()
        if "pyserini" in r and "ndcg@10" in r["pyserini"] and "error" not in r["pyserini"]
    ]
    our_scores = [
        r["ours"]["ndcg@10"] for r in results["datasets"].values()
        if "ours" in r and "ndcg@10" in r["ours"] and "error" not in r["ours"]
    ]
    ref_scores = list(BRIGHT_REFERENCE.values())
    
    results["summary"] = {
        "avg_reference": float(np.mean(ref_scores)),
        "avg_pyserini": float(np.mean(pyserini_scores)) if pyserini_scores else 0,
        "avg_ours": float(np.mean(our_scores)) if our_scores else 0,
    }
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Ref':>8} {'Pyserini':>10} {'Ours':>10} {'Gap':>8} {'Status':>8}")
    print("-" * 70)
    
    for domain in bright_splits:
        ref = BRIGHT_REFERENCE[domain]
        pyserini = results["datasets"][domain].get("pyserini", {}).get("ndcg@10", 0)
        ours = results["datasets"][domain].get("ours", {}).get("ndcg@10", 0)
        gap = ours - pyserini if pyserini > 0 else 0
        
        # Status indicator
        if "error" in results["datasets"][domain].get("ours", {}):
            status = "ERROR"
        elif pyserini == 0:
            status = "NO_REF"
        elif abs(gap) < 1.0:
            status = "✓"
        elif abs(gap) < 3.0:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{domain:<25} {ref:>7.1f}% {pyserini:>9.1f}% {ours:>9.1f}% {gap:>+7.1f} {status:>8}")
    
    print("-" * 70)
    avg_gap = results["summary"]["avg_ours"] - results["summary"]["avg_pyserini"]
    print(f"{'AVERAGE':<25} {results['summary']['avg_reference']:>7.1f}% "
          f"{results['summary']['avg_pyserini']:>9.1f}% {results['summary']['avg_ours']:>9.1f}% "
          f"{avg_gap:>+7.1f}")
    
    # Save results
    output_dir = Path("experiment_summaries/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use different filename for different implementations
    if implementation == "composable":
        output_path = output_dir / "bright_composable_comparison_results.json"
    elif implementation == "freeform":
        output_path = output_dir / "bright_freeform_comparison_results.json"
    else:
        output_path = output_dir / "bright_comparison_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    total_elapsed = time.time() - start_time
    print(f"\n✓ Results saved to: {output_path}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f} seconds)")
    print(f"Completed: {datetime.now().isoformat()}")


def run_beir_comparison():
    """Run BEIR benchmark comparison."""
    from ranking_evolved.datasets.beir import BEIRLoader
    
    print("=" * 70)
    print("BEIR BENCHMARK COMPARISON")
    print("Pyserini Native vs Our Implementation (bm25_pyserini.py)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("NOTE: We use Pyserini's tokenizer for accurate tokenization.")
    print("      Pyserini search will run if results are not cached.")
    print()
    
    loader = BEIRLoader()
    available_datasets = loader.list_datasets()
    total_datasets = len(available_datasets)
    
    print(f"Evaluating {total_datasets} BEIR datasets...")
    print()
    
    # Load previous results
    results_path = Path("experiment_summaries/test/beir_comparison_results.json")
    previous_results = load_previous_results(results_path)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
    }
    
    start_time = time.time()
    
    for idx, dataset_name in enumerate(available_datasets, 1):
        dataset_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{total_datasets}] Evaluating: {dataset_name}")
        print(f"{'='*70}")
        
        results["datasets"][dataset_name] = {}
        
        # Get reference score
        reference_score = BEIR_REFERENCE.get(dataset_name)
        if reference_score:
            results["datasets"][dataset_name]["reference"] = reference_score
            print(f"  Reference (paper): {reference_score:.2f}%")
        
        # Skip Pyserini for very large datasets
        skip_pyserini = dataset_name in BEIR_SKIP_PYSERINI
        
        if skip_pyserini:
            print(f"  Skipping Pyserini (dataset too large, known to hang)")
            results["datasets"][dataset_name]["pyserini"] = {
                "ndcg@10": 0.0,
                "error": "skipped - dataset too large, known to cause hangs"
            }
            pyserini_score = 0
        # Check if we have previous Pyserini results
        elif (
            dataset_name in previous_results and
            "pyserini" in previous_results[dataset_name] and
            "ndcg@10" in previous_results[dataset_name]["pyserini"]
        ):
            results["datasets"][dataset_name]["pyserini"] = previous_results[dataset_name]["pyserini"]
            pyserini_score = previous_results[dataset_name]["pyserini"]["ndcg@10"]
            print(f"  Pyserini (cached): {pyserini_score:.2f}%")
        else:
            # For now, skip Pyserini native evaluation (can be added later)
            print(f"  Skipping Pyserini native (not implemented in unified script)")
            results["datasets"][dataset_name]["pyserini"] = {
                "ndcg@10": 0.0,
                "error": "not implemented in unified script - use run_beir_comparison.py"
            }
            pyserini_score = 0
        
        # Our implementation (always run)
        print(f"  Running our implementation...")
        sys.stdout.flush()
        try:
            our_result = evaluate_our_implementation_beir(dataset_name)
            results["datasets"][dataset_name]["ours"] = our_result
            our_score = our_result["ndcg@10"]
            print(f"    nDCG@10: {our_score:.2f}% ({our_result['time_seconds']:.1f}s)")
            if f"recall@{loader.metadata.get('recall_k')}" in our_result:
                recall_k = loader.metadata.get("recall_k")
                recall_val = our_result[f"recall@{recall_k}"]
                print(f"    Recall@{recall_k}: {recall_val:.2f}%")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["datasets"][dataset_name]["ours"] = {"error": str(e)}
            our_score = 0
        
        # Summary
        gap = our_score - pyserini_score if pyserini_score > 0 else 0
        ref_gap = our_score - reference_score if reference_score else None
        
        summary_parts = []
        if reference_score:
            summary_parts.append(f"Ref={reference_score:.1f}%")
        summary_parts.append(f"Pyserini={pyserini_score:.1f}%")
        summary_parts.append(f"Ours={our_score:.1f}%")
        print(f"  Summary: {' | '.join(summary_parts)}")
        
        if pyserini_score > 0:
            status = "✓" if abs(gap) < 1.0 else "⚠️" if abs(gap) < 3.0 else "❌"
            print(f"  Gap: Ours-Pyserini={gap:+.1f}% {status}")
        
        if ref_gap is not None:
            ref_status = "✓" if abs(ref_gap) < 2.0 else "⚠️" if abs(ref_gap) < 5.0 else "❌"
            print(f"  Gap: Ours-Ref={ref_gap:+.1f}% {ref_status}")
        
        # Progress estimate
        elapsed = time.time() - start_time
        avg_time_per_dataset = elapsed / idx
        remaining_datasets = total_datasets - idx
        estimated_remaining = avg_time_per_dataset * remaining_datasets
        
        print(f"  Progress: {idx}/{total_datasets} datasets | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"Est. remaining: {estimated_remaining/60:.1f}m")
    
    # Compute averages
    pyserini_scores = [
        r["pyserini"]["ndcg@10"] for r in results["datasets"].values()
        if "pyserini" in r and "ndcg@10" in r["pyserini"] and "error" not in r["pyserini"]
    ]
    our_scores = [
        r["ours"]["ndcg@10"] for r in results["datasets"].values()
        if "ours" in r and "ndcg@10" in r["ours"] and "error" not in r["ours"]
    ]
    
    results["summary"] = {
        "avg_pyserini": float(np.mean(pyserini_scores)) if pyserini_scores else 0,
        "avg_ours": float(np.mean(our_scores)) if our_scores else 0,
    }
    
    # Compute reference average
    ref_scores = [BEIR_REFERENCE.get(ds) for ds in available_datasets if BEIR_REFERENCE.get(ds)]
    avg_reference = float(np.mean(ref_scores)) if ref_scores else 0
    results["summary"]["avg_reference"] = avg_reference
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Ref':>8} {'Pyserini':>10} {'Ours':>10} {'Gap':>8} {'Status':>8}")
    print("-" * 70)
    
    for dataset_name in available_datasets:
        reference = BEIR_REFERENCE.get(dataset_name, 0)
        pyserini = results["datasets"][dataset_name].get("pyserini", {}).get("ndcg@10", 0)
        ours = results["datasets"][dataset_name].get("ours", {}).get("ndcg@10", 0)
        gap = ours - pyserini if pyserini > 0 else 0
        
        # Status indicator
        if "error" in results["datasets"][dataset_name].get("ours", {}):
            status = "ERROR"
        elif pyserini == 0:
            status = "NO_REF"
        elif abs(gap) < 1.0:
            status = "✓"
        elif abs(gap) < 3.0:
            status = "⚠️"
        else:
            status = "❌"
        
        ref_str = f"{reference:.1f}%" if reference > 0 else "-"
        print(f"{dataset_name:<25} {ref_str:>8} {pyserini:>9.1f}% {ours:>9.1f}% {gap:>+7.1f} {status:>8}")
    
    print("-" * 70)
    avg_gap = results["summary"]["avg_ours"] - results["summary"]["avg_pyserini"]
    print(f"{'AVERAGE':<25} {avg_reference:>7.1f}% {results['summary']['avg_pyserini']:>9.1f}% "
          f"{results['summary']['avg_ours']:>9.1f}% {avg_gap:>+7.1f}")
    
    # Save results
    output_dir = Path("experiment_summaries/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    total_elapsed = time.time() - start_time
    print(f"\n✓ Results saved to: {results_path}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f} seconds)")
    print(f"Completed: {datetime.now().isoformat()}")


def run_trec_dl_comparison():
    """Run TREC DL benchmark comparison."""
    from ranking_evolved.datasets.trec_dl import TRECDLLoader
    
    print("=" * 70)
    print("TREC DL BENCHMARK COMPARISON")
    print("Pyserini Native vs Our Implementation (bm25_pyserini.py)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("NOTE: We use Pyserini's tokenizer for accurate tokenization.")
    print("      Pyserini search will run if results are not cached.")
    print()
    
    loader = TRECDLLoader()
    available_datasets = loader.list_datasets()
    total_datasets = len(available_datasets)
    
    print(f"Evaluating {total_datasets} TREC DL datasets...")
    print()
    
    # Load previous results
    results_path = Path("experiment_summaries/test/trec_dl_comparison_results.json")
    previous_results = load_previous_results(results_path)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
    }
    
    start_time = time.time()
    
    for idx, dataset_name in enumerate(available_datasets, 1):
        dataset_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{total_datasets}] Evaluating: {dataset_name}")
        print(f"{'='*70}")
        
        results["datasets"][dataset_name] = {}
        
        # Get reference score
        reference_ndcg = TREC_DL_REFERENCE.get(dataset_name)
        if reference_ndcg:
            results["datasets"][dataset_name]["reference"] = reference_ndcg
            print(f"  Reference (paper): nDCG@10={reference_ndcg:.2f}%")
        
        # Check if we have previous Pyserini results
        has_pyserini = (
            dataset_name in previous_results and
            "pyserini" in previous_results[dataset_name] and
            "ndcg@10" in previous_results[dataset_name]["pyserini"]
        )
        
        if has_pyserini:
            results["datasets"][dataset_name]["pyserini"] = previous_results[dataset_name]["pyserini"]
            pyserini_ndcg = previous_results[dataset_name]["pyserini"]["ndcg@10"]
            pyserini_recall = previous_results[dataset_name]["pyserini"].get("recall@1000", 0)
            print(f"  Pyserini (cached):")
            print(f"    nDCG@10: {pyserini_ndcg:.2f}%")
            print(f"    Recall@1000: {pyserini_recall:.2f}%")
        else:
            # For now, skip Pyserini native evaluation (can be added later)
            print(f"  Skipping Pyserini native (not implemented in unified script)")
            results["datasets"][dataset_name]["pyserini"] = {
                "ndcg@10": 0.0,
                "error": "not implemented in unified script - use run_trec_dl_comparison.py"
            }
            pyserini_ndcg = 0
            pyserini_recall = 0
        
        # Our implementation (always run)
        print(f"  Running our implementation...")
        sys.stdout.flush()
        try:
            our_result = evaluate_our_implementation_trec_dl(dataset_name)
            results["datasets"][dataset_name]["ours"] = our_result
            our_ndcg = our_result["ndcg@10"]
            our_recall = our_result.get("recall@1000", 0)
            print(f"    nDCG@10: {our_ndcg:.2f}% ({our_result['time_seconds']:.1f}s)")
            print(f"    Recall@1000: {our_recall:.2f}%")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["datasets"][dataset_name]["ours"] = {"error": str(e)}
            our_ndcg = 0
            our_recall = 0
        
        # Summary
        ndcg_gap = our_ndcg - pyserini_ndcg if pyserini_ndcg > 0 else 0
        recall_gap = our_recall - pyserini_recall if pyserini_recall > 0 else 0
        ref_ndcg_gap = our_ndcg - reference_ndcg if reference_ndcg else None
        
        summary_parts = []
        if reference_ndcg:
            summary_parts.append(f"Ref={reference_ndcg:.1f}%")
        summary_parts.append(f"Pyserini={pyserini_ndcg:.1f}%")
        summary_parts.append(f"Ours={our_ndcg:.1f}%")
        
        print(f"  Summary (nDCG@10): {' | '.join(summary_parts)}")
        print(f"    Gap: Ours-Pyserini={ndcg_gap:+.1f}%")
        if ref_ndcg_gap is not None:
            print(f"    Gap: Ours-Ref={ref_ndcg_gap:+.1f}%")
        print(f"  Summary (Recall@1000): Pyserini={pyserini_recall:.1f}%, Ours={our_recall:.1f}% (gap={recall_gap:+.1f}%)")
        
        if pyserini_ndcg > 0:
            status = "✓" if abs(ndcg_gap) < 1.0 else "⚠️" if abs(ndcg_gap) < 3.0 else "❌"
            print(f"  Status: {status}")
        
        # Progress estimate
        elapsed = time.time() - start_time
        avg_time_per_dataset = elapsed / idx
        remaining_datasets = total_datasets - idx
        estimated_remaining = avg_time_per_dataset * remaining_datasets
        
        print(f"  Progress: {idx}/{total_datasets} datasets | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"Est. remaining: {estimated_remaining/60:.1f}m")
    
    # Compute averages
    pyserini_ndcg_scores = [
        r["pyserini"]["ndcg@10"] for r in results["datasets"].values()
        if "pyserini" in r and "ndcg@10" in r["pyserini"] and "error" not in r["pyserini"]
    ]
    our_ndcg_scores = [
        r["ours"]["ndcg@10"] for r in results["datasets"].values()
        if "ours" in r and "ndcg@10" in r["ours"] and "error" not in r["ours"]
    ]
    pyserini_recall_scores = [
        r["pyserini"].get("recall@1000", 0) for r in results["datasets"].values()
        if "pyserini" in r and "recall@1000" in r["pyserini"] and "error" not in r["pyserini"]
    ]
    our_recall_scores = [
        r["ours"].get("recall@1000", 0) for r in results["datasets"].values()
        if "ours" in r and "recall@1000" in r["ours"] and "error" not in r["ours"]
    ]
    
    results["summary"] = {
        "avg_pyserini_ndcg@10": float(np.mean(pyserini_ndcg_scores)) if pyserini_ndcg_scores else 0,
        "avg_ours_ndcg@10": float(np.mean(our_ndcg_scores)) if our_ndcg_scores else 0,
        "avg_pyserini_recall@1000": float(np.mean(pyserini_recall_scores)) if pyserini_recall_scores else 0,
        "avg_ours_recall@1000": float(np.mean(our_recall_scores)) if our_recall_scores else 0,
    }
    
    # Compute reference average
    ref_scores = [TREC_DL_REFERENCE.get(ds) for ds in available_datasets if TREC_DL_REFERENCE.get(ds)]
    avg_reference = float(np.mean(ref_scores)) if ref_scores else 0
    results["summary"]["avg_reference_ndcg@10"] = avg_reference
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<10} {'Metric':<15} {'Ref':>8} {'Pyserini':>10} {'Ours':>10} {'Gap':>8} {'Status':>8}")
    print("-" * 70)
    
    for dataset_name in available_datasets:
        reference = TREC_DL_REFERENCE.get(dataset_name, 0)
        pyserini = results["datasets"][dataset_name].get("pyserini", {})
        ours = results["datasets"][dataset_name].get("ours", {})
        
        pyserini_ndcg = pyserini.get("ndcg@10", 0)
        our_ndcg = ours.get("ndcg@10", 0)
        ndcg_gap = our_ndcg - pyserini_ndcg if pyserini_ndcg > 0 else 0
        
        pyserini_recall = pyserini.get("recall@1000", 0)
        our_recall = ours.get("recall@1000", 0)
        recall_gap = our_recall - pyserini_recall if pyserini_recall > 0 else 0
        
        # Status indicators
        if "error" in ours:
            status_ndcg = "ERROR"
            status_recall = "ERROR"
        elif pyserini_ndcg == 0:
            status_ndcg = "NO_REF"
            status_recall = "NO_REF"
        else:
            status_ndcg = "✓" if abs(ndcg_gap) < 1.0 else "⚠️" if abs(ndcg_gap) < 3.0 else "❌"
            status_recall = "✓" if abs(recall_gap) < 1.0 else "⚠️" if abs(recall_gap) < 3.0 else "❌"
        
        ref_str = f"{reference:.1f}%" if reference > 0 else "-"
        print(f"{dataset_name:<10} {'nDCG@10':<15} {ref_str:>8} {pyserini_ndcg:>9.1f}% {our_ndcg:>9.1f}% {ndcg_gap:>+7.1f} {status_ndcg:>8}")
        print(f"{'':<10} {'Recall@1000':<15} {'-':>8} {pyserini_recall:>9.1f}% {our_recall:>9.1f}% {recall_gap:>+7.1f} {status_recall:>8}")
    
    print("-" * 70)
    avg_ndcg_gap = results["summary"]["avg_ours_ndcg@10"] - results["summary"]["avg_pyserini_ndcg@10"]
    avg_recall_gap = results["summary"]["avg_ours_recall@1000"] - results["summary"]["avg_pyserini_recall@1000"]
    print(f"{'AVERAGE':<10} {'nDCG@10':<15} {avg_reference:>7.1f}% {results['summary']['avg_pyserini_ndcg@10']:>9.1f}% "
          f"{results['summary']['avg_ours_ndcg@10']:>9.1f}% {avg_ndcg_gap:>+7.1f}")
    print(f"{'':<10} {'Recall@1000':<15} {'-':>8} {results['summary']['avg_pyserini_recall@1000']:>9.1f}% "
          f"{results['summary']['avg_ours_recall@1000']:>9.1f}% {avg_recall_gap:>+7.1f}")
    
    # Save results
    output_dir = Path("experiment_summaries/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    total_elapsed = time.time() - start_time
    print(f"\n✓ Results saved to: {results_path}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f} seconds)")
    print(f"Completed: {datetime.now().isoformat()}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare our BM25 implementations to Pyserini on different benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run BRIGHT benchmark comparison
  python scripts/compare_to_pyserini.py --benchmark bright

  # Run BEIR benchmark comparison
  python scripts/compare_to_pyserini.py --benchmark beir

  # Run TREC DL benchmark comparison
  python scripts/compare_to_pyserini.py --benchmark trec_dl

  # Run with different implementation
  python scripts/compare_to_pyserini.py --benchmark bright --implementation composable
  python scripts/compare_to_pyserini.py --benchmark bright --implementation freeform
        """,
    )
    parser.add_argument(
        "--benchmark",
        choices=["bright", "beir", "trec_dl"],
        required=True,
        help="Which benchmark to run (bright, beir, or trec_dl)",
    )
    parser.add_argument(
        "--implementation",
        choices=["pyserini", "composable", "freeform"],
        default="pyserini",
        help="Which implementation to evaluate (default: pyserini). Note: only bright supports all implementations.",
    )
    
    args = parser.parse_args()
    
    if args.benchmark == "bright":
        run_bright_comparison(args.implementation)
    elif args.benchmark == "beir":
        if args.implementation != "pyserini":
            print("WARNING: BEIR comparison only supports pyserini implementation")
        run_beir_comparison()
    elif args.benchmark == "trec_dl":
        if args.implementation != "pyserini":
            print("WARNING: TREC DL comparison only supports pyserini implementation")
        run_trec_dl_comparison()
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
