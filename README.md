# ranking-evolved

Evolving BM25 scoring functions with [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). Evaluated on the [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) and [BEIR](https://github.com/beir-cellar/beir) benchmarks.

## Quick Start

```bash
# Install dependencies (Python >= 3.11)
uv sync

# Run evaluation on BRIGHT
uv run python evaluator.py src/ranking_evolved/bm25_composable_fast.py --bright biology

# Run evaluation on BEIR
uv run python evaluator.py src/ranking_evolved/bm25_composable_fast.py --beir scifact
```

## Usage

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config, Corpus, LuceneTokenizer

tokenizer = LuceneTokenizer()

docs = [
    tokenizer("Protein folding is essential for cell function"),
    tokenizer("Machine learning models for protein structure prediction"),
    tokenizer("Cell biology and molecular mechanisms"),
]
corpus = Corpus(docs, ids=["doc1", "doc2", "doc3"])

config = BM25Config.evolved()
bm25 = BM25Unified(corpus, config)

indices, scores = bm25.rank(tokenizer("protein folding"), top_k=10)
```

### Preset Configurations

```python
BM25Config.evolved()    # Best overall (evolved TF + IDF, k1=0.9, b=0.4)
BM25Config.lucene()     # Lucene/Pyserini compatible
BM25Config.classic()    # Robertson BM25 (k1=1.5, b=0.75)
BM25Config.bm25l()      # BM25L (better for long documents)
BM25Config.bm25_plus()  # BM25+ (lower-bound guarantee)
```

## Evaluation

```bash
# Unified evaluator — mix BRIGHT and BEIR datasets
uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --bright biology,earth_science --beir scifact,nfcorpus

# All BRIGHT domains
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --bright all

# All BEIR datasets
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --beir all

# Fast iteration with query sampling
uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --bright biology --sample-queries 20
```

Options: `--tokenizer lucene|simple`, `--k 10`, `--sample-queries N`

### Running Baselines

```bash
# Compare all implementations and save to results/baselines/
./run_baselines.sh
```

## Running OpenEvolve

```bash
export EVAL_EXCLUDE_DATASETS="dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora,webis-touche2020,cqadupstack,leetcode,aops,theoremqa_questions,robotics,psychology,sustainable_living"
export OPENAI_API_KEY="your-key"

# Three seed program strategies
uv run python -m openevolve.cli src/ranking_evolved/bm25_constrained_fast.py evaluator_parallel.py \
  --config openevolve_config_constrained_fast.yaml --output openevolve_output_constrained_fast

uv run python -m openevolve.cli src/ranking_evolved/bm25_composable_fast.py evaluator_parallel.py \
  --config openevolve_config_composable.yaml --output openevolve_output_composable_fast

uv run python -m openevolve.cli src/ranking_evolved/bm25_freeform_fast.py evaluator_parallel.py \
  --config openevolve_config_freeform.yaml --output openevolve_output_freeform_fast
```

See [docs/OPENEVOLVE_RUN_GUIDE.md](docs/OPENEVOLVE_RUN_GUIDE.md) for details on prompts, outputs, and the evolution database.

## Project Structure

```
ranking-evolved/
├── src/ranking_evolved/
│   ├── bm25.py                    # Core BM25 implementation
│   ├── bm25_constrained_fast.py   # OpenEvolve seed: constrained search space
│   ├── bm25_composable_fast.py    # OpenEvolve seed: composable primitives
│   ├── bm25_freeform_fast.py      # OpenEvolve seed: freeform edits
│   ├── bm25_classic.py            # Vanilla Robertson BM25
│   ├── metrics.py                 # NDCG, MAP, MRR, precision, recall
│   └── datasets.py                # Dataset loading utilities
├── evaluator.py                   # Unified multi-benchmark evaluator
├── evaluator_parallel.py          # Parallel evaluator (used by OpenEvolve)
├── evaluator_bright.py            # BRIGHT-only evaluator
├── evaluator_beir.py              # BEIR-only evaluator
├── benchmarks/                    # Full benchmark runners
├── tests/                         # Unit tests
├── docs/                          # Detailed results and analysis
│   └── results.md                 # Full evaluation tables and scoring details
└── references/                    # BM25 formula derivations
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff format
uv run mypy src/
```

## References

- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — Evolutionary algorithm framework
- [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) — Benchmark for reasoning-intensive retrieval
- [BEIR](https://github.com/beir-cellar/beir) — Zero-shot information retrieval benchmark
- [BM25 Formulas](references/bm25_formulas.md) — Detailed formula derivations
- [Detailed Results](docs/results.md) — Full evaluation tables, scoring component details, hyperparameter search
