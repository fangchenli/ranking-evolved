# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). Evaluated on the [BRIGHT dataset](https://huggingface.co/datasets/xlangai/BRIGHT).

## Quick Start

```bash
# Install dependencies (Python >= 3.11 required)
uv sync

# Run evaluation on a single domain
uv run python evaluator_bright.py src/ranking_evolved/bm25.py --k 10 --domain biology

# Run comprehensive benchmark
uv run python -m benchmarks.bright_benchmark --domains biology earth_science --k 10
```

## Usage

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config, Corpus, tokenize

# Create corpus from documents
docs = [["hello", "world"], ["hello", "there"], ["world", "news"]]
corpus = Corpus(docs, ids=["doc1", "doc2", "doc3"])

# Use preset configurations
bm25 = BM25Unified(corpus, BM25Config.evolved())  # Best performer
bm25 = BM25Unified(corpus, BM25Config.lucene())   # Lucene-style
bm25 = BM25Unified(corpus, BM25Config.classic())  # Original Robertson BM25

# Or customize
config = BM25Config(
    idf="lucene",        # Options: classic, lucene, atire, bm25l, bm25+, clipped, evolved
    tf="evolved",        # Options: classic, bm25l, bm25+, atire, evolved
    query_mode="unique", # Options: unique, sum_all, saturated
    k1=0.9, b=0.4,       # Parameters (defaults: k1=1.2, b=0.75)
)
bm25 = BM25Unified(corpus, config)

# Rank documents
indices, scores = bm25.rank(tokenize("hello world"), top_k=10)
```

## Configuration Options

### IDF Strategies

| Strategy | Formula | Notes |
|----------|---------|-------|
| `classic` | $\log\frac{N - df + 0.5}{df + 0.5}$ | Can go negative for common terms |
| `lucene` | $\log\left(1 + \frac{N - df + 0.5}{df + 0.5}\right)$ | Always positive |
| `atire` | $\log\frac{N}{df}$ | Simpler formula |
| `evolved` | $\text{clip}\left(\log\frac{N + 0.5}{df + 0.5}, 0, 8\right)$ | Best performer |

### TF Strategies

| Strategy | Formula | Notes |
|----------|---------|-------|
| `classic` | $\frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \text{norm}}$ | Standard BM25 |
| `bm25l` | $\frac{(k_1 + 1)(c + \delta)}{k_1 + c + \delta}$ | Better for long docs |
| `bm25+` | Classic $+ \delta$ | Minimum boost for any match |
| `evolved` | $\log(1 + tf_{raw} \cdot tf_{sat})$ | Log-damped, best performer |

### Query Term Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `unique` | Each unique term contributes once | Default, best for short queries |
| `sum_all` | Sum scores for all occurrences | Pyserini/Anserini compatible (BoW) |
| `saturated` | Apply $\frac{(k_3 + 1) \cdot qtf}{k_3 + qtf}$ | Best for long queries with repetition |

**Query-Side BM25**: The `saturated` mode implements "Query-Side BM25" from the paper "Lighting the Way for BRIGHT" (Ge et al.), which applies BM25-style saturation to query term frequencies. This helps when query term repetition signals emphasis rather than being incidental.

## Best Configuration

**Lucene IDF + Evolved TF + saturated query mode (k1=0.9, b=0.4, k3=2.0)** achieves the best results on BRIGHT:

| Tokenizer | Query Mode | NDCG@10 | MAP | MRR |
|-----------|------------|---------|-----|-----|
| Lucene | saturated (k3=2.0) | **0.1451** | 0.1184 | 0.1988 |
| Lucene | unique | 0.1392 | 0.1122 | 0.1943 |
| Simple | saturated (k3=2.0) | 0.1350 | 0.1072 | 0.1947 |
| Simple | unique | 0.1284 | 0.1015 | 0.1894 |

*Macro average across all 12 BRIGHT domains*

Key findings:
- **Evolved TF** provides ~35% improvement over classic TF
- **k1=0.9, b=0.4** significantly outperforms default k1=1.2, b=0.75
- **Lucene tokenizer** adds ~8% improvement over simple whitespace
- **Saturated query mode** adds ~4% improvement over unique mode for long queries

## Full BRIGHT Evaluation

### With Simple Tokenizer

| Split | Queries | Docs | NDCG@10 | MAP | MRR |
|-------|--------:|-----:|--------:|----:|----:|
| biology | 103 | 57,359 | 0.2318 | 0.1830 | 0.3404 |
| earth_science | 116 | 121,249 | 0.2941 | 0.2457 | 0.4132 |
| economics | 103 | 50,220 | 0.1061 | 0.0823 | 0.1539 |
| psychology | 101 | 52,835 | 0.0772 | 0.0680 | 0.1097 |
| robotics | 101 | 61,961 | 0.0867 | 0.0739 | 0.1301 |
| stackoverflow | 117 | 107,081 | 0.1701 | 0.1474 | 0.2256 |
| sustainable_living | 108 | 60,792 | 0.1201 | 0.0977 | 0.1531 |
| pony | 112 | 7,894 | 0.2047 | 0.1226 | 0.4385 |
| leetcode | 142 | 413,932 | 0.1319 | 0.0982 | 0.1296 |
| aops | 111 | 188,002 | 0.0216 | 0.0162 | 0.0496 |
| theoremqa_theorems | 76 | 23,839 | 0.0387 | 0.0329 | 0.0663 |
| theoremqa_questions | 194 | 188,002 | 0.0583 | 0.0495 | 0.0624 |
| **macro avg** | 1,384 | 1,333,166 | **0.1284** | 0.1015 | 0.1894 |

### With Lucene Tokenizer

| Split | Queries | Docs | NDCG@10 | MAP | MRR |
|-------|--------:|-----:|--------:|----:|----:|
| biology | 103 | 57,359 | 0.2524 | 0.2036 | 0.3772 |
| earth_science | 116 | 121,249 | 0.3493 | 0.2884 | 0.4678 |
| economics | 103 | 50,220 | 0.1362 | 0.1083 | 0.1830 |
| psychology | 101 | 52,835 | 0.1053 | 0.0846 | 0.1441 |
| robotics | 101 | 61,961 | 0.1073 | 0.0882 | 0.1565 |
| stackoverflow | 117 | 107,081 | 0.1995 | 0.1623 | 0.2634 |
| sustainable_living | 108 | 60,792 | 0.1540 | 0.1243 | 0.2016 |
| pony | 112 | 7,894 | 0.0937 | 0.0715 | 0.2261 |
| leetcode | 142 | 413,932 | 0.1299 | 0.0939 | 0.1267 |
| aops | 111 | 188,002 | 0.0287 | 0.0180 | 0.0515 |
| theoremqa_theorems | 76 | 23,839 | 0.0543 | 0.0476 | 0.0648 |
| theoremqa_questions | 194 | 188,002 | 0.0604 | 0.0553 | 0.0694 |
| **macro avg** | 1,384 | 1,333,166 | **0.1392** | 0.1122 | 0.1943 |

**Improvement:** NDCG@10 0.1284 → 0.1392 (+8.4%)

### With Lucene Tokenizer + Saturated Query Mode (k3=2.0)

| Split | Queries | Docs | NDCG@10 | MAP | MRR |
|-------|--------:|-----:|--------:|----:|----:|
| biology | 103 | 57,359 | **0.2746** | 0.2309 | 0.4120 |
| earth_science | 116 | 121,249 | **0.3764** | 0.3074 | 0.4997 |
| economics | 103 | 50,220 | 0.1481 | 0.1190 | 0.1942 |
| psychology | 101 | 52,835 | 0.1253 | 0.1040 | 0.1743 |
| robotics | 101 | 61,961 | 0.1099 | 0.0902 | 0.1522 |
| stackoverflow | 117 | 107,081 | 0.1993 | 0.1669 | 0.2605 |
| sustainable_living | 108 | 60,792 | 0.1782 | 0.1469 | 0.2217 |
| pony | 112 | 7,894 | 0.0688 | 0.0548 | 0.1625 |
| leetcode | 142 | 413,932 | 0.1416 | 0.1019 | 0.1404 |
| aops | 111 | 188,002 | 0.0305 | 0.0218 | 0.0600 |
| theoremqa_theorems | 76 | 23,839 | 0.0382 | 0.0326 | 0.0497 |
| theoremqa_questions | 194 | 188,002 | 0.0501 | 0.0450 | 0.0585 |
| **macro avg** | 1,384 | 1,333,166 | **0.1451** | 0.1184 | 0.1988 |

**Improvement over unique mode:** NDCG@10 0.1392 → 0.1451 (+4.2%)

**Per-domain impact of saturated mode:**
- Best gains: psychology (+19%), sustainable_living (+16%), biology (+9%), economics (+9%)
- Losses: pony (-27%), theoremqa_theorems (-30%) — these domains have shorter queries

Run evaluation:
```bash
# Simple tokenizer (default: unique query mode)
uv run python -m benchmarks.full_bright_evaluation

# With saturated query mode
uv run python -m benchmarks.full_bright_evaluation --query-mode saturated --k3 2.0

# Lucene tokenizer (requires Java 21)
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib
uv run python -m benchmarks.full_bright_evaluation --lucene --query-mode saturated --k3 2.0
```

## Baseline Comparisons (Biology)

| Implementation | NDCG@10 | Notes |
|----------------|--------:|-------|
| **Our BM25 (evolved TF + saturated)** | **0.2746** | Lucene tokenizer, k1=0.9, b=0.4, k3=2.0 |
| Our BM25 (evolved TF + unique) | 0.2524 | Lucene tokenizer, k1=0.9, b=0.4 |
| Paper Query-Side BM25 | 0.197 | From "Lighting the Way for BRIGHT" |
| Paper Anserini BoW | 0.182 | From "Lighting the Way for BRIGHT" |
| Our BM25 (classic TF) | 0.1872 | Lucene tokenizer, k1=0.9, b=0.4 |
| Pyserini/Anserini | 0.1810 | Reference Lucene implementation |
| Gensim OkapiBM25 | 0.0900 | Vector-space IDF² issue |

### Why We Outperform Pyserini and Paper Results

1. **Evolved TF formula**: Our log-damped TF saturation provides better scoring than classic BM25
2. **Query-Side BM25 with k3=2.0**: We found k3=2.0 works better than the paper's k3=8.0 for BRIGHT
3. **Combined innovations**: Evolved TF + saturated query mode = 39% better than paper's Query-Side BM25 (0.2746 vs 0.197)

## Evolved Scoring Formula

The best-performing formula (from OpenEvolve iteration 113):

**IDF:**
$$\text{IDF}(t) = \text{clip}\left(\log\frac{N + 0.5}{df(t) + 0.5}, 0, 8\right)$$

**TF Saturation:**
$$\text{norm} = 1 - b + b \cdot \frac{|d|}{\text{avgdl}}$$
$$tf_{raw} = \frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \text{norm}}$$
$$tf_{sat} = \frac{tf}{tf + k_1 + 0.5}$$

**Score:**
$$\text{Score}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \log(1 + tf_{raw} \cdot tf_{sat})$$

**Variable definitions:**
- $tf$ — Raw term frequency (count of term $t$ in document $d$)
- $\text{norm}$ — Length normalization factor
- $tf_{raw}$ — Standard BM25 TF saturation (Robertson formula)
- $tf_{sat}$ — Additional saturation factor (evolved innovation)
- $qtf$ — Query term frequency (used in `saturated` query mode)

## Project Structure

```
ranking-evolved/
├── src/ranking_evolved/
│   ├── bm25.py          # Modular BM25 implementation
│   └── metrics.py       # Evaluation metrics (NDCG, MAP, MRR, etc.)
├── benchmarks/
│   ├── bright_benchmark.py    # Comprehensive benchmark runner
│   ├── cross_validation.py    # Cross-validation across implementations
│   └── baselines/             # External library wrappers
├── references/
│   ├── bm25_formulas.md       # BM25 variant formulas
│   └── evolved_variants.md    # Archive of evolved formulas
├── evaluator_bright.py        # OpenEvolve evaluator
└── openevolve_config.yaml     # OpenEvolve configuration
```

## Running OpenEvolve

```bash
export OPENAI_API_KEY="your-key"
uv run python openevolve/openevolve-run.py \
    src/ranking_evolved/bm25.py \
    evaluator_bright.py \
    --config openevolve_config.yaml
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff format
uv run mypy src/
```

## References

- [BM25 Formulas Reference](references/bm25_formulas.md)
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
- [BRIGHT Dataset](https://huggingface.co/datasets/xlangai/BRIGHT)
