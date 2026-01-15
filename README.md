# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). The BRIGHT dataset is used for evaluation.

## Project Structure

```
ranking-evolved/
├── src/ranking_evolved/
│   ├── bm25.py          # Single-file modular BM25 (OpenEvolve target)
│   └── metrics.py       # Evaluation metrics (NDCG, MAP, MRR, etc.)
├── benchmarks/
│   ├── bright_benchmark.py    # Comprehensive benchmark runner
│   ├── cross_validation.py    # Cross-validation across implementations/tokenizers
│   └── baselines/             # External library wrappers
│       ├── gensim_bm25.py     # Gensim BM25 models (OkapiBM25, AtireBM25, LuceneBM25)
│       └── pyserini_bm25.py   # Pyserini/Anserini (requires Java 21)
├── references/
│   ├── bm25_formulas.md       # BM25 variant formulas + common bugs
│   └── evolved_variants.md    # Archive of evolved formulas
├── evaluator_bright.py        # OpenEvolve evaluator
└── openevolve_config.yaml     # OpenEvolve configuration
```

## Quick Start

```bash
# Install dependencies (Python >= 3.11 required)
uv sync

# Run evaluation on a single domain
uv run python evaluator_bright.py src/ranking_evolved/bm25.py --k 10 --domain biology

# Run comprehensive benchmark
uv run python -m benchmarks.bright_benchmark --domains biology earth_science --k 10
```

## BM25 Implementation

The main BM25 implementation ([bm25.py](src/ranking_evolved/bm25.py)) provides a fully configurable BM25 scorer where you can independently select:

- **IDF strategy**: How to compute inverse document frequency
- **TF strategy**: How to compute term frequency saturation
- **Query term mode**: How to handle repeated terms in queries

### Quick Start

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config, Corpus, tokenize

# Create corpus from documents
docs = [["hello", "world"], ["hello", "there"], ["world", "news"]]
corpus = Corpus(docs, ids=["doc1", "doc2", "doc3"])

# Or load from HuggingFace dataset
corpus = Corpus.from_huggingface_dataset(dataset)

# Use a preset configuration
bm25 = BM25Unified(corpus, BM25Config.lucene())
indices, scores = bm25.rank(tokenize("hello world"), top_k=10)
```

### Preset Configurations

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config

# Standard variants
bm25 = BM25Unified(corpus, BM25Config.classic())     # Original Robertson BM25
bm25 = BM25Unified(corpus, BM25Config.lucene())      # Lucene-style (non-negative IDF)
bm25 = BM25Unified(corpus, BM25Config.atire())       # ATIRE variant
bm25 = BM25Unified(corpus, BM25Config.bm25l())       # Long document friendly
bm25 = BM25Unified(corpus, BM25Config.bm25_plus())   # Lower-bound bonus

# Special variants
bm25 = BM25Unified(corpus, BM25Config.pyserini())    # Pyserini-compatible (sums repeated query terms)
bm25 = BM25Unified(corpus, BM25Config.evolved())     # This project's best performer
```

### Custom Configuration

Mix and match any IDF strategy, TF strategy, and query term handling:

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config

config = BM25Config(
    idf="lucene",       # Options: classic, lucene, atire, bm25l, bm25+, clipped, evolved
    tf="evolved",       # Options: classic, bm25l, bm25+, atire, evolved
    query_mode="unique", # Options: unique, sum_all, saturated
    k1=0.9,             # TF saturation (default: 1.2)
    b=0.4,              # Length normalization (default: 0.75)
    k3=8.0,             # Query TF saturation, for query_mode="saturated" (default: 8.0)
    delta=0.5,          # Bonus for bm25l/bm25+ TF (default: 0.5)
)
bm25 = BM25Unified(corpus, config)
```

### Understanding the Options

#### IDF Strategies (Inverse Document Frequency)

Controls how rare/common terms are weighted:

| Strategy | Formula | When to use |
|----------|---------|-------------|
| `classic` | `log((N-df+0.5)/(df+0.5))` | Academic baseline (can go negative) |
| `lucene` | `log(1 + (N-df+0.5)/(df+0.5))` | General use (always positive) |
| `atire` | `log(N/df)` | Simpler formula |
| `bm25l` | `log((N+1)/(df+0.5))` | Non-negative, for BM25L |
| `bm25+` | `log((N+1)/df)` | Non-negative, for BM25+ |
| `clipped` | `clip(log(...), 0, 8)` | Prevents extreme values |
| `evolved` | `clip(log((N+0.5)/(df+0.5)), 0, 8)` | This project's best |

#### TF Strategies (Term Frequency Saturation)

Controls how term frequency affects scores:

| Strategy | Formula | When to use |
|----------|---------|-------------|
| `classic` | `tf*(k1+1)/(tf+k1*norm)` | Standard BM25 |
| `bm25l` | `((k1+1)*(c+δ))/(k1+c+δ)` | Long documents (adds bonus before saturation) |
| `bm25+` | `tf*(k1+1)/(tf+k1*norm) + δ` | Long documents (adds minimum boost) |
| `atire` | `(k1+1)*tf/(k1*norm+tf)` | Equivalent to classic, different form |
| `evolved` | `log1p(tf_raw * tf_sat)` | Log-damped, this project's best |

#### Query Term Modes

Controls how repeated query terms are handled:

| Mode | Behavior | When to use |
|------|----------|-------------|
| `unique` | Each unique term contributes once | Default, best for natural language queries |
| `sum_all` | Sum scores for all occurrences | Pyserini-compatible, for keyword queries |
| `saturated` | Apply `(k3+1)*qtf/(k3+qtf)` | When repetition signals emphasis |

### API Reference

```python
# Ranking
indices, scores = bm25.rank(query_tokens, top_k=10)

# Batch ranking
results = bm25.batch_rank([query1, query2, query3], top_k=10)

# Score a single document
score = bm25.score(query_tokens, doc_index)

# Inspect configuration
print(bm25.config)  # BM25Config(idf=LuceneIDF, tf=ClassicTF, query_mode=unique, k1=0.9, b=0.4)
```

### Performance Comparison (BRIGHT Biology Domain, Simple Tokenization)

All configurations tested with simple whitespace tokenizer on BRIGHT biology (57,359 documents, 103 queries):

| Configuration | NDCG@10 | MAP | MRR |
|--------------|---------|-----|-----|
| **Lucene IDF + Evolved TF (k1=0.9, b=0.4)** | **0.2318** | **0.1830** | **0.3404** |
| Lucene IDF + Evolved TF (k1=1.5, b=0.75) | 0.2235 | 0.1740 | 0.3237 |
| Evolved (k1=1.5, b=0.75) | 0.2219 | 0.1724 | 0.3188 |
| Lucene + saturated (k1=0.9, b=0.4) | 0.2167 | 0.1726 | 0.3226 |
| Lucene (k1=0.9, b=0.4) | 0.2100 | 0.1604 | 0.3154 |
| Pyserini-style (k1=0.9, b=0.4) | 0.2078 | 0.1685 | 0.3078 |
| BM25+ (k1=1.2, b=0.75, δ=1.0) | 0.1840 | 0.1437 | 0.2779 |
| Lucene (k1=1.2, b=0.75) | 0.0926 | 0.0702 | 0.1466 |
| ATIRE (k1=1.2, b=0.75) | 0.0924 | 0.0697 | 0.1470 |
| Clipped IDF + Classic TF (k1=1.5, b=0.75) | 0.0781 | 0.0557 | 0.1233 |
| Classic (k1=1.2, b=0.75) | 0.0761 | 0.0544 | 0.1225 |
| BM25L (k1=1.2, b=0.75, δ=0.5) | 0.0713 | 0.0531 | 0.1178 |
| Classic (k1=1.5, b=0.75) | 0.0665 | 0.0489 | 0.1074 |

**Key findings:**
- **Evolved TF** provides ~10% improvement over classic TF with same IDF
- **k1=0.9, b=0.4** significantly outperforms k1=1.2-1.5, b=0.75 (compensates for tokenization)
- **Lucene IDF** (non-negative) works better than Classic IDF (which can go negative)
- **query_mode=unique** (bag-of-words) is best for natural language queries

### Understanding BM25: The Core Formula

All BM25 variants share the same structure:

```
Score(doc, query) = Σ IDF(term) × TF(term, doc)
```

Where:
- **IDF (Inverse Document Frequency)**: How rare/discriminative is this term? Rare terms get higher weight.
- **TF (Term Frequency saturation)**: How relevant is this document to the term? More occurrences = higher score, but with diminishing returns.

The variants differ in **how they handle edge cases**:

### IDF Variants: Handling Common Terms

The classic IDF formula `log((N-df+0.5)/(df+0.5))` produces **negative values** for terms appearing in >50% of documents. Different variants handle this differently:

| Problem | Solution | Formula |
|---------|----------|---------|
| Negative IDF for common terms | **Add 1 inside log** (Lucene) | `log(1 + (N-df+0.5)/(df+0.5))` |
| Negative IDF for common terms | **Simpler ratio** (ATIRE) | `log(N/df)` |
| Negative IDF for common terms | **Non-negative numerator** (BM25L/BM25+) | `log((N+1)/(df+0.5))` |
| Extreme IDF values | **Clip to range** (Evolved) | `clip(log((N+0.5)/(df+0.5)), 0, 8)` |

### TF Variants: Handling Long Documents

The standard TF formula penalizes long documents. If a 1000-word doc and 100-word doc both mention "python" 5 times, the short doc scores higher. Sometimes this is wrong—the long doc might just have more context.

| Problem | Solution | How it works |
|---------|----------|--------------|
| Long docs unfairly penalized | **Add bonus δ** (BM25L) | Adds δ=0.5 to normalized TF before saturation |
| Long docs unfairly penalized | **Lower bound** (BM25+) | Adds δ=1.0 after TF saturation (minimum boost for any match) |
| TF saturation too aggressive | **Log damping** (Evolved) | `log1p(tf_raw × tf_sat)` instead of raw product |

### Query Term Handling

Most implementations treat queries as **bag-of-words**: each unique term contributes once, regardless of repetition. Some variants weight repeated query terms:

| Approach | When to use | Formula |
|----------|-------------|---------|
| **unique** (default) | Most cases—repetition in queries is usually incidental | Each term scores once |
| **sum_all** | Pyserini compatibility, keyword-style queries | Each occurrence adds to score |
| **saturated** | When repetition signals emphasis | `(k3+1)*qtf/(k3+qtf)` weights repeated terms with diminishing returns |

## Running OpenEvolve

```bash
# Set your API key
export OPENAI_API_KEY="your-key"

# Run evolution
uv run python openevolve/openevolve-run.py \
    src/ranking_evolved/bm25.py \
    evaluator_bright.py \
    --config openevolve_config.yaml
```

The evaluator computes NDCG@k, Precision@k, Recall@k, MAP, and MRR, using their average as `combined_score` for selection.

## Benchmarking

### Run Benchmark

```bash
# All variants on all domains
uv run python -m benchmarks.bright_benchmark

# Specific variants and domains
uv run python -m benchmarks.bright_benchmark \
    --variants evolved classic lucene bm25l \
    --domains biology earth_science economics \
    --k 10

# Include Gensim baseline (requires: uv sync --group benchmark)
uv run python -m benchmarks.bright_benchmark --include-gensim

# Run cross-validation (requires Java 21 for Pyserini)
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib
uv run python -m benchmarks.cross_validation
```

### BRIGHT Dataset Domains

- biology, earth_science, economics, psychology, robotics
- stackoverflow, sustainable_living, pony
- leetcode, aops, theoremqa_theorems, theoremqa_questions

## Evolved Scoring Formula

The current best-performing kernel (from OpenEvolve iteration 113):

**IDF:**
```
idf(t) = clip(log((N + 0.5) / (df(t) + 0.5)), 0, 8)
```

**Score kernel:**
```python
# Order-preserving unique query terms
unique_terms = list(dict.fromkeys(query))

# Standard BM25 TF saturation
norm = 1 - b + b * (doc_length / avg_doc_length)
tf_raw = (tf * (k1 + 1)) / (tf + k1 * norm)

# Additional saturation factor (evolved)
tf_sat = tf / (tf + k1 + 0.5)

# Log damping (evolved)
score = sum(idf * log1p(tf_raw * tf_sat))
```

**Parameters:** k1=1.5, b=0.75

## Benchmark Results

### Our BM25 Variants (Biology Domain)

| Variant | NDCG@10 | MAP | MRR | Precision@10 | Recall@10 |
|---------|---------|-----|-----|--------------|-----------|
| **evolved** | 0.2219 | 0.1724 | 0.3188 | 0.0796 | 0.2548 |
| **lucene** | 0.2235 | 0.1740 | 0.3237 | 0.0796 | 0.2548 |
| **bm25l** | 0.2235 | 0.1740 | 0.3237 | 0.0796 | 0.2548 |
| classic | 0.2050 | 0.1678 | 0.3152 | 0.0660 | 0.2153 |

### Cross-Validation: Implementation & Tokenization Comparison

We cross-validated our BM25 implementation against Pyserini/Anserini (the reference Lucene implementation) using different tokenization strategies.

| Implementation | Tokenizer | k1 | b | NDCG@10 | MAP | MRR |
|----------------|-----------|-----|------|---------|-----|-----|
| **Our BM25 (evolved TF)** | Lucene (via Pyserini) | 0.9 | 0.4 | **0.2524** | 0.2036 | 0.3772 |
| Our BM25 (evolved TF) | Simple whitespace | 0.9 | 0.4 | 0.2318 | 0.1830 | 0.3404 |
| Our BM25 (evolved TF) | Simple whitespace | 1.5 | 0.75 | 0.2235 | 0.1740 | 0.3237 |
| Our BM25 (classic TF) | Simple whitespace | 0.9 | 0.4 | 0.2100 | 0.1604 | 0.3154 |
| Our BM25 (evolved TF) | Lucene (via Pyserini) | 1.5 | 0.75 | 0.1875 | 0.1480 | 0.2830 |
| Our BM25 (classic TF) | Lucene (via Pyserini) | 0.9 | 0.4 | 0.1872 | 0.1506 | 0.2918 |
| Pyserini/Anserini | Lucene (native Java) | 0.9 | 0.4 | 0.1810 | 0.1420 | 0.2480 |
| Our BM25 (pyserini-style) | Lucene (via Pyserini) | 0.9 | 0.4 | 0.1719 | 0.1357 | 0.2338 |
| Our BM25 (classic TF) | Simple whitespace | 1.2 | 0.75 | 0.0926 | 0.0702 | 0.1466 |
| Pyserini/Anserini | Lucene (native Java) | 1.2 | 0.75 | 0.0793 | 0.0657 | 0.1431 |
| Our BM25 (classic TF) | Lucene (via Pyserini) | 1.2 | 0.75 | 0.0789 | 0.0624 | 0.1368 |

**Key Findings:**

1. **Evolved TF provides ~35% improvement** over classic TF with same tokenization (0.2524 vs 0.1872 with Lucene tokenizer).

2. **Tokenization has mixed effects** - Lucene tokenization helps evolved TF (+9%, 0.2318→0.2524) but doesn't help classic TF much (0.2100→0.1872).

3. **Our pyserini-style matches Pyserini** - When using `query_mode=sum_all`, our implementation (0.1719) closely matches actual Pyserini (0.1810), confirming the query term counting is the key difference.

4. **Parameters matter** - k1=0.9, b=0.4 outperforms k1=1.2, b=0.75 by ~2-3x across all configurations.

### Why Our Implementation Outperforms Pyserini

Investigation revealed the root cause of the 28% NDCG gap:

**Query Term Counting Difference:**
- **Our implementation**: Uses unique query terms (bag-of-words). Each term contributes to the score once, regardless of how many times it appears in the query.
- **Pyserini/Lucene**: Sums scores for ALL query term occurrences. If "light" appears 4x in the query, the IDF×TF contribution for "light" is added 4 times.

**Example Impact (Query 1 from BRIGHT Biology):**
```
Query contains: light (4x), heat (4x), insect (3x), led (3x)...

Document A: matches "light", "heat", "led" with high TF
  - Pyserini score: 31.86 (boosted by repeated query terms)
  - Our score: 8.47

Document B: matches "article", "number", "example" (low query TF)
  - Pyserini score: 14.26
  - Our score: 14.57
```

**Result**: Pyserini ranks Document A first (because matched terms repeat in query), while we rank Document B first. For BRIGHT's long-form queries, our bag-of-words approach better identifies relevant documents because query term repetition is often incidental (natural language), not a relevance signal.

**Additional Lucene 8.0+ Difference:**
Lucene 8.0+ removed the `(k1+1)` factor from the TF numerator for performance (see [LUCENE-8563](https://issues.apache.org/jira/browse/LUCENE-8563)). This doesn't affect ranking order but results in lower absolute scores.

### Baseline Comparisons (Biology Domain)

| Implementation | NDCG@10 | Notes |
|----------------|---------|-------|
| **Our BM25 (evolved TF)** | **0.2524** | Best: Lucene tokenizer, k1=0.9, b=0.4 |
| Our BM25 (evolved TF) | 0.2318 | Simple tokenizer, k1=0.9, b=0.4 |
| Our BM25 (classic TF) | 0.2100 | Simple tokenizer, k1=0.9, b=0.4 |
| Our BM25 (classic TF) | 0.1872 | Lucene tokenizer, k1=0.9, b=0.4 |
| Pyserini/Anserini | 0.1810 | Reference Lucene implementation |
| Our BM25 (pyserini-style) | 0.1719 | Lucene tokenizer, query_mode=sum_all |
| Gensim OkapiBM25 | 0.0900 | Vector-space IDF² issue + IDF clamping |
| Gensim LuceneBM25 | 0.0845 | Vector-space IDF² issue + missing (k1+1) in TF |
| Gensim AtireBM25 | 0.0838 | Vector-space IDF² issue |

**Why Gensim BM25 underperforms:**

Gensim's BM25 models use a vector-space approach where scoring is computed as `score = query_vec · doc_vec`. This has several issues:

1. **Zero IDF terms are ignored**: Terms appearing in exactly N/2 documents get IDF=0 (classic BM25 formula: `log((N-df+0.5)/(df+0.5))` = 0 when df=N/2). Gensim stores these as 0, so they contribute nothing to scores—even if they're discriminative.

2. **IDF² amplification**: The dot product computes `Σ (IDF × TF_q) × (IDF × TF_d)`, squaring the IDF contribution and over-weighting rare terms relative to common ones.

3. **LuceneBM25Model missing (k1+1)**: Uses `tf / (tf + k1 * norm)` instead of `tf * (k1+1) / (tf + k1 * norm)`, reducing scores by 2.5x (for k1=1.5).

4. **OkapiBM25Model inconsistent IDF clamping**: Negative IDF terms get clamped to `epsilon × avg_idf`, but zero-IDF terms are stored as 0—creating inconsistent treatment of common terms.

Run verification: `uv run python -m benchmarks.gensim_root_cause`

See [bm25_formulas.md](references/bm25_formulas.md) for detailed formula analysis.

## Full BRIGHT Evaluation (Evolved BM25, k=10)

| Split | Combined | Prec@10 | Rec@10 | nDCG@10 | MAP | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| biology | 0.2095 | 0.0796 | 0.2548 | 0.2219 | 0.1724 | 0.3188 |
| earth_science | 0.2790 | 0.1086 | 0.2999 | 0.2963 | 0.2483 | 0.4421 |
| economics | 0.1097 | 0.0534 | 0.1362 | 0.1155 | 0.0886 | 0.1549 |
| psychology | 0.0756 | 0.0337 | 0.0980 | 0.0749 | 0.0633 | 0.1080 |
| robotics | 0.1054 | 0.0426 | 0.1440 | 0.1063 | 0.0842 | 0.1499 |
| stackoverflow | 0.1622 | 0.0752 | 0.1909 | 0.1727 | 0.1474 | 0.2248 |
| sustainable_living | 0.1149 | 0.0463 | 0.1568 | 0.1194 | 0.0980 | 0.1542 |
| pony | 0.1802 | 0.1759 | 0.0917 | 0.1774 | 0.1067 | 0.3491 |
| aops | 0.0229 | 0.0135 | 0.0265 | 0.0191 | 0.0130 | 0.0422 |
| theoremqa_theorems | 0.0459 | 0.0118 | 0.0592 | 0.0472 | 0.0367 | 0.0744 |
| theoremqa_questions | 0.0532 | 0.0175 | 0.0872 | 0.0551 | 0.0483 | 0.0580 |
| leetcode | 0.1206 | 0.0430 | 0.2080 | 0.1267 | 0.0957 | 0.1295 |
| **macro avg** | **0.1233** | **0.0584** | **0.1461** | **0.1277** | **0.1002** | **0.1838** |

## Evolution History

| Date | Domain | NDCG@10 | Combined | Notes |
|------|--------|---------|----------|-------|
| 2025-12-06 | biology | 0.0813 | 0.0836 | Baseline BM25 |
| 2025-12-06 | biology | 0.1547 | 0.1506 | +smoothed IDF, +tf log damping |
| 2025-12-07 | biology | 0.1828 | 0.1748 | +tf saturation, tighter IDF |
| **2025-12-07** | **biology** | **0.2219** | **0.2095** | **+clipped IDF, unique terms, log1p** |
| 2025-12-08 | psychology | 0.0870 | 0.0847 | Psychology-focused run |

## References

- [BM25 Formulas Reference](references/bm25_formulas.md) - Complete mathematical formulas
- [Evolved Variants Archive](references/evolved_variants.md) - Historic evolved formulas
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) - Evolution framework
- [BRIGHT Dataset](https://huggingface.co/datasets/xlangai/BRIGHT) - Evaluation dataset

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run ruff format

# Type check
uv run mypy src/
```
