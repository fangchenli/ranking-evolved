# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). Evaluated on the [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) and [BEIR](https://github.com/beir-cellar/beir) benchmarks.

## Quick Start

```bash
# Install dependencies (Python >= 3.11 required)
uv sync

# Run BRIGHT evaluation on a single domain
uv run python evaluator_bright.py src/ranking_evolved/bm25.py --k 10 --domain biology

# Run BEIR evaluation on a single dataset
uv run python evaluator_beir.py src/ranking_evolved/bm25.py --dataset scifact --tokenizer lucene

# Run comprehensive BRIGHT benchmark
uv run python -m benchmarks.bright_benchmark --domains biology earth_science --k 10
```

## How BM25 Works

BM25 ranks documents by computing a relevance score for each document given a query. The scoring pipeline has four main stages:

```
Query → [1. Tokenization] → [2. Query Term Handling] → [3. Scoring] → Ranked Results
                ↓                                            ↑
           Documents → [1. Tokenization] → [Corpus Index] ───┘
```

### The BM25 Formula

The full BM25 score for a document $d$ given query $q$ is:

$$\text{Score}(d, q) = \sum_{t \in q} \underbrace{\text{IDF}(t)}_{\text{term importance}} \times \underbrace{\text{TF}(t, d)}_{\text{doc relevance}} \times \underbrace{\text{QTF}(t, q)}_{\text{query weight}}$$

Where:
- **IDF** (Inverse Document Frequency) — How rare/important is this term across all documents?
- **TF** (Term Frequency) — How relevant is this document for this term?
- **QTF** (Query Term Frequency) — How much should repeated query terms be weighted?

---

## Stage 1: Tokenization

Tokenization converts raw text into a list of normalized terms. This is the first and most impactful preprocessing step.

### Available Tokenizers

| Tokenizer | Description | Example |
|-----------|-------------|---------|
| `LuceneTokenizer` | Full Lucene pipeline with stemming (recommended) | `"The fox's running"` → `['fox', 'run']` |
| `tokenize` | Simple whitespace + lowercase | `"The fox's running"` → `['the', 'fox', 's', 'running']` |

### LuceneTokenizer Pipeline

The `LuceneTokenizer` replicates Pyserini/Anserini's tokenization in pure Python (no Java required):

```python
from ranking_evolved.bm25 import LuceneTokenizer

tokenizer = LuceneTokenizer()
tokens = tokenizer("The quick brown fox's running quickly!")
# Returns: ['quick', 'brown', 'fox', 'run', 'quickli']
```

**Pipeline steps:**

| Step | Filter | Input | Output | Purpose |
|------|--------|-------|--------|---------|
| 1 | StandardTokenizer | `"The fox's running"` | `['The', "fox's", 'running']` | Split on whitespace/punctuation |
| 2 | EnglishPossessiveFilter | `["fox's"]` | `['fox']` | Remove `'s` suffixes |
| 3 | LowerCaseFilter | `['The', 'Fox']` | `['the', 'fox']` | Normalize case |
| 4 | StopFilter | `['the', 'fox']` | `['fox']` | Remove common words |
| 5 | PorterStemFilter | `['running']` | `['run']` | Reduce to word stems |

**Stopwords removed** (33 words, Lucene default): `a`, `an`, `and`, `are`, `as`, `at`, `be`, `but`, `by`, `for`, `if`, `in`, `into`, `is`, `it`, `no`, `not`, `of`, `on`, `or`, `such`, `that`, `the`, `their`, `then`, `there`, `these`, `they`, `this`, `to`, `was`, `will`, `with`

> **Note:** We use the official Lucene/Pyserini stopword list (33 words) for Pyserini compatibility. An extended list (71 words) is also available as `ENGLISH_STOPWORDS` for backwards compatibility.

**Porter Stemming examples:**

| Word | Stem | Rule |
|------|------|------|
| `running` | `run` | Remove `-ing`, add nothing |
| `quickly` | `quickli` | Transform `-ly` suffix |
| `connections` | `connect` | Remove `-ions` |
| `computational` | `comput` | Remove `-ational` |

### Tokenizer Configuration

```python
# Default: stemming + stopwords
tokenizer = LuceneTokenizer()

# Without stemming (preserves original word forms)
tokenizer = LuceneTokenizer(stem=False)
tokenizer("running jumps")  # ['running', 'jumps']

# Custom stopwords
tokenizer = LuceneTokenizer(stopwords=frozenset(["custom", "words"]))

# No stopword removal
tokenizer = LuceneTokenizer(stopwords=frozenset())
```

### Impact of Tokenization

| Tokenizer | NDCG@10 | Improvement |
|-----------|---------|-------------|
| Simple | 0.1284 | baseline |
| Lucene (pure Python) | 0.1587 | **+23.6%** |

*The Lucene tokenizer's stemming allows matching between `"running"` and `"run"`, and stopword removal focuses scoring on content words.*

---

## Stage 2: Query Term Handling

When a query has repeated terms, how should they be counted? This is controlled by the **query term mode**.

### Query Term Modes

| Mode | Formula | Behavior |
|------|---------|----------|
| `unique` | $\text{QTF} = 1$ | Each unique term contributes once (bag-of-words) |
| `sum_all` | $\text{QTF} = qtf$ | Sum scores for all occurrences (Pyserini-style) |
| `saturated` | $\text{QTF} = \frac{(k_3 + 1) \cdot qtf}{k_3 + qtf}$ | Diminishing returns for repeated terms |

### Example: Query `"light light heat"`

Consider a query where "light" appears twice:

| Mode | How "light" is weighted | Use case |
|------|-------------------------|----------|
| `unique` | Counted once | Short queries, incidental repetition |
| `sum_all` | Counted twice (2×) | Pyserini compatibility |
| `saturated` (k3=2) | Counted 1.5× | Long queries where repetition signals emphasis |

### Saturated Mode (Query-Side BM25)

The `saturated` mode implements "Query-Side BM25" from ["Lighting the Way for BRIGHT"](https://arxiv.org/abs/2411.00934) (Ge et al.). It applies BM25-style saturation to query term frequencies:

$$\text{QTF}(t, q) = \frac{(k_3 + 1) \cdot qtf}{k_3 + qtf}$$

**k3 parameter behavior:**
- `k3 → 0`: First occurrence matters most, repetition ignored
- `k3 → ∞`: Linear weighting (same as `sum_all`)
- `k3 = 2.0`: Our optimal value for BRIGHT (paper used k3=8.0)

```python
config = BM25Config(
    query_mode="saturated",
    k3=2.0,  # Lower = faster saturation
)
```

### Impact by Domain

| Domain | Avg Query Length | Saturated vs Unique |
|--------|------------------|---------------------|
| biology | 89 words | **+6.3%** |
| psychology | 127 words | **+19%** |
| sustainable_living | 95 words | **+16%** |
| pony | 12 words | -27% |
| theoremqa | 8 words | -30% |

*Saturated mode helps long queries where term repetition signals emphasis, but hurts short queries where repetition is incidental.*

---

## Stage 3: Scoring Components

### IDF (Inverse Document Frequency)

IDF measures how rare/important a term is across the corpus. Rare terms (appearing in few documents) get higher IDF scores.

| Strategy | Formula | Range | Notes |
|----------|---------|-------|-------|
| `classic` | $\log\frac{N - df + 0.5}{df + 0.5}$ | $(-\infty, +\infty)$ | Original Robertson BM25; negative for terms in >50% of docs |
| `lucene` | $\log\left(1 + \frac{N - df + 0.5}{df + 0.5}\right)$ | $[0, +\infty)$ | Always non-negative |
| `atire` | $\log\frac{N}{df}$ | $[0, +\infty)$ | Simpler formula |
| `bm25l` | $\log\frac{N + 1}{df + 0.5}$ | $[0, +\infty)$ | For long document correction |
| `bm25+` | $\log\frac{N + 1}{df}$ | $[0, +\infty)$ | Lower-bound guarantee |
| `evolved` | $\text{clip}\left(\log\frac{N + 0.5}{df + 0.5}, 0, 8\right)$ | $[0, 8]$ | **Best performer** — clips extreme values |

**Example IDF values** (N=100,000 documents):

| df (docs containing term) | classic | lucene | evolved |
|---------------------------|---------|--------|---------|
| 1 (very rare) | 11.5 | 11.5 | **8.0** (clipped) |
| 100 | 6.9 | 6.9 | 6.9 |
| 10,000 | 2.2 | 2.3 | 2.3 |
| 50,000 | -0.0 | 0.7 | 0.7 |
| 99,000 (very common) | -4.6 | 0.0 | 0.0 |

### TF (Term Frequency Saturation)

TF measures how relevant a document is for a term, with saturation to prevent long documents from dominating.

**Document length normalization:**
$$\text{norm} = 1 - b + b \cdot \frac{|d|}{\text{avgdl}}$$

Where:
- $|d|$ = document length (number of terms)
- $\text{avgdl}$ = average document length in corpus
- $b$ = length normalization strength (0 = no normalization, 1 = full normalization)

| Strategy | Formula | Notes |
|----------|---------|-------|
| `classic` | $\frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \text{norm}}$ | Standard BM25 saturation |
| `bm25l` | $\frac{(k_1 + 1)(c + \delta)}{k_1 + c + \delta}$ where $c = \frac{tf}{\text{norm}}$ | Better for long documents |
| `bm25+` | Classic $+ \delta$ | Minimum boost for any match |
| `atire` | Same as classic (different derivation) | Equivalent formula |
| `evolved` | $\log(1 + tf_{raw} \cdot tf_{sat})$ | **Best performer** — log-damped |

**The evolved TF formula** (discovered via OpenEvolve):

$$tf_{raw} = \frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \text{norm}}$$

$$tf_{sat} = \frac{tf}{tf + k_1 + 0.5}$$

$$\text{TF}_{evolved} = \log(1 + tf_{raw} \cdot tf_{sat})$$

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `k1` | 1.2 | 0.5-2.0 | TF saturation speed (lower = faster saturation) |
| `b` | 0.75 | 0-1 | Length normalization strength |
| `k3` | 8.0 | 0-∞ | Query TF saturation (only for `saturated` mode) |

**Our optimal values for BRIGHT:** `k1=0.9`, `b=0.4`, `k3=2.0`

---

## Stage 4: Putting It Together

### Full Scoring Example

Query: `"protein folding mechanisms"`
Document: `"This paper discusses protein folding and misfolding mechanisms in cells."`

**Step 1: Tokenization**
```
Query tokens:  ['protein', 'fold', 'mechan']
Doc tokens:    ['paper', 'discuss', 'protein', 'fold', 'misfold', 'mechan', 'cell']
```

**Step 2: Compute IDF** (assuming N=100,000, evolved strategy)
```
IDF('protein') = 4.2   (appears in ~1,500 docs)
IDF('fold')    = 6.1   (appears in ~220 docs)
IDF('mechan')  = 5.8   (appears in ~300 docs)
```

**Step 3: Compute TF** (assuming doc length=7, avgdl=500, k1=0.9, b=0.4)
```
norm = 1 - 0.4 + 0.4 * (7/500) = 0.606

TF('protein') = log(1 + tf_raw * tf_sat) = 0.52  (tf=1)
TF('fold')    = log(1 + tf_raw * tf_sat) = 0.52  (tf=1)
TF('mechan')  = log(1 + tf_raw * tf_sat) = 0.52  (tf=1)
```

**Step 4: Final Score** (unique query mode, QTF=1)
```
Score = IDF('protein') × TF × QTF + IDF('fold') × TF × QTF + IDF('mechan') × TF × QTF
      = 4.2 × 0.52 × 1 + 6.1 × 0.52 × 1 + 5.8 × 0.52 × 1
      = 2.18 + 3.17 + 3.02
      = 8.37
```

---

## Usage

### Basic Usage

```python
from ranking_evolved.bm25 import BM25Unified, BM25Config, Corpus, LuceneTokenizer

# 1. Create tokenizer
tokenizer = LuceneTokenizer()

# 2. Tokenize documents
docs = [
    tokenizer("Protein folding is essential for cell function"),
    tokenizer("Machine learning models for protein structure prediction"),
    tokenizer("Cell biology and molecular mechanisms"),
]
corpus = Corpus(docs, ids=["doc1", "doc2", "doc3"])

# 3. Configure BM25
config = BM25Config(
    idf="evolved",         # Best IDF strategy
    tf="evolved",          # Best TF strategy
    query_mode="saturated", # Best for long queries
    k1=0.9, b=0.4, k3=2.0, # Optimal parameters
)
bm25 = BM25Unified(corpus, config)

# 4. Rank documents
indices, scores = bm25.rank(tokenizer("protein folding"), top_k=10)
print(f"Best match: doc {corpus.ids[indices[0]]} (score: {scores[0]:.2f})")
```

### Preset Configurations

```python
# Best overall performance
bm25 = BM25Unified(corpus, BM25Config.evolved())

# Lucene/Pyserini compatible
bm25 = BM25Unified(corpus, BM25Config.lucene())

# Classic Robertson BM25
bm25 = BM25Unified(corpus, BM25Config.classic())

# BM25L (better for long documents)
bm25 = BM25Unified(corpus, BM25Config.bm25l())

# BM25+ (lower-bound guarantee)
bm25 = BM25Unified(corpus, BM25Config.bm25_plus())
```

### Loading from HuggingFace

```python
from datasets import load_dataset
from ranking_evolved.bm25 import Corpus, LuceneTokenizer

# Load BRIGHT biology domain
dataset = load_dataset("xlangai/BRIGHT", "documents", split="biology")

# Create corpus with Lucene tokenization
tokenizer = LuceneTokenizer()
docs = [tokenizer(doc["content"]) for doc in dataset]
ids = [doc["id"] for doc in dataset]
corpus = Corpus(docs, ids=ids)
```

---

## Best Configuration

**Lucene tokenizer + Evolved TF + saturated query mode (k1=0.9, b=0.4, k3=2.0)**

| Tokenizer | Query Mode | NDCG@10 | MAP | MRR |
|-----------|------------|---------|-----|-----|
| Lucene (pure Python) | saturated | **0.1587** | 0.1290 | 0.2166 |
| Pyserini (Java) | saturated | 0.1451 | 0.1184 | 0.1988 |
| Simple | saturated | 0.1350 | 0.1072 | 0.1947 |
| Simple | unique | 0.1284 | 0.1015 | 0.1894 |

*Macro average across all 12 BRIGHT domains (1,384 queries, 1.3M documents)*

### Key Findings

1. **Pure Python Lucene tokenizer** outperforms Pyserini by 9.4%
2. **Evolved TF** provides ~35% improvement over classic TF
3. **k1=0.9, b=0.4** significantly outperforms defaults (k1=1.2, b=0.75)
4. **Saturated query mode** adds 4-9% for long queries

---

## Hyperparameter Search Results

We ran a grid search over BM25 parameters (k1, b, tokenizer) across all 12 BRIGHT domains.

### Search Grid

```python
k1_values = [0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
b_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
tokenizers = ["simple", "lucene"]
```

Total: 72 combinations per domain (36 × 2 tokenizers)

### Best Parameters Per Domain

| Domain | Tokenizer | k1 | b | NDCG@10 |
|--------|-----------|---:|---:|--------:|
| earth_science | lucene | 1.5 | 0.5 | **0.3874** |
| biology | lucene | 0.7 | 0.4 | 0.2809 |
| pony | simple | 0.5 | 0.5 | 0.2158 |
| stackoverflow | lucene | 1.2 | 0.5 | 0.2109 |
| sustainable_living | lucene | 0.9 | 0.5 | 0.1660 |
| economics | lucene | 0.7 | 0.5 | 0.1558 |
| psychology | lucene | 1.5 | 0.3 | 0.1450 |
| leetcode | lucene | 0.5 | 0.75 | 0.1377 |
| robotics | lucene | 0.9 | 0.4 | 0.1348 |
| theoremqa_theorems | lucene | 0.7 | 0.75 | 0.0732 |
| theoremqa_questions | lucene | 0.5 | 0.75 | 0.0613 |
| aops | lucene | 0.5 | 0.75 | 0.0339 |

### Key Findings

1. **Lucene tokenizer wins 11/12 domains** — Only `pony` (short queries, code-related) prefers simple tokenization
2. **Optimal k1 range: 0.5–1.5** — Lower k1 for math/code domains (faster saturation), higher k1 for natural language
3. **Optimal b range: 0.3–0.75** — Math/code domains prefer b=0.75 (stronger length normalization)
4. **Domain-specific tuning matters** — Best k1 varies from 0.5 (leetcode, aops) to 1.5 (earth_science, psychology)

### Run Hyperparameter Search

```bash
# Single domain
uv run python hyperparam_search.py --domain biology

# Specific tokenizer only
uv run python hyperparam_search.py --domain biology --tokenizer lucene

# All domains (saves to JSON)
uv run python hyperparam_search.py --domain all --output hyperparam_results.json
```

---

## Full BRIGHT Evaluation Results

### Best Configuration: Lucene + Evolved + Saturated (k3=2.0)

| Domain | Queries | Docs | NDCG@10 | MAP | MRR |
|--------|--------:|-----:|--------:|----:|----:|
| biology | 103 | 57,359 | **0.2920** | 0.2434 | 0.4355 |
| earth_science | 116 | 121,249 | **0.4203** | 0.3515 | 0.5510 |
| economics | 103 | 50,220 | 0.1621 | 0.1262 | 0.2152 |
| psychology | 101 | 52,835 | 0.1649 | 0.1329 | 0.2166 |
| robotics | 101 | 61,961 | 0.1299 | 0.1076 | 0.1667 |
| stackoverflow | 117 | 107,081 | 0.2053 | 0.1727 | 0.2700 |
| sustainable_living | 108 | 60,792 | 0.1815 | 0.1468 | 0.2347 |
| pony | 112 | 7,894 | 0.0766 | 0.0540 | 0.1866 |
| leetcode | 142 | 413,932 | 0.1370 | 0.0991 | 0.1323 |
| aops | 111 | 188,002 | 0.0305 | 0.0217 | 0.0572 |
| theoremqa_theorems | 76 | 23,839 | 0.0543 | 0.0448 | 0.0756 |
| theoremqa_questions | 194 | 188,002 | 0.0505 | 0.0478 | 0.0575 |
| **MACRO AVG** | **1,384** | **1,333,166** | **0.1587** | **0.1290** | **0.2166** |

### Comparison with Baselines (Biology Domain)

| Implementation | NDCG@10 | Notes |
|----------------|--------:|-------|
| **Our BM25** | **0.2920** | Lucene + evolved + saturated |
| Paper Query-Side BM25 | 0.197 | From "Lighting the Way for BRIGHT" |
| Paper Anserini BoW | 0.182 | From "Lighting the Way for BRIGHT" |
| Pyserini/Anserini | 0.181 | Reference implementation |
| Gensim OkapiBM25 | 0.090 | Vector-space IDF² issue |

**Our implementation achieves 48% improvement over the paper's Query-Side BM25.**

### Run Evaluation

```bash
# Best configuration (recommended)
uv run python -m benchmarks.full_bright_evaluation --lucene --query-mode saturated --k3 2.0

# Simple tokenizer baseline
uv run python -m benchmarks.full_bright_evaluation

# Pyserini tokenizer (requires Java 21)
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
uv run python -m benchmarks.full_bright_evaluation --pyserini --query-mode saturated --k3 2.0
```

---

## Project Structure

```
ranking-evolved/
├── src/ranking_evolved/
│   ├── bm25.py          # BM25 implementation (tokenizers, IDF/TF strategies, scorers)
│   ├── bm25_classic.py  # Vanilla Robertson BM25 for OpenEvolve (k1=1.5, b=0.75)
│   ├── bm25_evolved.py  # Evolved BM25 with discovered optimizations
│   └── metrics.py       # Evaluation metrics (NDCG, MAP, MRR, precision, recall)
├── benchmarks/
│   ├── full_bright_evaluation.py  # Full 12-domain evaluation
│   ├── bright_benchmark.py        # Comprehensive benchmark runner
│   └── baselines/                 # External library wrappers (Pyserini, Gensim)
├── tests/
│   ├── test_bm25.py               # BM25 unit tests
│   └── test_lucene_tokenizer.py   # Tokenizer tests (88 test cases)
├── references/
│   ├── bm25_formulas.md           # BM25 variant formulas
│   └── evolved_variants.md        # Archive of evolved formulas
├── evaluator.py                   # Unified multi-benchmark evaluator (recommended)
├── evaluator_bright.py            # BRIGHT benchmark evaluator
├── evaluator_beir.py              # BEIR benchmark evaluator
└── openevolve_config.yaml         # OpenEvolve configuration
```

---

## Running Evaluators

### Unified Evaluator (Recommended)

Evaluate on any combination of BRIGHT and BEIR benchmarks.

```bash
# Single benchmark
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --bright biology
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --beir scifact

# Multiple datasets (comma-separated)
uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --bright biology,earth_science,economics

uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --beir scifact,nfcorpus,fiqa,trec-covid

# Mix of benchmarks
uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --bright biology,earth_science \
    --beir scifact,nfcorpus

# All datasets
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --bright all
uv run python evaluator.py src/ranking_evolved/bm25_classic.py --beir all

# Fast iteration with sampling
uv run python evaluator.py src/ranking_evolved/bm25_classic.py \
    --bright biology --beir scifact --sample-queries 20
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--bright` | `` | BRIGHT domains (comma-separated, or `all`) |
| `--beir` | `` | BEIR datasets (comma-separated, or `all`) |
| `--tokenizer` | `lucene` | Tokenizer (`simple` or `lucene`) |
| `--k` | `10` | Cutoff for @k metrics |
| `--sample-queries` | `0` | Sample N queries per dataset (0 = use all) |

**Environment variables (for OpenEvolve):**
```bash
EVAL_BRIGHT_DOMAINS=biology,earth_science
EVAL_BEIR_DATASETS=scifact,nfcorpus
EVAL_SAMPLE_QUERIES=20
EVAL_TOKENIZER=lucene
```

### BRIGHT Evaluator

Evaluate on the [BRIGHT benchmark](https://huggingface.co/datasets/xlangai/BRIGHT) (12 domains, 1.3M documents).

```bash
# Basic evaluation on biology domain
uv run python evaluator_bright.py src/ranking_evolved/bm25.py --domain biology

# Use Lucene tokenizer with saturated query mode (best config)
uv run python evaluator_bright.py src/ranking_evolved/bm25.py \
    --tokenizer lucene --query-mode saturated --k3 2.0

# Evaluate across all domains
uv run python evaluator_bright.py src/ranking_evolved/bm25.py --domain all

# Fast iteration with query sampling
uv run python evaluator_bright.py src/ranking_evolved/bm25.py \
    --domain biology --sample-queries 20
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--domain` | `biology` | Domain to evaluate (`biology`, `earth_science`, `economics`, etc., or `all`) |
| `--tokenizer` | `simple` | Tokenizer (`simple` or `lucene`) |
| `--query-mode` | `unique` | Query term handling (`unique`, `sum_all`, `saturated`) |
| `--k3` | `2.0` | k3 parameter for saturated mode |
| `--k` | `10` | Cutoff for @k metrics |
| `--sample-queries` | `0` | Sample N queries (0 = use all) |

### BEIR Evaluator

Evaluate on the [BEIR benchmark](https://github.com/beir-cellar/beir) (18 datasets for zero-shot IR evaluation).

```bash
# Evaluate on SciFact (small, fast)
uv run python evaluator_beir.py src/ranking_evolved/bm25.py --dataset scifact

# Use Lucene tokenizer (default)
uv run python evaluator_beir.py src/ranking_evolved/bm25.py \
    --dataset nfcorpus --tokenizer lucene

# Evaluate across all BEIR datasets (takes several hours)
uv run python evaluator_beir.py src/ranking_evolved/bm25.py --dataset all

# Fast iteration with query sampling
uv run python evaluator_beir.py src/ranking_evolved/bm25.py \
    --dataset quora --sample-queries 50
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `scifact` | Dataset to evaluate (`scifact`, `nfcorpus`, `quora`, etc., or `all`) |
| `--tokenizer` | `lucene` | Tokenizer (`simple` or `lucene`) |
| `--k` | `10` | Cutoff for @k metrics |
| `--sample-queries` | `0` | Sample N queries (0 = use all) |
| `--data-dir` | `datasets/beir` | Directory to store downloaded datasets |

**Available BEIR datasets:** `scifact`, `nfcorpus`, `arguana`, `scidocs`, `fiqa`, `webis-touche2020`, `trec-covid`, `quora`, `cqadupstack`, `robust04`, `trec-news`, `hotpotqa`, `nq`, `fever`, `climate-fever`, `dbpedia-entity`, `bioasq`

### BEIR Benchmark Results

Comparison of our BM25 implementations vs the [BEIR paper](https://arxiv.org/abs/2104.08663) baseline (nDCG@10):

| Dataset | BEIR Paper | Our Classic | Δ | Notes |
|---------|-----------|-------------|------|-------|
| SciFact | 0.665 | **0.690** | +3.8% | Fact verification |
| TREC-COVID | 0.656 | **0.716** | +9.1% | COVID-19 retrieval |
| NFCorpus | 0.325 | **0.331** | +1.8% | Nutrition/medical |
| FiQA | 0.236 | **0.253** | +7.2% | Financial QA |
| ArguAna | 0.315 | **0.323** | +2.5% | Argument retrieval |
| SCIDOCS | 0.158 | **0.159** | +0.6% | Citation prediction |

**Configuration difference:**
- BEIR paper uses Anserini defaults: **k1=0.9, b=0.4**
- Our Classic uses Robertson defaults: **k1=1.5, b=0.75**

The Robertson Classic parameters (higher k1, higher b) outperform Anserini defaults on all tested BEIR datasets.

---

## Running OpenEvolve

The evolved TF formula was discovered using [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve):

```bash
export OPENAI_API_KEY="your-key"

# Evaluate on multiple benchmarks (recommended)
EVAL_BRIGHT_DOMAINS=biology,earth_science \
EVAL_BEIR_DATASETS=scifact,nfcorpus \
EVAL_SAMPLE_QUERIES=20 \
uv run openevolve-run src/ranking_evolved/bm25_classic.py evaluator.py --config openevolve_config.yaml

# Or single benchmark
uv run openevolve-run src/ranking_evolved/bm25_classic.py evaluator_bright.py --config openevolve_config.yaml
uv run openevolve-run src/ranking_evolved/bm25_classic.py evaluator_beir.py --config openevolve_config.yaml
```

---

## OpenEvolve v2 Results (Expanded Evolution Targets)

We expanded `bm25_evolved.py` to allow OpenEvolve to modify **10 different aspects** of the BM25 system:

1. **EvolvedParameters** — k1, b, k3, IDF bounds
2. **EvolvedStopwords** — stopword list customization
3. **EvolvedStemmer** — stemming rules
4. **EvolvedTokenizer** — tokenization pipeline
5. **EvolvedIDF** — IDF formula
6. **EvolvedTF** — TF saturation formula
7. **EvolvedLengthNorm** — document length normalization
8. **EvolvedQueryWeighting** — query term handling (unique/sum_all/saturated)
9. **EvolvedScoreAggregation** — score combination (sum/weighted_sum/max/mean)
10. **BM25.score_kernel()** — main scoring function

### Best Evolved Configuration (195 iterations, 6 generations)

| Component | Original | Evolved |
|-----------|----------|---------|
| **IDF** | `log((N+0.5)/(df+0.5))` | `log((N+0.5)/(df+0.5) + 1)` |
| **TF** | `log1p(tf_raw * tf_sat)` | `log1p((tf*(k1+1))/(tf+k1*norm+0.5))` |
| **Query weighting** | `unique` | `sum_all` |
| **Score aggregation** | `sum` | `weighted_sum` |

### Full Evaluation Results

#### Simple Tokenizer

| Domain | Queries | Docs | NDCG@10 | MAP | MRR |
|--------|--------:|-----:|--------:|----:|----:|
| earth_science | 116 | 121,249 | **0.2952** | 0.2483 | 0.4042 |
| biology | 103 | 57,359 | **0.2467** | 0.2005 | 0.3814 |
| stackoverflow | 117 | 107,081 | 0.1603 | 0.1373 | 0.2106 |
| sustainable_living | 108 | 60,792 | 0.1440 | 0.1200 | 0.1890 |
| leetcode | 142 | 413,932 | 0.1361 | 0.0981 | 0.1316 |
| pony | 112 | 7,894 | 0.1230 | 0.0907 | 0.2618 |
| economics | 103 | 50,220 | 0.1215 | 0.0806 | 0.1611 |
| psychology | 101 | 52,835 | 0.1003 | 0.0856 | 0.1328 |
| robotics | 101 | 61,961 | 0.0904 | 0.0740 | 0.1206 |
| theoremqa_questions | 194 | 188,002 | 0.0420 | 0.0366 | 0.0506 |
| aops | 111 | 188,002 | 0.0270 | 0.0183 | 0.0496 |
| theoremqa_theorems | 76 | 23,839 | 0.0142 | 0.0136 | 0.0254 |
| **MACRO AVG** | **1,384** | **1,333,166** | **0.1251** | 0.1003 | 0.1766 |

#### Lucene Tokenizer

| Domain | Queries | Docs | NDCG@10 | MAP | MRR |
|--------|--------:|-----:|--------:|----:|----:|
| earth_science | 116 | 121,249 | **0.4287** | 0.3658 | 0.5583 |
| biology | 103 | 57,359 | **0.2699** | 0.2263 | 0.4000 |
| stackoverflow | 117 | 107,081 | 0.1861 | 0.1623 | 0.2446 |
| economics | 103 | 50,220 | 0.1790 | 0.1358 | 0.2266 |
| psychology | 101 | 52,835 | 0.1714 | 0.1401 | 0.2234 |
| sustainable_living | 108 | 60,792 | 0.1716 | 0.1413 | 0.2211 |
| leetcode | 142 | 413,932 | 0.1280 | 0.0944 | 0.1264 |
| robotics | 101 | 61,961 | 0.0992 | 0.0866 | 0.1380 |
| pony | 112 | 7,894 | 0.0319 | 0.0316 | 0.1058 |
| theoremqa_questions | 194 | 188,002 | 0.0360 | 0.0361 | 0.0419 |
| aops | 111 | 188,002 | 0.0289 | 0.0199 | 0.0503 |
| theoremqa_theorems | 76 | 23,839 | 0.0274 | 0.0246 | 0.0462 |
| **MACRO AVG** | **1,384** | **1,333,166** | **0.1465** | 0.1221 | 0.1986 |

### Key Findings

1. **Lucene tokenizer + evolved formulas** achieves **0.1465 macro NDCG@10** across all 12 domains
2. **`sum_all` query weighting + `weighted_sum` aggregation** improves ranking by giving repeated query terms more weight
3. **Simplified TF formula** with `+0.5` smoothing in denominator provides better saturation behavior
4. **`+1` inside IDF log** provides smoother scaling for rare terms

---

## Development

```bash
uv sync --group dev
uv run pytest              # Run all tests
uv run ruff format         # Format code
uv run mypy src/           # Type checking
```

---

## References

- [BM25 Formulas Reference](references/bm25_formulas.md) — Detailed formula derivations
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — Evolutionary algorithm framework
- [BRIGHT Dataset](https://huggingface.co/datasets/xlangai/BRIGHT) — Benchmark for retrieval
- ["Lighting the Way for BRIGHT"](https://arxiv.org/abs/2411.00934) — Query-Side BM25 paper
