# BM25 Formulas Reference

This document contains the mathematical formulations for BM25 and its variants, optimized for LLM consumption during evolution.

## Standard BM25 (Robertson et al.)

The BM25 score for a document D given query Q is:

```
Score(D, Q) = Σ IDF(t) × TF_doc(t, D) × TF_query(t, Q)
              t∈Q
```

Where the sum is over all unique terms t in query Q.

**Note:** Most implementations treat query terms as binary (bag-of-words), ignoring
`TF_query`. The full BM25 formula includes query term frequency weighting.

### IDF Component (Inverse Document Frequency)

Classic Robertson IDF:
```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
```

Where:
- `N` = total number of documents in corpus
- `df(t)` = number of documents containing term t

Note: This can produce negative values for terms appearing in >50% of documents.

### Document TF Component (Term Frequency Saturation)

```
TF_doc(t, D) = (f(t, D) × (k1 + 1)) / (f(t, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:
- `f(t, D)` = frequency of term t in document D
- `|D|` = length of document D (in terms)
- `avgdl` = average document length in corpus
- `k1` = term frequency saturation parameter (default: 1.2-1.5)
- `b` = length normalization parameter (default: 0.75)

### Query TF Component (Query Term Weighting)

```
TF_query(t, Q) = (f(t, Q) × (k3 + 1)) / (f(t, Q) + k3)
```

Where:
- `f(t, Q)` = frequency of term t in query Q
- `k3` = query TF saturation parameter (default: 8.0)

**Bag-of-Words Mode:** When `TF_query = 1` for all terms (ignoring query TF),
this reduces to the common "bag-of-words" BM25 implementation.

**Query-Side Mode:** When query TF is computed, repeated terms in the query
get diminishing returns, which can help with keyword stuffing or emphasis.

### Combined Formula (Bag-of-Words)

```
Score(D, Q) = Σ log((N - df(t) + 0.5) / (df(t) + 0.5)) ×
              t∈Q
              (f(t, D) × (k1 + 1)) / (f(t, D) + k1 × (1 - b + b × |D| / avgdl))
```

### Combined Formula (Query-Side / Full BM25)

```
Score(D, Q) = Σ IDF(t) × TF_doc(t, D) × TF_query(t, Q)
              t∈Q

            = Σ log((N - df(t) + 0.5) / (df(t) + 0.5)) ×
              t∈Q
              (f(t, D) × (k1 + 1)) / (f(t, D) + k1 × norm) ×
              (f(t, Q) × (k3 + 1)) / (f(t, Q) + k3)
```

Where `norm = 1 - b + b × |D| / avgdl`

---

## BM25 Variants

### 1. Lucene BM25

Prevents negative IDF by adding 1 inside the log:

```
IDF_Lucene(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

TF component remains the same as standard BM25.

### 2. ATIRE BM25

Uses a simpler IDF formulation:

```
IDF_ATIRE(t) = log(N / df(t))
```

TF component includes document length in a different position:
```
TF_ATIRE(t, D) = ((k1 + 1) × f(t, D)) / (k1 × (1 - b + b × |D| / avgdl) + f(t, D))
```

This is mathematically equivalent to standard BM25 TF but written differently.

### 3. BM25L (Long Document Correction)

Addresses the penalty on long documents by adding a constant δ:

```
c(t, D) = f(t, D) / (1 - b + b × |D| / avgdl)

TF_BM25L(t, D) = ((k1 + 1) × (c(t, D) + δ)) / (k1 + c(t, D) + δ)
```

IDF uses a non-negative variant:
```
IDF_BM25L(t) = log((N + 1) / (df(t) + 0.5))
```

Recommended δ = 0.5

### 4. BM25+ (Lower-Bound Bonus)

Adds a lower-bound bonus when a term appears at least once:

```
TF_BM25+(t, D) = ((k1 + 1) × f(t, D)) / (k1 × (1 - b + b × |D| / avgdl) + f(t, D)) + δ
```

The δ is added AFTER the TF computation, ensuring terms that match get a minimum boost.

IDF typically uses non-negative variant:
```
IDF_BM25+(t) = log((N + 1) / (df(t)))
```

Recommended δ = 1.0

### 5. BM25F (Field-Weighted)

For documents with multiple fields (title, body, etc.):

```
tf_combined = Σ (w_f × f(t, D_f)) / (1 - b_f + b_f × |D_f| / avgdl_f)
              f∈fields

TF_BM25F(t, D) = (tf_combined × (k1 + 1)) / (tf_combined + k1)
```

Where:
- `w_f` = weight for field f
- `b_f` = length normalization for field f
- `D_f` = content of field f in document D

---

## Parameter Guidelines

### k1 (Term Frequency Saturation)
- Range: 0.0 to 3.0+
- Default: 1.2 (Lucene), 1.5 (academic)
- Lower values: faster saturation, diminishing returns from repeated terms
- Higher values: more linear relationship with term frequency
- At k1=0: binary presence/absence only
- Interpretation: "tf value at which half the maximum score is achieved"

### b (Length Normalization)
- Range: 0.0 to 1.0
- Default: 0.75
- b=0: no length normalization (favors longer documents)
- b=1: full normalization (strongly penalizes long documents)
- Optimal varies by collection and query type

### δ (BM25L/BM25+ bonus)
- BM25L: typically 0.5
- BM25+: typically 1.0
- Higher values increase long document scores

---

## Implementation Notes

### Numerical Stability

1. **Avoid division by zero**: Add small epsilon (1e-9) to denominators
2. **Clamp IDF**: Prevent extreme values with `clip(idf, 0, 8)` or similar
3. **Handle empty queries**: Return 0.0 immediately

### Optimization Opportunities

1. **Pre-compute normalization factors**:
   ```
   norm[d] = 1 - b + b × |D_d| / avgdl
   ```

2. **Vectorize IDF computation**:
   ```python
   idf = np.log((N - df + 0.5) / (df + 0.5))
   ```

3. **Early termination**: Skip documents with zero overlap with query terms

4. **Sparse representation**: Only store non-zero term frequencies

### Known Issues

1. **Negative IDF**: Standard formula gives negative IDF for terms in >50% of documents
   - Solution: Use Lucene or ATIRE variant, or clamp to 0

2. **Long document penalty**: Standard BM25 penalizes longer documents excessively
   - Solution: Use BM25L or BM25+ variants

3. **Common term domination**: Very common terms can dominate scoring
   - Solution: Query-side term weighting or IDF capping

---

## Common Implementation Bugs

### Bug 1: Misplaced +1 in IDF (causes ~60% NDCG drop)

**Wrong:**
```python
idf = log((N - df + 0.5) / (df + 0.5) + 1)  # +1 AFTER the ratio
```

**Correct (Classic):**
```python
idf = log((N - df + 0.5) / (df + 0.5))
```

**Correct (Lucene):**
```python
idf = log(1 + (N - df + 0.5) / (df + 0.5))  # +1 BEFORE the ratio
```

The bug compresses the IDF range and over-weights common terms. For a term in 50% of documents:
- Correct Classic: `log(1.0) = 0`
- Buggy version: `log(2.0) = 0.69`

This causes the buggy implementation to give significant weight to terms that should have zero discriminative power.

### Bug 2: Missing (k1 + 1) in TF numerator (causes ~60% score reduction)

**Wrong (found in Gensim LuceneBM25Model):**
```python
tf_weight = tf / (tf + k1 * norm)
```

**Correct:**
```python
tf_weight = tf * (k1 + 1) / (tf + k1 * norm)
```

The `(k1 + 1)` factor scales the TF component. Without it, for k1=1.5 and tf=1:
- Correct: `1 * 2.5 / (1 + 1.5) = 1.0`
- Wrong: `1 / (1 + 1.5) = 0.4`

This reduces scores by ~60%, severely impacting ranking quality.

### Note: Gensim LuceneBM25Model IDF is actually correct

Gensim's `LuceneBM25Model` uses:
```python
idf = log(N + 1) - log(df + 0.5)
    = log((N + 1) / (df + 0.5))
```

This is mathematically equivalent to the correct Lucene formula:
```python
idf = log(1 + (N - df + 0.5) / (df + 0.5))
    = log((df + 0.5 + N - df + 0.5) / (df + 0.5))
    = log((N + 1) / (df + 0.5))
```

**The IDF formula is correct.** The main issue with Gensim's LuceneBM25Model is the missing `(k1+1)` factor in the TF component (Bug 2 above).

### Bug 4: Gensim's vector-space approach is fundamentally wrong for BM25

**Two separate issues:**

1. **SparseMatrixSimilarity uses cosine similarity** - Gensim's recommended usage pattern
   computes cosine similarity between BM25-weighted vectors, which normalizes scores by
   vector lengths, destroying the BM25 ranking semantics.

2. **The dot product approach also fails** - Even with dot product (not cosine), the pattern
   `score = query_bm25_vec · doc_bm25_vec` multiplies the IDF and TF weights twice (once in
   query vector, once in document vector). This squares the IDF contribution, which is wrong.

**Correct BM25 scoring:**
```python
score = Σ IDF(t) × TF_doc(t, D)   # for each query term t
```

**Gensim's scoring (wrong even with dot product):**
```python
score = query_bm25_vec · doc_bm25_vec
      = Σ (IDF(t) × TF_q(t)) × (IDF(t) × TF_d(t))
      = Σ IDF(t)² × TF_q(t) × TF_d(t)  # IDF is squared!
```

Benchmarks confirm: fixing cosine→dot product only marginally improves results (0.09→0.08-0.09),
still far below correct implementations (0.22).

### Bug 5: Gensim OkapiBM25Model clamps negative IDF too aggressively

`OkapiBM25Model` clamps negative IDF values to `epsilon × average_idf` (default epsilon=0.25).
For very common terms like "the" and "of", this gives them IDF ≈ 2.4 instead of near-zero.

| Term | Correct IDF | Gensim OkapiBM25 IDF |
|------|-------------|---------------------|
| the | -0.07 | 2.39 (clamped) |
| of | -0.03 | 2.39 (clamped) |
| atp | 5.37 | 5.37 |

This over-weights common terms, hurting ranking quality.

### Benchmark Results Showing Bug Impact

| Implementation | NDCG@10 | Issue |
|----------------|---------|-------|
| Correct Lucene | 0.2235 | Reference |
| Correct Classic | 0.2050 | Negative IDF hurts slightly |
| Gensim OkapiBM25 (dot product fix) | 0.0900 | Bugs 5, squared IDF |
| Gensim LuceneBM25 (dot product fix) | 0.0845 | Bugs 2, 3, squared IDF |
| Gensim AtireBM25 (dot product fix) | 0.0838 | Squared IDF |
| Gensim (cosine similarity) | ~0.09 | Bug 4 + all above |
| Buggy +1 IDF | ~0.08 | Bug 1 |

**Conclusion:** Do not use Gensim for BM25 retrieval benchmarks. Even after fixing
the cosine similarity issue (Bug 4), the vector-space approach `query_vec · doc_vec`
fundamentally computes the wrong thing - it multiplies BM25 weights twice, effectively
squaring the IDF contribution. All three models produce incorrect rankings.

---

## Experimental Findings (from this project)

### What Worked (Biology domain, BRIGHT dataset)

1. **Clipped IDF**: `min(8, max(0, log((N + 0.5) / (df + 0.5))))`
2. **Unique query terms**: Deduplicate query before scoring
3. **Log-damped TF saturation**: `log(1 + tf_raw × tf_sat)` instead of raw product
4. **TF saturation factor**: Additional `tf / (tf + k1 + 0.5)` multiplier

Best kernel (iteration 113):
```python
tf_raw = (tf × (k1 + 1)) / (tf + k1 × norm)
tf_sat = tf / (tf + k1 + 0.5)
score = idf × log(1 + tf_raw × tf_sat)
```

Results: NDCG@10 improved from 0.0813 (baseline) to 0.2219 (evolved)

### Cross-Validation: Tokenization Impact

Cross-validation comparing implementations across tokenization methods (Biology domain):

| Implementation | Tokenizer | k1 | b | NDCG@10 |
|----------------|-----------|-----|------|---------|
| Our Lucene BM25 | Lucene (Python) | 0.9 | 0.4 | **0.2524** |
| Our Lucene BM25 | Simple | 0.9 | 0.4 | 0.2318 |
| Our Lucene BM25 | Simple | 1.5 | 0.75 | 0.2235 |
| Our Lucene BM25 | Lucene (Python) | 1.2 | 0.75 | 0.1942 |
| Pyserini/Anserini | Lucene (Java) | 0.9 | 0.4 | 0.1810 |
| Pyserini/Anserini | Lucene (Java) | 1.2 | 0.75 | 0.0793 |

**Key findings:**

1. **Tokenization matters more than BM25 parameters** - The gap between tokenizers
   is larger than the gap between k1/b settings.

2. **Optimal parameters depend on tokenizer:**
   - Simple tokenization: k1=0.9-1.5, b=0.4-0.75 all work well
   - Lucene tokenization: k1=0.9, b=0.4 is clearly best (compensates for shorter tokens)

3. **Pyserini underperforms by ~28%** compared to our implementation with the same
   tokenization (0.18 vs 0.25 at k1=0.9, b=0.4).

### Root Cause: Query Term Counting

Investigation revealed the cause of the Pyserini performance gap:

**Our implementation (bag-of-words):**
```python
unique_terms = list(dict.fromkeys(query))  # Deduplicate
score = sum(IDF(t) × TF(t, doc) for t in unique_terms)
```

**Pyserini/Lucene (sum over all occurrences):**
```python
# No deduplication - repeated query terms contribute multiple times
score = sum(IDF(t) × TF(t, doc) for t in query)  # Including duplicates!
```

**Impact Example:**
```
Query: "light light light heat heat insect" (after tokenization)

Document with tf(light)=5, tf(heat)=2:
  - Our score: IDF(light)×TF(light) + IDF(heat)×TF(heat) ≈ 8.5
  - Pyserini: 3×IDF(light)×TF(light) + 2×IDF(heat)×TF(heat) ≈ 31.9
```

For BRIGHT's long-form queries where term repetition is incidental (natural language),
the bag-of-words approach produces better rankings because it doesn't over-weight
documents that match frequently-repeated query terms.

**Additional Note:** Lucene 8.0+ removed `(k1+1)` from the TF numerator for performance
([LUCENE-8563](https://issues.apache.org/jira/browse/LUCENE-8563)). This doesn't affect
ranking but results in lower absolute scores (~0.5x our scores for single-term queries).

Run cross-validation: `uv run python -m benchmarks.cross_validation`

---

## References

1. Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.

2. Kamphuis, C., de Vries, A. P., Boytsov, L., & Lin, J. (2020). Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants. ECIR 2020.

3. Lv, Y., & Zhai, C. (2011). Lower-Bounding Term Frequency Normalization. CIKM 2011. (BM25+)

4. Lv, Y., & Zhai, C. (2011). When Documents Are Very Long, BM25 Fails! SIGIR 2011. (BM25L)

### Library Documentation

- bm25s: https://bm25s.github.io/
- Pyserini: https://github.com/castorini/pyserini
- Gensim: https://radimrehurek.com/gensim/models/bm25model.html
- Lucene: https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
