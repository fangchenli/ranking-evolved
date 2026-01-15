# Evolved BM25 Variants Archive

This document preserves the key formulas discovered through OpenEvolve experiments
on the BRIGHT dataset. These variants showed significant improvements over baseline BM25.

## Biology-Tuned Variant (bm25_biology.py)

Found through OpenEvolve iteration 113 on the biology domain.
NDCG@10 improved from 0.0813 (baseline) to 0.2219.

### IDF Formula
```
idf(t) = min(8, max(0, log((N + 0.5) / (df(t) + 0.5))))
```

**Key characteristics:**
- Clipping to [0, 8] prevents ultra-rare terms from dominating
- Uses (N + 0.5) / (df + 0.5) instead of standard (N - df + 0.5) / (df + 0.5)
- Always non-negative

### TF/Score Formula
```python
def score_kernel(query, norm, frequencies, idf, k1):
    # Order-preserving unique query terms
    terms = list(dict.fromkeys(query))

    tf = [frequencies.get(term, 0) for term in terms]
    idf_values = [idf.get(term, 0.0) for term in terms]

    denom = tf + k1 * norm

    # Standard BM25 TF saturation
    tf_raw = (tf * (k1 + 1.0)) / max(denom, 1e-9)

    # Additional saturation factor (evolved)
    tf_sat = tf / (tf + k1 + 0.5)

    # Log damping to prevent runaway boosts (evolved)
    scores = idf_values * log1p(tf_raw * tf_sat)

    return sum(scores)
```

**Key innovations:**
1. Order-preserving unique query terms (emphasizes distinct concepts)
2. Additional `tf_sat` saturation factor: `tf / (tf + k1 + 0.5)`
3. Log damping: `log1p(tf_raw * tf_sat)` instead of direct `tf_raw`
4. Early exit if no terms match

### Parameters
- k1 = 1.5
- b = 0.75

---

## Psychology-Tuned Variant (bm25_psychology.py)

Evolved for the psychology domain with multi-aspect queries.
Combined score improved from ~0.0756 (baseline) to ~0.0847.

### IDF Formula
```python
def idf(df, N):
    num = N - df + 0.5
    den = df + 0.5
    base = log(max(num / max(den, 1e-9), 1e-9))

    # Offset and clip (evolved)
    idf = clip(base + 0.63, 0.07, 5.0)

    # Mid-frequency boost (evolved)
    df_ratio = df / max(N, 1.0)
    mid_mask = (df_ratio > 0.015) & (df_ratio < 0.20)
    idf[mid_mask] += 0.05

    return idf
```

**Key characteristics:**
- Offset of +0.63 shifts all IDF values up
- Clipping to [0.07, 5.0] (tighter upper bound than biology variant)
- Special boost (+0.05) for mid-frequency terms (1.5%-20% of docs)

### TF/Score Formula
```python
def _tf_component(tf, norm, k1):
    denom = tf + k1 * norm
    base = (tf * (k1 + 1.0)) / max(denom, 1e-9)

    # Extra soft saturation (evolved)
    tf_soft = tf / (tf + k1 + 0.75)  # Note: 0.75 vs 0.5 in biology
    return base * (0.60 + 0.40 * tf_soft)  # Weighted combination

def _coverage_boost(tf, n_terms):
    """Small bonus for matching more distinct query terms."""
    matched = (tf > 0).sum()
    if matched <= 0:
        return 1.0
    frac = matched / n_terms
    # Asymmetric reward for high coverage
    return 1.0 + (0.11 if frac >= 0.75 else 0.07) * frac

def score_kernel(query, norm, frequencies, idf, k1, len_adj=0.0):
    terms = unique_terms(query)
    tf = [frequencies.get(t, 0) for t in terms]
    idf_vals = [idf.get(t, 0.0) for t in terms]

    tf_part = _tf_component(tf, norm, k1) * _coverage_boost(tf, len(terms))
    score_vec = idf_vals * log1p(tf_part)
    score = sum(score_vec)

    # Length adjustment (evolved)
    if len_adj:
        score *= 1.0 + len_adj

    return score
```

**Key innovations:**
1. Coverage boost for multi-aspect queries (rewards matching more terms)
2. Weighted TF combination: `base * (0.60 + 0.40 * tf_soft)`
3. Length adjustment based on z-score of log document length
4. Asymmetric coverage bonus (higher for >75% term match)

### Length Adjustment
```python
log_dl = log1p(document_lengths)
m = mean(log_dl)
s = std(log_dl)
z = (log_dl - m) / s
len_adj = clip(z * 0.02, -0.03, 0.03)  # Tiny adjustment
```

### Parameters
- k1 = 1.31 (lower than standard)
- b = 0.64 (lower than standard, friendlier to longer documents)

---

## Classic BM25 (bm25_classic.py)

Standard BM25 for reference/comparison.

### IDF Formula
```
idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### TF Formula
```
tf_saturation = (tf * (k1 + 1)) / (tf + k1 * norm)
score = idf * tf_saturation
```

### Parameters
- k1 = 1.5
- b = 0.75

---

## Summary of Key Differences

| Feature | Classic | Biology | Psychology |
|---------|---------|---------|------------|
| IDF base | (N-df+0.5)/(df+0.5)+1 | (N+0.5)/(df+0.5) | (N-df+0.5)/(df+0.5)+0.63 |
| IDF bounds | unbounded | [0, 8] | [0.07, 5.0] |
| TF saturation | single | double (tf_raw × tf_sat) | weighted (0.6 + 0.4×tf_soft) |
| Score combination | direct | log1p damped | log1p damped |
| Coverage boost | no | no | yes |
| Length adjustment | no | no | yes |
| k1 default | 1.5 | 1.5 | 1.31 |
| b default | 0.75 | 0.75 | 0.64 |

## Evolution Insights

1. **IDF capping is beneficial**: Prevents rare terms from overwhelming scores
2. **Double saturation works**: Multiple layers of TF damping improve rankings
3. **log1p damping**: Smooths the score contribution curve
4. **Coverage matters for complex queries**: Psychology's multi-aspect queries benefit from term diversity bonuses
5. **Length normalization tuning**: Lower b values help domains with longer average documents
