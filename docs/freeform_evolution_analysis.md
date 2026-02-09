# Freeform Evolution Analysis: From BM25 to Concave Surprisal Retrieval

## Executive Summary

A 200-iteration OpenEvolve freeform run (~45 hours) evolved a standard Lucene BM25 seed into a fundamentally different scoring algorithm. The evolved program replaces BM25's saturated TF with log-evidence accumulation, adds query-side intelligence (clarity gating, facet priors), and introduces multi-channel lexical matching (prefix + character n-gram).

| Metric | Pyserini Official | Our BM25 (Pyserini-style) | Evolved Best |
|--------|:-:|:-:|:-:|
| avg NDCG@10 | 0.268 | 0.268 | **0.286** (+6.9%) |
| avg Recall@100 | 0.437 | 0.444 | **0.482** (+10.3%) |
| Combined Score | 0.403 | 0.409 | **0.443** (+9.8%) |

- **Run**: `output/openevolve_output_freeform_fast/20260202_075458/`
- **Best score found at**: iteration 188 (avg NDCG@10 = 0.291 on evolution eval set)
- **Seed**: `src/ranking_evolved/bm25_freeform_fast.py` (Lucene BM25, k1=0.9, b=0.4)

---

## Per-Dataset Comparison

| Dataset | Pyserini NDCG@10 | Evolved NDCG@10 | Delta | Pyserini R@100 | Evolved R@100 | Delta |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| beir_trec-covid | 0.670 | **0.725** | +8.2% | 0.109 | **0.124** | +13.3% |
| beir_scifact | 0.679 | **0.683** | +0.6% | 0.925 | **0.928** | +0.3% |
| beir_nfcorpus | 0.322 | **0.335** | +4.1% | 0.246 | **0.263** | +6.9% |
| beir_arguana | 0.301 | 0.289 | -4.1% | 0.936 | 0.922 | -1.5% |
| beir_fiqa | 0.236 | 0.234 | -0.7% | 0.539 | 0.526 | -2.6% |
| beir_scidocs | 0.149 | **0.151** | +1.3% | 0.348 | **0.351** | +1.1% |
| bright_earth_science | 0.281 | 0.277 | -1.4% | 0.596 | **0.653** | +9.7% |
| bright_biology | 0.181 | **0.247** | +36.7% | 0.421 | **0.572** | +36.1% |
| bright_stackoverflow | 0.162 | **0.197** | +21.8% | 0.409 | **0.531** | +29.8% |
| bright_economics | 0.164 | 0.124 | -24.5% | 0.408 | 0.378 | -7.3% |
| bright_pony | 0.043 | **0.132** | +205% | 0.172 | **0.331** | +92.3% |
| bright_theoremqa | 0.021 | **0.038** | +77.3% | 0.134 | **0.200** | +49.1% |

Largest wins: **bright_pony** (+205% NDCG), **bright_biology** (+36.7%), **bright_stackoverflow** (+21.8%). These are BRIGHT domains with complex/long queries where the evolved algorithm's coverage and clarity features excel.

---

## Improvements Discovered

### 1. IDF: Surprisal instead of Odds-Ratio

**Seed (Lucene BM25):**
```python
idf = log(1 + (N - df + 0.5) / (df + 0.5))
```

**Evolved:**
```python
idf = log1p(N / df)
```

The evolved IDF interprets df/N as an occurrence probability and uses self-information (surprisal). This is smoother across corpora and avoids the odds-ratio edge cases of BM25 IDF (which can produce extreme values for very rare or very common terms).

### 2. TF: Log-Evidence instead of BM25 Saturation

**Seed:**
```python
tf_part = tf / (tf + k1 * norm)    # bounded in [0, 1)
```

**Evolved:**
```python
evidence = log1p(tf / (tf_log_base + eps))    # unbounded but concave
```

BM25's saturated TF is bounded — once TF is high, additional occurrences contribute almost nothing. The evolved log-evidence is still concave (diminishing returns) but never fully saturates. A document with tf=100 still scores higher than tf=50, unlike BM25 where both would score ~1.0.

### 3. Document Length Normalization: Log-Damping

**Seed:**
```python
norm = 1 - b + b * (dl / avgdl)    # linear in doc length
score *= tf / (tf + k1 * norm)      # embedded in TF denominator
```

**Evolved:**
```python
length_ratio = (dl + 1) / (avgdl + 1)
dl_damp = 1 + dl_alpha * log1p(length_ratio)    # dl_alpha = 0.15
score /= dl_damp                                 # applied to final score
```

Length normalization is separated from TF and applied as a final divisor. The log transform is much gentler than BM25's linear penalty — a document 10x longer than average is only moderately penalized, not crushed. This directly addresses the "BM25 fails on long documents" problem (Lv & Zhai, SIGIR 2011).

### 4. Query Term Weighting (New)

**Seed:** All query terms have weight 1.0 (bag of words).

**Evolved:**
```python
weight = count ^ 0.5    # qtf_power = 0.5
```

Repeated query terms get sublinear weighting. If a term appears 4 times in the query, its weight is 2.0 (not 4.0). This prevents verbose queries from being dominated by repeated tokens while still recognizing emphasis.

### 5. Query Clarity Gating (New)

**Seed:** None.

**Evolved:**
```python
rarity = idf / (idf + 1)           # maps IDF to [0, 1)
clarity = rarity ^ q_clarity_power  # q_clarity_power = 0.6
# Each term's contribution is multiplied by clarity
```

This gates each term's contribution by how discriminative it is. Common terms (low IDF → low clarity) contribute less to the score. Rare, specific terms dominate. The power parameter 0.6 softens the gating so common terms aren't completely suppressed.

### 6. Facet Prior: Reweighting Toward Decisive Terms (New)

**Seed:** None.

**Evolved:**
```python
facet = (idf ^ 1.6) / mean(idf ^ 1.6)
idf_used = 0.88 * idf + 0.12 * facet    # facet_mix = 0.12
```

The "facet prior" treats a query as a mixture of decisive facets (high-IDF constraints) and background hints (common terms). By raising IDF to a power > 1 and normalizing, it sharpens the weight distribution toward the most discriminative terms. This improves precision (NDCG@10) without hurting recall because the effect is small (12% mix).

### 7. Coverage / Soft-AND (New)

**Seed:** Pure additive scoring — matching 1 of 5 query terms can score well if that term has high TF.

**Evolved:**
```python
coverage = matched_idf_mass / total_query_idf_mass
score *= 1 + coverage_gamma * coverage    # coverage_gamma = 0.25
```

Coverage measures what fraction of the query's informative mass a document matches. A document matching 4/5 query terms gets a multiplicative bonus over one matching 1/5. This acts as a soft AND — it doesn't require all terms (like Boolean AND) but rewards breadth of matching.

### 8. Coordination Bonus (New)

**Seed:** None.

**Evolved:**
```python
score *= 1 + coord_beta * (1 - exp(-3 * coverage))    # coord_beta = 0.08
```

An additional saturating coordination bonus on top of coverage. The exponential form means the first few matched terms give the biggest boost, with diminishing returns. This further encourages documents that satisfy multiple query constraints.

### 9. Rare-Key Presence Boost (New)

**Seed:** None.

**Evolved:**
```python
if idf > rare_idf_pivot:  # pivot = 4.5
    rare_hits += (idf - pivot) / (idf + eps)
score *= 1 + rare_boost * log1p(rare_hits)    # rare_boost = 0.12
```

A bounded multiplicative bonus for matching very rare terms (IDF > 4.5). This is particularly useful for technical/specialized queries where a single rare keyword match (like a function name or scientific term) is highly indicative of relevance.

### 10. Prefix Matching Channel (New)

**Seed:** None — only exact token matches.

**Evolved:**
```python
# 5-character prefix matching as secondary channel (weight = 0.18)
prefix_tokens = [t[:5] for t in query if len(t) >= 5]
score += 0.18 * retrieval_score(prefix_query, prefix_doc_tf, ...)
```

A secondary scoring channel that matches on 5-character prefixes of tokens. This provides robustness to morphological variants (e.g., "compute" vs "computing"), identifiers, and partial matches that the stemmer misses.

### 11. Character N-gram Channel (New)

**Seed:** None.

**Evolved:**
```python
# 4-gram character matching (weight = 0.10, max 2 grams per token)
grams = [t[j:j+4] for t in query for j in spaced_positions]
score += 0.10 * retrieval_score(gram_query, gram_doc_tf, ...)
```

A third scoring channel using character 4-grams. This survives tokenization mismatches common with URLs, code snippets, and hyphenated words. Only a few grams per token are extracted (spaced evenly) to keep it cheap.

### 12. Score Compression (New)

**Seed:** Raw additive score returned directly.

**Evolved:**
```python
score = log1p(sum_evidence)    # before applying multipliers
```

The raw evidence sum is compressed through log1p before coverage/coordination multipliers are applied. This prevents a single high-TF term from dominating the score and compresses the dynamic range, making the multiplicative bonuses more effective.

---

## Parameter Summary

| Parameter | Value | Role |
|-----------|-------|------|
| `tf_log_base` | 1.0 | Base for log TF evidence |
| `dl_alpha` | 0.15 | Document length damping strength |
| `q_clarity_power` | 0.6 | Query clarity gating exponent |
| `coverage_gamma` | 0.25 | Coverage bonus strength |
| `coord_beta` | 0.08 | Coordination bonus strength |
| `qtf_power` | 0.5 | Query term frequency sublinearity |
| `facet_mix` | 0.12 | Facet prior mixing weight |
| `facet_power` | 1.6 | Facet prior IDF exponent |
| `prefix_len` | 5 | Character prefix length |
| `prefix_weight` | 0.18 | Prefix channel weight |
| `ngram_n` | 4 | Character n-gram size |
| `ngram_max_per_token` | 2 | Max n-grams extracted per token |
| `ngram_weight` | 0.10 | N-gram channel weight |
| `rare_idf_pivot` | 4.5 | IDF threshold for rare-key boost |
| `rare_boost` | 0.12 | Rare-key presence multiplier |

---

## Evolution Timeline

- **Total iterations**: 200 (199 trace entries)
- **Duration**: ~45 hours (Feb 2 08:01 UTC - Feb 4 04:35 UTC)
- **Islands**: 4 (parallel evolution populations)
- **Generations**: 9
- **Best score**: iteration 188 (avg NDCG@10 = 0.291)

### Top 5 Iterations by NDCG@10
| Iteration | avg NDCG@10 |
|:-:|:-:|
| 188 | 0.2909 |
| 199 | 0.2903 |
| 144 | 0.2866 |
| 177 | 0.2863 |
| 131 | 0.2861 |

### Key Innovation Timeline
1. **Early** (iter ~15): Shifted from BM25 saturated TF to log-evidence accumulation
2. **Mid** (iter ~44-64): Coverage/soft-AND mechanism, query clarity gating
3. **Late** (iter ~100+): Facet prior, rare-key boost, prefix/n-gram channels refined
4. **Final** (iter ~188): Parameter tuning, coordination bonus optimized

---

## Architectural Comparison

| Aspect | BM25 (Seed) | Evolved |
|--------|-------------|---------|
| IDF | Odds-ratio: `log(1 + (N-df+0.5)/(df+0.5))` | Surprisal: `log1p(N/df)` |
| TF | Saturated: `tf/(tf+k1*norm)` | Log-evidence: `log1p(tf/base)` |
| Length norm | Linear in TF denominator | Log-damping on final score |
| Query weighting | Uniform (1.0) | Sublinear (`count^0.5`) |
| Term importance | IDF only | IDF * clarity * facet prior |
| Score structure | Additive | Additive + multiplicative bonuses |
| Matching channels | 1 (exact tokens) | 3 (tokens + prefix + n-gram) |
| Document interaction | Independent per-term | Coverage-aware (soft AND) |
| Parameters | 2 (k1, b) | 15 |

---

## Key Takeaways

1. **Log-evidence > saturated TF**: The biggest single change. BM25's hard saturation is replaced with a gentler log that still rewards additional occurrences.

2. **Coverage matters**: The soft-AND mechanism provides the largest gains on BRIGHT domains where queries are long and complex. Documents matching more query facets score disproportionately higher.

3. **Multi-channel matching is cheap and effective**: Prefix and n-gram channels add only ~10-18% weight but dramatically improve performance on domains with tokenization challenges (pony: +205%, stackoverflow: +22%).

4. **Query-side intelligence scales**: Clarity gating + facet priors help without any training data — they're computed from corpus statistics alone.

5. **Not a free lunch**: The evolved algorithm is slower (more indexing overhead for prefix/n-gram channels) and has more parameters to tune. It also slightly underperforms BM25 on some datasets (arguana: -4%, economics: -25%) where the additional complexity isn't warranted.

---

## Prior Work

Each evolved feature has precedents in the IR literature. The evolution independently rediscovered and combined techniques spanning 50+ years of research. Below we map each feature to its closest prior art.

### 1. Surprisal IDF

The evolved `log1p(N/df)` is essentially the **original 1972 Sparck Jones IDF** [1], which is the self-information (surprisal) `-log(df/N)`. Aizawa [2] formally proved this information-theoretic interpretation. The ATIRE search engine [3] used `log(N/df)` and showed it performs comparably to BM25's odds-ratio IDF. BM25+ [4] uses the nearly identical `log((N+1)/df)`. Kamphuis et al. [5] systematically compared IDF variants and found surprisal-family formulas competitive with Robertson's odds-ratio.

**Verdict**: Rediscovery of the oldest IDF formula. Well-established since 1972.

### 2. Log-Evidence TF

The evolved `log1p(tf/base)` belongs to the **logarithmic TF family** from the SMART system [6, 7], which used `1 + log(tf)` as the standard TF transformation. This predates BM25's saturated TF by two decades. Manning et al. [8] codified it as "sublinear TF scaling." Fang et al. [9] showed that both log TF and BM25 saturation satisfy the core axioms of retrieval. The key structural difference: log TF is concave but unbounded, while BM25's `tf/(tf+k1)` has a hard ceiling.

**Verdict**: Rediscovery of pre-BM25 log TF. The parameterized `base` adds a degree of freedom the classical `1+log(tf)` lacks.

### 3. Log-Damped Length Normalization

Lv & Zhai [4, 10] diagnosed that BM25's linear length normalization is too aggressive on long documents. BM25L floors the normalized TF; BM25+ adds an additive constant. Singhal et al. [11] established the pivoted normalization framework. Paik [12] and Na [13] argued for separating length normalization from TF computation. The evolved formula's specific form `score / (1 + alpha * log1p(dl/avgdl))` as a post-hoc log divisor appears novel -- it combines Singhal's pivot concept, Lv & Zhai's diagnosis, and Paik/Na's decoupling argument.

**Verdict**: Novel combination. The log-separated form synthesizes ideas from [4, 11, 12, 13] into a new formula.

### 4. Sublinear Query TF

The full BM25 formula includes `qtf*(k3+1)/(qtf+k3)` for query term frequency [14, 15], though most implementations ignore it. The SMART system [7] applied `1+log(tf)` to both query and document sides. Bendersky & Croft [16] showed that repeated terms in verbose queries are typically incidental. The evolved `qtf^0.5` is a simpler power-law alternative to BM25's k3 rational function.

**Verdict**: Variant of BM25's k3 component. Most implementations omit query TF entirely; the evolution rediscovered its value.

### 5. Query Clarity Gating

Cronen-Townsend et al. [17] introduced the clarity score as a query-level KL divergence for predicting query difficulty. He & Ounis [18] proposed pre-retrieval IDF-based predictors. Salton et al. [19] formalized term discrimination value. The evolved per-term clarity gate `(idf/(idf+1))^0.6` applies these concepts at the term level rather than the query level, without requiring pseudo-relevance feedback.

**Verdict**: Novel per-term operationalization of query-level clarity concepts from [17, 18].

### 6. Facet Prior (IDF Sharpening)

Fang et al.'s [9] TDC axiom requires that more discriminative terms should have greater influence. Clinchant & Gaussier [20] derived non-linear IDF from information-theoretic models. Modern learned sparse models (SPLADE [21]) converge on super-linear IDF through gradient descent. The evolved `idf^1.6` sharpening is a closed-form approximation of what neural models learn.

**Verdict**: Consistent with axiomatic IR theory [9] and information-based models [20]. The specific mixing form is novel.

### 7--8. Coverage and Coordination

Coordination-level matching dates to the Cranfield experiments [22]. Apache Lucene's `coord(q,d) = overlap/maxOverlap` was deprecated in Lucene 7.0 when BM25 became default. Salton, Fox & Wu [23] formalized soft Boolean / p-norm retrieval. Metzler & Croft [24] captured term dependencies via the MRF model. The evolved IDF-weighted coverage is a generalization of Lucene's coord factor, and the coordination bonus provides a bounded soft-AND.

**Verdict**: Modernized, IDF-weighted version of Lucene's deprecated coord factor, with roots in extended Boolean models [23].

### 9. Rare-Key Presence Boost

Robertson & Walker's [25] 2-Poisson/eliteness model posits that rare terms in a document signal topical "eliteness." Church & Gale [26] showed rare terms exhibit "burstiness." Amati & Van Rijsbergen's [27] Divergence from Randomness framework measures term informativeness multiplicatively. The evolved IDF-threshold bonus is a simple operationalization of eliteness detection.

**Verdict**: Operationalization of the eliteness concept [25] with an explicit IDF pivot, similar in spirit to DFR [27].

### 10--11. Prefix and N-gram Channels

McNamee & Mayfield [28, 29] showed that 4-5 character n-grams match or beat word-level indexing across languages. Mayfield & McNamee [29] showed that 5-character prefix truncation performs comparably to Porter stemming. Robertson, Zaragoza & Taylor [30] introduced BM25F for scoring across multiple document fields. The evolved multi-channel approach (exact + prefix + n-gram) is structurally equivalent to BM25F with three synthetic "fields" representing different text granularities.

**Verdict**: Rediscovery of established techniques. 4-gram [28] and 5-char prefix [29] match literature-optimal values. The multi-channel combination parallels BM25F [30].

### 12. Score Compression

The score normalization for fusion literature [31, 32] shows that compressing score distributions improves downstream combination. The SMART system's log TF [7] applies the same principle per-term. Clinchant & Gaussier's [20] log-logistic model uses `log(1 + c*tf/mean_tf)`. The evolved `log1p(sum_evidence)` applies compression at the aggregate score level.

**Verdict**: Extension of per-term log compression [7] to the aggregate score, consistent with fusion normalization [31, 32].

---

## References

1. Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11--21.
2. Aizawa, A. (2003). An information-theoretic perspective of tf-idf measures. *Information Processing and Management*, 39(1), 45--65.
3. Trotman, A., Puurula, A., & Burgess, B. (2014). Improvements to BM25 and language models examined. *ADCS 2014*.
4. Lv, Y. & Zhai, C. (2011). Lower-bounding term frequency normalization. *CIKM 2011*.
5. Kamphuis, C., de Vries, A.P., Boytsov, L., & Lin, J. (2020). Which BM25 do you mean? A large-scale reproducibility study of scoring variants. *ECIR 2020*.
6. Salton, G., Wong, A., & Yang, C.S. (1975). A vector space model for automatic indexing. *Communications of the ACM*, 18(11), 613--620.
7. Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513--523.
8. Manning, C.D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
9. Fang, H., Tao, T., & Zhai, C. (2004). A formal study of information retrieval heuristics. *SIGIR 2004*.
10. Lv, Y. & Zhai, C. (2011). When documents are very long, BM25 fails! *SIGIR 2011*.
11. Singhal, A., Buckley, C., & Mitra, M. (1996). Pivoted document length normalization. *SIGIR 1996*.
12. Paik, J.H. (2013). A novel TF-IDF weighting scheme for effective ranking. *SIGIR 2013*.
13. Na, S.-H. (2015). Single-term frequency normalization. *CIKM 2015*.
14. Robertson, S.E. et al. (1995). Okapi at TREC-3. *NIST Special Publication 500-225*.
15. Robertson, S.E. & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333--389.
16. Bendersky, M. & Croft, W.B. (2008). Discovering key concepts in verbose queries. *SIGIR 2008*.
17. Cronen-Townsend, S., Zhou, Y., & Croft, W.B. (2002). Predicting query performance. *SIGIR 2002*.
18. He, B. & Ounis, I. (2004). Inferring query performance using pre-retrieval predictors. *SPIRE 2004*.
19. Salton, G., Yang, C.S., & Yu, C.T. (1975). A theory of term importance in automatic text analysis. *JASIS*, 26(1), 33--44.
20. Clinchant, S. & Gaussier, E. (2010). Information-based models for ad hoc IR. *SIGIR 2010*.
21. Formal, T. et al. (2021). SPLADE: Sparse lexical and expansion model for first stage ranking. *SIGIR 2021*.
22. Cleverdon, C.W. (1967). The Cranfield tests on index language devices. *Aslib Proceedings*, 19(6), 173--194.
23. Salton, G., Fox, E., & Wu, H. (1983). Extended Boolean information retrieval. *Communications of the ACM*, 26(11), 1022--1036.
24. Metzler, D. & Croft, W.B. (2005). A Markov random field model for term dependencies in information retrieval. *SIGIR 2005*.
25. Robertson, S.E. & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. *SIGIR 1994*.
26. Church, K. & Gale, W. (1995). Poisson mixtures. *Natural Language Engineering*, 1(2), 163--190.
27. Amati, G. & Van Rijsbergen, C.J. (2002). Probabilistic models of information retrieval based on measuring the divergence from randomness. *ACM TOIS*, 20(4), 357--389.
28. McNamee, P. & Mayfield, J. (2004). Character n-gram tokenization for European language text retrieval. *Information Retrieval*, 7(1--2), 73--97.
29. Mayfield, J. & McNamee, P. (2003). Single n-gram stemming. *SIGIR 2003*.
30. Robertson, S.E., Zaragoza, H., & Taylor, M. (2004). Simple BM25 extension to multiple weighted fields. *CIKM 2004*.
31. Lee, J.H. (1997). Analyses of multiple evidence combination. *SIGIR 1997*.
32. Montague, M. & Aslam, J.A. (2001). Relevance score normalization for metasearch. *CIKM 2001*.
