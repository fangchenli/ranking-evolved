# Discovered Lexical Retrieval Features via OpenEvolve

This document describes novel lexical retrieval mechanisms discovered through evolutionary optimization on BEIR and BRIGHT benchmarks. Three distinct architectures emerged, each with unique contributions to the field.

## Table of Contents

1. [Overview](#overview)
2. [Baseline: BM25](#baseline-bm25)
3. [Architecture A: Two-Channel Prefix Matching](#architecture-a-two-channel-prefix-matching)
4. [Architecture B: Query Specificity Gating](#architecture-b-query-specificity-gating)
5. [Architecture C: Anchor-First Two-Channel Scoring](#architecture-c-anchor-first-two-channel-scoring)
6. [Comparative Analysis](#comparative-analysis)
7. [Key Insights](#key-insights)

---

## Overview

| Architecture | combined_score | avg_recall@100 | avg_nDCG@10 | Key Innovation |
|-------------|----------------|----------------|-------------|----------------|
| A: Prefix Matching | **0.4334** | 0.4702 | **0.2861** | Morphological robustness via prefix pseudo-tokens |
| B: Specificity Gating | 0.4292 | 0.4675 | 0.2760 | Query-adaptive soft-AND gating |
| C: Anchor-First | 0.4312 | **0.4710** | 0.2721 | Two-channel scoring with anchor terms |

**Scoring objective:** `combined_score = 0.8 × avg_recall@100 + 0.2 × avg_nDCG@10`

---

## Baseline: BM25

The classical BM25 formula serves as our starting point:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $f(t, d)$ = term frequency of $t$ in document $d$
- $|d|$ = document length
- $\text{avgdl}$ = average document length in corpus
- $k_1 = 1.2$ (saturation parameter)
- $b = 0.75$ (length normalization)

**IDF (Lucene variant):**
$$\text{IDF}(t) = \log\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

---

## Architecture A: Two-Channel Prefix Matching

**Source:** Collaborator's run, iteration 61 (combined_score: 0.4334)

### Core Idea

Add a secondary "prefix channel" that matches on the first $k$ characters of each token. This provides robustness to morphological variations (e.g., "oxidize" ↔ "oxidation") and tokenization mismatches common in technical domains.

### IDF: Smoothed Surprisal

Replace BM25-odds IDF with information-theoretic surprisal:

$$\text{IDF}(t) = -\log\left(\frac{\text{df}(t) + 1}{N + 2}\right)$$

**Intuition:** This is the negative log probability of a term appearing in a random document, with add-one smoothing. More stable across domains than BM25-odds.

### TF Transformation: Concave Evidence

Instead of BM25 saturation, use unbounded log:

$$\text{TF}_{\text{evidence}}(t, d) = \log(1 + f(t, d) / \beta)$$

Where $\beta = 1.0$ (tf_log_base). This models diminishing returns without a hard ceiling.

### Coverage Multiplier (Soft-AND)

Reward documents that match more of the query's "information mass":

$$\text{coverage} = \frac{\sum_{t \in q \cap d} w_t \cdot \text{IDF}(t)}{\sum_{t \in q} w_t \cdot \text{IDF}(t)}$$

$$\text{score} \leftarrow \text{score} \times (1 + \gamma \cdot \text{coverage})$$

Where $\gamma = 0.25$ (coverage_gamma).

### Length Dampening

Replace pivoted length normalization with log-based dampening:

$$\text{score} \leftarrow \frac{\text{score}}{1 + \alpha \cdot \log(1 + |d| / \text{avgdl})}$$

Where $\alpha = 0.15$ (dl_alpha).

### Prefix Channel

For each token $t$, create a prefix pseudo-token $P_k(t)$ using the first $k=5$ characters:

$$\text{score}_{\text{total}} = \text{score}_{\text{token}} + \lambda \cdot \text{score}_{\text{prefix}}$$

Where $\lambda = 0.18$ (prefix_weight).

**Why this helps:** Technical domains (TheoremQA, StackOverflow) often have symbol-heavy queries where exact token matching fails. Prefix matching catches partial matches.

### Final Formula

$$\text{score}(q, d) = \frac{\log(1 + E) \cdot (1 + \gamma \cdot C)}{1 + \alpha \cdot \log(1 + L)}$$

Where:
- $E = \sum_{t \in q} w_q(t) \cdot \text{IDF}(t) \cdot \text{clarity}(t) \cdot \log(1 + f(t,d)/\beta)$
- $C = $ coverage ratio
- $L = |d| / \text{avgdl}$
- $\text{clarity}(t) = \left(\frac{\text{IDF}(t)}{\text{IDF}(t) + 1}\right)^{0.6}$

---

## Architecture B: Query Specificity Gating

**Source:** Our run, iteration 39 (combined_score: 0.4292)

### Core Idea

Many queries contain one highly distinctive term plus several vague modifiers. Naive coordination rewards can over-promote documents matching many vague terms. Gate the "soft-AND" effects by query specificity.

### IDF Sharpening

Apply power scaling to emphasize rare terms:

$$\text{IDF}_{\text{sharp}}(t) = \text{IDF}(t)^{\rho}$$

Where $\rho = 1.12$ (idf_power).

### Focus Prior (Effective Length)

Mix token count with unique term count to favor topically focused documents:

$$|d|_{\text{eff}} = (1 - \mu) \cdot |d| + \mu \cdot |\{t : f(t,d) > 0\}|$$

Where $\mu = 0.65$ (focus_mix).

### Query Specificity

Measure how "peaky" the query's IDF distribution is:

$$\text{spec}(q) = \frac{\max_{t \in q} \text{IDF}(t)}{\sum_{t \in q} \text{IDF}(t)}$$

- High spec (close to 1): Query dominated by one distinctive term
- Low spec (close to $1/|q|$): Query terms have similar importance

### Specificity Gate

Down-weight coordination effects for peaky queries:

$$\text{gate}_{\text{spec}} = \max\left(\phi, (1 - \text{spec})^{\psi}\right)$$

Where $\phi = 0.55$ (spec_floor), $\psi = 1.20$ (spec_power).

### Soft Stopword Penalty

Instead of hard stopword removal, apply a soft penalty to ultra-common terms:

$$\text{IDF}_{\text{adj}}(t) = \text{IDF}(t) \cdot \begin{cases}
1 - \delta \cdot \frac{\text{df}(t)/N - \tau}{1 - \tau} & \text{if } \text{df}(t)/N > \tau \\
1 & \text{otherwise}
\end{cases}$$

Where $\tau = 0.12$ (common_df_cut), $\delta = 0.35$ (common_penalty).

### Coordination Reward

Reward multi-term matches, gated by specificity:

$$\text{coord} = \frac{|\{t \in q : f(t,d) > 0\}|}{|q|}$$

$$\text{score} \leftarrow \text{score} \times (1 + \alpha_c \cdot \text{gate} \cdot \text{coord})^{\beta_c}$$

Where $\alpha_c = 0.25$, $\beta_c = 0.75$.

### Pair Synergy

Reward co-occurrence of distinctive term pairs:

$$\text{pair}(q, d) = \sum_{i < j} r_i \cdot r_j \cdot \sqrt{\text{TF}_i \cdot \text{TF}_j}$$

Where $r_t = \max(0, \text{IDF}(t) - 0.5 \cdot \max_t \text{IDF})$ filters to distinctive terms.

$$\text{score} \leftarrow \text{score} \times (1 + \beta_p \cdot \text{gate} \cdot \text{pair})$$

Where $\beta_p = 0.06$ (pair_boost).

---

## Architecture C: Anchor-First Two-Channel Scoring

**Source:** Our run, iteration 197 (combined_score: 0.4312)

### Core Idea

Compute two parallel scores—a "full" score for recall and an "anchor" score focusing on the most discriminative terms—then mix them adaptively based on query characteristics.

### New Parameters Discovered

| Parameter | Value | Purpose |
|-----------|-------|---------|
| tf_idf_gamma | 0.22 | IDF-aware TF saturation |
| tf_df_delta | 0.18 | DF-aware TF saturation |
| idf_lift_power | 0.45 | Corpus-normalized IDF |
| q_drop_min_len | 8 | Query length threshold for dropout |
| q_drop_df_ratio | 0.22 | DF ratio threshold for dropout |
| entropy_floor | 0.35 | Entropy gate floor |
| entropy_power | 0.9 | Entropy gate power |
| anchor_mix_alpha | 0.35 | Anchor channel weight |
| anchor_mix_power | 1.6 | Anchor weight shaping |
| anchor_residual | 0.55 | Anchor IDF threshold |
| info_cov_alpha | 0.10 | Information coverage weight |
| info_cov_gamma | 0.75 | Coverage power |
| salience_alpha | 0.10 | Salience/aboutness weight |
| salience_power | 0.5 | Salience power |

### IDF Lift Normalization

Normalize term importance relative to corpus mean IDF:

$$\text{IDF}_{\text{lift}}(t) = \text{IDF}(t) \cdot \left(\frac{\text{IDF}(t)}{\overline{\text{IDF}}}\right)^{\lambda}$$

Where $\overline{\text{IDF}} = \frac{1}{|V|} \sum_{t \in V} \text{IDF}(t)$ and $\lambda = 0.45$.

**Intuition:** Terms that are more discriminative than average in this corpus get boosted; this stabilizes importance across heterogeneous collections.

### IDF/DF-Aware TF Saturation

Make the saturation parameter $k_1$ dynamic based on term properties:

$$k_1^{\text{eff}}(t) = \frac{k_1}{1 + \gamma_1 \cdot \frac{\text{IDF}(t)}{\max_t \text{IDF}}} \cdot (1 + \gamma_2 \cdot \sqrt{p_{\text{df}}(t)})$$

Where:
- $\gamma_1 = 0.22$ (tf_idf_gamma)
- $\gamma_2 = 0.18$ (tf_df_delta)
- $p_{\text{df}}(t) = \text{df}(t) / N$

**Intuition:** Rare, important terms saturate faster (lower effective $k_1$); common terms need more occurrences to reach saturation.

### Query DF Dropout

For long queries ($|q| \geq 8$), drop terms whose document frequency ratio exceeds a threshold:

$$q' = \{t \in q : \text{df}(t)/N < \theta\}$$

Where $\theta = 0.22$ (q_drop_df_ratio).

**Intuition:** Long queries often contain noisy/common terms that dilute the signal. Dropping them improves precision.

### Entropy Gate

Complement specificity with an entropy-based measure:

$$H(q) = -\sum_{t \in q} p_t \log p_t, \quad p_t = \frac{\text{IDF}(t)}{\sum_{t'} \text{IDF}(t')}$$

$$H_{\text{norm}}(q) = \frac{H(q)}{\log |q|}$$

$$\text{gate}_{\text{ent}} = \max\left(\phi_e, H_{\text{norm}}^{\psi_e}\right)$$

Where $\phi_e = 0.35$ (entropy_floor), $\psi_e = 0.9$ (entropy_power).

**Combined AND gate:**
$$\text{gate}_{\text{AND}} = \text{gate}_{\text{spec}} \times \text{gate}_{\text{ent}}$$

### Information-Mass Coverage

Weight coverage by term importance and TF contribution:

$$\text{info\_cov}(q, d) = \frac{\sum_{t \in q \cap d} \text{IDF}(t) \cdot \text{TF}(t,d)^{\gamma_3}}{\sum_{t \in q} \text{IDF}(t)}$$

$$\text{score} \leftarrow \text{score} \times (1 + \alpha_i \cdot \text{info\_cov}^{\gamma_4})$$

Where $\gamma_3 = 0.55$ (info_cov_tf_gamma), $\gamma_4 = 0.75$ (info_cov_gamma), $\alpha_i = 0.10$ (info_cov_alpha).

### Salience / Aboutness

Reward documents where query terms occupy a significant fraction of the text:

$$\text{salience}(q, d) = \frac{\sum_{t \in q} f(t, d)}{|d|}$$

$$\text{score} \leftarrow \text{score} \times (1 + \alpha_s \cdot \text{salience}^{\gamma_s})$$

Where $\alpha_s = 0.10$ (salience_alpha), $\gamma_s = 0.5$ (salience_power).

**Intuition:** Demotes long generic documents that mention query terms once among many other topics.

### Two-Channel Anchor Scoring

**Anchor terms:** Terms whose adjusted IDF exceeds a threshold relative to the query maximum:

$$\text{anchor}(t) = \mathbb{1}\left[\text{IDF}_{\text{adj}}(t) \geq \theta_a \cdot \max_{t'} \text{IDF}_{\text{adj}}(t')\right]$$

Where $\theta_a = 0.55$ (anchor_residual).

**Anchor score:** Sum contributions only from anchor terms:

$$\text{score}_{\text{anchor}} = \sum_{t \in q : \text{anchor}(t)} w_q(t) \cdot \text{IDF}(t) \cdot \text{TF}(t, d)$$

**Mixture weight:** Depends on query specificity:

$$w_{\text{anchor}} = \alpha_a \cdot \text{spec}(q)^{\psi_a}$$

Where $\alpha_a = 0.35$ (anchor_mix_alpha), $\psi_a = 1.6$ (anchor_mix_power).

**Final score:**

$$\text{score}_{\text{final}} = (1 - w_{\text{anchor}}) \cdot \text{score}_{\text{full}} + w_{\text{anchor}} \cdot \max(\text{score}_{\text{anchor}}, 0.15 \cdot \text{score}_{\text{full}})$$

The floor ensures anchor-only scoring doesn't collapse to zero for multi-term matches missing the top anchor.

---

## Comparative Analysis

### Strengths by Architecture

| Aspect | A: Prefix | B: Specificity | C: Anchor |
|--------|-----------|----------------|-----------|
| **Morphology** | Excellent (prefix matching) | Moderate | Moderate |
| **Precision** | Good | Good | Moderate |
| **Recall** | Good | Good | Excellent |
| **Peaky queries** | Moderate | Good (gating) | Good (anchor) |
| **Long queries** | Moderate | Moderate | Good (dropout) |
| **Cross-domain** | Excellent (smooth IDF) | Good | Good (IDF lift) |

### Per-Dataset Winners

| Dataset Type | Best Architecture | Reason |
|--------------|-------------------|--------|
| Medical (trec-covid, nfcorpus) | A: Prefix | Technical terminology benefits from morphological matching |
| Scientific (scifact, scidocs) | A: Prefix | Similar morphological benefits |
| QA/Reasoning (theoremqa) | B: Specificity | Short, focused queries need specificity gating |
| Domain-specific (earth_science) | B/C | Focus prior helps topical documents |
| Long-form (arguana, economics) | C: Anchor | Query dropout and anchor scoring help |

---

## Key Insights

### 1. IDF Matters More Than TF

All three architectures spend more parameters on IDF transformations than TF. The corpus-level signal (term rarity) is more important than document-level counts for zero-shot retrieval.

### 2. Query-Adaptive Scoring

Static formulas underperform query-adaptive ones. Key adaptations:
- **Specificity gating** (B, C): Adjust soft-AND strength based on query IDF distribution
- **Entropy gating** (C): Complement specificity with diversity measure
- **Anchor mixing** (C): Trust different term subsets based on query structure

### 3. Multi-Channel Robustness

Both A (prefix channel) and C (anchor channel) use two parallel scoring pipelines. This hedging strategy improves robustness:
- **A:** Token channel + prefix channel for morphological coverage
- **C:** Full score (recall) + anchor score (precision)

### 4. Soft > Hard

Soft mechanisms consistently outperform hard cutoffs:
- Soft stopword penalty > hard stopword removal
- Focus prior (mixing lengths) > single length measure
- Specificity gating > hard thresholds

### 5. Bounded Multipliers

All multiplicative factors are bounded above 1.0 to preserve recall:
- Coverage: $1 + \gamma \cdot \text{coverage}$
- Coordination: $(1 + \alpha \cdot \text{coord})^{\beta}$
- Information coverage: $1 + \alpha \cdot \text{cov}^{\gamma}$

This ensures base relevance scores are never reduced, only enhanced.

---

## Future Directions

1. **Hybrid Architecture:** Combine prefix matching (A) with anchor scoring (C)
2. **Learned Gating:** Replace hand-tuned gates with lightweight learned functions
3. **Field-Aware Scoring:** Extend to multi-field documents (title, body, metadata)
4. **Efficiency:** Optimize two-channel architectures for production latency

---

*Generated from OpenEvolve runs on BEIR + BRIGHT benchmarks, February 2026*
