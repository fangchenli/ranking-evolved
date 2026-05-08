# Late Interaction Recent Experiment Summary

This summarizes the three newest runs under `output/late_interaction_freeform_latency_aware/` plus the toy run
`output/late_interaction_freeform_latency_aware_toy/20260504_132707_1c4fa0`.

Analyzed runs:

| Label | Run | Config | Best step | Best program |
|---|---|---|---:|---|
| A | `20260503_130818_4cf5b9` | `freeform_latency_aware.yaml` | 17 | `1f6b15e8-eeef-4cf1-a86a-bdea911eb728` |
| B | `20260503_175442_510eda` | `freeform_latency_aware.yaml` | 17 | `5edeed5f-5ab0-4c06-b57b-ca3145c97067` |
| C | `20260504_015558_ef58b3` | `freeform_latency_aware.yaml` | 24 | `5de37301-244b-4d62-8b32-fe6fcffa87c1` |
| Toy | `20260504_132707_1c4fa0` | `freeform_latency_aware_toy.yaml` | 12 | `98d37cac-0836-4407-a03c-7ea22c88647a` |

## Objective

All four runs optimize `recall1000_ndcg10_latency`:

```text
latency_ratio_d = candidate_median_query_latency_ms_d / exact_maxsim_median_query_latency_ms_d
latency_score_d = 1 / (1 + latency_ratio_d)

combined_score = w_recall * agg_recall@1000
               + w_ndcg   * agg_ndcg@10
               + w_latency * agg_latency_score
```

Shared latency policy:

- Baseline is external exact MaxSim: `tasks/late_interaction/baselines/exact_maxsim.cpu.json`.
- Latency is median per query, after warmup.
- If any dataset is more than `8x` slower than exact MaxSim, that dataset's latency score is set to `0`.

Run-specific objective differences:

| Runs | Weights | Dataset aggregation | Recall floor |
|---|---:|---|---:|
| A | `0.40 recall + 0.20 nDCG + 0.40 latency` | arithmetic mean | none |
| B, C, Toy | `0.40 recall + 0.30 nDCG + 0.30 latency` | geometric mean with `eps=0.001` | any dataset `recall@1000 < 0.10` zeroes score |

The geometric runs are stricter: a candidate cannot win by being very good on one dataset while collapsing on another.

Geometric aggregation is calculated separately for each objective signal
(`recall@1000`, `ndcg@10`, and `latency_score`) across the evaluated datasets:

```text
agg_metric = exp(mean(log(max(metric_d, 0.001))))
           = exp((log(max(metric_1, 0.001))
                + log(max(metric_2, 0.001))
                + ...
                + log(max(metric_D, 0.001))) / D)
```

So a zero or near-zero dataset metric is floored to `0.001` before taking the
log. This avoids `log(0)`, but still strongly penalizes any candidate that
collapses on one dataset. For the toy run, `D=1`, so the geometric aggregate is
just the single dataset value unless that value is below `0.001`.

## Evolution Pipeline Differences

| Run | Search and proposer setup | Prompt structure | Evaluator setup |
|---|---|---|---|
| A | 20 iterations, population 30, archive 6, 3 islands, migration every 8, 5 inspirations, GPT-5.2 medium. | Diff-mode prompt: the current parent program is shown in full; recent/top/inspiration programs are shown as compact metrics plus unified diffs against the parent; the LLM must answer with SEARCH/REPLACE diffs. Context includes 1 recent, 1 top, 2 diverse programs, and artifacts. System prompt emphasizes novelty, fixed embeddings, prior-art warnings, and arithmetic objective. | CPU inline evaluator. Full query set. Datasets: `bright_theoremqa_theorems`, `bright_economics`, `beir_trec-covid`. Warmup 10. |
| B | Same as A. | Same context shape as A, but prompt changes the objective text to geometric aggregation, 0.10 recall floor, and weights `0.4/0.3/0.3`. Same datasets. | Same datasets and evaluator settings as A. |
| C | 50 iterations instead of 20. Same island/population settings as B. Adds `candidate_retries: 3`. GPT-5.2 medium. | Same diff/context structure as B. Prompt changes evaluator description to a larger-corpus trio spanning about an 8x size range. | CPU inline evaluator. Full query set. Datasets: `beir_fiqa`, `bright_stackoverflow`, `bright_theoremqa_questions`. Warmup 10. |
| Toy | 20 iterations, population 16, archive 8, 2 islands, migration every 4 with rate 0.5, 8 inspirations, AST-node complexity, failure buffer 12. GPT-5.2 low, `candidate_retries: 2`. | Same diff-mode structure, but shorter toy system prompt. It also configures compact metric keys, per-dataset suffixes, max diff lines 120, and recent failed attempts. | CPU inline evaluator. Only `beir_fiqa`, only 20 sampled queries, warmup 5, timeout 600s. |

In short: yes, all four runs use diff-mode prompting. The parent candidate is
the only full program in the prompt. Other context programs are not pasted in
full; they are represented as compact metric summaries plus unified diffs
against that parent.

## Aggregate Results

These objective aggregates are the values that actually feed `combined_score`.

| Run | Seed score | Best score | agg R@1000 | agg nDCG@10 | agg latency score | Objective components, best |
|---|---:|---:|---:|---:|---:|---|
| A | 0.4720 | 0.4986 | 0.4287 | 0.2910 | 0.6723 | `0.1715 + 0.0582 + 0.2689` |
| B | 0.3988 | 0.4367 | 0.4155 | 0.1433 | 0.7583 | `0.1662 + 0.0430 + 0.2275` |
| C | 0.4403 | 0.5595 | 0.6329 | 0.1379 | 0.8833 | `0.2532 + 0.0414 + 0.2650` |
| Toy | 0.6342 | 0.7017 | 0.8667 | 0.3981 | 0.7852 | `0.3467 + 0.1194 + 0.2356` |

## Per-Dataset Results

`p50` is query median latency in ms. `Coverage` is the fraction of the corpus reranked exactly by the best program.

| Run | Dataset | Seed R@1000 | Best R@1000 | Seed nDCG@10 | Best nDCG@10 | Seed p50 | Best p50 | Best coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | `bright_theoremqa_theorems` | 0.373 | 0.373 | 0.037 | 0.037 | 291 | 294 | 100.0% |
| A | `bright_economics` | 0.673 | 0.732 | 0.114 | 0.135 | 514 | 397 | 47.8% |
| A | `beir_trec-covid` | 0.438 | 0.181 | 0.714 | 0.701 | 2643 | 549 | 14.0% |
| B | `bright_theoremqa_theorems` | 0.373 | 0.391 | 0.037 | 0.039 | 287 | 153 | 31.1% |
| B | `bright_economics` | 0.673 | 0.518 | 0.114 | 0.105 | 518 | 222 | 21.4% |
| B | `beir_trec-covid` | 0.438 | 0.354 | 0.714 | 0.715 | 2454 | 593 | 11.6% |
| C | `beir_fiqa` | 0.925 | 0.883 | 0.420 | 0.420 | 705 | 116 | 13.3% |
| C | `bright_stackoverflow` | 0.664 | 0.693 | 0.080 | 0.090 | 1262 | 170 | 9.8% |
| C | `bright_theoremqa_questions` | 0.409 | 0.414 | 0.072 | 0.069 | 2663 | 238 | 7.4% |
| Toy | `beir_fiqa` sample | 0.917 | 0.867 | 0.398 | 0.398 | 697 | 186 | 26.7% |

Main read:

- A won by speed and economics quality, but the arithmetic objective hid a large `beir_trec-covid` recall loss.
- B fixed that failure mode partially through geometric aggregation and a recall floor. It still accepted recall loss for latency, but less catastrophically.
- C is the strongest full run: it keeps aggregate recall roughly flat versus seed while cutting median latency by about an order of magnitude on the two largest datasets.
- Toy is useful as a fast search fixture, but the single-dataset 20-query evaluator makes its best candidate less trustworthy.

## Best Candidate Code Findings

### A: Token-Max-Coordinate Scan

Source: `output/late_interaction_freeform_latency_aware/20260503_130818_4cf5b9/best/program.py`

What it discovered:

1. Precompute per-document coordinate extrema and quantize them to int8.
2. Let each query choose a small set of high-importance embedding dimensions.
3. Use the coordinate envelope as a cheap full-corpus scan, then exact MaxSim rerank a fixed `24 * top_k` shortlist.

Minimal code:

```python
mx = tok.max(axis=0)
mn = tok.min(axis=0)
doc_max_q[d_idx] = np.rint(mx / scale).clip(-127, 127)
doc_min_q[d_idx] = np.rint(mn / scale).clip(-127, 127)

dim_budget = int(np.ceil(self.cfg.dim_budget_multiplier * np.sqrt(dim) * np.sqrt(Q)))
dims = np.argpartition(np.max(np.abs(q_tokens), axis=0), -dim_budget)[-dim_budget:]
approx[start:end] = (mx @ w_pos_sel) + (mn @ w_neg_sel)
```

Assessment: discovered at step 17, generation 1. This is a real algorithmic attempt, not a direct copy of a named system. It is still fairly heuristic: the coordinate envelope is a plausible upper-bound-style proxy, but the fixed `24 * top_k` rerank budget overfits corpus size. The result is fast on `beir_trec-covid` but loses too much recall there.

### B: PCA-Projected Anchor MaxSim

Source: `output/late_interaction_freeform_latency_aware/20260503_175442_510eda/best/program.py`

What it discovered:

1. Compute a deterministic PCA projection from per-document mean directions.
2. Represent each document with a few high-L2 token anchors plus a mean anchor.
3. Run a projected-space MaxSim scan over anchors, then exact MaxSim rerank with `max(4 * top_k, 48 * sqrt(N))`.

Minimal code:

```python
sum_xx += np.outer(xf, xf)
eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
P = eigvecs[:, order[-proj_dim:]]

sel = _select_topk_rows_by_norm(t, Config.anchors_per_doc)
full = np.vstack([a[:L], mean_dir[None, :]])
doc_scores += sim.reshape(n_docs, A, c).max(axis=1).sum(axis=1)
```

Assessment: discovered at step 17, generation 2. This is mostly a principled recombination of known ideas: PCA projection, document token sketches, approximate MaxSim, and exact reranking. It is more coherent than A's coordinate scan and improves latency on all three datasets, but it still pays with meaningful recall loss on economics and TREC-COVID.

### C: Progressive-Prefix Centered PCA-SimHash

Source: `output/late_interaction_freeform_latency_aware/20260504_015558_ef58b3/best/program.py`

What it discovered:

1. Build centered PCA-SimHash tables from corpus mean vectors instead of random hyperplanes.
2. Use fine buckets first, then back off to shorter coarse-prefix buckets when fine hits are too sparse.
3. Union IDF-weighted LSH votes with a dense mean-vector backstop before exact MaxSim reranking.

Minimal code:

```python
C = (Xc.T @ Xc) / max(Xc.shape[0], 1)
eigvals, eigvecs = np.linalg.eigh(C)
H = eigvecs[:, idx].T.astype(np.float32)
hyper[t] = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-6)

if lsh_hits_fine < int(coarse_trigger_hit_frac * rerank_n):
    base_c = (base_codes & coarse_mask)
    np.add.at(counts, arr, np.float32(w * coarse_w))

union = np.unique(np.concatenate([lsh_top, mean_top]))
pre = counts[union] + np.maximum(0.0, mean_scores[union])
```

Assessment: discovered at step 24, generation 4. This is the most convincing full-run result. The components are not novel research by themselves; SimHash, multi-probe LSH, IDF bucket weighting, dense backstops, and reranking are all established patterns. The useful discovery is the package: progressive coarse-prefix widening recovered enough recall while keeping exact rerank coverage under 14% on all datasets.

### Toy: Mean/Max Dense Summary Shortlist

Source: `output/late_interaction_freeform_latency_aware_toy/20260504_132707_1c4fa0/best/program.py`

What it discovered:

1. Precompute normalized mean-pooled and max-pooled document vectors.
2. Build query mean and query max vectors, then score each document by the best of four cosine proxies.
3. Rerank `max(top_k, 64 * sqrt(N))` candidates with exact MaxSim.

Minimal code:

```python
doc_means[i] = d.mean(axis=0, dtype=np.float32)
doc_maxs[i] = d.max(axis=0)
doc_means = doc_means / np.maximum(np.linalg.norm(doc_means, axis=1, keepdims=True), 1e-12)
doc_maxs = doc_maxs / np.maximum(np.linalg.norm(doc_maxs, axis=1, keepdims=True), 1e-12)

approx_scores = np.maximum(
    np.maximum(doc_means @ q_mean, doc_means @ q_max),
    np.maximum(doc_maxs @ q_mean, doc_maxs @ q_max),
)
rerank_n = max(top_k, int(64.0 * np.sqrt(n_docs)))
```

Assessment: discovered at step 12, generation 2. This is not very novel. It is a simple dense-summary shortlist plus exact rerank, close to a common bi-encoder-style first stage. On the 20-query FiQA toy evaluator it preserved nDCG@10 and recall@10/100 while sacrificing recall@1000 for latency, which the objective rewarded. Treat it as a quick heuristic, not as evidence of a robust late-interaction discovery.

## Bottom Line

The key evolution setup change was moving from arithmetic aggregation in A to geometric aggregation plus a recall floor in B/C/Toy. That changed selection pressure from "win one axis hard" toward "do not collapse on any dataset." The evaluator change in C also mattered: FiQA, StackOverflow, and TheoremQA questions rewarded corpus-size-scaled candidate generation more clearly than the earlier theorem/economics/TREC trio.

Among the full runs, C is the best candidate to continue from. It is not a fundamentally new retrieval family, but it is the most robust discovered engineering pattern here: adaptive LSH candidate generation, dense fallback, and exact MaxSim reranking with small corpus coverage.
