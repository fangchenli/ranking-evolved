# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). The BRIGHT biology split is used as the evaluation task.

## Run the OpenEvolve experiment

1. Install deps (needs Python >=3.11): `uv sync` (or `pip install -e .` if you prefer pip)
2. Set your LLM key (uses OpenAI-compatible API): `export OPENAI_API_KEY="your-key"`
3. Run evolution (config currently set to 200 iterations):  
   `uv run python openevolve/openevolve-run.py src/ranking_evolved/bm25.py evaluator_bright.py --config openevolve_config.yaml`
4. Inspect metrics in the CLI output and artifacts in `openevolve_output/` (best program and logs).

The evaluator (`evaluator_bright.py`) computes precision/recall@k, NDCG@k, MAP, and MRR, and uses their average as `combined_score` for selection. For direct evaluation without OpenEvolve:  
`uv run python evaluator_bright.py src/ranking_evolved/bm25.py --k 10 --sample-queries 0 --domain biology`

### Scoring equations

Current kernel (iteration 113, integrated):
- Clipped IDF:  
  `idf(t) = min(8, max(0, log((N + 0.5) / (df(t) + 0.5))))`
- Per-term (with unique query terms):  
  `tf_raw = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avg_dl))`  
  `tf_sat = tf / (tf + k1 + 0.5)`  
  `score(t, d) = idf(t) * log(1 + tf_raw * tf_sat)`  
  where `k1=1.5`, `b=0.75`, `|d|` is document length, and `avg_dl` is average document length. Early exit if no terms match.

Overnight kernel (2025-12-07, before iteration 113):
- Smoothed IDF: `idf(t) = log((N + 1) / (df(t) + ε))`, with `ε ≈ 1e-5`
- Per-term: `score(t, d) = idf(t) * ( tf / (tf + k1 + 1) ) * ( tf * (k1 + 1) / ( tf + k1 * (1 - b + b * |d| / avg_dl) ) )`

Previous kernel (2025-12-06 run):
- IDF smoothing: `idf(t) = log((N + 1) / (df(t) + 1))`
- Per-term: `score(t, d) = idf(t) * (1 + log(1 + tf)) * ( tf * (k1 + 1) / ( tf + k1 * (1 - b + b * |d| / avg_dl) ) )`

## Experiment journal

- 2025-12-06 (commit before 19050cf): Baseline BM25 scored combined ~0.0836 on full BRIGHT biology (precision@10 ~0.0369, recall@10 ~0.1135, NDCG@10 ~0.0813, MAP ~0.0592, MRR ~0.1269).
- 2025-12-06 (commit around 19050cf): OpenEvolve run (80 iterations, smoothed IDF + tf log damping integrated) produced best combined ~0.1506 on full BRIGHT biology (precision@10 ~0.0515, recall@10 ~0.1644, NDCG@10 ~0.1547, MAP ~0.1297, MRR ~0.2526). Updated `src/ranking_evolved/bm25.py` with those kernel tweaks (removed self-normalization).
- 2025-12-07 (commit 19050cf+): Overnight OpenEvolve run (200 iterations, tf saturation + tighter IDF smoothing) yielded combined ~0.1748 on full BRIGHT biology (precision@10 ~0.0641, recall@10 ~0.1955, NDCG@10 ~0.1828, MAP ~0.1469, MRR ~0.2846). `src/ranking_evolved/bm25.py` now uses the smoothed IDF (ε≈1e-5) and tf saturation factor.
- 2025-12-07 later: Best in-run candidate (iteration 113, ID 7cc8c383…) re-evaluated at combined ~0.2095 on full BRIGHT biology (precision@10 ~0.0796, recall@10 ~0.2548, NDCG@10 ~0.2219, MAP ~0.1724, MRR ~0.3188). This kernel (clipped IDF, unique query terms, log-damped TF saturation) is integrated into `src/ranking_evolved/bm25.py`.
- 2025-12-08: Psychology-focused run (gpt-5.1, 100 iterations) yielded a modest psych candidate (saved as `openevolve_output/best/best_program_psychology.py`) with combined ~0.0847 on psychology (prec@10 ~0.0386, rec@10 ~0.1145, nDCG@10 ~0.0870, MAP ~0.0707, MRR ~0.1125), up from the baseline psych score (~0.0756). Not integrated into main `bm25.py`.

## Gensim BM25 baseline

- Script: `uv run python scripts/eval_bright_gensim_lucene.py --domain biology --k 10` (add `--top-k` to truncate ranking or adjust `--k1`/`--b`).
- Pipeline: tokenizes BRIGHT, builds a `Dictionary`, fits `LuceneBM25Model`, converts docs with `model[bow]`, ranks via a `SparseMatrixSimilarity` index, and reports JSON metrics (Prec@k, Rec@k, nDCG@k, MAP, MRR, combined).

## Full BRIGHT evaluation (current bm25.py, k=10, full queries)

| Split | Combined | Prec@10 | Rec@10 | nDCG@10 | MAP | MRR | Queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biology | 0.2095 | 0.0796 | 0.2548 | 0.2219 | 0.1724 | 0.3188 | 103 |
| earth_science | 0.2790 | 0.1086 | 0.2999 | 0.2963 | 0.2483 | 0.4421 | 116 |
| economics | 0.1097 | 0.0534 | 0.1362 | 0.1155 | 0.0886 | 0.1549 | 103 |
| psychology | 0.0756 | 0.0337 | 0.0980 | 0.0749 | 0.0633 | 0.1080 | 101 |
| robotics | 0.1054 | 0.0426 | 0.1440 | 0.1063 | 0.0842 | 0.1499 | 101 |
| stackoverflow | 0.1622 | 0.0752 | 0.1909 | 0.1727 | 0.1474 | 0.2248 | 117 |
| sustainable_living | 0.1149 | 0.0463 | 0.1568 | 0.1194 | 0.0980 | 0.1542 | 108 |
| pony | 0.1802 | 0.1759 | 0.0917 | 0.1774 | 0.1067 | 0.3491 | 112 |
| aops | 0.0229 | 0.0135 | 0.0265 | 0.0191 | 0.0130 | 0.0422 | 111 |
| theoremqa_theorems | 0.0459 | 0.0118 | 0.0592 | 0.0472 | 0.0367 | 0.0744 | 76 |
| theoremqa_questions | 0.0532 | 0.0175 | 0.0872 | 0.0551 | 0.0483 | 0.0580 | 194 |
| leetcode | 0.1206 | 0.0430 | 0.2080 | 0.1267 | 0.0957 | 0.1295 | 142 |
| **macro avg** | **0.1233** | **0.0584** | **0.1461** | **0.1277** | **0.1002** | **0.1838** | — |

Macro combined (all splits): ~0.1233. Low performers: aops, theoremqa_*; consider targeted tuning.
