# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). The BRIGHT biology split is used as the evaluation task.

## Run the OpenEvolve experiment

1. Install deps (needs Python >=3.11): `uv sync` (or `pip install -e .` if you prefer pip)
2. Set your LLM key (uses OpenAI-compatible API): `export OPENAI_API_KEY="your-key"`
3. Run evolution (config currently set to 200 iterations):  
   `python openevolve/openevolve-run.py src/ranking_evolved/bm25.py evaluator_bright.py --config openevolve_config.yaml`
4. Inspect metrics in the CLI output and artifacts in `openevolve_output/` (best program and logs).

The evaluator (`evaluator_bright.py`) computes precision/recall@k, NDCG@k, MAP, and MRR, and uses their average as `combined_score` for selection. For faster exploratory runs, set `BRIGHT_SAMPLE_QUERIES` (e.g., 32) and optionally `BRIGHT_SAMPLE_SEED` to subsample queries during evaluation.

Quick exploratory run (shorter, higher temperature):
- Set sampling if desired: `export BRIGHT_SAMPLE_QUERIES=32` (or unset for full queries)
- Run: `python openevolve/openevolve-run.py src/ranking_evolved/bm25.py evaluator_bright.py --config openevolve_config_explore.yaml`

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
