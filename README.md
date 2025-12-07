# ranking-evolved

BM25 ranking experiments with evolution via [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). The BRIGHT biology split is used as the evaluation task.

## Run the OpenEvolve experiment

1. Install deps (needs Python >=3.11): `uv sync` (or `pip install -e .` if you prefer pip)
2. Set your LLM key (uses OpenAI-compatible API): `export OPENAI_API_KEY="your-key"`
3. Run evolution:  
   `python openevolve/openevolve-run.py src/ranking_evolved/bm25.py evaluator_bright_biology.py --config openevolve_config.yaml --iterations 40`
4. Inspect metrics in the CLI output and artifacts in `openevolve_output/`.

The evaluator (`evaluator_bright_biology.py`) computes precision/recall@k, NDCG@k, MAP, and MRR, and uses their average as `combined_score` for selection.

## Experiment journal

- 2025-12-06: Baseline BM25 scored combined ~0.0836 on full BRIGHT biology (precision@10 ~0.0369, recall@10 ~0.1135, NDCG@10 ~0.0813, MAP ~0.0592, MRR ~0.1269).
- 2025-12-06: OpenEvolve run (80 iterations, smoothed IDF + tf log damping integrated) produced best combined ~0.1506 on full BRIGHT biology (precision@10 ~0.0515, recall@10 ~0.1644, NDCG@10 ~0.1547, MAP ~0.1297, MRR ~0.2526). Updated `src/ranking_evolved/bm25.py` with those kernel tweaks (removed self-normalization).
