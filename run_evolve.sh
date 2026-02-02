# Constrained fast
OUT_CONSTRAINED=output/openevolve_output_constrained_fast/$(date +%Y%m%d_%H%M%S)
uv run python -m openevolve.cli src/ranking_evolved/bm25_constrained_fast.py evaluator_parallel.py --config openevolve_config_constrained.yaml --output "$OUT_CONSTRAINED"
uv run python scripts/plot_evolution_metrics.py "$OUT_CONSTRAINED" --save "$OUT_CONSTRAINED/evolution_metrics.png" --no-show

# Composable fast
OUT_COMPOSABLE=output/openevolve_output_composable_fast/$(date +%Y%m%d_%H%M%S)
uv run python -m openevolve.cli src/ranking_evolved/bm25_composable_fast.py evaluator_parallel.py --config openevolve_config_composable.yaml --output "$OUT_COMPOSABLE"
uv run python scripts/plot_evolution_metrics.py "$OUT_COMPOSABLE" --save "$OUT_COMPOSABLE/evolution_metrics.png" --no-show

# Freeform fast
OUT_FREEFORM=output/openevolve_output_freeform_fast/$(date +%Y%m%d_%H%M%S)
uv run python -m openevolve.cli src/ranking_evolved/bm25_freeform_fast.py evaluator_parallel.py --config openevolve_config_freeform.yaml --output "$OUT_FREEFORM"
uv run python scripts/plot_evolution_metrics.py "$OUT_FREEFORM" --save "$OUT_FREEFORM/evolution_metrics.png" --no-show

# Resume from existing run (after initial 70 steps):
# 1. Find the latest checkpoint: ls -t <run_dir>/checkpoints/ | head -1
# 2. Use --checkpoint <run_dir>/checkpoints/checkpoint_N and --output <run_dir>
# 3. For resume: set max_iterations to REMAINING steps (e.g. 130 to go from 70 → 200 total)
#
# Example resume commands:
# RUN_DIR=output/openevolve_output_constrained_fast/20260201_184826
# LATEST_CHECKPOINT=$(ls -t "$RUN_DIR/checkpoints/" | head -1)
# uv run python -m openevolve.cli src/ranking_evolved/bm25_constrained_fast.py evaluator_parallel.py --config openevolve_config_constrained.yaml --output "$RUN_DIR" --checkpoint "$RUN_DIR/checkpoints/$LATEST_CHECKPOINT"

