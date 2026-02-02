# Constrained fast
uv run python -m openevolve.cli src/ranking_evolved/bm25_constrained_fast.py evaluator_parallel.py --config openevolve_config_constrained.yaml --output output/openevolve_output_constrained_fast/$(date +%Y%m%d_%H%M%S)

# Composable fast
uv run python -m openevolve.cli src/ranking_evolved/bm25_composable_fast.py evaluator_parallel.py --config openevolve_config_composable.yaml --output output/openevolve_output_composable_fast/$(date +%Y%m%d_%H%M%S)

# Freeform fast
uv run python -m openevolve.cli src/ranking_evolved/bm25_freeform_fast.py evaluator_parallel.py --config openevolve_config_freeform.yaml --output output/openevolve_output_freeform_fast/$(date +%Y%m%d_%H%M%S)

# Resume from existing run: use --checkpoint <run_dir>/checkpoints/checkpoint_N and --output <run_dir>
# e.g. --output output/openevolve_output_constrained_fast/20260201_184826 --checkpoint output/openevolve_output_constrained_fast/20260201_184826/checkpoints/checkpoint_10

