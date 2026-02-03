#!/bin/bash
# Run full evaluation (all datasets) for the three best programs IN PARALLEL.
#
# Use this inside a single allocation that has enough resources for 3 runs:
#   e.g. 96 CPUs, 300 GB RAM (see docs/SLURM_FULL_EVALUATION.md).
#
# Optional: set EVAL_MAX_WORKERS to cap workers per run (e.g. 24 so 3×24=72 CPUs).

set -e

export EVAL_EXCLUDE_DATASETS="${EVAL_EXCLUDE_DATASETS:-}"

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

BEST_FREEFORM="output/openevolve_output_freeform_fast/20260202_075458/best/best_program.py"
BEST_COMPOSABLE="output/openevolve_output_composable_fast/20260202_030814/best/best_program.py"
BEST_CONSTRAINED="output/openevolve_output_constrained_fast/20260201_231921/best/best_program.py"

for path in "$BEST_FREEFORM" "$BEST_COMPOSABLE" "$BEST_CONSTRAINED"; do
  if [ ! -f "$path" ]; then
    echo "Error: Best program not found: $path"
    exit 1
  fi
done

echo "=============================================="
echo "Full evaluation (all datasets) — 3 runs in parallel"
echo "EVAL_EXCLUDE_DATASETS=$EVAL_EXCLUDE_DATASETS"
echo "EVAL_MAX_WORKERS=${EVAL_MAX_WORKERS:-auto}"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# Run all three in background
uv run python evaluator_parallel.py "$BEST_FREEFORM" \
  --save "$RESULTS_DIR/best_freeform_full.json" --verbose \
  > "$RESULTS_DIR/best_freeform_full.log" 2>&1 &
PID_FREEFORM=$!

uv run python evaluator_parallel.py "$BEST_COMPOSABLE" \
  --save "$RESULTS_DIR/best_composable_full.json" --verbose \
  > "$RESULTS_DIR/best_composable_full.log" 2>&1 &
PID_COMPOSABLE=$!

uv run python evaluator_parallel.py "$BEST_CONSTRAINED" \
  --save "$RESULTS_DIR/best_constrained_full.json" --verbose \
  > "$RESULTS_DIR/best_constrained_full.log" 2>&1 &
PID_CONSTRAINED=$!

echo "Started 3 evaluations: PIDs $PID_FREEFORM (freeform), $PID_COMPOSABLE (composable), $PID_CONSTRAINED (constrained)"
echo "Logs: $RESULTS_DIR/best_*_full.log"
echo "Waiting for all to finish..."
echo ""

wait $PID_FREEFORM && echo "Freeform done." || { echo "Freeform failed (PID $PID_FREEFORM)."; exit 1; }
wait $PID_COMPOSABLE && echo "Composable done." || { echo "Composable failed (PID $PID_COMPOSABLE)."; exit 1; }
wait $PID_CONSTRAINED && echo "Constrained done." || { echo "Constrained failed (PID $PID_CONSTRAINED)."; exit 1; }

echo ""
echo "=============================================="
echo "Done. Results:"
echo "  $RESULTS_DIR/best_freeform_full.json"
echo "  $RESULTS_DIR/best_composable_full.json"
echo "  $RESULTS_DIR/best_constrained_full.json"
echo "=============================================="
