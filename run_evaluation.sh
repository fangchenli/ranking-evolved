#!/bin/bash
# Run full evaluation (all datasets) on the three best evolved programs.
#
# Uses the same best-program paths as in the three-way comparison:
#   - Freeform:   output/openevolve_output_freeform_fast/20260202_075458/best/best_program.py
#   - Composable: output/openevolve_output_composable_fast/20260202_030814/best/best_program.py
#   - Constrained: output/openevolve_output_constrained_fast/20260201_231921/best/best_program.py
#
# To evaluate all datasets (BRIGHT + BEIR + TREC DL), EVAL_EXCLUDE_DATASETS must be unset or empty.
# Evolution runs use a reduced set; this script runs the full benchmark suite.

set -e

# Evaluate on ALL datasets (no exclusions)
export EVAL_EXCLUDE_DATASETS=""

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
echo "Full evaluation (all datasets)"
echo "EVAL_EXCLUDE_DATASETS=$EVAL_EXCLUDE_DATASETS"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# 1. Freeform best
echo ""
echo ">>> Evaluating Freeform best: $BEST_FREEFORM"
uv run python evaluator_parallel.py "$BEST_FREEFORM" \
  --save "$RESULTS_DIR/best_freeform_full.json" \
  --verbose

# 2. Composable best
echo ""
echo ">>> Evaluating Composable best: $BEST_COMPOSABLE"
uv run python evaluator_parallel.py "$BEST_COMPOSABLE" \
  --save "$RESULTS_DIR/best_composable_full.json" \
  --verbose

# 3. Constrained best
echo ""
echo ">>> Evaluating Constrained best: $BEST_CONSTRAINED"
uv run python evaluator_parallel.py "$BEST_CONSTRAINED" \
  --save "$RESULTS_DIR/best_constrained_full.json" \
  --verbose

echo ""
echo "=============================================="
echo "Done. Results saved to:"
echo "  $RESULTS_DIR/best_freeform_full.json"
echo "  $RESULTS_DIR/best_composable_full.json"
echo "  $RESULTS_DIR/best_constrained_full.json"
echo "=============================================="
