#!/bin/bash
# Run full evaluation (all datasets) for the three best programs AND the three seed
# programs — all 6 runs in parallel (separate processes).
#
# Use inside a single allocation with enough resources for 6 parallel evals:
#   e.g. 192 CPUs, 600 GB (see scripts/slurm_eval_best_and_seeds.sh).

set -e

export EVAL_EXCLUDE_DATASETS="${EVAL_EXCLUDE_DATASETS:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

BEST_FREEFORM="output/openevolve_output_freeform_fast/20260202_075458/best/best_program.py"
BEST_COMPOSABLE="output/openevolve_output_composable_fast/20260202_030814/best/best_program.py"
BEST_CONSTRAINED="output/openevolve_output_constrained_fast/20260201_231921/best/best_program.py"
SEED_CONSTRAINED="src/ranking_evolved/bm25_constrained_fast.py"
SEED_COMPOSABLE="src/ranking_evolved/bm25_composable_fast.py"
SEED_FREEFORM="src/ranking_evolved/bm25_freeform_fast.py"

for path in "$BEST_FREEFORM" "$BEST_COMPOSABLE" "$BEST_CONSTRAINED" "$SEED_CONSTRAINED" "$SEED_COMPOSABLE" "$SEED_FREEFORM"; do
  if [ ! -f "$path" ]; then
    echo "Error: Program not found: $path"
    exit 1
  fi
done

echo "=============================================="
echo "Full evaluation: 6 runs in parallel (3 best + 3 seeds)"
echo "EVAL_EXCLUDE_DATASETS=$EVAL_EXCLUDE_DATASETS"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# Start all 6 in parallel
uv run python evaluator_parallel.py "$BEST_FREEFORM"   --save "$RESULTS_DIR/best_freeform_full.json"   --verbose > "$RESULTS_DIR/best_freeform_full.log"   2>&1 & P1=$!
uv run python evaluator_parallel.py "$BEST_COMPOSABLE" --save "$RESULTS_DIR/best_composable_full.json" --verbose > "$RESULTS_DIR/best_composable_full.log" 2>&1 & P2=$!
uv run python evaluator_parallel.py "$BEST_CONSTRAINED" --save "$RESULTS_DIR/best_constrained_full.json" --verbose > "$RESULTS_DIR/best_constrained_full.log" 2>&1 & P3=$!
uv run python evaluator_parallel.py "$SEED_FREEFORM"   --save "$RESULTS_DIR/seed_freeform_full.json"   --verbose > "$RESULTS_DIR/seed_freeform_full.log"   2>&1 & P4=$!
uv run python evaluator_parallel.py "$SEED_COMPOSABLE" --save "$RESULTS_DIR/seed_composable_full.json" --verbose > "$RESULTS_DIR/seed_composable_full.log" 2>&1 & P5=$!
uv run python evaluator_parallel.py "$SEED_CONSTRAINED" --save "$RESULTS_DIR/seed_constrained_full.json" --verbose > "$RESULTS_DIR/seed_constrained_full.log" 2>&1 & P6=$!

echo "Started 6 evaluations. PIDs: $P1 $P2 $P3 $P4 $P5 $P6"
echo "Logs: $RESULTS_DIR/{best,seed}_*_full.log"
echo "Waiting for all to finish..."
echo ""

FAIL=0
wait $P1 && echo "  best freeform done."   || { echo "  best freeform failed."; FAIL=1; }
wait $P2 && echo "  best composable done." || { echo "  best composable failed."; FAIL=1; }
wait $P3 && echo "  best constrained done." || { echo "  best constrained failed."; FAIL=1; }
wait $P4 && echo "  seed freeform done."   || { echo "  seed freeform failed."; FAIL=1; }
wait $P5 && echo "  seed composable done." || { echo "  seed composable failed."; FAIL=1; }
wait $P6 && echo "  seed constrained done." || { echo "  seed constrained failed."; FAIL=1; }

echo ""
echo "=============================================="
echo "Done. Results:"
echo "  Best:  $RESULTS_DIR/best_{freeform,composable,constrained}_full.json"
echo "  Seeds: $RESULTS_DIR/seed_{freeform,composable,constrained}_full.json"
echo "=============================================="

[ $FAIL -eq 0 ] || exit 1
