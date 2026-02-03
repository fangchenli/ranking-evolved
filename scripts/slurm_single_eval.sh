#!/bin/bash
# Run a single full evaluation. Usage:
#   sbatch [sbatch-args] scripts/slurm_single_eval.sh <program.py> <output.json>
# Example:
#   sbatch --cpus-per-task=32 --mem=100G --time=2-00:00:00 \
#     scripts/slurm_single_eval.sh output/.../best_program.py results/best_freeform_full.json

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err

PROGRAM_PATH="$1"
OUTPUT_JSON="$2"

if [ -z "$PROGRAM_PATH" ] || [ -z "$OUTPUT_JSON" ]; then
  echo "Usage: $0 <program.py> <output.json>"
  exit 1
fi

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export EVAL_EXCLUDE_DATASETS=""

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Program: $PROGRAM_PATH"
echo "Output: $OUTPUT_JSON"
echo "Start: $(date)"

uv run python evaluator_parallel.py "$PROGRAM_PATH" --save "$OUTPUT_JSON" --verbose

echo "End: $(date)"
