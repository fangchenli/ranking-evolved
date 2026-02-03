#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eval-all-%j.out
#SBATCH --error=slurm-eval-all-%j.err

# Run all three full evaluations in parallel inside one job.
# Adjust --cpus-per-task and --mem if your cluster has different limits.
# Optional: uncomment to cap workers per eval (e.g. 24 each).
# export EVAL_MAX_WORKERS=24

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export EVAL_EXCLUDE_DATASETS=""

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK Mem=${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"

./run_evaluation_parallel.sh

echo "End: $(date)"
