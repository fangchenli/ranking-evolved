#!/bin/bash
#SBATCH --job-name=eval-best-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eval-best-seeds-%j.out
#SBATCH --error=slurm-eval-best-seeds-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnian@scu.edu

# Run full evaluation for 3 best + 3 seed programs (all datasets) — all 6 in parallel.
# Resources: 6 × (~32 CPUs, ~100 GB) = 192 CPUs, 600 GB.
# Set --mail-user to your email for job completion/failure notifications.

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export EVAL_EXCLUDE_DATASETS=""

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK Mem=${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"

./run_evaluation_and_seeds_parallel.sh

echo "End: $(date)"
