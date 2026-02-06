#!/bin/bash
#SBATCH --job-name=rerun-fail-seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/log/slurm-rerun-failed-sequential-%j.out
#SBATCH --error=results/log/slurm-rerun-failed-sequential-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnian@scu.edu

# Sequential rerun of failed datasets for all 7 programs.
# Progress (Evaluating... / Batch X / dataset OK or ERROR) goes to stderr → .err
# file. To watch: tail -f results/log/slurm-rerun-failed-sequential-<jobid>.err

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/log

YFANG_LAB_DATA="${YFANG_LAB_DATA:-/WAVE/datasets/yfang_lab/jnian/ranking-evolved-data}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$YFANG_LAB_DATA/hf_cache}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK Mem=${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"

./run_rerun_failed_sequential.sh

echo "End: $(date)"
