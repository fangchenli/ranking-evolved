#!/bin/bash
#SBATCH --job-name=eval-wave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/log/slurm-eval-wave-%j.out
#SBATCH --error=results/log/slurm-eval-wave-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnian@scu.edu

# Single-program wave evaluation. Submit with:
#   sbatch --export=ALL,PROGRAM_PATH=<path>,RESULT_NAME=<name> -D /path/to/repo scripts/slurm_eval_one_wave.sh
# PROGRAM_PATH: path to best_program.py (relative to repo root or absolute).
# RESULT_NAME: output filename without .json (e.g. seed_freeform_full or composable_step141_full).

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/log

if [ -z "${PROGRAM_PATH}" ] || [ -z "${RESULT_NAME}" ]; then
  echo "ERROR: PROGRAM_PATH and RESULT_NAME must be set (e.g. sbatch --export=ALL,PROGRAM_PATH=...,RESULT_NAME=...)" >&2
  exit 1
fi

YFANG_LAB_DATA="${YFANG_LAB_DATA:-/WAVE/datasets/yfang_lab/jnian/ranking-evolved-data}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$YFANG_LAB_DATA/hf_cache}"
export PYTHONUNBUFFERED=1
# Use all allocated CPUs for tokenization (indexing)
# Note: batch_rank query parallelism is capped at 8-16 threads for huge corpora (see bm25_*_fast.py)
export EVAL_THREADS_PER_WORKER="${SLURM_CPUS_PER_TASK:-16}"
export BM25_QUERY_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PROGRAM_PATH=$PROGRAM_PATH"
echo "RESULT_NAME=$RESULT_NAME"
echo "CPUs=$SLURM_CPUS_PER_TASK Mem=${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"

uv run python evaluator_parallel_wave.py "$PROGRAM_PATH" --save "results/${RESULT_NAME}.json" --verbose

echo "End: $(date)"
