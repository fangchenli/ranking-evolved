#!/bin/bash
#SBATCH --job-name=resume-evolve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-resume-evolve-%j.out
#SBATCH --error=slurm-resume-evolve-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnian@scu.edu

# Continue the three OpenEvolve experiments for 130 more steps (70 + 130 = 200 total).
# Set EVAL_EXCLUDE_DATASETS so each candidate is evaluated on the fast 12-dataset set.
#
# Before submitting: export OPENAI_API_KEY on the login node; sbatch copies your
# environment to the job, so the job will see it. Example:
#   export OPENAI_API_KEY="sk-..."
#   export EVAL_EXCLUDE_DATASETS="dl19,dl20,..."   # optional; script sets it below
#   sbatch scripts/slurm_resume_evolve.sh
#
# Edit the RUN_DIR_* below if your run directories differ.

RUN_DIR_CONSTRAINED="output/openevolve_output_constrained_fast/20260201_231921"
RUN_DIR_COMPOSABLE="output/openevolve_output_composable_fast/20260202_030814"
RUN_DIR_FREEFORM="output/openevolve_output_freeform_fast/20260202_075458"

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Fast evaluation: exclude large datasets (same as commands.txt)
export EVAL_EXCLUDE_DATASETS="dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora,webis-touche2020,cqadupstack,leetcode,aops,theoremqa_questions,robotics,psychology,sustainable_living"

# OPENAI_API_KEY is inherited from the shell that ran sbatch. Ensure it's set.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is not set. Export it on the login node before sbatch."
  exit 1
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK Mem=${SLURM_MEM_PER_NODE}MB"
echo "EVAL_EXCLUDE_DATASETS=$EVAL_EXCLUDE_DATASETS"
echo "Start: $(date)"

resume_one() {
  local RUN_DIR="$1"
  local CONFIG="$2"
  local SEED="$3"
  local NAME="$4"
  local CHECKPOINT_DIR="$RUN_DIR/checkpoints"
  local LATEST
  LATEST=$(ls -t "$CHECKPOINT_DIR" 2>/dev/null | grep "^checkpoint_" | head -1)
  if [ -z "$LATEST" ]; then
    echo "Error: No checkpoint in $CHECKPOINT_DIR"
    return 1
  fi
  echo "--- Resuming $NAME from $CHECKPOINT_DIR/$LATEST ---"
  uv run python -m openevolve.cli "$SEED" evaluator_parallel.py --config "$CONFIG" --output "$RUN_DIR" --checkpoint "$CHECKPOINT_DIR/$LATEST"
  uv run python scripts/plot_evolution_metrics.py "$RUN_DIR" --save "$RUN_DIR/evolution_metrics.png" --no-show
}

resume_one "$RUN_DIR_CONSTRAINED" openevolve_config_constrained.yaml src/ranking_evolved/bm25_constrained_fast.py "constrained"
resume_one "$RUN_DIR_COMPOSABLE" openevolve_config_composable.yaml src/ranking_evolved/bm25_composable_fast.py "composable"
resume_one "$RUN_DIR_FREEFORM"   openevolve_config_freeform.yaml   src/ranking_evolved/bm25_freeform_fast.py "freeform"

echo "End: $(date)"
