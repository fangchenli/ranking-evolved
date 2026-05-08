#!/usr/bin/env bash
# Run all 3 baseline programs (exact MaxSim, FastPLAID, freeform seed) on
# the 12-dataset suite, full eval (all queries per dataset), CPU.
#
# Why `script -q FILE`: the evaluator's per-query progress bars (tqdm) write
# to stderr and self-disable when stderr is not a TTY. Even `| tee` makes
# stdout a pipe, and some terminals/shells trip tqdm's TTY detection on
# stderr too. `script` allocates a real PTY for the child process, so tqdm
# always renders live bars on screen, and the same byte stream is captured
# to the log file. The log file contains tqdm's carriage-return updates
# (greppable for `[eval]` summary lines but visually noisy in an editor).
#
# `PYTHONUNBUFFERED=1` prevents Python from block-buffering its output when
# it sees a non-interactive stream; with a PTY this rarely matters but is
# cheap insurance for the long runs.
set -euo pipefail

DATASETS="beir_arguana,beir_fiqa,beir_nfcorpus,beir_scifact,beir_scidocs,beir_trec-covid,bright_biology,bright_earth_science,bright_economics,bright_pony,bright_stackoverflow,bright_theoremqa_theorems"

mkdir -p tasks/late_interaction/baselines

# 1/3: exact MaxSim correctness anchor
EVAL_DEVICE=cpu PYTHONUNBUFFERED=1 \
  script -q tasks/late_interaction/baselines/exact_maxsim.cpu.log \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/programs/exact_maxsim.py \
    --datasets "$DATASETS" \
    --warmup-queries 10 --timed-repeats 3

# 2/3: FastPLAID external speed reference
EVAL_DEVICE=cpu PYTHONUNBUFFERED=1 \
  script -q tasks/late_interaction/baselines/fastplaid.cpu.log \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/programs/fastplaid.py \
    --datasets "$DATASETS" \
    --warmup-queries 10 --timed-repeats 3

# 3/3: freeform seed equivalence gate
EVAL_DEVICE=cpu PYTHONUNBUFFERED=1 \
  script -q tasks/late_interaction/baselines/freeform.cpu.log \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/seeds/freeform.py \
    --datasets "$DATASETS" \
    --warmup-queries 10 --timed-repeats 3


# chmod +x bash_scripts/baselines_late_interaction.sh
# ./bash_scripts/baselines_late_interaction.sh