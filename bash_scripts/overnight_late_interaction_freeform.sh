#!/usr/bin/env bash
# Overnight run for late_interaction/freeform on the 3-dataset suite:
#   beir_fiqa, bright_stackoverflow, bright_theoremqa_questions
#
# Pipeline (sequential — each step requires the previous one's output):
#   Step A — Encode embeddings for bright_theoremqa_questions if missing.
#            (beir_fiqa + bright_stackoverflow are already cached.)
#   Step B — Run exact MaxSim baseline so the latency baseline JSON has all
#            three datasets. Resume-mode skips datasets already present
#            with a matching run config (so fiqa + stackoverflow stay).
#   Step C — Run the freeform seed and the composable seed on the same
#            three datasets. Both should produce recall@1000 / ndcg@10
#            BIT-IDENTICAL to exact MaxSim (any divergence > 1e-4 means a
#            seed regressed and the evolve loop's baseline would be wrong).
#   Step D — Verify equivalence; abort the script if seeds drifted.
#   Step E — Launch the evolve loop with the freeform seed (--max-iterations
#            controllable via the env var below).
#
# Logs:
#   bash_scripts/logs/<timestamp>/{encode,baseline_exact,baseline_freeform,
#                                  baseline_composable,equiv_check,evolve}.log
#   The evolve run additionally writes its own
#   `output/late_interaction_freeform_latency_aware/<run_id>/run.log`.
#
# Usage:
#   chmod +x bash_scripts/overnight_late_interaction_freeform.sh
#   ./bash_scripts/overnight_late_interaction_freeform.sh
#
# Tunable via env (override at invocation):
#   MAX_ITERATIONS   default 100  — total iterations for the evolve loop
#   EVAL_DEVICE      default cpu  — passed to encode + baselines + evolve
#   ENCODE_BATCH     default 16   — encoding batch size
#   EQUIV_TOL        default 1e-3 — recall/ndcg drift tolerance vs exact MaxSim
#
# Why `set -euo pipefail`: a partial pipeline overnight is worse than no
# pipeline. If encoding or a baseline run fails we want to halt before
# burning hours on a broken evolve run that reads stale baselines.
#
# Why `script -q`: the evaluator's tqdm bars self-disable when stderr is
# not a TTY. `script` allocates a real PTY for the child process so the
# bars render live AND the same byte stream is captured to a log file.

set -euo pipefail

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATASETS_LI="beir_fiqa,bright_stackoverflow,bright_theoremqa_questions"
ENCODE_SPECS=("bright:theoremqa_questions")  # only this one needs encoding
CACHE_ROOT="cache/late_interaction"
MODEL_DIR_NAME="lightonai__LateOn"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
ENCODE_BATCH="${ENCODE_BATCH:-16}"
EQUIV_TOL="${EQUIV_TOL:-1e-3}"

CONFIG_FREEFORM="tasks/late_interaction/configs/freeform_latency_aware.yaml"

BASELINE_DIR="tasks/late_interaction/baselines"
EXACT_JSON="${BASELINE_DIR}/exact_maxsim.${EVAL_DEVICE}.json"
FREEFORM_JSON="${BASELINE_DIR}/freeform.${EVAL_DEVICE}.json"
COMPOSABLE_JSON="${BASELINE_DIR}/composable.${EVAL_DEVICE}.json"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="bash_scripts/logs/${TS}"
mkdir -p "${LOG_DIR}"

echo "==============================================================="
echo "Overnight late_interaction freeform pipeline"
echo "  timestamp:        ${TS}"
echo "  datasets:         ${DATASETS_LI}"
echo "  device:           ${EVAL_DEVICE}"
echo "  max_iterations:   ${MAX_ITERATIONS}"
echo "  log dir:          ${LOG_DIR}"
echo "==============================================================="

# -----------------------------------------------------------------------------
# Sanity gate: OPENAI_API_KEY must be readable BEFORE we start encoding,
# so we don't burn hours on prep and then bail at the evolve step.
# -----------------------------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]] && ! grep -qE '^(export[[:space:]]+)?OPENAI_API_KEY=' .env 2>/dev/null; then
  echo "[fatal] OPENAI_API_KEY is not set in the env and not present in .env."
  echo "        The evolve step needs it. Set it and re-run."
  exit 2
fi

# -----------------------------------------------------------------------------
# Step A — Encode missing embeddings.
# -----------------------------------------------------------------------------
echo
echo "[A] Encoding missing embedding caches"
for spec in "${ENCODE_SPECS[@]}"; do
  ds_name="${spec/:/_}"   # bright:theoremqa_questions -> bright_theoremqa_questions
  cache_subdir="${CACHE_ROOT}/${MODEL_DIR_NAME}/${ds_name}"
  if [[ -f "${cache_subdir}/metadata.json" ]]; then
    echo "[A] skip ${spec} — cache already present at ${cache_subdir}"
    continue
  fi
  echo "[A] encoding ${spec} -> ${cache_subdir}"
  EVAL_DEVICE="${EVAL_DEVICE}" PYTHONUNBUFFERED=1 \
    script -q "${LOG_DIR}/encode_${ds_name}.log" \
    uv run python -m tasks.late_interaction.encode_embeddings \
      --dataset "${spec}" \
      --output-dir "${CACHE_ROOT}" \
      --batch-size "${ENCODE_BATCH}" \
      --dtype float16
done

# -----------------------------------------------------------------------------
# Step B — exact MaxSim baseline (refreshes / extends exact_maxsim.cpu.json).
# -----------------------------------------------------------------------------
echo
echo "[B] Running exact MaxSim baseline -> ${EXACT_JSON}"
EVAL_DEVICE="${EVAL_DEVICE}" PYTHONUNBUFFERED=1 \
  script -q "${LOG_DIR}/baseline_exact.log" \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/programs/exact_maxsim.py \
    --datasets "${DATASETS_LI}" \
    --warmup-queries 10 --timed-repeats 3

# -----------------------------------------------------------------------------
# Step C — freeform + composable seed equivalence runs.
# `--no-resume` so we always overwrite previous numbers (different dataset
# suite from the historical baseline file).
# -----------------------------------------------------------------------------
echo
echo "[C] Running freeform seed -> ${FREEFORM_JSON}"
EVAL_DEVICE="${EVAL_DEVICE}" PYTHONUNBUFFERED=1 \
  script -q "${LOG_DIR}/baseline_freeform.log" \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/seeds/freeform.py \
    --datasets "${DATASETS_LI}" \
    --warmup-queries 10 --timed-repeats 3 \
    --output "${FREEFORM_JSON}" \
    --no-resume

echo
echo "[C] Running composable seed -> ${COMPOSABLE_JSON}"
EVAL_DEVICE="${EVAL_DEVICE}" PYTHONUNBUFFERED=1 \
  script -q "${LOG_DIR}/baseline_composable.log" \
  uv run python -m tasks.late_interaction.evaluator \
    --program tasks/late_interaction/seeds/composable.py \
    --datasets "${DATASETS_LI}" \
    --warmup-queries 10 --timed-repeats 3 \
    --output "${COMPOSABLE_JSON}" \
    --no-resume

# -----------------------------------------------------------------------------
# Step D — equivalence check. Both seeds must match exact MaxSim on
# recall@1000 / ndcg@10 to within EQUIV_TOL on every dataset, otherwise
# the evolve loop's latency baseline would be measuring a different
# scoring function than the seed. Abort if they drift.
# -----------------------------------------------------------------------------
echo
echo "[D] Equivalence check (tolerance ${EQUIV_TOL})"
.venv/bin/python - "${EXACT_JSON}" "${FREEFORM_JSON}" "${COMPOSABLE_JSON}" \
  "${EQUIV_TOL}" "${DATASETS_LI}" >"${LOG_DIR}/equiv_check.log" 2>&1 <<'PY'
import json, sys
exact_path, freeform_path, composable_path, tol_s, datasets_csv = sys.argv[1:6]
tol = float(tol_s)
datasets = [d.strip() for d in datasets_csv.split(",") if d.strip()]
exact = json.load(open(exact_path))
free = json.load(open(freeform_path))
comp = json.load(open(composable_path))
errors = []
for ds in datasets:
    for metric in ("recall_at_1000", "ndcg_at_10"):
        e = exact[ds][metric]
        f = free[ds][metric]
        c = comp[ds][metric]
        if abs(e - f) > tol:
            errors.append(f"FREEFORM drift on {ds}/{metric}: exact={e:.6f} freeform={f:.6f} delta={f-e:+.6f}")
        if abs(e - c) > tol:
            errors.append(f"COMPOSABLE drift on {ds}/{metric}: exact={e:.6f} composable={c:.6f} delta={c-e:+.6f}")
        print(f"{ds:34s} {metric:15s}  exact={e:.4f}  freeform={f:.4f}  composable={c:.4f}")
if errors:
    print("\nFAIL:", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    sys.exit(1)
print("\nOK: both seeds match exact MaxSim within tolerance.")
PY
EQUIV_RC=$?
cat "${LOG_DIR}/equiv_check.log"
if [[ "${EQUIV_RC}" -ne 0 ]]; then
  echo
  echo "[fatal] seed/exact-MaxSim equivalence check FAILED — aborting before the evolve run."
  echo "        Inspect ${LOG_DIR}/equiv_check.log and the per-seed baseline JSONs."
  exit "${EQUIV_RC}"
fi

# -----------------------------------------------------------------------------
# Step E — Verify the evolve config points at the same dataset suite, then
# launch. This is a defense-in-depth check; the YAML was edited to match
# but a future edit could drift and cause the evolve loop to read a
# baseline that doesn't cover its datasets.
# -----------------------------------------------------------------------------
echo
echo "[E] Verifying evolve config dataset alignment"
CFG_DATASETS="$(grep -E '^[[:space:]]+EVAL_DATASETS:' "${CONFIG_FREEFORM}" | head -1 | sed -E 's/^[[:space:]]+EVAL_DATASETS:[[:space:]]*//; s/^["'\'']//; s/["'\'']$//')"
if [[ "${CFG_DATASETS}" != "${DATASETS_LI}" ]]; then
  echo "[fatal] EVAL_DATASETS in ${CONFIG_FREEFORM} is:"
  echo "          ${CFG_DATASETS}"
  echo "        expected:"
  echo "          ${DATASETS_LI}"
  echo "        Edit the YAML and re-run."
  exit 3
fi
echo "[E] OK: config EVAL_DATASETS matches the suite we just baselined."

echo
echo "[E] Launching evolve loop (max_iterations=${MAX_ITERATIONS})"
EVAL_DEVICE="${EVAL_DEVICE}" PYTHONUNBUFFERED=1 \
  script -q "${LOG_DIR}/evolve.log" \
  uv run ranking-evolved run \
    --config "${CONFIG_FREEFORM}" \
    --replay \
    --max-iterations "${MAX_ITERATIONS}"

echo
echo "==============================================================="
echo "Pipeline complete."
echo "  Logs:    ${LOG_DIR}"
echo "  Run dir: $(ls -td output/late_interaction_freeform_latency_aware/* | head -1)"
echo "==============================================================="
