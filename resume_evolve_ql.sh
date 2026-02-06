#!/bin/bash
# Resume Query Likelihood evolution from a checkpoint
#
# Usage: ./resume_evolve_ql.sh <output_dir> <config_yaml> <checkpoint_name>
#
# Example:
#   ./resume_evolve_ql.sh \
#     output/openevolve_output_QL_freeform_fast/20260205_120000 \
#     openevolve_config_QL_freeform.yaml \
#     checkpoint_70

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <config_yaml> <checkpoint_name>"
    echo ""
    echo "Example:"
    echo "  $0 output/openevolve_output_QL_freeform_fast/20260205_120000 openevolve_config_QL_freeform.yaml checkpoint_70"
    exit 1
fi

OUTPUT_DIR="$1"
CONFIG="$2"
CHECKPOINT_NAME="$3"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/${CHECKPOINT_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming evolution from: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo ""

uv run python -m openevolve.cli \
  src/ranking_evolved/ql_freeform_fast.py \
  evaluator_ql_parallel.py \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT_PATH"
