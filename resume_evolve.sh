#!/bin/bash
# Resume an OpenEvolve run from the latest checkpoint
#
# Usage:
#   ./resume_evolve.sh <run_dir> [config_file]
#
# Before resuming: set max_iterations in the config to REMAINING steps (not total).
#   e.g. after 70 steps, to reach 200 total set max_iterations: 130 in the YAML.
#
# Example:
#   ./resume_evolve.sh output/openevolve_output_constrained_fast/20260201_184826 openevolve_config_constrained.yaml

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [config_file]"
    echo ""
    echo "Example:"
    echo "  $0 output/openevolve_output_constrained_fast/20260201_184826 openevolve_config_constrained.yaml"
    exit 1
fi

RUN_DIR="$1"
CONFIG_FILE="${2:-openevolve_config_constrained.yaml}"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: No checkpoints directory found: $CHECKPOINT_DIR"
    exit 1
fi

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR" 2>/dev/null | grep "^checkpoint_" | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT_DIR/$LATEST_CHECKPOINT"
echo "Resuming from: $CHECKPOINT_PATH"
echo "Output directory: $RUN_DIR"
echo "Config: $CONFIG_FILE"
echo ""

# Determine seed file based on config
if [[ "$CONFIG_FILE" == *"constrained"* ]]; then
    SEED="src/ranking_evolved/bm25_constrained_fast.py"
elif [[ "$CONFIG_FILE" == *"composable"* ]]; then
    SEED="src/ranking_evolved/bm25_composable_fast.py"
elif [[ "$CONFIG_FILE" == *"freeform"* ]]; then
    SEED="src/ranking_evolved/bm25_freeform_fast.py"
else
    echo "Error: Could not determine seed file from config name. Please specify manually."
    exit 1
fi

echo "Running resume..."
uv run python -m openevolve.cli "$SEED" evaluator_parallel.py --config "$CONFIG_FILE" --output "$RUN_DIR" --checkpoint "$CHECKPOINT_PATH"

# Generate plot after resume
if [ $? -eq 0 ]; then
    echo ""
    echo "Generating evolution metrics plot..."
    uv run python scripts/plot_evolution_metrics.py "$RUN_DIR" --save "$RUN_DIR/evolution_metrics.png" --no-show
fi
