#!/bin/bash
# Run OpenEvolve's evolution visualizer from project root.
#
# The visualizer lives in the openevolve repo (scripts/visualizer.py), not in the
# installed package. Clone the repo once and point this script at it.
#
# One-time setup:
#   git clone https://github.com/algorithmicsuperintelligence/openevolve.git ../openevolve
#   uv pip install flask  # or: pip install -r ../openevolve/scripts/requirements.txt
#
# Usage:
#   ./scripts/run_visualizer.sh
#   ./scripts/run_visualizer.sh output/openevolve_output_freeform_fast/20260202_075458
#   ./scripts/run_visualizer.sh output/openevolve_output_freeform_fast/20260202_075458/checkpoints/checkpoint_80
#
# The visualizer lives in algorithmicsuperintelligence/openevolve (not codelion/openevolve).
# Optional: set OPENEVOLVE_REPO to the path to a clone that has scripts/visualizer.py (default: ../openevolve-as).

set -e
REPO="${OPENEVOLVE_REPO:-$(cd "$(dirname "$0")/.." && pwd)/../openevolve-as}"
# Default path: project root, then look for a recent output
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATH_ARG="${1:-}"

if [ ! -d "$REPO" ]; then
    echo "Error: OpenEvolve repo not found at: $REPO"
    echo "The visualizer is in algorithmicsuperintelligence/openevolve (not codelion/openevolve)."
    echo "Clone it with: git clone https://github.com/algorithmicsuperintelligence/openevolve.git $REPO"
    echo "Then install deps: uv pip install flask"
    exit 1
fi

VISUALIZER="$REPO/scripts/visualizer.py"
if [ ! -f "$VISUALIZER" ]; then
    echo "Error: visualizer.py not found at: $VISUALIZER"
    echo "Your clone may be from codelion/openevolve, which does not include the visualizer."
    echo "Use a clone that has scripts/visualizer.py, e.g.:"
    echo "  git clone https://github.com/algorithmicsuperintelligence/openevolve.git $REPO"
    echo "Or set OPENEVOLVE_REPO to the path of that clone."
    exit 1
fi

# Resolve path: if empty, use latest freeform run or similar
if [ -z "$PATH_ARG" ]; then
    # Default: latest run under output/openevolve_output_freeform_fast
    LATEST=$(ls -td "$PROJECT_ROOT"/output/openevolve_output_freeform_fast/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        PATH_ARG="$LATEST"
        echo "Using latest run: $PATH_ARG"
    else
        echo "Usage: $0 [path_to_run_or_checkpoint]"
        echo "Example: $0 output/openevolve_output_freeform_fast/20260202_075458"
        exit 1
    fi
fi

# If relative, make it absolute so the visualizer (run from repo) can find it
if [[ "$PATH_ARG" != /* ]]; then
    PATH_ARG="$PROJECT_ROOT/$PATH_ARG"
fi
if [ ! -d "$PATH_ARG" ]; then
    echo "Error: Path not found: $PATH_ARG"
    exit 1
fi

cd "$REPO/scripts"
echo "Starting visualizer at http://127.0.0.1:8080 (path: $PATH_ARG)"
exec python visualizer.py --path "$PATH_ARG" --host 127.0.0.1 --port 8080
