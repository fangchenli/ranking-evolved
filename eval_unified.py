#!/usr/bin/env python3
"""
Unified evaluator wrapper for BM25 and QL implementations.

Automatically detects the model type (BM25 or QL) from the program file
and dispatches to the appropriate evaluator with WAVE optimizations.

Usage:
    python eval_unified.py <program_path> --save <output.json> [--verbose]

Examples:
    # BM25 program
    python eval_unified.py output/best/best_program.py --save results/bm25_run.json --verbose

    # QL program
    python eval_unified.py output/ql_best/best_program.py --save results/ql_run.json --verbose
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def detect_model_type(program_path: Path) -> str:
    """
    Detect if program is BM25 or QL by inspecting class definitions.

    Returns:
        "bm25" or "ql"
    """
    content = program_path.read_text()

    # Look for class definitions
    # BM25 programs have: class BM25(...)
    # QL programs have: class QL(...)

    if re.search(r'\bclass\s+BM25\b', content):
        return "bm25"
    elif re.search(r'\bclass\s+QL\b', content):
        return "ql"
    else:
        # Fallback: check for other indicators
        if 'def batch_rank' in content and 'idf' in content.lower():
            return "bm25"
        elif 'query likelihood' in content.lower() or 'dirichlet' in content.lower():
            return "ql"
        else:
            # Default to BM25 if unclear
            print(f"Warning: Could not reliably detect model type, assuming BM25", file=sys.stderr)
            return "bm25"


def main():
    parser = argparse.ArgumentParser(description="Unified evaluator for BM25 and QL models")
    parser.add_argument("program_path", type=Path, help="Path to program .py file")
    parser.add_argument("--save", type=Path, help="Path to save results JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sample-queries", type=int, help="Sample N queries per dataset (for faster iteration)")

    args = parser.parse_args()

    if not args.program_path.exists():
        print(f"Error: Program file not found: {args.program_path}", file=sys.stderr)
        sys.exit(1)

    # Detect model type
    model_type = detect_model_type(args.program_path)
    print(f"Detected model type: {model_type.upper()}", file=sys.stderr)

    # Select appropriate evaluator
    if model_type == "bm25":
        evaluator = "evaluator_parallel_wave.py"
    else:  # ql
        evaluator = "evaluator_ql_parallel.py"

    # Build command
    cmd = ["python", evaluator, str(args.program_path)]

    if args.save:
        cmd.extend(["--save", str(args.save)])

    if args.verbose:
        cmd.append("--verbose")

    if args.sample_queries:
        cmd.extend(["--sample-queries", str(args.sample_queries)])

    # Run evaluator
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
