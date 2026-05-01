from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import uuid


def _load_module(program_path: str):
    path = pathlib.Path(program_path)
    module_name = f"evolution_algo_candidate_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate program: {program_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str) -> dict[str, float]:
    module = _load_module(program_path)

    score = float(getattr(module, "SCORE", 0.0))
    complexity = float(getattr(module, "COMPLEXITY", 0.0))
    diversity = float(getattr(module, "DIVERSITY", 0.0))

    values = [score, complexity, diversity]
    if any(not math.isfinite(v) for v in values):
        return {
            "combined_score": -1e9,
            "score": -1e9,
            "complexity": 0.0,
            "diversity": 0.0,
            "error": 1.0,
        }

    return {
        "combined_score": score,
        "score": score,
        "complexity": complexity,
        "diversity": diversity,
        "error": 0.0,
    }