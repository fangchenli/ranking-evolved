"""
Optuna hyperparameter search for the evolved freeform retrieval program.

Tunes the 15 Config parameters of the best evolved program using Bayesian
optimization (TPE sampler), evaluating on 11 small/medium datasets
(excludes trec-covid which alone takes ~28 min to index).

Usage:
    uv run python optuna_search.py --n-trials 100
    uv run python optuna_search.py --n-trials 50 --max-workers 4 --study-name my_study
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

# Set dataset exclusions BEFORE importing evaluator_parallel
# (it reads EVAL_EXCLUDE_DATASETS at import time)
EXCLUDE_DATASETS = (
    "dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora,"
    "webis-touche2020,cqadupstack,trec-covid,"
    "leetcode,aops,theoremqa_questions,"
    "robotics,psychology,sustainable_living"
)
os.environ["EVAL_EXCLUDE_DATASETS"] = EXCLUDE_DATASETS

import optuna  # noqa: E402
from evaluator_parallel import EvalConfig, evaluate_parallel  # noqa: E402

BEST_PROGRAM = "output/openevolve_output_freeform_fast/20260202_075458/best/best_program.py"

# Current (evolution-discovered) parameter values
DEFAULTS = {
    "tf_log_base": 1.0,
    "dl_alpha": 0.15,
    "q_clarity_power": 0.6,
    "coverage_gamma": 0.25,
    "qtf_power": 0.5,
    "facet_mix": 0.12,
    "facet_power": 1.6,
    "coord_beta": 0.08,
    "prefix_weight": 0.18,
    "ngram_weight": 0.10,
    "rare_idf_pivot": 4.5,
    "rare_boost": 0.12,
    "prefix_len": 5,
    "ngram_n": 4,
    "ngram_max_per_token": 2,
}

# Regex patterns for patching Config class attributes
# Matches lines like: "    tf_log_base: float = 1.0"
_FLOAT_PATTERN = r"(    {name}:\s*float\s*=\s*)\S+"
_INT_PATTERN = r"(    {name}:\s*int\s*=\s*)\S+"


def write_trial_program(params: dict, source_text: str) -> str:
    """Write a copy of best_program.py with Config values replaced.

    Returns the path to the temporary file.
    """
    text = source_text
    for name, value in params.items():
        if isinstance(value, int) and name in ("prefix_len", "ngram_n", "ngram_max_per_token"):
            pattern = _INT_PATTERN.format(name=name)
            replacement = rf"\g<1>{value}"
        else:
            pattern = _FLOAT_PATTERN.format(name=name)
            replacement = rf"\g<1>{value}"
        text, count = re.subn(pattern, replacement, text)
        if count == 0:
            raise ValueError(f"Failed to patch parameter '{name}' in Config class")

    fd, path = tempfile.mkstemp(suffix=".py", prefix="optuna_trial_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def make_objective(
    source_text: str,
    max_workers: int,
) -> callable:
    """Create the Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Core scoring
            "tf_log_base": trial.suggest_float("tf_log_base", 0.1, 5.0, log=True),
            "dl_alpha": trial.suggest_float("dl_alpha", 0.0, 1.0),
            "q_clarity_power": trial.suggest_float("q_clarity_power", 0.1, 2.0),
            "coverage_gamma": trial.suggest_float("coverage_gamma", 0.0, 1.0),
            "qtf_power": trial.suggest_float("qtf_power", 0.1, 1.5),
            # Facet prior
            "facet_mix": trial.suggest_float("facet_mix", 0.0, 0.5),
            "facet_power": trial.suggest_float("facet_power", 0.5, 3.0),
            # Coordination
            "coord_beta": trial.suggest_float("coord_beta", 0.0, 0.5),
            # Prefix channel
            "prefix_len": trial.suggest_int("prefix_len", 3, 8),
            "prefix_weight": trial.suggest_float("prefix_weight", 0.0, 0.5),
            # N-gram channel
            "ngram_n": trial.suggest_int("ngram_n", 3, 6),
            "ngram_max_per_token": trial.suggest_int("ngram_max_per_token", 1, 4),
            "ngram_weight": trial.suggest_float("ngram_weight", 0.0, 0.5),
            # Rare-key boost
            "rare_idf_pivot": trial.suggest_float("rare_idf_pivot", 2.0, 7.0),
            "rare_boost": trial.suggest_float("rare_boost", 0.0, 0.5),
        }

        program_path = write_trial_program(params, source_text)
        try:
            t0 = time.time()
            config = EvalConfig(
                max_workers=max_workers,
                tokenizer="lucene",
                include_beir=True,
                include_bright=True,
                include_trec_dl=False,
            )
            results = evaluate_parallel(program_path, config, verbose=False)
            score = results["combined_score"]
            elapsed = time.time() - t0

            # Store per-dataset metrics as user attributes
            trial.set_user_attr("avg_ndcg@10", results.get("avg_ndcg@10", 0.0))
            trial.set_user_attr("avg_recall@100", results.get("avg_recall@100", 0.0))

            print(
                f"  Trial {trial.number}: score={score:.4f} "
                f"(ndcg={results.get('avg_ndcg@10', 0):.4f}, "
                f"recall={results.get('avg_recall@100', 0):.4f}) "
                f"[{elapsed/60:.1f} min]"
            )

            return score
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}", file=sys.stderr)
            return 0.0
        finally:
            os.unlink(program_path)

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for evolved retrieval program"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Max parallel dataset evaluation workers (0=auto, default: 0)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="freeform_hparam_search_v2",
        help="Optuna study name (default: freeform_hparam_search_v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/optuna_search.json",
        help="Output JSON file (default: results/optuna_search.json)",
    )
    args = parser.parse_args()

    # Read source program once
    with open(BEST_PROGRAM) as f:
        source_text = f.read()

    # Create study with SQLite storage for persistence
    storage = f"sqlite:///results/{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    # Enqueue the current (evolved) defaults as the first trial
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULTS)

    objective = make_objective(source_text, args.max_workers)

    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Trials: {args.n_trials} (existing: {len(study.trials)})")
    print(f"Max workers: {args.max_workers}")
    print(f"Datasets: 11 (BRIGHT + BEIR, excluding large + trec-covid)")
    print(f"Objective: combined_score = 0.8 * avg_recall@100 + 0.2 * avg_ndcg@10")
    print()

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"Optimization complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best combined_score: {study.best_value:.4f}")
    print(f"Best avg_ndcg@10: {study.best_trial.user_attrs.get('avg_ndcg@10', 'N/A')}")
    print(f"Best avg_recall@100: {study.best_trial.user_attrs.get('avg_recall@100', 'N/A')}")
    print(f"\nBest parameters:")
    for name, value in sorted(study.best_params.items()):
        default = DEFAULTS.get(name)
        delta = ""
        if default is not None:
            if isinstance(value, int):
                delta = f" (was {default}, delta={value - default:+d})"
            else:
                delta = f" (was {default:.4f}, delta={value - default:+.4f})"
        print(f"  {name}: {value}{delta}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "elapsed_minutes": elapsed / 60,
        "best_trial": study.best_trial.number,
        "best_combined_score": study.best_value,
        "best_avg_ndcg@10": study.best_trial.user_attrs.get("avg_ndcg@10"),
        "best_avg_recall@100": study.best_trial.user_attrs.get("avg_recall@100"),
        "best_params": study.best_params,
        "defaults": DEFAULTS,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
