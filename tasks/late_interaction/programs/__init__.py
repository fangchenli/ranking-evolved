"""Importable late-interaction retrieval programs.

Each program in this directory exposes a `LateInteractionRetriever` class that
satisfies `tasks.late_interaction.interface.LateInteractionRetriever`. The
evaluator loads any of them by file path, so the same evaluator drives:

  - reference baselines (exact_maxsim, fastplaid)
  - the freeform seed (`tasks/late_interaction/seeds/freeform.py`)
  - evolved candidates (any file with the same Protocol shape)

Use:
    EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \\
        --program tasks/late_interaction/programs/fastplaid.py \\
        --datasets beir_scifact,beir_nfcorpus
"""
