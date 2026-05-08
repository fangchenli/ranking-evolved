"""Per-program, per-device baseline metric JSONs.

The evaluator CLI writes one JSON per (program, device) combination here:

    fastplaid.{cpu|cuda}.json
    exact_maxsim.{cpu|cuda}.json
    freeform.{cpu|cuda}.json

These files are read by:
  - the latency-aware objective (controller's external-baseline loader) when
    `objective.latency.baseline_source: external`,
  - the recall-floor wrapper in the evaluator,
  - any human looking at "what does each retriever do on this hardware?"

The retriever programs themselves live in `tasks/late_interaction/programs/`
and `tasks/late_interaction/seeds/`.
"""
