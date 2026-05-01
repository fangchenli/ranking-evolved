# tasks/ql — Query Likelihood (NOT YET MIGRATED)

The QL files in this directory were structurally relocated during the
Phase-2 modular restructuring but their imports, evaluator wiring, and
seed contract have **NOT** been validated against the new
`ranking-evolved` framework.

Do **NOT** run via `ranking-evolved run --config tasks/ql/...` — there
are no configs yet, and the seed/library imports still reference the
pre-Phase-2 layout (`from ranking_evolved.bm25 import ...` etc.).

To migrate QL in a future phase:

1. Rewrite the imports in `tasks/ql/seeds/freeform.py` and
   `tasks/ql/library.py` to point at `tasks.bm25.library` (or, if QL
   needs its own tokenizer, factor the tokenizer into
   `tasks/_shared/`).
2. Add `tasks/ql/evaluator.py` and `tasks/ql/evaluator_worker.py`,
   either as a copy of `tasks/bm25/evaluator.py` adapted for QL, or
   by extracting a shared `tasks/_shared/ir_evaluator.py` if both
   tasks converge.
3. Add `tasks/ql/configs/freeform.yaml` pointing at the new seed and
   evaluator.
4. Validate end-to-end with the same `--replay` smoke that BM25 passes.
5. Remove the `STATUS: STRUCTURALLY RELOCATED` headers from each file.

Until then, every Python file under `tasks/ql/` carries a loud
`STATUS:` header at the top to make it crystal clear that the file is
not part of the maintained surface.
