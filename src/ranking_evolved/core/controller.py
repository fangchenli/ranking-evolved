"""Evolution controller — the main loop.

Orchestrates one iteration:
  sample parent + inspirations  ->  build prompt  ->  propose  ->  parse diff
  ->  apply  ->  evaluate child  ->  admit  ->  trace + replay + commit DB

All state writes (program insert + iteration row + archive update + trace
line) happen inside one `RunStore.transaction()` so `run.db` and
`trace.jsonl` cannot disagree about what happened at iteration N.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config.base import Config
from ..config.objective import ObjectiveConfig
from ..evaluation.experiment_summary import (
    BASELINE_LATENCY_FILE,
    append_program_metrics,
    write_baseline_latency,
    write_experiment_summary,
)
from ..evaluation.external_baseline import (
    detect_runtime_device,
    load_external_baseline,
    resolve_baseline_path,
)
from ..evaluation.objective_math import ObjectiveOutcome, compute_objective
from ..evaluation.plots import generate_run_plots
from ..prompts.diff import DiffError, apply_search_replace
from ..prompts.sampler import PromptSampler
from ..search.map_elites_islands import (
    MapElitesIslandsConfig,
    MapElitesIslandsStrategy,
)
from .replay import ReplayWriter, build_step
from .run_store import RunStore
from .trace import TraceWriter
from .types import (
    DiffApplication,
    EvaluationResult,
    Program,
)


def _now() -> float:
    return time.time()


class Controller:
    """Drives the evolution loop end-to-end against a single run directory."""

    def __init__(
        self,
        *,
        config: Config,
        run_dir: Path,
        proposer: Any,
        runner: Any,
        logger: logging.Logger | None = None,
        config_path: Path | None = None,
        run_id: str | None = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.proposer = proposer
        self.runner = runner
        self.logger = logger or logging.getLogger("ranking_evolved.controller")
        self._config_path = Path(config_path) if config_path else None
        self._run_id = run_id or self.run_dir.name

        # Run-state surfaces.
        self.store = RunStore(self.run_dir / "run.db")
        self.trace = TraceWriter(self.run_dir / "trace.jsonl")
        self.replay = ReplayWriter(self.run_dir) if config.evolution.capture_replay else None
        (self.run_dir / "best").mkdir(parents=True, exist_ok=True)

        # Build strategy with the user's search config.
        if not isinstance(config.search, MapElitesIslandsConfig):
            raise TypeError("Phase-1 supports only MapElitesIslandsConfig.")
        self.strategy = MapElitesIslandsStrategy(config.search)

        # Prompt builder.
        self.prompt_sampler = PromptSampler(config.prompt)

        # Latency-aware objective state. Populated from the seed evaluation
        # (when `objective.latency.enabled`) and used to recompute every
        # candidate's `combined_score` from raw metrics. None when latency
        # is disabled (legacy path: evaluator's combined_score is used as-is).
        self._baseline_latency_by_dataset: dict[str, float] | None = None
        self._dataset_names: list[str] = []
        self._seed_program_id: str | None = None
        self._timestamp_start = _iso_now_str()

    # ------------------------------------------------------------------
    # public entrypoint
    # ------------------------------------------------------------------

    async def run(self, *, seed_path: Path, resume: bool = False) -> Program:
        if resume:
            start_iteration = self._resume_existing_run()
        else:
            seed_program = await self._build_and_eval_seed(seed_path)
            self._seed_program_id = seed_program.id
            self.strategy.initialize(seed_program)
            # The strategy stamps authoritative complexity/diversity/feature_coords
            # during admission; refresh our reference so the persisted seed
            # matches what's in self.strategy.programs.
            seed_program = self.strategy.programs.get(seed_program.id, seed_program)
            self._persist_program(
                seed_program,
                prompt_system=None,
                prompt_user=None,
                llm_raw=None,
                iteration=0,
                parent_metrics=None,
                eval_duration=0.0,
            )
            self._persist_seed_archive_cells()
            self._export_best(seed_program)
            # Bootstrap diagnostic: confirm the diversity anchor was pinned,
            # the seed populates every island, and complexity was stamped
            # by the strategy (not left at the controller's placeholder).
            anchor_present = (
                getattr(self.strategy, "_diversity_seed_anchor", None) is not None
            )
            self.logger.info(
                "[evo-diag] bootstrap seed=%s complexity=%.0f diversity=%.3f "
                "feature_coords=%s diversity_anchor_pinned=%s "
                "n_programs=%d island_sizes=%s archive_size=%d",
                _short_id(seed_program.id),
                float(seed_program.complexity),
                float(seed_program.diversity),
                dict(seed_program.feature_coords),
                anchor_present,
                len(self.strategy.programs),
                [len(i) for i in self.strategy.islands],
                len(self.strategy.archive),
            )
            # Append the seed as the first row of program_metrics.jsonl so the
            # comparison baseline is in the same file as the children.
            self._append_program_row(
                program=seed_program,
                parent=None,
                iteration=0,
                generation=0,
            )
            start_iteration = 1

        for it in range(start_iteration, self.config.evolution.max_iterations + 1):
            try:
                await self._step(it)
            except _SkipIteration as skip:
                self.logger.warning("iteration %d skipped: %s", it, skip)

        best = self.strategy.best()
        self._export_best(best)
        self._write_experiment_summary(best)
        return best

    def close(self) -> None:
        try:
            self.trace.close()
        finally:
            self.store.close(vacuum=self.config.run_store.vacuum_on_close)

    # ------------------------------------------------------------------
    # one iteration
    # ------------------------------------------------------------------

    async def _step(self, iteration: int) -> None:
        snap_before = self.strategy.snapshot()
        parent, inspirations = self.strategy.sample(iteration)
        sampling = self.strategy.sampling_decisions(iteration, parent, inspirations)
        parent_island = sampling.parent_island
        self.logger.info(
            "[ranking-evolved] iter=%04d island=%d parent=%s sampled %d inspirations",
            iteration,
            parent_island,
            _short_id(parent.id),
            len(inspirations),
        )
        top_programs = self.strategy.top_programs(n=5, island_idx=parent_island)
        previous_programs = self.strategy.recent_programs(
            n=3, island_idx=parent_island, exclude_program_id=parent.id
        )
        # Surface recent score=0/crashed candidates so the LLM can avoid
        # repeating them. The strategy keeps a bounded ring buffer of
        # rejected children — see `MapElitesIslandsStrategy._record_failure`.
        recent_failures: list[Any] = []
        if hasattr(self.strategy, "recent_failures"):
            try:
                n_failures = max(0, int(self.config.prompt.num_failed_attempts))
            except (AttributeError, TypeError, ValueError):
                n_failures = 0
            if n_failures > 0:
                recent_failures = self.strategy.recent_failures(
                    n=n_failures,
                    island_idx=parent_island,
                )

        # Diagnostic line for post-run sanity checks: which programs were
        # picked for context, with their scores + feature coords. Lets us
        # verify in 20-step toy runs that island isolation, dedup, and the
        # diverse-pick pipeline are doing what we think.
        self.logger.info(
            "[evo-diag] iter=%04d sample island=%d parent=%s(score=%s, c=%.0f, d=%.3f) "
            "top=%s recent=%s inspirations=%s failures=%d",
            iteration,
            parent_island,
            _short_id(parent.id),
            _fmt_metric(parent.metrics.get("combined_score")),
            float(parent.complexity),
            float(parent.diversity),
            [
                f"{_short_id(p.id)}({_fmt_metric(p.metrics.get('combined_score'))})"
                for p in top_programs
            ],
            [
                f"{_short_id(p.id)}({_fmt_metric(p.metrics.get('combined_score'))})"
                for p in previous_programs
            ],
            [
                f"{_short_id(p.id)}({_fmt_metric(p.metrics.get('combined_score'))})"
                for p in inspirations
            ],
            len(recent_failures),
        )

        # Retry loop: up to `proposer.candidate_retries` fresh LLM proposals
        # per iteration. Each retry's prompt includes a footer summarizing
        # the failure modes of the prior attempts. Retries do NOT bump the
        # iteration counter — the iteration is "owed" one usable candidate.
        max_attempts = max(1, int(self.config.proposer.candidate_retries))
        retry_feedback: list[dict[str, Any]] = []
        attempt_outcome: dict[str, Any] | None = None
        for attempt in range(1, max_attempts + 1):
            prompt = self.prompt_sampler.build(
                iteration=iteration,
                parent=parent,
                inspirations=inspirations,
                top_programs=top_programs,
                previous_programs=previous_programs,
                parent_artifacts=parent.artifacts or None,
                retry_feedback=retry_feedback or None,
                recent_failures=recent_failures or None,
            )
            self.logger.info(
                "[ranking-evolved] iter=%04d attempt=%d/%d requesting proposal from LLM",
                iteration,
                attempt,
                max_attempts,
            )
            candidate = await self.proposer.propose(prompt)
            self.logger.info(
                "[ranking-evolved] iter=%04d attempt=%d proposal received; applying diff",
                iteration,
                attempt,
            )

            try:
                child_code, diff_app = apply_search_replace(
                    parent.source_code, candidate.raw_response
                )
            except DiffError as exc:
                self.logger.warning(
                    "iteration %d attempt %d/%d: diff did not apply (%s); retrying.",
                    iteration,
                    attempt,
                    max_attempts,
                    exc.application.fatal_error,
                )
                retry_feedback.append(
                    {"kind": "diff_failed", "detail": exc.application.fatal_error or "unknown"}
                )
                # Save the last failed attempt so we can record-skipped if
                # all retries are exhausted.
                attempt_outcome = {
                    "kind": "diff_failed",
                    "prompt": prompt,
                    "candidate": candidate,
                    "diff": exc.application,
                }
                continue

            child_id = str(uuid.uuid4())
            self.logger.info(
                "[ranking-evolved] iter=%04d attempt=%d evaluating candidate %s",
                iteration,
                attempt,
                _short_id(child_id),
            )
            eval_result, child_program = await self._evaluate_child(
                child_id=child_id,
                child_code=child_code,
                parent=parent,
                iteration=iteration,
            )

            if eval_result.error:
                # Evaluator crashed — feed the tail of the traceback back so
                # the next attempt can avoid the same pitfall (NumPy-version
                # mismatches, missing attributes, etc.).
                err_tail = (eval_result.error or "").splitlines()
                err_summary = err_tail[-1].strip() if err_tail else "unknown crash"
                self.logger.warning(
                    "iteration %d attempt %d/%d: evaluator crashed (%s); retrying.",
                    iteration,
                    attempt,
                    max_attempts,
                    err_summary[:200],
                )
                retry_feedback.append({"kind": "eval_crashed", "detail": err_summary})
                attempt_outcome = {
                    "kind": "eval_crashed",
                    "prompt": prompt,
                    "candidate": candidate,
                    "diff": diff_app,
                    "eval_result": eval_result,
                    "child_program": child_program,
                    "child_code": child_code,
                }
                continue

            # Success: leave the retry loop with the working candidate.
            attempt_outcome = {
                "kind": "ok",
                "prompt": prompt,
                "candidate": candidate,
                "diff": diff_app,
                "eval_result": eval_result,
                "child_program": child_program,
                "child_code": child_code,
            }
            if attempt > 1:
                self.logger.info(
                    "[ranking-evolved] iter=%04d recovered on attempt %d/%d",
                    iteration,
                    attempt,
                    max_attempts,
                )
            break

        assert attempt_outcome is not None  # loop ran at least once
        # If every attempt failed, log a clear "gave up" line and skip-record.
        if attempt_outcome["kind"] != "ok":
            self.logger.warning(
                "iteration %d gave up after %d attempts (last failure: %s); skipping admit.",
                iteration,
                max_attempts,
                attempt_outcome["kind"],
            )
            self._record_skipped_iteration(
                iteration=iteration,
                parent=parent,
                prompt=attempt_outcome["prompt"],
                candidate=attempt_outcome["candidate"],
                diff=attempt_outcome["diff"],
                snap_before=snap_before,
            )
            return

        prompt = attempt_outcome["prompt"]
        candidate = attempt_outcome["candidate"]
        diff_app = attempt_outcome["diff"]
        eval_result = attempt_outcome["eval_result"]
        child_program = attempt_outcome["child_program"]
        child_code = attempt_outcome["child_code"]

        # Admit + record DB-state diff. The strategy stamps authoritative
        # complexity/diversity/feature_coords on its internal copy during
        # admission; refresh our local reference so persistence matches.
        best_before = self.strategy.best_program_id
        admission = self.strategy.admit(child_program, iteration=iteration)
        child_program = self.strategy.programs.get(child_program.id, child_program)
        is_new_best = (
            self.strategy.best_program_id == child_program.id and best_before != child_program.id
        )
        snap_after = self.strategy.snapshot()

        # Persist program + iteration row (one transaction).
        improvement = _improvement(parent.metrics, child_program.metrics)
        self._persist_program(
            child_program,
            prompt_system=prompt.system if self.config.trace.include_prompts else None,
            prompt_user=prompt.user if self.config.trace.include_prompts else None,
            llm_raw=candidate.raw_response,
            iteration=iteration,
            parent_metrics=parent.metrics,
            eval_duration=eval_result.duration_s,
            improvement=improvement,
            diff_summary={
                "n_extracted": diff_app.n_extracted,
                "n_applied": diff_app.n_applied,
            },
            island=parent_island,
            admission=admission,
        )

        # Admission-outcome diagnostic. Surfaces (a) whether the child was
        # accepted into a MAP-Elites cell or rejected as a failure, (b) the
        # cell coordinates it took (or would have taken), (c) which program
        # it evicted, and (d) the post-step archive composition by island.
        # Together with the [evo-diag] sample line above this is what we
        # grep in the 20-step toy run to confirm the new behaviors fire.
        admitted = bool(admission.cell_key)
        archive_islands_after = [0, 0, 0]
        for pid in self.strategy.archive:
            p = self.strategy.programs.get(pid)
            if p is not None and 0 <= p.island < len(archive_islands_after):
                archive_islands_after[p.island] += 1
        self.logger.info(
            "[evo-diag] iter=%04d admit island=%d admitted=%s cell=%s "
            "child=%s(score=%s, c=%.0f, d=%.3f) evicted=%s "
            "n_programs=%d island_sizes=%s archive_size=%d archive_by_island=%s "
            "failure_buffer=%d prompt_chars=%d",
            iteration,
            parent_island,
            "yes" if admitted else "REJECT",
            admission.cell_key or "(none)",
            _short_id(child_program.id),
            _fmt_metric(child_program.metrics.get("combined_score")),
            float(child_program.complexity),
            float(child_program.diversity),
            _short_id(admission.evicted_program_id) if admission.evicted_program_id else "(none)",
            len(self.strategy.programs),
            [len(i) for i in self.strategy.islands],
            len(self.strategy.archive),
            archive_islands_after,
            len(getattr(self.strategy, "_recent_failures", []) or []),
            len(prompt.user) if prompt and prompt.user else 0,
        )

        self._log_iteration_progress(
            iteration=iteration,
            island=parent_island,
            child=child_program,
            improvement=improvement,
            is_new_best=is_new_best,
        )
        self._log_iteration_banner(
            iteration=iteration,
            score=_fmt_metric(child_program.metrics.get("combined_score")),
            is_new_best=is_new_best,
        )

        # Best snapshot (always overwrite on improvement; cheap).
        if self.strategy.best_program_id == child_program.id:
            self._export_best(child_program)

        # Replay capture. Always snapshot step 1, the final step, and any
        # step that produced a new global best — regardless of the
        # `capture_replay_every` sampling rate. This guarantees that even on
        # sparsely-sampled runs the interesting moments (initial state,
        # successful improvements, end state) are always inspectable.
        if self.replay is not None and self._should_capture_replay(iteration, is_new_best):
            step = build_step(
                iteration=iteration,
                sampling=sampling,
                parent=parent,
                inspirations=inspirations,
                top_programs=top_programs,
                previous_programs=previous_programs,
                parent_artifacts=parent.artifacts or None,
                prompt=prompt,
                llm_proposer=candidate.proposer,
                llm_model=candidate.model,
                llm_raw=candidate.raw_response,
                llm_tokens_in=candidate.tokens_in,
                llm_tokens_out=candidate.tokens_out,
                llm_latency_ms=candidate.latency_ms,
                diff=diff_app,
                child_code=child_code,
                child_eval=eval_result,
                db_before=snap_before,
                db_after=snap_after,
                admission=admission,
            )
            self.replay.write(step)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _should_capture_replay(self, iteration: int, is_new_best: bool) -> bool:
        """Decide whether to write `replay/step_NNNN.json` for this iteration.

        Always captures: step 1, the final step, and any new-global-best step.
        Otherwise samples every `evolution.capture_replay_every` iterations.
        Default `capture_replay_every=1` means every step (legacy behavior).
        """
        every = max(1, int(self.config.evolution.capture_replay_every))
        if iteration == 1:
            return True
        if iteration >= self.config.evolution.max_iterations:
            return True
        if is_new_best:
            return True
        return iteration % every == 0

    def _resume_existing_run(self) -> int:
        state = self.store.get_meta("strategy_state")
        if state is None:
            raise RuntimeError(
                f"cannot resume {self.run_dir}: run.db has no strategy_state. "
                "Only runs created with resume-capable ranking-evolved can be resumed."
            )
        self.strategy.load_state_dict(state)
        best = self.strategy.best()
        self._seed_program_id = self.store.get_meta("seed_program_id", "seed")
        self._timestamp_start = self.store.get_meta("timestamp_start", self._timestamp_start)
        self._load_baseline_latency_from_disk()
        self._export_best(best)

        last_iter = self.store.get_meta("last_iter", self.store.last_iteration())
        last_iter = int(last_iter or 0)
        self.logger.info("resuming run %s from iteration %d", self.run_dir, last_iter + 1)
        return last_iter + 1

    def _load_baseline_latency_from_disk(self) -> None:
        path = self.run_dir / BASELINE_LATENCY_FILE
        if not path.exists():
            self._baseline_latency_by_dataset = None
            self._dataset_names = []
            return
        data = json.loads(path.read_text())
        baseline = {
            str(name): float(value)
            for name, value in (data.get("baseline_latency_by_dataset") or {}).items()
        }
        self._baseline_latency_by_dataset = baseline
        self._dataset_names = sorted(baseline)

    def _load_external_baseline(self) -> dict[str, float]:
        """Read the external baseline JSON; assert device fingerprint matches.

        Used when `objective.latency.baseline_source == "external"` instead of
        capturing the seed's own latencies. The path may include
        `${EVAL_DEVICE}` which is interpolated to the active device.
        """
        device = detect_runtime_device()
        path = resolve_baseline_path(
            self.config.objective.latency.baseline_path,
            device=device,
        )
        return load_external_baseline(path, runtime_device=device)

    async def _build_and_eval_seed(self, seed_path: Path) -> Program:
        seed_code = Path(seed_path).read_text()
        self.logger.info(
            "[ranking-evolved] evaluating seed program %s",
            seed_path,
        )
        eval_result = await self.runner.evaluate(seed_path)
        self.logger.info("[ranking-evolved] seed evaluation complete")
        seed_id = "seed"

        metrics = dict(eval_result.metrics) if eval_result.metrics else {"combined_score": 0.0}

        # When latency-aware: choose a baseline source, persist it, and
        # recompute the seed's `combined_score` under the configured objective
        # so it sits on the same scale as the children.
        if self.config.objective.latency.enabled:
            per_ds = _explode_per_dataset(metrics)
            self._dataset_names = sorted(per_ds.keys())

            source = self.config.objective.latency.baseline_source
            if source == "seed":
                baseline = {
                    name: float(metrics[f"{name}_query_latency_median_ms"])
                    for name in self._dataset_names
                    if f"{name}_query_latency_median_ms" in metrics
                }
            elif source == "external":
                baseline = self._load_external_baseline()
            else:
                raise ValueError(
                    f"Unsupported objective.latency.baseline_source={source!r}; "
                    "expected 'seed' or 'external'."
                )

            self._baseline_latency_by_dataset = baseline
            write_baseline_latency(
                self.run_dir,
                objective=self.config.objective,
                baseline_latency_by_dataset=baseline,
            )
            outcome = compute_objective(per_ds, baseline, self.config.objective)
            _merge_outcome_into_metrics(metrics, outcome, self.config.objective)

        # complexity / diversity / feature_coords are stamped authoritatively
        # by MapElitesIslandsStrategy._admit_into_island; the values here are
        # placeholders that get overwritten before the program is persisted.
        return Program(
            id=seed_id,
            source_code=seed_code,
            parent_id=None,
            generation=0,
            iteration_found=0,
            timestamp=_now(),
            metrics=metrics,
            complexity=float(len(seed_code)),
            diversity=0.0,
            island=0,
            feature_coords={},
            changes_description="seed",
            artifacts=dict(eval_result.artifacts) if eval_result.artifacts else {},
            metadata={},
        )

    async def _evaluate_child(
        self,
        *,
        child_id: str,
        child_code: str,
        parent: Program,
        iteration: int,
    ) -> tuple[EvaluationResult, Program]:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tf:
            tf.write(child_code)
            tmp_path = tf.name
        try:
            result = await self.runner.evaluate(tmp_path)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

        metrics = dict(result.metrics) if result.metrics else {"combined_score": 0.0}

        # Persist the evaluator's error string into the program's metrics so
        # post-hoc analysis can distinguish "candidate code crashed" from
        # "candidate ran but tripped the recall floor" without cracking open
        # the per-step replay JSONs. Truncated to keep DB rows reasonable.
        if result.error:
            metrics["eval_error"] = str(result.error)[:2000]
            metrics["eval_crashed"] = 1.0

        # Latency-aware objective: replace the evaluator's placeholder
        # combined_score with `weights.recall * avg_recall@k + weights.ndcg
        # * avg_ndcg@k + weights.latency * avg_latency_score`, where
        # latency_score uses the seed-program baseline captured earlier.
        if (
            self.config.objective.latency.enabled
            and self._baseline_latency_by_dataset is not None
            and result.metrics
        ):
            per_ds = _explode_per_dataset(metrics)
            outcome = compute_objective(
                per_ds,
                self._baseline_latency_by_dataset,
                self.config.objective,
            )
            _merge_outcome_into_metrics(metrics, outcome, self.config.objective)

        program = Program(
            id=child_id,
            source_code=child_code,
            parent_id=parent.id,
            generation=parent.generation + 1,
            iteration_found=iteration,
            timestamp=_now(),
            metrics=metrics,
            complexity=float(len(child_code)),
            diversity=0.0,
            island=parent.island,
            feature_coords={},
            changes_description="evolved",
            artifacts=dict(result.artifacts) if result.artifacts else {},
            metadata={},
        )
        self._append_program_row(
            program=program,
            parent=parent,
            iteration=iteration,
            generation=program.generation,
        )
        return result, program

    def _persist_program(
        self,
        program: Program,
        *,
        prompt_system: str | None,
        prompt_user: str | None,
        llm_raw: str | None,
        iteration: int,
        parent_metrics: dict[str, float] | None,
        eval_duration: float,
        improvement: float | None = None,
        diff_summary: dict[str, Any] | None = None,
        island: int | None = None,
        admission: Any = None,
    ) -> None:
        with self.store.transaction() as conn:
            self.store.add_program(
                program,
                prompt_system=prompt_system,
                prompt_user=prompt_user,
                llm_raw_response=llm_raw,
                conn=conn,
            )
            # Mirror the full MAP-Elites cell map so offline tooling that
            # joins `programs` to `archive_cells` sees the current grid. A
            # successful admission may rebucket existing programs, so updating
            # only the child's cell would leave stale rows behind.
            if admission is not None and admission.cell_key:
                self.store.replace_archive_cells(
                    self._strategy_archive_cell_rows(),
                    conn=conn,
                )
            if iteration > 0:
                self.store.add_iteration(
                    iteration=iteration,
                    parent_id=program.parent_id,
                    child_id=program.id,
                    prompt_hash=_hash(prompt_user) if prompt_user else None,
                    llm_latency_ms=None,
                    diff_n_extracted=(diff_summary or {}).get("n_extracted"),
                    diff_n_applied=(diff_summary or {}).get("n_applied"),
                    eval_duration_s=eval_duration,
                    child_score=program.metrics.get("combined_score"),
                    improvement_delta=improvement,
                    conn=conn,
                )
            self.store.set_meta("last_iter", iteration, conn=conn)
            self.store.set_meta("seed_program_id", self._seed_program_id or "seed", conn=conn)
            self.store.set_meta("timestamp_start", self._timestamp_start, conn=conn)
            self.store.set_meta("strategy_state", self.strategy.state_dict(), conn=conn)
        # Trace projection (outside the transaction is OK; trace is append-only):
        self.trace.append(
            iteration=iteration,
            parent_id=program.parent_id,
            child_id=program.id,
            parent_metrics=parent_metrics,
            child_metrics=program.metrics,
            improvement_delta=improvement,
            prompt={"system": prompt_system or "", "user": prompt_user or ""}
            if prompt_user
            else None,
            llm_response=llm_raw,
            diff_summary=diff_summary,
            island=island,
            eval_duration_s=eval_duration,
        )

    def _persist_seed_archive_cells(self) -> None:
        """Mirror the strategy's per-island feature maps for the seed into SQLite.

        `initialize()` admits a seed copy into every island, each with its own
        feature-coord cell. We replay those into `archive_cells` so offline
        tooling sees the initial grid state.
        """
        with self.store.transaction() as conn:
            self.store.replace_archive_cells(self._strategy_archive_cell_rows(), conn=conn)

    def _strategy_archive_cell_rows(self) -> list[tuple[int, str, str]]:
        return [
            (island_idx, str(cell_key), str(program_id))
            for island_idx, cell_map in enumerate(self.strategy.island_feature_maps)
            for cell_key, program_id in cell_map.items()
        ]

    def _log_iteration_banner(
        self,
        *,
        iteration: int,
        score: str,
        is_new_best: bool,
    ) -> None:
        """Print a visually distinctive line at end-of-step / new-best.

        Uses ANSI color when stdout is a TTY (and `NO_COLOR` is not set).
        Always logs through the framework logger so the run.log captures
        the banner too.
        """
        max_iter = self.config.evolution.max_iterations
        if is_new_best:
            body = f"NEW BEST  iter {iteration:04d}/{max_iter:04d}  combined_score={score}"
            line = _color(f">>> {body} <<<", "bright_green_bold")
            sep = _color("=" * 64, "bright_green_bold")
            self.logger.info(sep)
            self.logger.info(line)
            self.logger.info(sep)
        else:
            body = f"step {iteration:04d}/{max_iter:04d} done  combined_score={score}"
            self.logger.info(_color(f"--- {body} ---", "cyan"))

    def _log_iteration_progress(
        self,
        *,
        iteration: int,
        island: int,
        child: Program,
        improvement: float | None,
        is_new_best: bool,
    ) -> None:
        metrics = child.metrics or {}
        recall_key = f"avg_recall@{self.config.objective.recall_k}"
        ndcg_key = f"avg_ndcg@{self.config.objective.ndcg_k}"
        score = _fmt_metric(metrics.get("combined_score"))
        recall = _fmt_metric(metrics.get(recall_key))
        ndcg = _fmt_metric(metrics.get(ndcg_key))
        latency_score = _fmt_metric(metrics.get("avg_latency_score"))
        latency_ms = _fmt_metric(metrics.get("avg_query_latency_median_ms"), digits=2)
        recall_component = _fmt_metric(metrics.get("objective_recall_component"))
        ndcg_component = _fmt_metric(metrics.get("objective_ndcg_component"))
        latency_component = _fmt_metric(metrics.get("objective_latency_component"))
        delta = "n/a" if improvement is None else f"{improvement:+.6f}"
        beat_parent = "n/a" if improvement is None else ("yes" if improvement > 0 else "no")
        markers: list[str] = []
        # EVAL_CRASHED takes priority over RECALL_FLOOR — a crash means we
        # never even computed per-dataset recalls, so the floor flag is moot.
        # Include the truncated error tail so the operator can grep run.log
        # without opening the replay JSON.
        if metrics.get("eval_crashed"):
            err = str(metrics.get("eval_error") or "").splitlines()
            err_summary = err[-1].strip() if err else "unknown"
            markers.append(f"EVAL_CRASHED ({err_summary[:120]})")
        elif metrics.get("recall_floor_triggered"):
            markers.append("RECALL_FLOOR_TRIGGERED")
        if metrics.get("latency_penalty_triggered"):
            markers.append("LATENCY_PENALTY_TRIGGERED")
        if is_new_best:
            markers.append("NEW BEST FOUND!")
        marker = (" " + " | ".join(markers)) if markers else ""
        self.logger.info(
            "[ranking-evolved] iter=%04d island=%d gen=%d score=%s "
            "parent_delta=%s beat_parent=%s recall@%d=%s ndcg@%d=%s "
            "latency_score=%s query_latency=%sms components=(recall=%s,ndcg=%s,latency=%s)%s",
            iteration,
            island,
            child.generation,
            score,
            delta,
            beat_parent,
            self.config.objective.recall_k,
            recall,
            self.config.objective.ndcg_k,
            ndcg,
            latency_score,
            latency_ms,
            recall_component,
            ndcg_component,
            latency_component,
            marker,
        )

    def _record_skipped_iteration(
        self,
        *,
        iteration: int,
        parent: Program,
        prompt,
        candidate,
        diff: DiffApplication,
        snap_before,
    ) -> None:
        # Persist a row marking the skip so the trace doesn't have a gap.
        with self.store.transaction() as conn:
            self.store.add_iteration(
                iteration=iteration,
                parent_id=parent.id,
                child_id=None,
                prompt_hash=_hash(prompt.user),
                llm_latency_ms=candidate.latency_ms,
                diff_n_extracted=diff.n_extracted,
                diff_n_applied=diff.n_applied,
                eval_duration_s=None,
                child_score=None,
                improvement_delta=None,
                conn=conn,
            )
            self.store.set_meta("last_iter", iteration, conn=conn)
            self.store.set_meta("seed_program_id", self._seed_program_id or "seed", conn=conn)
            self.store.set_meta("timestamp_start", self._timestamp_start, conn=conn)
            self.store.set_meta("strategy_state", self.strategy.state_dict(), conn=conn)
        self.trace.append(
            iteration=iteration,
            parent_id=parent.id,
            child_id=None,
            parent_metrics=parent.metrics,
            child_metrics=None,
            improvement_delta=None,
            prompt={"system": prompt.system, "user": prompt.user}
            if self.config.trace.include_prompts
            else None,
            llm_response=candidate.raw_response,
            diff_summary={
                "n_extracted": diff.n_extracted,
                "n_applied": diff.n_applied,
                "fatal_error": diff.fatal_error,
            },
            island=parent.island,
            eval_duration_s=None,
            extra={"skipped": True},
        )

    def _export_best(self, program: Program) -> None:
        export_best_program(self.run_dir, program)

    # ------------------------------------------------------------------
    # Per-program JSONL + run-level summary (objective experiment outputs)
    # ------------------------------------------------------------------

    def _append_program_row(
        self,
        *,
        program: Program,
        parent: Program | None,
        iteration: int,
        generation: int,
    ) -> None:
        """One row per evaluated program in `program_metrics.jsonl`.

        Schema includes: identifying fields, the resolved objective
        components, top-line aggregates, the per-dataset block. Cross-run
        comparison reads this file directly without unpacking SQLite.
        """
        objective_name = self.config.objective.name
        recall_k = self.config.objective.recall_k
        ndcg_k = self.config.objective.ndcg_k
        m = program.metrics or {}
        per_dataset = _explode_per_dataset(m) if m else {}
        row: dict[str, Any] = {
            "program_id": program.id,
            "parent_id": program.parent_id,
            "island": program.island,
            "generation": generation,
            "iteration_found": iteration,
            "objective_name": objective_name,
            "combined_score": float(m.get("combined_score", 0.0)),
            f"avg_recall@{recall_k}": float(m.get(f"avg_recall@{recall_k}", 0.0)),
            f"avg_ndcg@{ndcg_k}": float(m.get(f"avg_ndcg@{ndcg_k}", 0.0)),
            "avg_latency_score": float(m.get("avg_latency_score", 0.0)),
            "avg_query_latency_median_ms": float(m.get("avg_query_latency_median_ms", 0.0)),
            "latency_penalty_triggered": float(m.get("latency_penalty_triggered", 0.0)),
            "objective_recall_component": float(m.get("objective_recall_component", 0.0)),
            "objective_ndcg_component": float(m.get("objective_ndcg_component", 0.0)),
            "objective_latency_component": float(m.get("objective_latency_component", 0.0)),
            "per_dataset": per_dataset,
        }
        try:
            append_program_metrics(self.run_dir, row)
        except OSError as exc:
            self.logger.warning("could not append program_metrics.jsonl: %s", exc)

    def _write_experiment_summary(self, best: Program) -> None:
        try:
            write_experiment_summary(
                self.run_dir,
                run_id=self._run_id,
                config_path=self._config_path or Path(""),
                objective=self.config.objective,
                seed_program_id=self._seed_program_id or "seed",
                best_program_id=best.id,
                best_combined_score=float(best.metrics.get("combined_score", 0.0)),
                best_metrics=best.metrics,
                baseline_latency_by_dataset=self._baseline_latency_by_dataset,
                datasets=self._dataset_names,
                timestamp_start=self._timestamp_start,
                timestamp_end=_iso_now_str(),
            )
        except OSError as exc:
            self.logger.warning("could not write experiment_summary.json: %s", exc)

        try:
            written = generate_run_plots(self.run_dir)
            if written:
                self.logger.info(
                    "[ranking-evolved] plots: %s",
                    ", ".join(str(path) for path in written),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            self.logger.warning("could not generate run plots: %s", exc)


class _SkipIteration(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Helpers shared by the seed-eval and child-eval paths.
# ----------------------------------------------------------------------


def export_best_program(run_dir: Path, program: Program) -> None:
    """Write the current best program plus visible provenance into `<run>/best`."""
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "program.py").write_text(program.source_code)
    (best_dir / "metrics.json").write_text(json.dumps(program.metrics, indent=2))

    for marker in best_dir.glob("created_at_step_*.txt"):
        marker.unlink()

    step = int(program.iteration_found)
    replay_step = f"step_{step:04d}.json" if step > 0 else None
    replay_path = f"../replay/{replay_step}" if replay_step else None
    score = program.metrics.get("combined_score")
    metadata = {
        "program_id": program.id,
        "parent_id": program.parent_id,
        "iteration_found": step,
        "generation": int(program.generation),
        "island": int(program.island),
        "combined_score": float(score) if isinstance(score, int | float) else None,
        "replay_step": replay_step,
        "replay_path": replay_path,
        "exported_at": _iso_now_str(),
    }
    (best_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    marker_name = f"created_at_step_{step:04d}.txt"
    marker_lines = [
        f"created_at_step: {step}",
        f"program_id: {program.id}",
        f"parent_id: {program.parent_id or 'none'}",
        f"generation: {program.generation}",
        f"island: {program.island}",
        f"combined_score: {metadata['combined_score']}",
        f"replay: {replay_path or 'seed evaluation; no replay step'}",
        "",
    ]
    (best_dir / marker_name).write_text("\n".join(marker_lines))

    readme_lines = [
        "# Best Program",
        "",
        f"- Created at step: {step}",
        f"- Program id: `{program.id}`",
        f"- Parent id: `{program.parent_id or 'none'}`",
        f"- Generation: {program.generation}",
        f"- Island: {program.island}",
        f"- Combined score: {metadata['combined_score']}",
        f"- Replay: `{replay_path or 'seed evaluation; no replay step'}`",
        "",
        "Files:",
        "- `program.py` is the current best source.",
        "- `metrics.json` is the current best metric payload.",
        "- `metadata.json` is the machine-readable provenance summary.",
        f"- `{marker_name}` makes the creating step visible in directory listings.",
        "",
    ]
    (best_dir / "README.md").write_text("\n".join(readme_lines))


# Suffixes the evaluator emits per dataset; presence lets us reverse-engineer
# the dataset list from the flat metrics dict that evaluators return today.
_PER_DATASET_SUFFIXES = (
    "_query_latency_median_ms",
    "_index_time_ms",
    "_query_time_ms",
)


def _explode_per_dataset(flat_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Group a flat `<dataset>_<metric>` dict into a nested `dataset -> metric -> value`.

    Robust to dataset names that themselves contain underscores (e.g.
    `bright_biology`) because we identify dataset names by the unique
    suffixes the evaluator emits per dataset.
    """
    datasets: set[str] = set()
    for key in flat_metrics:
        for suf in _PER_DATASET_SUFFIXES:
            if (
                key.endswith(suf)
                and not key.endswith("_baseline_query_latency_median_ms")
                and not key.startswith("avg_")
                and not key.startswith("total_")
            ):
                datasets.add(key[: -len(suf)])
                break

    out: dict[str, dict[str, float]] = {}
    for ds in datasets:
        prefix = ds + "_"
        bucket: dict[str, float] = {}
        for key, val in flat_metrics.items():
            if not key.startswith(prefix):
                continue
            metric_name = key[len(prefix) :]
            if isinstance(val, int | float):
                bucket[metric_name] = float(val)
        out[ds] = bucket
    return out


def _merge_outcome_into_metrics(
    metrics: dict[str, Any],
    outcome: ObjectiveOutcome,
    objective: ObjectiveConfig,
) -> None:
    """Replace `combined_score` and add objective components / per-dataset latency keys."""
    metrics["combined_score"] = float(outcome.combined_score)
    metrics["objective_name"] = objective.name
    metrics["objective_recall_component"] = float(outcome.objective_recall_component)
    metrics["objective_ndcg_component"] = float(outcome.objective_ndcg_component)
    metrics["objective_latency_component"] = float(outcome.objective_latency_component)
    metrics[f"avg_recall@{objective.recall_k}"] = float(outcome.avg_recall)
    metrics[f"avg_ndcg@{objective.ndcg_k}"] = float(outcome.avg_ndcg)
    metrics["avg_latency_score"] = float(outcome.avg_latency_score)
    metrics["avg_latency_ratio"] = float(outcome.avg_latency_ratio)
    metrics["avg_query_latency_median_ms"] = float(outcome.avg_query_latency_median_ms)
    metrics["avg_baseline_query_latency_median_ms"] = float(
        outcome.avg_baseline_query_latency_median_ms
    )
    metrics["latency_penalty_triggered"] = float(outcome.latency_penalty_triggered)
    metrics["recall_floor_triggered"] = float(outcome.recall_floor_triggered)
    metrics["aggregation_mode"] = outcome.aggregation_mode
    # Drop the placeholder marker once we've actually computed the score.
    metrics.pop("combined_score_pending", None)
    # Stamp per-dataset latency stats back into the flat dict so existing
    # tools (replay dashboard, evolution_trace) see them too.
    for ds, ds_metrics in outcome.per_dataset.items():
        for metric_name, value in ds_metrics.items():
            metrics[f"{ds}_{metric_name}"] = float(value)


def _fmt_metric(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _short_id(program_id: str | None) -> str:
    if not program_id:
        return "n/a"
    return program_id[:8]


def _iso_now_str() -> str:
    return datetime.now(UTC).isoformat()


def _hash(s: str | None) -> str | None:
    if s is None:
        return None
    return hashlib.sha256(s.encode()).hexdigest()[:16]


_ANSI_CODES = {
    "cyan": "\x1b[36m",
    "bright_green_bold": "\x1b[1;92m",
    "reset": "\x1b[0m",
}


def _color_supported() -> bool:
    """True when ANSI color is appropriate for stdout."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("RANKING_EVOLVED_NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _color(text: str, style: str) -> str:
    """Wrap `text` in an ANSI style if the terminal supports it."""
    code = _ANSI_CODES.get(style)
    if not code or not _color_supported():
        return text
    return f"{code}{text}{_ANSI_CODES['reset']}"


def _improvement(parent: dict[str, float] | None, child: dict[str, float] | None) -> float | None:
    if not parent or not child:
        return None
    p = parent.get("combined_score")
    c = child.get("combined_score")
    if p is None or c is None:
        return None
    return float(c - p)
