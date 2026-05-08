"""MAP-Elites + Islands — port of OpenEvolve's database.py loop dynamics.

Mirrors OpenEvolve's behavior at the field level so the replay dashboard
shows equivalent decisions step-for-step. Specifically:

  * Parent sampling uses three modes (exploration | exploitation | random)
    keyed by exploration_ratio + exploitation_ratio.
  * Inspirations come from the parent's island (island isolation), include
    the island's best program, the top `elite_selection_ratio * n`, then
    feature-cell-perturbed neighbors, then random fill.
  * Admission follows the same precedence: feature_coords -> island map
    -> replace if better fitness -> archive update -> population cap ->
    best-program tracking.
  * Migration fires when `max(island_generations) - last_migration_gen`
    crosses `migration_interval`; each island's top `migration_rate` migrates
    to the adjacent islands (ring topology), with the duplicate-code and
    migrant-flag guards from OpenEvolve.

Things deliberately omitted from this v1 port (keep the surface small):

  * Novelty rejection sampling via embeddings + LLM judge — disabled.
  * Persistent on-disk artifact storage during sampling — handled by
    `core.run_store`, not here.

Reference (read-only): see the corresponding OpenEvolve symbols cited in
the plan: `ProgramDatabase.add` (database.py:211), `_calculate_feature_coords`
(database.py:834), `should_migrate` (database.py:1775), `migrate_programs`
(database.py:1780), `sample` (database.py:382), `_sample_parent`/`_sample_inspirations`.
"""

from __future__ import annotations

import ast
import hashlib
import math
import random
import time
import uuid
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..core.types import (
    AdmissionDecisions,
    PopulationSnapshot,
    Program,
    SamplingDecisions,
)
from .base import register_strategy

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------


@dataclass
class MapElitesIslandsConfig:
    population_size: int = 200
    archive_size: int = 50
    num_islands: int = 3
    migration_interval: int = 20
    migration_rate: float = 0.1
    feature_dimensions: list[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    feature_bins_per_dim: dict[str, int] | None = None
    elite_selection_ratio: float = 0.2  # OpenEvolve default
    exploration_ratio: float = 0.2  # OpenEvolve default
    exploitation_ratio: float = 0.7  # OpenEvolve default
    num_inspirations: int = 5
    random_seed: int = 42
    # `complexity` feature dimension uses AST node count of the program's
    # source (comments + docstrings are absent from the AST), so cosmetic
    # comment edits don't move programs between cells. Set to "char_count"
    # to fall back to len(source_code) (legacy behavior).
    complexity_metric: str = "ast_nodes"
    # Complexity is binned by live-population percentile rank by default.
    # When a new min/max or cluster shape emerges, existing programs are
    # rebucketed against the current live island population. Set to "minmax"
    # to use the older running-min/running-max scaler.
    complexity_binning: str = "adaptive_percentile"
    # `diversity` is computed against (a) the seed's normalized core source
    # (always pinned) and (b) the live archive's normalized core sources.
    # The reference is "rolling" because archive membership changes as the
    # search progresses; failed (score=0) programs are never admitted to the
    # archive so they cannot pollute the diversity yardstick.
    failure_buffer_size: int = 12

    def bins_for(self, dim: str) -> int:
        if self.feature_bins_per_dim and dim in self.feature_bins_per_dim:
            return self.feature_bins_per_dim[dim]
        return self.feature_bins


@dataclass(frozen=True)
class FailureRecord:
    """A score=0 / crashed candidate captured for prompt feedback.

    Carries enough state for the prompt builder to render `(error_summary,
    diff_against_parent)` even after the parent has been evicted from the
    live population. Bounded ring buffer; see `failure_buffer_size`.
    """

    iteration: int
    parent_id: str | None
    parent_island: int
    parent_source_code: str
    child_source_code: str
    error_summary: str


# ----------------------------------------------------------------------------
# implementation
# ----------------------------------------------------------------------------


@register_strategy("map_elites_islands")
class MapElitesIslandsStrategy:
    """Single-process, in-memory MAP-Elites + islands.

    Holds the full population in `self.programs`. The controller must call
    `admit(child, iteration)` once per loop step; admission also triggers
    migration when due. Persistence is the controller's job (it copies
    programs into `core.run_store` inside the same transaction).
    """

    def __init__(self, config: MapElitesIslandsConfig):
        if config.complexity_binning not in {"adaptive_percentile", "minmax"}:
            raise ValueError(
                "complexity_binning must be 'adaptive_percentile' or 'minmax', "
                f"got {config.complexity_binning!r}"
            )
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.programs: dict[str, Program] = {}
        self.islands: list[set[str]] = [set() for _ in range(config.num_islands)]
        self.island_feature_maps: list[dict[str, str]] = [{} for _ in range(config.num_islands)]
        self.archive: set[str] = set()
        self.island_generations: list[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.current_island: int = 0
        self.best_program_id: str | None = None
        self.island_best_programs: list[str | None] = [None] * config.num_islands

        # Running stats per dim for minmax feature scaling.
        self._feature_stats: dict[str, dict[str, float]] = {}
        # Sorted live values per adaptive-percentile dimension. Complexity is
        # the only built-in adaptive dimension today.
        self._feature_percentile_values: dict[str, list[float]] = {}
        # Cache of `_normalized_core_source(code)` keyed by raw source. Strips
        # comments + docstrings via the AST, so identical-algorithm-different-
        # comments programs collapse to the same string for both complexity
        # (AST node count) and diversity (string distance against archive).
        self._core_source_cache: dict[str, str] = {}
        # Pinned seed reference for the diversity metric. Set in `initialize`.
        # The seed always anchors the diversity yardstick even after it is
        # evicted from the archive — otherwise the metric would silently
        # change definition mid-run.
        self._diversity_seed_anchor: str | None = None
        # Bounded ring buffer of recent score=0 / crashed candidates. The
        # controller surfaces these to the next prompt under "Recent failed
        # attempts" so the LLM avoids repeating the same broken approach.
        self._recent_failures: deque[FailureRecord] = deque(
            maxlen=max(1, int(config.failure_buffer_size))
        )

    # ------------------------------------------------------------------
    # SearchStrategy Protocol surface
    # ------------------------------------------------------------------

    def initialize(self, seed: Program) -> None:
        """Seed all islands with copies of the seed program.

        OpenEvolve does this lazily inside `_sample_exploration_parent` when
        an island is empty; doing it eagerly here makes the first iteration's
        replay snapshot match without special-casing.
        """
        # Pin the seed's normalized core source as the diversity anchor.
        # Children's diversity is measured against this anchor + the live
        # archive's core sources (rolling sample).
        self._diversity_seed_anchor = self._core_source(seed.source_code)
        seed_with_island = _with_island(seed, 0)
        # `force=True` bypasses the score=0 rejection that applies to evolved
        # children: even a zero-scoring seed must bootstrap MAP-Elites or the
        # population is empty and `_sample_parent` has nothing to return.
        # Test fixtures (e.g. evolution_algo_test) intentionally seed at
        # SCORE=0 to drive proposal-script behavior — that path stays alive.
        self._admit_into_island(seed_with_island, target_island=0, iteration=0, force=True)
        for island_idx in range(1, self.config.num_islands):
            copy = _with_id_and_island(seed, island_idx, parent_id=seed.id)
            self._admit_into_island(copy, target_island=island_idx, iteration=0, force=True)
        self.best_program_id = seed.id

    def sample(self, iteration: int) -> tuple[Program, list[Program]]:
        parent = self._sample_parent()
        inspirations = self._sample_inspirations(parent, n=self.config.num_inspirations)
        return parent, inspirations

    def admit(self, child: Program, iteration: int) -> AdmissionDecisions:
        # Step 1: increment generation for the parent's island (the iteration
        # was conducted "in" that island). OpenEvolve increments per iteration.
        # NOTE: explicit None-check — `or` would treat island 0 as falsy.
        parent_island = self._island_of(child.parent_id)
        if parent_island is None:
            parent_island = self.current_island
        self.island_generations[parent_island] += 1

        # Step 2: admit the child.
        admission = self._admit_into_island(child, target_island=parent_island, iteration=iteration)

        # Step 3: trigger migration if due.
        if self._should_migrate():
            details = self._migrate(iteration)
            admission = AdmissionDecisions(
                target_island=admission.target_island,
                feature_coords=admission.feature_coords,
                cell_key=admission.cell_key,
                evicted_program_id=admission.evicted_program_id,
                migration_fired=True,
                migration_details=details,
            )

        # Step 4: round-robin advance current_island so the next iteration's
        # exploration draws from a different island. (Matches OpenEvolve's
        # process_parallel.next_island() cadence.)
        self.current_island = (self.current_island + 1) % self.config.num_islands
        return admission

    def best(self) -> Program:
        if self.best_program_id and self.best_program_id in self.programs:
            return self.programs[self.best_program_id]
        # Fallback: scan
        return max(self.programs.values(), key=_fitness)

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot sufficient for exact in-run resume."""
        return {
            "schema_version": 3,
            "programs": {pid: asdict(program) for pid, program in self.programs.items()},
            "islands": [sorted(island) for island in self.islands],
            "island_feature_maps": [dict(m) for m in self.island_feature_maps],
            "archive": sorted(self.archive),
            "island_generations": list(self.island_generations),
            "last_migration_generation": self.last_migration_generation,
            "current_island": self.current_island,
            "best_program_id": self.best_program_id,
            "island_best_programs": list(self.island_best_programs),
            "feature_stats": self._feature_stats,
            "feature_percentile_values": self._feature_percentile_values,
            "diversity_seed_anchor": self._diversity_seed_anchor,
            "recent_failures": [asdict(f) for f in self._recent_failures],
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`.

        This intentionally restores the active search population, archive,
        island maps, feature-scaling stats, diversity reference, and RNG state
        instead of reconstructing approximately from evaluated programs.
        """
        schema_version = int(state.get("schema_version", 0))
        if schema_version not in (1, 2, 3):
            raise ValueError(
                f"unsupported MAP-Elites state schema: {state.get('schema_version')!r}"
            )

        programs = state.get("programs") or {}
        self.programs = {
            pid: Program(
                id=str(data["id"]),
                source_code=str(data["source_code"]),
                parent_id=data.get("parent_id"),
                generation=int(data["generation"]),
                iteration_found=int(data["iteration_found"]),
                timestamp=float(data["timestamp"]),
                metrics=dict(data.get("metrics") or {}),
                complexity=float(data["complexity"]),
                diversity=float(data["diversity"]),
                island=int(data["island"]),
                feature_coords=dict(data.get("feature_coords") or {}),
                changes_description=str(data.get("changes_description") or ""),
                artifacts=dict(data.get("artifacts") or {}),
                metadata=dict(data.get("metadata") or {}),
            )
            for pid, data in programs.items()
        }
        self.islands = [set(map(str, island)) for island in state.get("islands", [])]
        if len(self.islands) != self.config.num_islands:
            raise ValueError(
                f"resume state has {len(self.islands)} islands, config expects "
                f"{self.config.num_islands}"
            )
        self.island_feature_maps = [
            {str(cell): str(pid) for cell, pid in mapping.items()}
            for mapping in state.get("island_feature_maps", [])
        ]
        if len(self.island_feature_maps) != self.config.num_islands:
            raise ValueError(
                f"resume state has {len(self.island_feature_maps)} island feature maps, "
                f"config expects {self.config.num_islands}"
            )
        self.archive = set(map(str, state.get("archive", [])))
        self.island_generations = [int(x) for x in state.get("island_generations", [])]
        if len(self.island_generations) != self.config.num_islands:
            raise ValueError(
                f"resume state has {len(self.island_generations)} island generation counters, "
                f"config expects {self.config.num_islands}"
            )
        self.last_migration_generation = int(state.get("last_migration_generation", 0))
        self.current_island = int(state.get("current_island", 0))
        self.best_program_id = state.get("best_program_id")
        self.island_best_programs = [
            None if pid is None else str(pid)
            for pid in state.get("island_best_programs", [None] * self.config.num_islands)
        ]
        self._feature_stats = {
            str(dim): {str(k): float(v) for k, v in stats.items()}
            for dim, stats in (state.get("feature_stats") or {}).items()
        }
        self._feature_percentile_values = {
            str(dim): [float(v) for v in values]
            for dim, values in (state.get("feature_percentile_values") or {}).items()
        }
        # Schema v2: persistent diversity anchor + recent-failure ring buffer.
        # Schema v1 carried a `diversity_reference` list (now unused — diversity
        # rolls over the live archive); just ignore it on resume.
        anchor = state.get("diversity_seed_anchor")
        self._diversity_seed_anchor = str(anchor) if anchor is not None else None
        self._core_source_cache.clear()
        self._recent_failures.clear()
        for entry in state.get("recent_failures", []) or []:
            try:
                self._recent_failures.append(
                    FailureRecord(
                        iteration=int(entry["iteration"]),
                        parent_id=entry.get("parent_id"),
                        parent_island=int(entry.get("parent_island", 0)),
                        parent_source_code=str(entry.get("parent_source_code") or ""),
                        child_source_code=str(entry.get("child_source_code") or ""),
                        error_summary=str(entry.get("error_summary") or ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self.rng.setstate(_to_tuple(state["rng_state"]))
        if not self._feature_percentile_values:
            self._rebucket_feature_maps()

    def snapshot(self) -> PopulationSnapshot:
        # Flatten archive cells across islands for the snapshot:
        flat_cells: dict[str, str] = {}
        for island_idx, m in enumerate(self.island_feature_maps):
            for cell, pid in m.items():
                flat_cells[f"{island_idx}:{cell}"] = pid
        return PopulationSnapshot(
            n_programs=len(self.programs),
            islands=[sorted(island) for island in self.islands],
            island_generations=list(self.island_generations),
            current_island=self.current_island,
            last_migration_generation=self.last_migration_generation,
            archive_cells=flat_cells,
            archive_size=len(self.archive),
            best_program_id=self.best_program_id,
            island_best_programs=list(self.island_best_programs),
        )

    # ------------------------------------------------------------------
    # sampling decisions (exposed so the controller can record the rng path)
    # ------------------------------------------------------------------

    def sampling_decisions(
        self, iteration: int, parent: Program, inspirations: list[Program]
    ) -> SamplingDecisions:
        rng_seed_hash = hashlib.sha256(
            f"{self.config.random_seed}:{iteration}".encode()
        ).hexdigest()[:16]
        parent_island = self._island_of(parent.id)
        if parent_island is None:
            parent_island = self.current_island
        top_ids = [p.id for p in self._top_programs(n=5, island_idx=parent_island)]
        prev_ids = [
            p.id
            for p in self._recent_programs(
                n=3, island_idx=parent_island, exclude_program_id=parent.id
            )
        ]
        return SamplingDecisions(
            rng_seed_hash=rng_seed_hash,
            parent_id=parent.id,
            parent_island=parent_island,
            inspiration_ids=[p.id for p in inspirations],
            inspiration_strategy="default",
            top_program_ids=top_ids,
            previous_program_ids=prev_ids,
        )

    def top_programs(self, n: int, island_idx: int | None = None) -> list[Program]:
        return self._top_programs(n=n, island_idx=island_idx)

    def recent_programs(
        self,
        n: int,
        island_idx: int | None = None,
        *,
        exclude_program_id: str | None = None,
    ) -> list[Program]:
        return self._recent_programs(
            n=n, island_idx=island_idx, exclude_program_id=exclude_program_id
        )

    # ------------------------------------------------------------------
    # internals: admission
    # ------------------------------------------------------------------

    def _admit_into_island(
        self,
        program: Program,
        *,
        target_island: int,
        iteration: int,
        force: bool = False,
    ) -> AdmissionDecisions:
        # Stamp authoritative complexity/diversity from the source code BEFORE
        # computing feature coords, so callers (controller) don't need to set
        # these fields. Mirrors OpenEvolve's `_calculate_feature_coords` which
        # owns both the metric and the binning.
        complexity = self._compute_complexity(program.source_code)
        diversity = self._compute_diversity(program)
        program = _replace(program, complexity=complexity, diversity=diversity)

        # Reject score=0 / crashed candidates from MAP-Elites entirely. They
        # never own a cell, never enter the archive, never join the live
        # population (so they cannot be sampled as parents). The failure
        # record IS retained so the next prompt can show "do not repeat this".
        # The bootstrap seed bypasses this check (`force=True`) — a zero-
        # scoring seed still has to populate islands at iteration 0 or the
        # search has nothing to sample as its first parent.
        if not force and _is_failure(program):
            self._record_failure(
                program=program,
                target_island=target_island,
                iteration=iteration,
            )
            return AdmissionDecisions(
                target_island=target_island,
                feature_coords={},
                cell_key="",
                evicted_program_id=None,
                migration_fired=False,
                migration_details=None,
            )

        old_feature_maps = [dict(m) for m in self.island_feature_maps]
        program = _replace(program, island=target_island, feature_coords={})
        self.programs[program.id] = program
        self.islands[target_island].add(program.id)

        evicted: str | None = None
        self._update_archive(program)
        self._enforce_population_limit(exclude_program_id=program.id)
        self._update_best_program(program)
        self._update_island_best(program, target_island)
        self._rebucket_feature_maps()

        program = self.programs.get(program.id, program)
        feature_coords_dict = dict(program.feature_coords)
        feature_coords_int = [
            int(feature_coords_dict[dim]) for dim in self.config.feature_dimensions
        ]
        cell_key = "-".join(str(c) for c in feature_coords_int)
        final_host = self.island_feature_maps[target_island].get(cell_key)
        old_host = old_feature_maps[target_island].get(cell_key)
        if final_host == program.id and old_host and old_host != program.id:
            evicted = old_host
            # Keep the existing admission semantics for direct cell takeover:
            # the displaced cell host leaves the island-local sampling pool and
            # archive, while the program row remains available for audit/resume.
            final_hosts = set(self.island_feature_maps[target_island].values())
            if old_host not in final_hosts:
                self.islands[target_island].discard(old_host)
                self.archive.discard(old_host)

        return AdmissionDecisions(
            target_island=target_island,
            feature_coords=feature_coords_dict,
            cell_key=cell_key,
            evicted_program_id=evicted,
            migration_fired=False,
            migration_details=None,
        )

    def _record_failure(
        self,
        *,
        program: Program,
        target_island: int,
        iteration: int,
    ) -> None:
        """Save a compact failure record for the next prompt's feedback section."""
        parent = self.programs.get(program.parent_id) if program.parent_id else None
        parent_code = parent.source_code if parent else ""
        err = str((program.metrics or {}).get("eval_error") or "").strip()
        if err:
            tail = err.splitlines()[-1] if "\n" in err else err
            error_summary = tail[:240]
        elif (program.metrics or {}).get("recall_floor_triggered"):
            error_summary = "recall_floor_triggered (recall < min_recall on at least one dataset)"
        else:
            error_summary = "combined_score=0 (rejected by objective)"
        self._recent_failures.append(
            FailureRecord(
                iteration=iteration,
                parent_id=program.parent_id,
                parent_island=target_island,
                parent_source_code=parent_code,
                child_source_code=program.source_code,
                error_summary=error_summary,
            )
        )

    def recent_failures(self, *, n: int, island_idx: int | None = None) -> list[FailureRecord]:
        """Return up to `n` most-recent failures, newest first.

        With `island_idx` set, restrict to failures whose attempted island
        matches. The controller calls this once per iteration to feed the
        prompt builder's "do not repeat" section.
        """
        if n <= 0 or not self._recent_failures:
            return []
        items = list(self._recent_failures)
        if island_idx is not None:
            items = [f for f in items if f.parent_island == island_idx]
        items.reverse()
        return items[:n]

    def _update_archive(self, program: Program) -> None:
        # Cell-eviction in `_admit_into_island` may have already swapped this
        # program into the archive. Short-circuit so we don't accidentally
        # discard the worst-archive entry without re-adding (a real bug in
        # OpenEvolve's equivalent code path that shrinks the archive on every
        # cell takeover that touches an archived program).
        if program.id in self.archive:
            return
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return
        # Drop stale references first.
        for pid in list(self.archive):
            if pid not in self.programs:
                self.archive.discard(pid)
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return
        # Replace the worst if `program` is better.
        worst_id = min(self.archive, key=lambda pid: _fitness(self.programs[pid]))
        if _fitness(program) > _fitness(self.programs[worst_id]):
            self.archive.discard(worst_id)
            self.archive.add(program.id)

    def _enforce_population_limit(self, *, exclude_program_id: str | None) -> None:
        if len(self.programs) <= self.config.population_size:
            return
        n_remove = len(self.programs) - self.config.population_size
        protected = {self.best_program_id, exclude_program_id} - {None}
        sorted_by_fitness = sorted(
            self.programs.values(),
            key=_fitness,  # ascending: worst first
        )
        to_remove = []
        for p in sorted_by_fitness:
            if len(to_remove) >= n_remove:
                break
            if p.id in protected:
                continue
            to_remove.append(p)
        for p in to_remove:
            self.programs.pop(p.id, None)
            for m in self.island_feature_maps:
                stale = [k for k, v in m.items() if v == p.id]
                for k in stale:
                    m.pop(k, None)
            for island in self.islands:
                island.discard(p.id)
            self.archive.discard(p.id)

    def _update_best_program(self, program: Program) -> None:
        if self.best_program_id is None or self.best_program_id not in self.programs:
            self.best_program_id = program.id
            return
        if _fitness(program) > _fitness(self.programs[self.best_program_id]):
            self.best_program_id = program.id

    def _update_island_best(self, program: Program, island_idx: int) -> None:
        cur = self.island_best_programs[island_idx]
        if cur is None or cur not in self.programs:
            self.island_best_programs[island_idx] = program.id
            return
        if _fitness(program) > _fitness(self.programs[cur]):
            self.island_best_programs[island_idx] = program.id

    # ------------------------------------------------------------------
    # internals: sampling
    # ------------------------------------------------------------------

    def _sample_parent(self) -> Program:
        rand = self.rng.random()
        if rand < self.config.exploration_ratio:
            return self._sample_exploration_parent()
        if rand < self.config.exploration_ratio + self.config.exploitation_ratio:
            return self._sample_exploitation_parent()
        return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        ids = [pid for pid in self.islands[self.current_island] if pid in self.programs]
        if not ids:
            return self._fallback_seed_into_current_island()
        return self.programs[self.rng.choice(ids)]

    def _sample_exploitation_parent(self) -> Program:
        archive = [pid for pid in self.archive if pid in self.programs]
        if not archive:
            return self._sample_exploration_parent()
        in_island = [pid for pid in archive if self.programs[pid].island == self.current_island]
        pool = in_island or archive
        return self.programs[self.rng.choice(pool)]

    def _sample_random_parent(self) -> Program:
        if not self.programs:
            raise ValueError("No programs available for sampling")
        return self.programs[self.rng.choice(list(self.programs.keys()))]

    def _fallback_seed_into_current_island(self) -> Program:
        if self.best_program_id and self.best_program_id in self.programs:
            best = self.programs[self.best_program_id]
            copy = _with_id_and_island(best, self.current_island, parent_id=best.id)
            self.programs[copy.id] = copy
            self.islands[self.current_island].add(copy.id)
            return copy
        return next(iter(self.programs.values()))

    def _sample_inspirations(self, parent: Program, *, n: int) -> list[Program]:
        if n <= 0:
            return []
        parent_island = parent.island
        island_ids = [pid for pid in self.islands[parent_island] if pid in self.programs]
        if not island_ids:
            return []
        chosen: list[Program] = []
        chosen_ids: set[str] = {parent.id}

        # Island best (excluding parent).
        island_best_id = self.island_best_programs[parent_island]
        if island_best_id and island_best_id in self.programs and island_best_id not in chosen_ids:
            chosen.append(self.programs[island_best_id])
            chosen_ids.add(island_best_id)

        # Top elite_selection_ratio * n elites.
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        for p in self._top_programs(n=top_n, island_idx=parent_island):
            if p.id not in chosen_ids:
                chosen.append(p)
                chosen_ids.add(p.id)
                if len(chosen) >= n:
                    return chosen[:n]

        # Diverse via perturbed feature cells.
        if len(chosen) < n:
            parent_coords = self._compute_feature_coords(parent)
            cell_to_id = self.island_feature_maps[parent_island]
            attempts = (n - len(chosen)) * 3
            for _ in range(attempts):
                perturbed = [
                    max(0, min(self.config.bins_for(dim) - 1, c + self.rng.randint(-2, 2)))
                    for c, dim in zip(
                        parent_coords, self.config.feature_dimensions, strict=True
                    )
                ]
                key = "-".join(str(c) for c in perturbed)
                pid = cell_to_id.get(key)
                if pid and pid not in chosen_ids and pid in self.programs:
                    chosen.append(self.programs[pid])
                    chosen_ids.add(pid)
                    if len(chosen) >= n:
                        return chosen[:n]

        # Random fill from remaining island programs.
        remaining = [pid for pid in island_ids if pid not in chosen_ids]
        if remaining and len(chosen) < n:
            picks = self.rng.sample(remaining, min(n - len(chosen), len(remaining)))
            chosen.extend(self.programs[pid] for pid in picks)

        return chosen[:n]

    def _top_programs(self, *, n: int, island_idx: int | None) -> list[Program]:
        if island_idx is None:
            candidates: Iterable[Program] = self.programs.values()
        else:
            candidates = (
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            )
        ordered = sorted(candidates, key=_fitness, reverse=True)
        return ordered[:n]

    def _recent_programs(
        self,
        *,
        n: int,
        island_idx: int | None,
        exclude_program_id: str | None,
    ) -> list[Program]:
        if n <= 0:
            return []
        if island_idx is None:
            candidates: Iterable[Program] = self.programs.values()
        else:
            candidates = (
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            )
        if exclude_program_id is not None:
            candidates = (p for p in candidates if p.id != exclude_program_id)
        ordered = sorted(
            candidates,
            key=lambda p: (p.iteration_found, p.timestamp),
            reverse=True,
        )
        return ordered[:n]

    # ------------------------------------------------------------------
    # internals: migration
    # ------------------------------------------------------------------

    def _should_migrate(self) -> bool:
        if self.config.num_islands < 2:
            return False
        return (
            max(self.island_generations) - self.last_migration_generation
            >= self.config.migration_interval
        )

    def _migrate(self, iteration: int) -> dict:
        events: list[dict] = []
        n_islands = len(self.islands)
        for src in range(n_islands):
            island_ids = [pid for pid in self.islands[src] if pid in self.programs]
            if not island_ids:
                continue
            sorted_pids = sorted(
                island_ids, key=lambda pid: _fitness(self.programs[pid]), reverse=True
            )
            n_migrants = max(1, int(len(sorted_pids) * self.config.migration_rate))
            migrants = sorted_pids[:n_migrants]
            for migrant_id in migrants:
                migrant = self.programs[migrant_id]
                if migrant.metadata.get("migrant"):
                    continue
                for dst in ((src + 1) % n_islands, (src - 1) % n_islands):
                    dst_codes = {
                        self.programs[pid].source_code
                        for pid in self.islands[dst]
                        if pid in self.programs
                    }
                    if migrant.source_code in dst_codes:
                        continue
                    copy = _with_id_and_island(
                        migrant, dst, parent_id=migrant.id, mark_migrant=True
                    )
                    self._admit_into_island(copy, target_island=dst, iteration=iteration)
                    events.append(
                        {
                            "iteration": iteration,
                            "src": src,
                            "dst": dst,
                            "src_program_id": migrant.id,
                            "dst_program_id": copy.id,
                        }
                    )
        self.last_migration_generation = max(self.island_generations)
        return {"events": events, "last_migration_generation": self.last_migration_generation}

    # ------------------------------------------------------------------
    # internals: feature coords + scaling
    # ------------------------------------------------------------------

    def _compute_feature_coords(self, program: Program) -> list[int]:
        if self._feature_scaling_missing():
            self._refresh_feature_scaling()
        coords: list[int] = []
        for dim in self.config.feature_dimensions:
            n_bins = self.config.bins_for(dim)
            value = self._feature_value(program, dim)
            if self._uses_adaptive_percentile(dim):
                idx = self._percentile_bin(dim, value, n_bins)
            else:
                scaled = self._scale(dim, value)
                idx = int(scaled * n_bins)
                idx = max(0, min(n_bins - 1, idx))
            coords.append(idx)
        return coords

    def _rebucket_feature_maps(self) -> None:
        """Recompute feature coords and island cell hosts from live programs.

        Adaptive complexity binning can move old programs when a new shape of
        complexity values appears. Rebuilding keeps every program on the same
        bin scale and resolves collisions with the existing strict fitness rule:
        a cell host is replaced only by a program with strictly higher fitness.
        """
        self._refresh_feature_scaling()
        new_maps: list[dict[str, str]] = [{} for _ in range(self.config.num_islands)]
        active_ids = set(self._active_program_ids())
        for pid in self.programs:
            program = self.programs.get(pid)
            if program is None:
                continue
            coords = self._compute_feature_coords(program)
            feature_coords = {
                dim: float(coords[i]) for i, dim in enumerate(self.config.feature_dimensions)
            }
            program = _replace(program, feature_coords=feature_coords)
            self.programs[pid] = program
            if pid not in active_ids:
                continue
            cell_key = "-".join(str(c) for c in coords)
            island_idx = program.island
            if not 0 <= island_idx < self.config.num_islands:
                continue
            incumbent_id = new_maps[island_idx].get(cell_key)
            if incumbent_id is None or _fitness(program) > _fitness(self.programs[incumbent_id]):
                new_maps[island_idx][cell_key] = pid
        self.island_feature_maps = new_maps

    # ------------------------------------------------------------------
    # internals: complexity + diversity (cosmetic-edit invariant)
    # ------------------------------------------------------------------

    def _core_source(self, code: str) -> str:
        """Return `code` with comments + module/class/function docstrings
        removed via the AST. Cached by raw source so we don't re-parse on
        every diversity comparison."""
        cached = self._core_source_cache.get(code)
        if cached is not None:
            return cached
        normalized = _normalized_core_source(code)
        # Bound the cache so a long run doesn't accumulate every program's
        # source twice. Old entries are evicted on overflow.
        if len(self._core_source_cache) > 256:
            self._core_source_cache.pop(next(iter(self._core_source_cache)))
        self._core_source_cache[code] = normalized
        return normalized

    def _compute_complexity(self, code: str) -> float:
        """Complexity proxy used as the MAP-Elites x-axis. Defaults to AST
        node count of `code` with comments+docstrings stripped, so cosmetic
        comment edits don't move programs between cells. Fallback is
        len(source_code) (legacy behavior, opt-in via complexity_metric)."""
        if self.config.complexity_metric == "char_count":
            return float(len(code))
        return float(_ast_node_count(self._core_source(code)))

    def _compute_diversity(self, program: Program) -> float:
        """Mean string-distance of `program`'s normalized core source against
        (a) the seed anchor and (b) every program currently in the archive.

        The archive is the rolling reference: as it changes, so does the
        diversity yardstick. Failed (score=0) programs are never admitted to
        the archive, so they cannot pollute it.
        """
        target_core = self._core_source(program.source_code)
        if not target_core:
            return 0.0
        ref_cores: list[str] = []
        if self._diversity_seed_anchor is not None:
            ref_cores.append(self._diversity_seed_anchor)
        for pid in self.archive:
            other = self.programs.get(pid)
            if other is None or other.id == program.id:
                continue
            ref_cores.append(self._core_source(other.source_code))
        if not ref_cores:
            return 0.0
        # Deduplicate identical references (e.g. when archive holds multiple
        # seed copies on first few iterations) so the metric isn't biased
        # toward whichever code happens to be replicated.
        seen: set[str] = set()
        distances: list[float] = []
        for ref in ref_cores:
            if ref in seen:
                continue
            seen.add(ref)
            distances.append(_code_distance(target_core, ref))
        if not distances:
            return 0.0
        return sum(distances) / len(distances)

    def _active_program_ids(self) -> list[str]:
        active: set[str] = set()
        for island in self.islands:
            for pid in island:
                if pid in self.programs:
                    active.add(pid)
        return [pid for pid in self.programs if pid in active]

    def _feature_value(self, program: Program, dim: str) -> float:
        if dim in program.metrics:
            return float(program.metrics[dim])
        if dim == "complexity":
            return float(program.complexity)
        if dim == "diversity":
            return float(program.diversity)
        if dim == "score":
            return _fitness(program)
        raise ValueError(
            f"Feature dimension '{dim}' not in program.metrics and not a built-in. "
            f"Available: {list(program.metrics.keys())}; built-ins: complexity, diversity, score."
        )

    def _uses_adaptive_percentile(self, dim: str) -> bool:
        return dim == "complexity" and self.config.complexity_binning == "adaptive_percentile"

    def _feature_scaling_missing(self) -> bool:
        for dim in self.config.feature_dimensions:
            if self._uses_adaptive_percentile(dim):
                if dim not in self._feature_percentile_values:
                    return True
            elif dim not in self._feature_stats:
                return True
        return False

    def _refresh_feature_scaling(self) -> None:
        active = list(self.programs.values())
        for dim in self.config.feature_dimensions:
            values = [self._feature_value(program, dim) for program in active]
            if not values:
                continue
            if self._uses_adaptive_percentile(dim):
                self._feature_percentile_values[dim] = sorted(values)
                self._feature_stats.pop(dim, None)
            else:
                self._feature_stats[dim] = {
                    "min": min(values),
                    "max": max(values),
                    "n": float(len(values)),
                }
                self._feature_percentile_values.pop(dim, None)

    def _scale(self, dim: str, value: float) -> float:
        s = self._feature_stats[dim]
        if math.isclose(s["max"], s["min"]):
            return 0.5
        return max(0.0, min(1.0, (value - s["min"]) / (s["max"] - s["min"])))

    def _percentile_bin(self, dim: str, value: float, n_bins: int) -> int:
        values = self._feature_percentile_values.get(dim) or []
        if n_bins <= 1:
            return 0
        if not values:
            return n_bins // 2
        if math.isclose(values[0], values[-1]):
            return n_bins // 2

        # Percentile-rank binning keeps equal complexity values in the same
        # cell while still letting new extremes rebucket older programs.
        lo = bisect_left(values, value)
        hi = bisect_right(values, value)
        if hi == lo:
            # Program is outside the current live values (possible for direct
            # helper calls before admission). Place by insertion percentile.
            rank = float(lo)
        else:
            # Average tied ranks so identical AST counts never get split
            # across bins by insertion order.
            rank = ((lo + hi - 1) / 2.0)
        percentile = rank / max(1.0, float(len(values) - 1))
        idx = int(percentile * n_bins)
        return max(0, min(n_bins - 1, idx))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _island_of(self, program_id: str | None) -> int | None:
        if program_id is None:
            return None
        p = self.programs.get(program_id)
        return p.island if p else None


# ----------------------------------------------------------------------------
# private utilities
# ----------------------------------------------------------------------------


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def _is_failure(p: Program) -> bool:
    """A program is a 'failure' if the evaluator crashed or its objective
    score is exactly zero (recall floor / penalty hit). These never enter
    MAP-Elites cells, the archive, or the live population — they only get
    surfaced to the LLM as 'do not repeat' context."""
    m = p.metrics or {}
    if m.get("eval_crashed"):
        return True
    score = m.get("combined_score")
    if score is None:
        # No score at all = treat as failure (defensive; shouldn't normally happen).
        return True
    try:
        return float(score) <= 0.0
    except (TypeError, ValueError):
        return True


def _normalized_core_source(code: str) -> str:
    """Return `code` with comments + module/class/function docstrings
    stripped via the AST.

    Comments are absent from the AST by construction; docstrings are dropped
    by removing the leading string-Constant Expr from each Module / ClassDef
    / FunctionDef / AsyncFunctionDef body. Falls back to the raw source on a
    SyntaxError (LLMs occasionally emit invalid Python under retries).
    """
    if not code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    docstring_nodes = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    for node in ast.walk(tree):
        if isinstance(node, docstring_nodes):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                replacement = body[1:] or [ast.Pass()]
                node.body = replacement
    try:
        return ast.unparse(tree)
    except (AttributeError, ValueError):
        return code


def _ast_node_count(code: str) -> int:
    """Count AST nodes in `code`. Comment lines are not nodes; this is the
    intended behavior — we want a metric that ignores cosmetic edits.

    Falls back to a line-count proxy on syntax errors so callers don't have
    to special-case malformed candidates.
    """
    if not code:
        return 1
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return max(1, code.count("\n") + 1)
    return sum(1 for _ in ast.walk(tree))


def _code_distance(a_core: str, b_core: str) -> float:
    """Normalized string-distance in [0, 1]. 0 = identical, 1 = no overlap.

    Uses `difflib.SequenceMatcher.ratio()` (Ratcliff-Obershelp) for a
    deterministic, dependency-free proxy. Computed against `_core_source`
    output, so identical algorithms with different comments score 0.
    """
    if not a_core and not b_core:
        return 0.0
    if not a_core or not b_core:
        return 1.0
    if a_core == b_core:
        return 0.0
    return 1.0 - SequenceMatcher(None, a_core, b_core, autojunk=False).ratio()


def _fitness(p: Program) -> float:
    """Fitness ignoring MAP-Elites feature dimensions; defaults to combined_score.

    Matches OpenEvolve's `get_fitness_score(metrics, feature_dimensions)`
    behavior: prefer `combined_score` if present, else mean of the remaining
    numeric metrics.
    """
    m = p.metrics
    if "combined_score" in m:
        return float(m["combined_score"])
    nums = [float(v) for v in m.values() if isinstance(v, int | float) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else 0.0


def _replace(p: Program, **changes) -> Program:
    """Like dataclasses.replace, but works for our frozen Program."""
    from dataclasses import replace as _r

    return _r(p, **changes)


def _with_island(p: Program, island: int) -> Program:
    return _replace(p, island=island)


def _with_id_and_island(
    p: Program, island: int, *, parent_id: str | None, mark_migrant: bool = False
) -> Program:
    new_meta = dict(p.metadata)
    if mark_migrant:
        new_meta["migrant"] = True
    return Program(
        id=str(uuid.uuid4()),
        source_code=p.source_code,
        parent_id=parent_id,
        generation=p.generation,
        iteration_found=p.iteration_found,
        timestamp=time.time(),
        metrics=dict(p.metrics),
        complexity=p.complexity,
        diversity=p.diversity,
        island=island,
        feature_coords=dict(p.feature_coords),
        changes_description=p.changes_description,
        artifacts=dict(p.artifacts),
        metadata=new_meta,
    )
