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

import hashlib
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

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
    exploration_ratio: float = 0.2      # OpenEvolve default
    exploitation_ratio: float = 0.7     # OpenEvolve default
    num_inspirations: int = 5
    random_seed: int = 42
    diversity_reference_size: int = 20

    def bins_for(self, dim: str) -> int:
        if self.feature_bins_per_dim and dim in self.feature_bins_per_dim:
            return self.feature_bins_per_dim[dim]
        return self.feature_bins


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
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.programs: dict[str, Program] = {}
        self.islands: list[set[str]] = [set() for _ in range(config.num_islands)]
        self.island_feature_maps: list[dict[str, str]] = [
            {} for _ in range(config.num_islands)
        ]
        self.archive: set[str] = set()
        self.island_generations: list[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.current_island: int = 0
        self.best_program_id: str | None = None
        self.island_best_programs: list[str | None] = [None] * config.num_islands

        # Running stats per dim for minmax feature scaling.
        self._feature_stats: dict[str, dict[str, float]] = {}
        # Reference set used to compute "diversity" via average edit distance.
        self._diversity_reference: list[str] = []

    # ------------------------------------------------------------------
    # SearchStrategy Protocol surface
    # ------------------------------------------------------------------

    def initialize(self, seed: Program) -> None:
        """Seed all islands with copies of the seed program.

        OpenEvolve does this lazily inside `_sample_exploration_parent` when
        an island is empty; doing it eagerly here makes the first iteration's
        replay snapshot match without special-casing.
        """
        seed_with_island = _with_island(seed, 0)
        self._admit_into_island(seed_with_island, target_island=0, iteration=0)
        for island_idx in range(1, self.config.num_islands):
            copy = _with_id_and_island(seed, island_idx, parent_id=seed.id)
            self._admit_into_island(copy, target_island=island_idx, iteration=0)
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
        admission = self._admit_into_island(
            child, target_island=parent_island, iteration=iteration
        )

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
            "schema_version": 1,
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
            "diversity_reference": list(self._diversity_reference),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`.

        This intentionally restores the active search population, archive,
        island maps, feature-scaling stats, diversity reference, and RNG state
        instead of reconstructing approximately from evaluated programs.
        """
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported MAP-Elites state schema: {state.get('schema_version')!r}")

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
        self._diversity_reference = [str(x) for x in state.get("diversity_reference", [])]
        self.rng.setstate(_to_tuple(state["rng_state"]))

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
        # Top + previous: island-local, mirrors OpenEvolve iteration.py:60-61.
        parent_island = self._island_of(parent.id)
        if parent_island is None:
            parent_island = self.current_island
        top_ids = [p.id for p in self._top_programs(n=5, island_idx=parent_island)]
        prev_ids = [p.id for p in self._top_programs(n=3, island_idx=parent_island)]
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

    # ------------------------------------------------------------------
    # internals: admission
    # ------------------------------------------------------------------

    def _admit_into_island(
        self, program: Program, *, target_island: int, iteration: int
    ) -> AdmissionDecisions:
        feature_coords_int = self._compute_feature_coords(program)
        feature_coords_dict = {
            dim: float(feature_coords_int[i])
            for i, dim in enumerate(self.config.feature_dimensions)
        }
        program = _replace(program, island=target_island, feature_coords=feature_coords_dict)
        self.programs[program.id] = program

        cell_key = "-".join(str(c) for c in feature_coords_int)
        island_map = self.island_feature_maps[target_island]
        evicted: str | None = None

        existing_id = island_map.get(cell_key)
        should_replace = (
            existing_id is None
            or existing_id not in self.programs
            or _fitness(program) > _fitness(self.programs[existing_id])
        )
        if should_replace:
            if existing_id and existing_id in self.programs and existing_id != program.id:
                evicted = existing_id
                # OpenEvolve removes the evicted program from the island set
                # when its cell is taken over.
                self.islands[target_island].discard(existing_id)
                if existing_id in self.archive:
                    self.archive.discard(existing_id)
                    self.archive.add(program.id)
            island_map[cell_key] = program.id

        self.islands[target_island].add(program.id)
        self._update_archive(program)
        self._enforce_population_limit(exclude_program_id=program.id)
        self._update_best_program(program)
        self._update_island_best(program, target_island)
        # Reference set for diversity stays bounded.
        if len(self._diversity_reference) < self.config.diversity_reference_size:
            self._diversity_reference.append(program.source_code)

        return AdmissionDecisions(
            target_island=target_island,
            feature_coords=feature_coords_dict,
            cell_key=cell_key,
            evicted_program_id=evicted,
            migration_fired=False,
            migration_details=None,
        )

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
            self.programs.values(), key=_fitness  # ascending: worst first
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
        in_island = [
            pid for pid in archive if self.programs[pid].island == self.current_island
        ]
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
                    for c, dim in zip(parent_coords, self.config.feature_dimensions)
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
                self.programs[pid]
                for pid in self.islands[island_idx]
                if pid in self.programs
            )
        ordered = sorted(candidates, key=_fitness, reverse=True)
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
            sorted_pids = sorted(island_ids, key=lambda pid: _fitness(self.programs[pid]), reverse=True)
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
        coords: list[int] = []
        for dim in self.config.feature_dimensions:
            if dim in program.metrics:
                value = float(program.metrics[dim])
            elif dim == "complexity":
                value = float(len(program.source_code))
            elif dim == "diversity":
                value = self._compute_diversity(program)
            elif dim == "score":
                value = _fitness(program)
            else:
                raise ValueError(
                    f"Feature dimension '{dim}' not in program.metrics and not a built-in. "
                    f"Available: {list(program.metrics.keys())}; built-ins: complexity, diversity, score."
                )
            self._update_feature_stats(dim, value)
            scaled = self._scale(dim, value)
            n_bins = self.config.bins_for(dim)
            idx = int(scaled * n_bins)
            coords.append(max(0, min(n_bins - 1, idx)))
        return coords

    def _update_feature_stats(self, dim: str, value: float) -> None:
        s = self._feature_stats.setdefault(dim, {"min": value, "max": value, "n": 0})
        s["min"] = min(s["min"], value)
        s["max"] = max(s["max"], value)
        s["n"] = s.get("n", 0) + 1

    def _scale(self, dim: str, value: float) -> float:
        s = self._feature_stats[dim]
        if math.isclose(s["max"], s["min"]):
            return 0.5
        return max(0.0, min(1.0, (value - s["min"]) / (s["max"] - s["min"])))

    def _compute_diversity(self, program: Program) -> float:
        if len(self._diversity_reference) < 2:
            return 0.0
        # Cheap proxy for edit distance: token-set Jaccard over whitespace-split.
        # Fast, deterministic, and bounded — this matches OpenEvolve's "fast"
        # mode (it also uses a cheap proxy by default).
        target_tokens = set(program.source_code.split())
        if not target_tokens:
            return 0.0
        sims: list[float] = []
        for ref in self._diversity_reference:
            ref_tokens = set(ref.split())
            if not ref_tokens:
                continue
            inter = len(target_tokens & ref_tokens)
            union = len(target_tokens | ref_tokens)
            sims.append(inter / union if union else 0.0)
        if not sims:
            return 0.0
        return 1.0 - (sum(sims) / len(sims))

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


def _fitness(p: Program) -> float:
    """Fitness ignoring MAP-Elites feature dimensions; defaults to combined_score.

    Matches OpenEvolve's `get_fitness_score(metrics, feature_dimensions)`
    behavior: prefer `combined_score` if present, else mean of the remaining
    numeric metrics.
    """
    m = p.metrics
    if "combined_score" in m:
        return float(m["combined_score"])
    nums = [float(v) for v in m.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
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
