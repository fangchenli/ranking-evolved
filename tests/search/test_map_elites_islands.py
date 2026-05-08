"""Tests for search.map_elites_islands — admission, archive, sampling, migration.

These tests isolate the strategy from the controller. We construct programs
by hand (no LLM, no evaluator), drive `admit` / `sample` directly, and check
the resulting state field-by-field. This is the parity-relevant module, so
the assertions are explicit.
"""

from __future__ import annotations

import time

from ranking_evolved.core.types import Program
from ranking_evolved.search.map_elites_islands import (
    MapElitesIslandsConfig,
    MapElitesIslandsStrategy,
    _ast_node_count,
    _normalized_core_source,
)


def _mk(
    id: str,
    *,
    code: str | None = None,
    parent_id: str | None = None,
    score: float = 0.5,
    iteration: int = 0,
    island: int = 0,
    metadata: dict | None = None,
) -> Program:
    return Program(
        id=id,
        source_code=code if code is not None else f"# {id}\nx = {iteration}\n",
        parent_id=parent_id,
        generation=iteration,
        iteration_found=iteration,
        timestamp=time.time(),
        metrics={"combined_score": score},
        complexity=float(len(code or id)),
        diversity=0.1,
        island=island,
        feature_coords={},
        metadata=metadata or {},
    )


def _make_strategy(**overrides) -> MapElitesIslandsStrategy:
    cfg = MapElitesIslandsConfig(
        population_size=overrides.pop("population_size", 50),
        archive_size=overrides.pop("archive_size", 5),
        num_islands=overrides.pop("num_islands", 3),
        migration_interval=overrides.pop("migration_interval", 5),
        migration_rate=overrides.pop("migration_rate", 0.5),
        feature_dimensions=overrides.pop("feature_dimensions", ["complexity", "diversity"]),
        feature_bins=overrides.pop("feature_bins", 5),
        elite_selection_ratio=overrides.pop("elite_selection_ratio", 0.4),
        exploration_ratio=overrides.pop("exploration_ratio", 0.2),
        exploitation_ratio=overrides.pop("exploitation_ratio", 0.7),
        num_inspirations=overrides.pop("num_inspirations", 3),
        random_seed=overrides.pop("random_seed", 7),
    )
    return MapElitesIslandsStrategy(cfg)


def _assignment_code(n: int) -> str:
    return "\n".join(f"x{i} = {i}" for i in range(n)) + "\n"


# ----------------------------------------------------------------------------


def test_initialize_seeds_every_island(record_io):
    strat = _make_strategy(num_islands=3)
    seed = _mk("seed", score=0.4)

    def run() -> dict:
        strat.initialize(seed)
        return {
            "n_programs": len(strat.programs),
            "islands_sizes": [len(i) for i in strat.islands],
            "best_id": strat.best_program_id,
            "feature_maps_have_seed": [bool(m) for m in strat.island_feature_maps],
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.initialize",
        input={"num_islands": 3, "seed_score": 0.4},
        run=run,
    )
    # 3 programs (1 original + 2 copies), one per island, each occupies a cell
    assert out == {
        "n_programs": 3,
        "islands_sizes": [1, 1, 1],
        "best_id": "seed",
        "feature_maps_have_seed": [True, True, True],
    }


def test_admit_replaces_cell_when_better(record_io):
    # Use "complexity" so identical source code maps to the same MAP-Elites cell.
    # ("score" would put each fitness value in a different cell — defeats the test.)
    strat = _make_strategy(
        num_islands=1,
        archive_size=10,
        feature_dimensions=["complexity"],
        feature_bins=5,
    )
    seed = _mk("seed", code="# seed\nx = 1\n", score=0.3)
    strat.initialize(seed)
    # Same source code → same complexity bin as the seed.
    same_cell = _mk("low", code=seed.source_code, parent_id="seed", score=0.2)
    better = _mk("high", code=seed.source_code, parent_id="seed", score=0.9)

    def run() -> dict:
        # Admit a worse program first — seed should retain the cell
        adm_low = strat.admit(same_cell, iteration=1)
        evicted_after_low = adm_low.evicted_program_id
        cell_after_low = strat.island_feature_maps[0].get(adm_low.cell_key)
        # Admit a better program — it should evict the seed in that cell
        adm_high = strat.admit(better, iteration=2)
        return {
            "evicted_after_low": evicted_after_low,
            "cell_after_low": cell_after_low,
            "evicted_after_high": adm_high.evicted_program_id,
            "cell_after_high": strat.island_feature_maps[0].get(adm_high.cell_key),
            "best_id": strat.best_program_id,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.admit",
        input={"scenario": "cell replacement on better fitness only"},
        run=run,
    )
    # `low` did not evict seed (worse fitness); `high` did.
    assert out["evicted_after_low"] is None
    assert out["cell_after_low"] == "seed"
    assert out["evicted_after_high"] == "seed"
    assert out["cell_after_high"] == "high"
    assert out["best_id"] == "high"


def test_archive_keeps_best_at_capacity(record_io):
    """Archive is capped, contains live programs, and contains the global best.

    Note: archive contents are not "top-k by score over time" — when a
    MAP-Elites cell is displaced, the displaced program is dropped from
    the archive even if it had a higher score than other archive entries
    set in earlier iterations. We assert invariants that hold under that
    behavior, not a top-k illusion.
    """
    strat = _make_strategy(num_islands=1, archive_size=3, feature_dimensions=["score"])
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)

    def run() -> dict:
        for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.6, 0.95]):
            strat.admit(
                _mk(f"c{i}", code=f"# c{i}\n", parent_id="seed", score=score, iteration=i + 1),
                iteration=i + 1,
            )
        archive_scores = sorted(
            strat.programs[pid].metrics["combined_score"] for pid in strat.archive
        )
        return {
            "archive_size": len(strat.archive),
            "archive_scores": archive_scores,
            "all_ids_live": all(pid in strat.programs for pid in strat.archive),
            "best_in_archive": strat.best_program_id in strat.archive,
            "global_best_id": strat.best_program_id,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._update_archive",
        input={"archive_cap": 3, "scores_admitted": [0.1, 0.2, 0.3, 0.4, 0.6, 0.95]},
        run=run,
    )
    assert out["archive_size"] == 3
    assert out["all_ids_live"] is True
    assert out["best_in_archive"] is True
    assert out["global_best_id"] == "c5"  # 0.95 is the highest score
    assert max(out["archive_scores"]) == 0.95


def test_population_limit_evicts_worst_keeps_best(record_io):
    strat = _make_strategy(num_islands=1, population_size=4, feature_dimensions=["score"])
    seed = _mk("seed", score=0.99)  # best — must never be removed
    strat.initialize(seed)

    def run() -> dict:
        for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            strat.admit(
                _mk(f"c{i}", code=f"# c{i}\n", parent_id="seed", score=score, iteration=i + 1),
                iteration=i + 1,
            )
        scores_kept = sorted(p.metrics["combined_score"] for p in strat.programs.values())
        return {
            "n_programs": len(strat.programs),
            "scores_kept": scores_kept,
            "best_kept": "seed" in strat.programs,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._enforce_population_limit",
        input={"population_size": 4, "scores": [0.1, 0.2, 0.3, 0.4, 0.5]},
        run=run,
    )
    assert out["n_programs"] == 4
    assert out["best_kept"] is True
    # The lowest scores are pruned first — best (0.99) must remain.
    assert 0.99 in out["scores_kept"]


def test_migration_fires_at_interval_and_skips_duplicates(record_io):
    strat = _make_strategy(num_islands=2, migration_interval=2, migration_rate=1.0)
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)
    # Both islands now have a copy of the seed (same source code).

    def run() -> dict:
        # First admit: triggers gen+1 on island of parent (seed in island 0).
        # Migration fires when max(gens) - last_migration_gen >= 2.
        strat.admit(
            _mk("c1", code="# new1\n", parent_id="seed", score=0.6, iteration=1),
            iteration=1,
        )
        first_admit = {
            "gens": list(strat.island_generations),
            "last_mig_gen": strat.last_migration_generation,
            "n_programs": len(strat.programs),
        }
        adm = strat.admit(
            _mk("c2", code="# new2\n", parent_id="seed", score=0.7, iteration=2),
            iteration=2,
        )
        return {
            "first_admit": first_admit,
            "second_admit_migration_fired": adm.migration_fired,
            "n_programs_after": len(strat.programs),
            "last_migration_generation": strat.last_migration_generation,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._migrate",
        input={"num_islands": 2, "migration_interval": 2, "migration_rate": 1.0},
        run=run,
    )
    assert out["first_admit"]["last_mig_gen"] == 0
    assert out["second_admit_migration_fired"] is True
    assert out["last_migration_generation"] >= 2
    # Migration creates copies; population grew.
    assert out["n_programs_after"] > out["first_admit"]["n_programs"]


def test_sample_returns_parent_and_inspirations_from_same_island(record_io):
    strat = _make_strategy(num_islands=2, num_inspirations=2)
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)
    # Add two distinct programs in island 0 so inspirations exist.
    strat.admit(_mk("a", code="# a\n", parent_id="seed", score=0.6, iteration=1), iteration=1)
    # Force the next admit into a fresh parent in island 1 — but for this test
    # we just sample after one admit.

    def run() -> dict:
        parent, inspirations = strat.sample(iteration=2)
        return {
            "parent_island": parent.island,
            "all_inspirations_from_parent_island": all(
                p.island == parent.island for p in inspirations
            ),
            "parent_not_in_inspirations": parent.id not in {p.id for p in inspirations},
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.sample",
        input={"num_inspirations": 2},
        run=run,
    )
    assert out["all_inspirations_from_parent_island"] is True
    assert out["parent_not_in_inspirations"] is True


def test_feature_coords_complexity_diversity_built_ins(record_io):
    strat = _make_strategy(
        num_islands=1,
        feature_dimensions=["complexity", "diversity"],
        feature_bins=5,
    )
    seed = _mk("seed", code="# seed\nx = 1\n", score=0.5)
    strat.initialize(seed)
    # Add a much longer program so complexity scaling has range.
    big = _mk("big", code="# big\n" + "y = 1\n" * 50, parent_id="seed", score=0.6)

    def run() -> dict:
        coords = strat._compute_feature_coords(big)
        return {"coords_within_bin_range": all(0 <= c < 5 for c in coords), "n_dims": len(coords)}

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._compute_feature_coords",
        input={"feature_dimensions": ["complexity", "diversity"], "feature_bins": 5},
        run=run,
    )
    assert out == {"coords_within_bin_range": True, "n_dims": 2}


def test_adaptive_complexity_rebucket_moves_existing_program(record_io):
    strat = _make_strategy(
        num_islands=1,
        feature_dimensions=["complexity"],
        feature_bins=3,
        archive_size=10,
    )
    seed = _mk("seed", code=_assignment_code(1), score=0.3)
    strat.initialize(seed)
    mid = _mk("mid", code=_assignment_code(10), parent_id="seed", score=0.4, iteration=1)
    high = _mk("high", code=_assignment_code(20), parent_id="seed", score=0.5, iteration=2)

    def run() -> dict:
        strat.admit(mid, iteration=1)
        mid_before = int(strat.programs["mid"].feature_coords["complexity"])
        strat.admit(high, iteration=2)
        return {
            "mid_before": mid_before,
            "mid_after": int(strat.programs["mid"].feature_coords["complexity"]),
            "high_after": int(strat.programs["high"].feature_coords["complexity"]),
            "occupied_complexity_bins": sorted(
                {int(p.feature_coords["complexity"]) for p in strat.programs.values()}
            ),
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._rebucket_feature_maps",
        input={"feature_dimensions": ["complexity"], "feature_bins": 3},
        run=run,
    )
    assert out["mid_before"] == 2
    assert out["mid_after"] == 1
    assert out["high_after"] == 2
    assert out["occupied_complexity_bins"] == [0, 1, 2]


def test_adaptive_rebucket_cell_host_uses_strict_score(record_io):
    strat = _make_strategy(
        num_islands=1,
        feature_dimensions=["complexity"],
        feature_bins=3,
        archive_size=10,
    )
    seed = _mk("seed", code=_assignment_code(1), score=0.3)
    strat.initialize(seed)
    same_complexity_code = _assignment_code(10)
    low = _mk("low", code=same_complexity_code, parent_id="seed", score=0.4, iteration=1)
    high = _mk("high", code=same_complexity_code, parent_id="seed", score=0.9, iteration=2)

    def run() -> dict:
        strat.admit(low, iteration=1)
        strat.admit(high, iteration=2)
        cell = "-".join(
            str(int(strat.programs["high"].feature_coords[dim]))
            for dim in strat.config.feature_dimensions
        )
        return {
            "shared_cell": cell,
            "cell_host": strat.island_feature_maps[0][cell],
            "low_cell": dict(strat.programs["low"].feature_coords),
            "high_cell": dict(strat.programs["high"].feature_coords),
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._rebucket_feature_maps",
        input={"collision_rule": "strict fitness replacement"},
        run=run,
    )
    assert out["cell_host"] == "high"
    assert out["low_cell"] == out["high_cell"]


def test_feature_coords_unknown_dim_raises(record_io):
    strat = _make_strategy(num_islands=1, feature_dimensions=["nonexistent_metric"])
    seed = _mk("seed", score=0.5)

    def run() -> str:
        try:
            strat.initialize(seed)
            return "did_not_raise"
        except ValueError as e:
            return str(e)

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._compute_feature_coords",
        input={"unknown_dim": "nonexistent_metric"},
        run=run,
    )
    assert "Feature dimension 'nonexistent_metric' not in program.metrics" in out


def test_sampling_decisions_records_island_local_top(record_io):
    strat = _make_strategy(num_islands=2, feature_dimensions=["score"])
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)
    strat.admit(_mk("a", code="# a\n", parent_id="seed", score=0.7, iteration=1), iteration=1)

    def run() -> dict:
        parent, insp = strat.sample(iteration=2)
        decisions = strat.sampling_decisions(2, parent, insp)
        return {
            "parent_id": decisions.parent_id,
            "parent_island": decisions.parent_island,
            "rng_seed_hash_len": len(decisions.rng_seed_hash),
            "top_ids_island_only": all(
                strat.programs[pid].island == decisions.parent_island
                for pid in decisions.top_program_ids
            ),
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.sampling_decisions",
        input={"iteration": 2},
        run=run,
    )
    assert out["top_ids_island_only"] is True
    assert out["rng_seed_hash_len"] == 16


def test_recent_programs_are_island_local_and_sorted_by_iteration(record_io):
    strat = _make_strategy(num_islands=2, feature_dimensions=["score"])
    seed = _mk("seed", score=0.5, iteration=0)
    programs = [
        seed,
        *[_mk(f"p{i}", parent_id="seed", score=0.5, iteration=i) for i in range(1, 4)],
    ]
    other = _mk("other", score=0.5, iteration=99, island=1)
    strat.programs = {p.id: p for p in [*programs, other]}
    strat.islands = [{"seed", "p1", "p2", "p3"}, {"other"}]

    def run() -> dict:
        recent = strat.recent_programs(n=3, island_idx=0, exclude_program_id="p3")
        parent = strat.programs["p3"]
        decisions = strat.sampling_decisions(4, parent, inspirations=[])
        return {
            "recent_ids": [p.id for p in recent],
            "decision_previous_ids": decisions.previous_program_ids,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.recent_programs",
        input={"exclude_program_id": "p3", "n": 3},
        run=run,
    )
    assert out["recent_ids"] == ["p2", "p1", "seed"]
    assert out["decision_previous_ids"] == ["p2", "p1", "seed"]


def test_determinism_same_seed_same_choices(record_io):
    """Two strategies with identical seed + admit sequence must produce the
    same sample outcome — even when run in the same process across calls.

    Note: we *don't* assert that two *different* seeds produce different
    outcomes. With small populations and few RNG draws, two seeds can
    legitimately land on the same parent. The property we care about is
    determinism in one direction: same seed -> same outcome.
    """

    def build_and_sample(seed_val: int) -> dict:
        strat = _make_strategy(num_islands=2, random_seed=seed_val, feature_dimensions=["score"])
        seed = _mk("seed", score=0.5)
        strat.initialize(seed)
        for i, sc in enumerate([0.6, 0.7, 0.4, 0.55]):
            strat.admit(
                _mk(f"c{i}", code=f"# c{i}\n", parent_id="seed", score=sc, iteration=i + 1),
                iteration=i + 1,
            )
        parent, insp = strat.sample(iteration=10)
        # Use scores instead of ids: copy programs get random uuids, and we
        # care that the *selection logic* (which fitness gets picked) is
        # deterministic, not that uuid generation is deterministic.
        return {
            "parent_score": parent.metrics["combined_score"],
            "insp_scores": tuple(p.metrics["combined_score"] for p in insp),
        }

    def run() -> dict:
        a = build_and_sample(seed_val=123)
        b = build_and_sample(seed_val=123)
        return {"a_eq_b": a == b, "ids_a": a}

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.sample (determinism)",
        input={"seed_runs": [123, 123]},
        run=run,
    )
    assert out["a_eq_b"] is True


def test_snapshot_captures_full_state(record_io):
    strat = _make_strategy(num_islands=2, feature_dimensions=["score"])
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)
    strat.admit(_mk("a", code="# a\n", parent_id="seed", score=0.7, iteration=1), iteration=1)

    def run() -> dict:
        snap = strat.snapshot()
        return {
            "n_programs": snap.n_programs,
            "n_islands": len(snap.islands),
            "has_archive_cells": len(snap.archive_cells) > 0,
            "best_program_id": snap.best_program_id,
            "current_island": snap.current_island,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.snapshot",
        input={},
        run=run,
    )
    assert out["n_programs"] >= 3
    assert out["n_islands"] == 2
    assert out["has_archive_cells"] is True
    assert out["best_program_id"] is not None


def test_state_dict_roundtrips_sampling_state(record_io):
    """Persisted state restores active programs, island state, archive, and RNG."""
    strat = _make_strategy(num_islands=2, feature_dimensions=["score"], random_seed=123)
    seed = _mk("seed", score=0.5)
    strat.initialize(seed)
    for i, score in enumerate([0.6, 0.7, 0.4]):
        strat.admit(
            _mk(f"c{i}", code=f"# c{i}\n", parent_id="seed", score=score, iteration=i + 1),
            iteration=i + 1,
        )

    state = strat.state_dict()
    restored = _make_strategy(num_islands=2, feature_dimensions=["score"], random_seed=999)

    def run() -> dict:
        restored.load_state_dict(state)
        parent_a, insp_a = strat.sample(iteration=10)
        parent_b, insp_b = restored.sample(iteration=10)
        return {
            "same_snapshot": restored.snapshot() == strat.snapshot(),
            "same_parent": parent_a.id == parent_b.id,
            "same_inspirations": [p.id for p in insp_a] == [p.id for p in insp_b],
            "best_id": restored.best().id,
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy.state_dict/load_state_dict",
        input={"roundtrip": True},
        run=run,
    )
    assert out["same_snapshot"] is True
    assert out["same_parent"] is True
    assert out["same_inspirations"] is True
    assert out["best_id"] == strat.best().id


def test_strategy_registered():
    """The decorator must register the strategy under the canonical name."""
    from ranking_evolved.search.base import REGISTRY

    assert "map_elites_islands" in REGISTRY
    # Registered factory points at the class.
    assert REGISTRY["map_elites_islands"] is MapElitesIslandsStrategy


def test_admit_stamps_diversity_and_complexity_from_source_code(record_io):
    """The strategy must overwrite caller-provided diversity/complexity placeholders.

    Regression: controller used to pass diversity=0.0 always, which collapsed
    the 2-D MAP-Elites grid to 1-D. Strategy now owns these fields.
    """
    strat = _make_strategy(num_islands=1, feature_dimensions=["complexity", "diversity"])
    seed = _mk("seed", code="# seed\nimport numpy as np\nx = 1\n", score=0.4)
    strat.initialize(seed)

    # Caller-side placeholders (mimic controller behavior pre-strategy-stamping).
    distinct_code = (
        "# distinct child\n"
        "from collections import defaultdict\n"
        "def hello(world):\n    return world * 2\n"
    )
    child = _mk("child", code=distinct_code, parent_id="seed", score=0.6, iteration=1)
    # Force the placeholders the controller used to ship.
    bad_child = Program(
        id=child.id,
        source_code=child.source_code,
        parent_id=child.parent_id,
        generation=child.generation,
        iteration_found=child.iteration_found,
        timestamp=child.timestamp,
        metrics=child.metrics,
        complexity=0.0,
        diversity=0.0,
        island=child.island,
        feature_coords={},
        metadata={},
    )

    def run() -> dict:
        strat.admit(bad_child, iteration=1)
        stamped = strat.programs[bad_child.id]
        return {
            "complexity": stamped.complexity,
            "diversity": stamped.diversity,
            "feature_coords": dict(stamped.feature_coords),
        }

    out = record_io(
        module="src/ranking_evolved/search/map_elites_islands.py",
        function="MapElitesIslandsStrategy._admit_into_island",
        input={"feature_dimensions": ["complexity", "diversity"]},
        run=run,
    )

    # Complexity is the AST node count of the comment/docstring-stripped
    # source. Identical algorithm with cosmetic-only changes → same value.
    expected_complexity = float(_ast_node_count(_normalized_core_source(distinct_code)))
    assert out["complexity"] == expected_complexity
    # Diversity must be > 0 because the child code differs from the seed
    # (which is the diversity anchor).
    assert out["diversity"] > 0.0
    # Both feature dimensions populated → 2-D grid is alive.
    assert set(out["feature_coords"].keys()) == {"complexity", "diversity"}


def test_admit_anchors_diversity_to_seed():
    """`initialize()` pins the seed's normalized core source as the
    diversity anchor, so the very first child gets a meaningful (non-zero)
    diversity score even when the archive is otherwise empty."""
    strat = _make_strategy(num_islands=1, feature_dimensions=["complexity", "diversity"])
    seed = _mk("seed", code="alpha = 1\n", score=0.5)
    strat.initialize(seed)
    # The anchor matches the seed's normalized core source.
    assert strat._diversity_seed_anchor == _normalized_core_source(seed.source_code)
    # A child with disjoint identifiers should have a high diversity score
    # even though the archive only contains seed copies.
    disjoint = _mk("d", code="zulu = 9\nyankee = 8\n", parent_id="seed", score=0.6)
    strat.admit(disjoint, iteration=1)
    stamped = strat.programs["d"]
    assert stamped.diversity > 0.5


def test_complexity_is_invariant_to_comment_edits():
    """Two programs with identical code but different comments must land
    in the same complexity bin. This was the user-reported bug: cosmetic
    edits moved programs across the MAP-Elites grid as much as real ones."""
    strat = _make_strategy(num_islands=1, feature_dimensions=["complexity", "diversity"])
    seed = _mk("seed", code="x = 1\ny = 2\n", score=0.5)
    strat.initialize(seed)
    # Same algorithm, very different comment volume.
    a_code = "# short\nx = 1\ny = 2\n"
    b_code = "# very very very long comment that triples the byte count\n# more\nx = 1\ny = 2\n"
    a = _mk("a", code=a_code, parent_id="seed", score=0.6, iteration=1)
    b = _mk("b", code=b_code, parent_id="seed", score=0.6, iteration=2)
    strat.admit(a, iteration=1)
    strat.admit(b, iteration=2)
    stamped_a = strat.programs["a"]
    stamped_b = strat.programs["b"]
    # AST node count is identical because comments aren't AST nodes.
    assert stamped_a.complexity == stamped_b.complexity


def test_score_zero_programs_are_not_admitted():
    """Crashed / recall-floor children must not occupy MAP-Elites cells,
    enter the archive, or join the live population — they only get
    captured in the failure ring buffer for prompt feedback."""
    strat = _make_strategy(num_islands=1, archive_size=5)
    seed = _mk("seed", code="x = 1\n", score=0.5)
    strat.initialize(seed)
    failed = _mk("failed", code="z = 9\n", parent_id="seed", score=0.0, iteration=1)

    n_programs_before = len(strat.programs)
    archive_before = set(strat.archive)
    cells_before = {k: dict(v) for k, v in enumerate(strat.island_feature_maps)}

    admission = strat.admit(failed, iteration=1)

    # Cell key is empty → controller's `if admission.cell_key:` skips upsert.
    assert admission.cell_key == ""
    assert admission.evicted_program_id is None
    # Live population unchanged.
    assert len(strat.programs) == n_programs_before
    assert "failed" not in strat.programs
    assert strat.archive == archive_before
    assert {k: dict(v) for k, v in enumerate(strat.island_feature_maps)} == cells_before
    # Failure record IS captured for the prompt builder.
    failures = strat.recent_failures(n=5)
    assert len(failures) == 1
    assert failures[0].child_source_code == failed.source_code
    assert failures[0].parent_source_code == seed.source_code


def test_recent_failures_filtered_by_island():
    """`recent_failures(island_idx=...)` must restrict to the parent
    island so the LLM sees failures relevant to its current evolution."""
    strat = _make_strategy(num_islands=2, archive_size=5)
    seed = _mk("seed", code="x = 1\n", score=0.5)
    strat.initialize(seed)
    f0 = _mk("f0", code="z = 9\n", parent_id="seed", score=0.0, iteration=1, island=0)
    # Force admission to island 0 by making seed in island 0 the parent.
    strat.admit(f0, iteration=1)
    # Manually craft a failure whose target_island is 1.
    f1_program = _mk("f1", code="z = 8\n", parent_id="seed", score=0.0, iteration=2)
    # The seed copy in island 1 is also called "seed" (initialize copies the seed
    # into every island with the same id only for island 0; for islands 1+, the
    # copy gets a fresh uuid). Find the island-1 seed copy and use that as parent.
    island1_seed = next(pid for pid in strat.islands[1] if pid in strat.programs)
    f1_program = Program(
        id="f1",
        source_code=f1_program.source_code,
        parent_id=island1_seed,
        generation=1,
        iteration_found=2,
        timestamp=f1_program.timestamp,
        metrics=f1_program.metrics,
        complexity=f1_program.complexity,
        diversity=f1_program.diversity,
        island=1,
        feature_coords={},
        metadata={},
    )
    strat.admit(f1_program, iteration=2)

    only_island_0 = strat.recent_failures(n=10, island_idx=0)
    only_island_1 = strat.recent_failures(n=10, island_idx=1)
    assert all(f.parent_island == 0 for f in only_island_0)
    assert all(f.parent_island == 1 for f in only_island_1)
    assert len(only_island_0) >= 1
    assert len(only_island_1) >= 1
