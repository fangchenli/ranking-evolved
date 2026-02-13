#!/usr/bin/env python3
"""
Print latency tables from results/*.json.

Shows index time, query time (in minutes), plus per-document indexing latency
and per-query latency (in ms) for every dataset, benchmark averages, and
overall 28-dataset averages.

Best = bold bright green, 2nd = light yellow (lower is better for latency).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Datasets used during evolution (seen); excluded with --only-unseen
SEEN_DATASETS: set[str] = {
    # BEIR
    "beir_arguana", "beir_fiqa", "beir_nfcorpus", "beir_scifact",
    "beir_scidocs", "beir_trec-covid",
    # BRIGHT
    "bright_biology", "bright_earth_science", "bright_economics",
    "bright_pony", "bright_stackoverflow",
    "bright_theoremqa_theorems",
}

# ANSI colours
GREEN = "\033[1;92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

NUM_W = 8
ROW_W = 28
DISPLAY_DECIMALS = 2

# ── Dataset sizes (docs, queries) ──────────────────────────────────────────
# Sources: BEIR corpus.jsonl/queries.jsonl line counts, BRIGHT HF arrow files,
# TREC DL ir-datasets / eval logs.
DATASET_INFO: dict[str, tuple[int, int]] = {
    # BEIR (14 datasets) — (num_docs, num_queries)
    "beir_nfcorpus":          (3_633,     3_237),
    "beir_scifact":           (5_183,     1_109),
    "beir_arguana":           (8_674,     1_406),
    "beir_scidocs":           (25_657,    1_000),
    "beir_fiqa":              (57_638,    6_648),
    "beir_trec-covid":        (171_332,   50),
    "beir_webis-touche2020":  (382_545,   49),
    "beir_cqadupstack":       (457_199,   3_026),
    "beir_quora":             (522_931,   15_000),
    "beir_nq":                (2_681_468, 3_452),
    "beir_dbpedia-entity":    (4_635_922, 467),
    "beir_hotpotqa":          (5_233_329, 97_852),
    "beir_fever":             (5_416_568, 123_142),
    "beir_climate-fever":     (5_416_593, 1_535),
    # BRIGHT (12 datasets)
    "bright_pony":                  (7_894,   112),
    "bright_biology":               (57_359,  103),
    "bright_economics":             (50_220,  103),
    "bright_psychology":            (52_835,  101),
    "bright_robotics":              (61_961,  101),
    "bright_sustainable_living":    (60_792,  108),
    "bright_earth_science":         (121_249, 116),
    "bright_stackoverflow":         (107_081, 117),
    "bright_aops":                  (188_002, 111),
    "bright_theoremqa_theorems":    (23_839,  76),
    "bright_theoremqa_questions":   (188_002, 194),
    "bright_leetcode":              (413_932, 142),
    # TREC DL (2 datasets)
    "trec_dl_dl19": (8_841_823, 200),
    "trec_dl_dl20": (8_841_823, 200),
}

BENCHMARKS = [
    ("bright", "BRIGHT"),
    ("beir",   "BEIR"),
    ("dl",     "TREC DL"),
]


def _display_key(v: float) -> float:
    return round(v, DISPLAY_DECIMALS)


def _mark_best_and_second(values: list[float | None], lower_is_better: bool = True) -> tuple[list[str], list[str]]:
    """Return ANSI prefix/suffix for best (green) and 2nd-best (yellow). Lower is better for latency."""
    n = len(values)
    prefix = [""] * n
    suffix = [""] * n
    keyed = [(i, _display_key(float(v))) for i, v in enumerate(values) if v is not None]
    if not keyed:
        return prefix, suffix
    unique = sorted({s for _, s in keyed}, reverse=not lower_is_better)
    best = unique[0]
    second = unique[1] if len(unique) >= 2 else None
    for i, score in keyed:
        if score == best:
            prefix[i], suffix[i] = GREEN, RESET
        elif second is not None and score == second:
            prefix[i], suffix[i] = YELLOW, RESET
    return prefix, suffix


def _fmt(v: float | None, w: int = NUM_W, decimals: int = 2) -> str:
    if v is None:
        return " — ".rjust(w)
    raw = f"{v:.{decimals}f}"
    return raw.rjust(w)


def _fmt_colored(v: float | None, pfx: str, sfx: str, w: int = NUM_W, decimals: int = 2) -> str:
    if v is None:
        return " — ".rjust(w + 1)
    raw = f"{v:.{decimals}f}"
    pad = max(0, (w + 1) - len(raw))
    return f"{pfx}{' ' * pad}{raw}{sfx}"


def _get_timing(data: dict, dataset_prefix: str, kind: str) -> float | None:
    """Get index_time_ms or query_time_ms for a dataset. Returns ms or None."""
    key = f"{dataset_prefix}_{kind}_time_ms"
    val = data.get(key)
    if val is not None:
        return float(val)
    return None


def _classify(prefix: str) -> str:
    if prefix.startswith("bright_"):
        return "bright"
    if prefix.startswith("beir_"):
        return "beir"
    if prefix.startswith("trec_dl_"):
        return "dl"
    return "unknown"


def _short(prefix: str) -> str:
    for p in ("bright_", "beir_", "trec_dl_"):
        if prefix.startswith(p):
            return prefix[len(p):]
    return prefix


def main() -> None:
    only_unseen = "--only-unseen" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--only-unseen"]
    name_filter = args[0].lower() if args else None

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    json_files = sorted(
        f for f in results_dir.iterdir()
        if f.suffix == ".json" and f.is_file()
        and "perquery" not in f.name.lower()
        and (name_filter is None or name_filter in f.name.lower())
    )

    if not json_files:
        print("No JSON files in results/")
        return

    all_data: dict[str, dict] = {}
    for f in json_files:
        try:
            with open(f) as fp:
                all_data[f.name] = json.load(fp)
        except Exception as e:
            print(f"Skip {f.name}: {e}")

    # Apply same filter as print_results_tables.py
    filtered: dict[str, dict] = {}
    for fname, data in all_data.items():
        zero_count = sum(
            1 for k in data if k.endswith("_ndcg@10")
            and data.get(k) is not None and float(data[k]) == 0.0
        )
        if zero_count <= 5 and data.get("datasets_evaluated", 0) >= 20:
            filtered[fname] = data
    all_data = filtered
    if not all_data:
        print("No valid result files")
        return

    # Discover all dataset prefixes from timing keys
    all_prefixes: set[str] = set()
    for data in all_data.values():
        for key in data:
            if key.endswith("_index_time_ms"):
                all_prefixes.add(key[: -len("_index_time_ms")])
            elif key.endswith("_query_time_ms"):
                all_prefixes.add(key[: -len("_query_time_ms")])

    # Group by benchmark
    bench_prefixes: dict[str, list[str]] = {"bright": [], "beir": [], "dl": []}
    for p in sorted(all_prefixes):
        c = _classify(p)
        if c in bench_prefixes:
            bench_prefixes[c].append(p)

    if only_unseen:
        for key in bench_prefixes:
            bench_prefixes[key] = [p for p in bench_prefixes[key] if p not in SEEN_DATASETS]

    tag = " (unseen)" if only_unseen else ""
    n_datasets = sum(len(v) for v in bench_prefixes.values())
    ds_label = f"{n_datasets} unseen datasets" if only_unseen else "28 datasets"

    file_order = sorted(all_data.keys())
    w = NUM_W

    # ── Helper: print a detailed time table ─────────────────────────────────
    def print_time_table(
        title: str,
        prefixes: list[str],
        kind: str,           # "index" or "query"
        unit_label: str,     # e.g. "min" or "ms/doc"
        transform,           # callable(ms_value, dataset_prefix) -> display_value
        decimals: int = 2,
        lower_better: bool = True,
        extra_summary_col: str | None = None,  # "avg" column label
    ) -> None:
        col_names = [_short(p) for p in prefixes]
        if extra_summary_col:
            col_names.append(extra_summary_col)
        ncols = len(col_names)

        # Build rows
        rows: list[list[float | None]] = []
        for fname in file_order:
            data = all_data[fname]
            vals: list[float | None] = []
            valid: list[float] = []
            for p in prefixes:
                raw_ms = _get_timing(data, p, kind)
                if raw_ms is not None and raw_ms > 0:
                    v = transform(raw_ms, p)
                    vals.append(v)
                    valid.append(v)
                else:
                    vals.append(None)
            if extra_summary_col:
                vals.append(sum(valid) / len(valid) if valid else None)
            rows.append(vals)

        # Per-column colouring
        cpfx = [[""] * len(file_order) for _ in range(ncols)]
        csfx = [[""] * len(file_order) for _ in range(ncols)]
        for j in range(ncols):
            col = [rows[i][j] for i in range(len(rows))]
            p, s = _mark_best_and_second(col, lower_is_better=lower_better)
            for i in range(len(rows)):
                cpfx[j][i] = p[i]
                csfx[j][i] = s[i]

        # Print
        col_w = max(w, max((len(c) for c in col_names), default=w))
        sep_len = ROW_W + 1 + (col_w + 1) * ncols
        print()
        print("=" * sep_len)
        print(f"  {title}  ({unit_label})")
        print("=" * sep_len)
        header = "".join(
            (c[:col_w] if len(c) > col_w else c).rjust(col_w + 1)
            for c in col_names
        )
        print(" " * ROW_W + header)
        print("-" * sep_len)
        for i, fname in enumerate(file_order):
            parts = []
            for j, v in enumerate(rows[i]):
                parts.append(_fmt_colored(v, cpfx[j][i], csfx[j][i], w=col_w, decimals=decimals))
            print(f"  {fname[:ROW_W]:<{ROW_W}}" + "".join(parts))
        print()

    # ── Helper: print a single-column summary table ─────────────────────────
    def print_summary_table(title: str, values: list[float | None], unit: str, decimals: int = 2, lower_better: bool = True) -> None:
        print()
        col_w = max(w, len("value"))
        sep_len = ROW_W + 1 + col_w + 1
        print("=" * sep_len)
        print(f"  {title}  ({unit})")
        print("=" * sep_len)
        print(" " * (ROW_W + 1) + "value".rjust(col_w))
        print("-" * sep_len)
        pfx, sfx = _mark_best_and_second(values, lower_is_better=lower_better)
        for i, fname in enumerate(file_order):
            cell = _fmt_colored(values[i], pfx[i], sfx[i], w=col_w, decimals=decimals)
            print(f"  {fname[:ROW_W]:<{ROW_W}} {cell}")
        print()

    # =====================================================================
    #  SECTION 1: Total time per dataset (minutes)
    # =====================================================================
    ms_to_min = lambda ms, _p: ms / 60_000.0

    for bench_key, bench_label in BENCHMARKS:
        prefs = bench_prefixes.get(bench_key, [])
        if not prefs:
            continue
        print_time_table(
            f"{bench_label}{tag} — Index Time", prefs, "index",
            "minutes", ms_to_min, decimals=2, extra_summary_col="avg",
        )
        print_time_table(
            f"{bench_label}{tag} — Query Time", prefs, "query",
            "minutes", ms_to_min, decimals=2, extra_summary_col="avg",
        )

    # ── Benchmark-level and overall summaries (minutes) ──────────────────
    for kind, kind_label in [("index", "Index"), ("query", "Query")]:
        bench_avgs: dict[str, list[float | None]] = {}
        for bench_key, bench_label in BENCHMARKS:
            prefs = bench_prefixes.get(bench_key, [])
            if not prefs:
                continue
            avgs: list[float | None] = []
            for fname in file_order:
                data = all_data[fname]
                valid = []
                for p in prefs:
                    raw = _get_timing(data, p, kind)
                    if raw is not None and raw > 0:
                        valid.append(raw / 60_000.0)
                avgs.append(sum(valid) / len(valid) if valid else None)
            bench_avgs[bench_key] = avgs
            print_summary_table(
                f"{bench_label}{tag} — avg {kind_label} Time per dataset",
                avgs, "minutes",
            )

        # Overall average across all 28 datasets
        overall: list[float | None] = []
        for i in range(len(file_order)):
            all_vals = []
            for bench_key, _ in BENCHMARKS:
                prefs = bench_prefixes.get(bench_key, [])
                data = all_data[file_order[i]]
                for p in prefs:
                    raw = _get_timing(data, p, kind)
                    if raw is not None and raw > 0:
                        all_vals.append(raw / 60_000.0)
            overall.append(sum(all_vals) / len(all_vals) if all_vals else None)
        print_summary_table(
            f"Overall ({ds_label}) — avg {kind_label} Time per dataset",
            overall, "minutes",
        )

    # ── Total wall-clock time (minutes) ──────────────────────────────────
    total_mins: list[float | None] = []
    for fname in file_order:
        d = all_data[fname]
        t = d.get("total_time_ms")
        total_mins.append(float(t) / 60_000.0 if t is not None else None)
    print_summary_table(f"Total evaluation time (index + query, {ds_label})", total_mins, "minutes")

    # =====================================================================
    #  SECTION 2: Per-document indexing latency (ms/doc)
    # =====================================================================
    def ms_per_doc(ms: float, prefix: str) -> float | None:
        info = DATASET_INFO.get(prefix)
        if info is None or info[0] == 0:
            return None
        return ms / info[0]

    for bench_key, bench_label in BENCHMARKS:
        prefs = bench_prefixes.get(bench_key, [])
        if not prefs:
            continue
        print_time_table(
            f"{bench_label}{tag} — Per-Document Indexing Latency", prefs, "index",
            "ms/doc", ms_per_doc, decimals=4, extra_summary_col="avg",
        )

    # Benchmark and overall summaries for ms/doc
    for bench_key, bench_label in BENCHMARKS:
        prefs = bench_prefixes.get(bench_key, [])
        if not prefs:
            continue
        avgs: list[float | None] = []
        for fname in file_order:
            data = all_data[fname]
            valid = []
            for p in prefs:
                raw = _get_timing(data, p, "index")
                if raw is not None and raw > 0:
                    v = ms_per_doc(raw, p)
                    if v is not None:
                        valid.append(v)
            avgs.append(sum(valid) / len(valid) if valid else None)
        print_summary_table(f"{bench_label}{tag} — avg Per-Doc Indexing Latency", avgs, "ms/doc", decimals=4)

    # Overall avg ms/doc
    overall_msdoc: list[float | None] = []
    for i, fname in enumerate(file_order):
        data = all_data[fname]
        valid = []
        for prefs in bench_prefixes.values():
            for p in prefs:
                raw = _get_timing(data, p, "index")
                if raw is not None and raw > 0:
                    v = ms_per_doc(raw, p)
                    if v is not None:
                        valid.append(v)
        overall_msdoc.append(sum(valid) / len(valid) if valid else None)
    print_summary_table(f"Overall ({ds_label}) — avg Per-Doc Indexing Latency", overall_msdoc, "ms/doc", decimals=4)

    # =====================================================================
    #  SECTION 3: Per-query latency (ms/query)
    # =====================================================================
    def ms_per_query(ms: float, prefix: str) -> float | None:
        info = DATASET_INFO.get(prefix)
        if info is None or info[1] == 0:
            return None
        return ms / info[1]

    for bench_key, bench_label in BENCHMARKS:
        prefs = bench_prefixes.get(bench_key, [])
        if not prefs:
            continue
        print_time_table(
            f"{bench_label}{tag} — Per-Query Latency", prefs, "query",
            "ms/query", ms_per_query, decimals=2, extra_summary_col="avg",
        )

    # Benchmark and overall summaries for ms/query
    for bench_key, bench_label in BENCHMARKS:
        prefs = bench_prefixes.get(bench_key, [])
        if not prefs:
            continue
        avgs: list[float | None] = []
        for fname in file_order:
            data = all_data[fname]
            valid = []
            for p in prefs:
                raw = _get_timing(data, p, "query")
                if raw is not None and raw > 0:
                    v = ms_per_query(raw, p)
                    if v is not None:
                        valid.append(v)
            avgs.append(sum(valid) / len(valid) if valid else None)
        print_summary_table(f"{bench_label}{tag} — avg Per-Query Latency", avgs, "ms/query")

    # Overall avg ms/query
    overall_msq: list[float | None] = []
    for i, fname in enumerate(file_order):
        data = all_data[fname]
        valid = []
        for prefs in bench_prefixes.values():
            for p in prefs:
                raw = _get_timing(data, p, "query")
                if raw is not None and raw > 0:
                    v = ms_per_query(raw, p)
                    if v is not None:
                        valid.append(v)
        overall_msq.append(sum(valid) / len(valid) if valid else None)
    print_summary_table(f"Overall ({ds_label}) — avg Per-Query Latency", overall_msq, "ms/query")


if __name__ == "__main__":
    main()
