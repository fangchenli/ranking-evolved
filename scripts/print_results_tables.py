#!/usr/bin/env python3
"""
Print 6 benchmark tables + 4 overall tables from results/*.json.
Numbers as percentage (×100, 2 decimals). Best=bold bright green, 2nd=light yellow.
Overall: avg nDCG@10, avg Recall@100, avg(R,nDCG), optimization target (0.8×R+0.2×nDCG).

Tie handling:
- If multiple runs tie for best (after rounding to the displayed 2 decimals), they all get GREEN.
- If multiple runs tie for second best, they all get YELLOW.
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

# ANSI: best = bold bright green, 2nd = light yellow
GREEN = "\033[1;92m"  # bold bright green
YELLOW = "\033[93m"   # bright / light yellow
RESET = "\033[0m"

# Column width: "100.00" = 6 chars
NUM_W = 6
ROW_W = 28

# Tie key: match what is displayed (percent, 2 decimals)
DISPLAY_DECIMALS = 2


def _display_key(v: float) -> float:
    """Key used for ranking/highlighting: percent value rounded to the displayed precision."""
    return round(v * 100.0, DISPLAY_DECIMALS)


def _mark_best_and_second(values_by_row: list[float | None]) -> tuple[list[str], list[str]]:
    """
    Given column values for each row, return ANSI prefix/suffix arrays.
    Ties are based on displayed (rounded) values.
    """
    n = len(values_by_row)
    prefix = [""] * n
    suffix = [""] * n

    keyed = []
    for i, v in enumerate(values_by_row):
        if v is None:
            continue
        keyed.append((i, _display_key(float(v))))

    if not keyed:
        return prefix, suffix

    # Find best and second-best DISTINCT keys (descending)
    unique_scores = sorted({score for _, score in keyed}, reverse=True)
    best_score = unique_scores[0]
    second_score = unique_scores[1] if len(unique_scores) >= 2 else None

    for i, score in keyed:
        if score == best_score:
            prefix[i], suffix[i] = GREEN, RESET
        elif second_score is not None and score == second_score:
            prefix[i], suffix[i] = YELLOW, RESET

    return prefix, suffix


def main() -> None:
    only_unseen = "--only-unseen" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--only-unseen"]
    name_filter = args[0].lower() if args else None

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    json_files = sorted(
        f for f in results_dir.iterdir()
        if f.suffix == ".json" and f.is_file()
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

    if not all_data:
        return

    # Filter: exclude files with >2 datasets having 0.0 values OR too few datasets evaluated
    filtered_data: dict[str, dict] = {}
    for fname, data in all_data.items():
        zero_count = 0
        for key in data:
            if key.endswith("_ndcg@10"):
                val = data.get(key)
                if val is not None and float(val) == 0.0:
                    zero_count += 1
        datasets_evaluated = data.get("datasets_evaluated", 0)
        # Exclude if >2 zeros OR if evaluated < 20 datasets (incomplete evaluation)
        if zero_count <= 5 and datasets_evaluated >= 20:
            filtered_data[fname] = data

    all_data = filtered_data
    if not all_data:
        print("No valid result files (all filtered out)")
        return

    bright_prefixes: set[str] = set()
    beir_prefixes: set[str] = set()
    dl_prefixes: set[str] = set()
    for data in all_data.values():
        for key in data:
            if key.endswith("_ndcg@10"):
                prefix = key[: -len("_ndcg@10")]
                if prefix.startswith("bright_"):
                    bright_prefixes.add(prefix)
                elif prefix.startswith("beir_"):
                    beir_prefixes.add(prefix)
                elif prefix in ("trec_dl_dl19", "trec_dl_dl20"):
                    dl_prefixes.add(prefix)

    bright_list = sorted(bright_prefixes)
    beir_list = sorted(beir_prefixes)
    dl_list = sorted(dl_prefixes)

    if only_unseen:
        bright_list = [p for p in bright_list if p not in SEEN_DATASETS]
        beir_list = [p for p in beir_list if p not in SEEN_DATASETS]
        dl_list = [p for p in dl_list if p not in SEEN_DATASETS]

    def short_name(prefix: str, benchmark: str) -> str:
        if benchmark == "bright":
            return prefix.replace("bright_", "")
        if benchmark == "beir":
            return prefix.replace("beir_", "")
        if benchmark == "dl":
            return prefix.replace("trec_dl_", "")
        return prefix

    def get_values(data: dict, prefixes: list[str], metric: str) -> tuple[list[float | None], float | None]:
        suffix = "_ndcg@10" if metric == "ndcg" else "_recall@100"
        values: list[float | None] = []
        valid: list[float] = []
        for p in prefixes:
            val = data.get(p + suffix)
            if data.get(p + "_error"):
                values.append(None)
            elif val is not None:
                v = float(val)
                values.append(v)
                valid.append(v)
            else:
                values.append(None)
        macro = sum(valid) / len(valid) if valid else None
        return values, macro

    def print_table(title: str, prefixes: list[str], benchmark: str, metric: str) -> None:
        col_names = [short_name(p, benchmark) for p in prefixes] + ["macro"]
        w = NUM_W  # fixed narrow column so table fits on screen
        file_order = sorted(all_data.keys())
        rows: list[tuple[str, list[float | None]]] = []
        for fname in file_order:
            values, macro = get_values(all_data[fname], prefixes, metric)
            rows.append((fname, values + [macro]))

        ncols = len(prefixes) + 1

        # Per-column: tie-aware best/second based on displayed values
        color_prefix: list[list[str]] = [[""] * len(file_order) for _ in range(ncols)]
        color_suffix: list[list[str]] = [[""] * len(file_order) for _ in range(ncols)]
        for j in range(ncols):
            col = [rows[i][1][j] for i in range(len(rows))]
            pfx, sfx = _mark_best_and_second(col)
            for i in range(len(rows)):
                color_prefix[j][i] = pfx[i]
                color_suffix[j][i] = sfx[i]

        print()
        sep_len = ROW_W + 1 + (w + 1) * ncols
        print("=" * sep_len)
        print(f"  {title}")
        print("=" * sep_len)
        header = "".join((c[:w] if len(c) > w else c).ljust(w + 1) for c in col_names)
        print(" " * (ROW_W + 1) + header)
        print("-" * sep_len)

        for i, fname in enumerate(file_order):
            out_parts = []
            for j, v in enumerate(rows[i][1]):
                if v is None:
                    out_parts.append(" — ".rjust(w + 1))
                else:
                    raw = f"{v * 100:.2f}"
                    pad = max(0, (w + 1) - len(raw))
                    cell = f"{color_prefix[j][i]}{' ' * pad}{raw}{color_suffix[j][i]}"
                    out_parts.append(cell)
            print(f"  {fname[:ROW_W]:<{ROW_W}} " + "".join(out_parts))
        print()

    # Benchmark tables
    tag = " (unseen)" if only_unseen else ""
    print_table(f"BRIGHT{tag} — nDCG@10", bright_list, "bright", "ndcg")
    print_table(f"BRIGHT{tag} — Recall@100", bright_list, "bright", "recall")
    print_table(f"BEIR{tag} — nDCG@10", beir_list, "beir", "ndcg")
    print_table(f"BEIR{tag} — Recall@100", beir_list, "beir", "recall")
    print_table(f"TREC DL (dl19, dl20){tag} — nDCG@10", dl_list, "dl", "ndcg")
    print_table(f"TREC DL (dl19, dl20){tag} — Recall@100", dl_list, "dl", "recall")

    # Overall tables
    file_order = sorted(all_data.keys())
    all_prefixes = bright_list + beir_list + dl_list
    n_datasets = len(all_prefixes)

    ndcg_vals: list[float | None] = []
    recall_vals: list[float | None] = []
    for fname in file_order:
        d = all_data[fname]
        if only_unseen:
            # Recompute from filtered datasets
            _, n_macro = get_values(d, all_prefixes, "ndcg")
            _, r_macro = get_values(d, all_prefixes, "recall")
            ndcg_vals.append(n_macro)
            recall_vals.append(r_macro)
        else:
            ndcg_vals.append(d.get("avg_ndcg@10"))
            recall_vals.append(d.get("avg_recall@100"))

    def print_overall(title: str, values: list[float | None]) -> None:
        print()
        w = NUM_W
        sep_len = ROW_W + 1 + (w + 1)
        print("=" * sep_len)
        print(f"  {title}")
        print("=" * sep_len)
        print(" " * (ROW_W + 1) + "value".ljust(w + 1))
        print("-" * sep_len)

        pfx, sfx = _mark_best_and_second(values)

        for i, fname in enumerate(file_order):
            v = values[i]
            if v is None:
                cell = " — ".rjust(w + 1)
            else:
                raw = f"{v * 100:.2f}"
                pad = max(0, (w + 1) - len(raw))
                cell = f"{pfx[i]}{' ' * pad}{raw}{sfx[i]}"
            print(f"  {fname[:ROW_W]:<{ROW_W}} {cell}")
        print()

    ds_label = f"{n_datasets} unseen datasets" if only_unseen else "all datasets"
    print_overall(f"Overall — avg nDCG@10 ({ds_label})", ndcg_vals)
    print_overall(f"Overall — avg Recall@100 ({ds_label})", recall_vals)

    # avg(Recall@100, nDCG@10)
    avg_both: list[float | None] = []
    for i in range(len(file_order)):
        r, n = recall_vals[i], ndcg_vals[i]
        if r is not None and n is not None:
            avg_both.append((float(r) + float(n)) / 2.0)
        else:
            avg_both.append(None)
    print_overall(f"Overall — avg(Recall@100, nDCG@10) ({ds_label})", avg_both)

    # optimization target: 0.8*R@100 + 0.2*nDCG@10
    opt_target: list[float | None] = []
    for i in range(len(file_order)):
        r, n = recall_vals[i], ndcg_vals[i]
        if r is not None and n is not None:
            opt_target.append(0.8 * float(r) + 0.2 * float(n))
        else:
            opt_target.append(None)
    print_overall(f"Overall — optimization target (0.8×R@100 + 0.2×nDCG@10) ({ds_label})", opt_target)


if __name__ == "__main__":
    main()
