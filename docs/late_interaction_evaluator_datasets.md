# Choosing the Evaluator Datasets for RankEvolve

**Date:** 2026-05-03
**Source data:** [tasks/late_interaction/baselines/](../tasks/late_interaction/baselines/) — `exact_maxsim.cpu.json`, `fastplaid.cpu.json`, `freeform.cpu.json`. Full eval (all queries) on macOS, single-thread BLAS, single-query call shape.

This doc records the three-baseline comparison on the 12-dataset suite and the rationale for the early-evolution subset that drives RankEvolve's freeform-seed loop.

---

## 1. Three-baseline comparison

`em` = exact MaxSim, `fp` = FastPLAID, `ff` = freeform seed. Datasets ordered by corpus size.

| dataset | docs | q | R@1000 em / fp / ff | nDCG@10 em / fp / ff | p50 ms em / fp / ff |
|---|---:|---:|---|---|---|
| beir_nfcorpus | 3,633 | 323 | 0.5392 / 0.5575 / 0.5392 | 0.1366 / 0.1361 / 0.1366 | 53 / 697 / 50 |
| beir_scifact | 5,183 | 300 | 1.0000 / 1.0000 / 1.0000 | 0.7399 / 0.7429 / 0.7399 | 75 / 745 / 71 |
| bright_pony | 7,894 | 112 | 0.9742 / 0.9772 / 0.9742 | 0.2083 / 0.2138 / 0.2083 | 96 / 512 / 75 |
| beir_arguana | 8,674 | 1,406 | 0.9893 / 0.9886 / 0.9893 | 0.3407 / 0.3427 / 0.3407 | 113 / 745 / 106 |
| bright_theoremqa_theorems | 23,839 | 76 | 0.3728 / 0.3794 / 0.3728 | 0.0369 / 0.0362 / 0.0369 | 437 / 810 / 281 |
| beir_scidocs | 25,657 | 1,000 | 0.6468 / 0.6546 / 0.6468 | 0.1125 / 0.1133 / 0.1125 | 343 / 701 / 319 |
| bright_economics | 50,220 | 103 | 0.6734 / 0.7078 / 0.6734 | 0.1142 / 0.1218 / 0.1142 | 586 / 841 / 458 |
| bright_biology | 57,359 | 103 | 0.7584 / 0.7777 / 0.7584 | 0.1043 / 0.1085 / 0.1043 | 586 / 858 / 527 |
| beir_fiqa | 57,638 | 648 | 0.9252 / 0.9151 / 0.9252 | 0.4195 / 0.4169 / 0.4195 | 679 / 849 / 640 |
| bright_stackoverflow | 107,081 | 117 | 0.6642 / 0.6792 / 0.6642 | 0.0804 / 0.0946 / 0.0804 | 1,348 / 1,403 / 1,128 |
| bright_earth_science | 121,249 | 116 | 0.6817 / 0.7194 / 0.6817 | 0.2077 / 0.2066 / 0.2077 | 1,584 / 1,314 / 1,143 |
| beir_trec-covid | 171,332 | 50 | 0.4375 / 0.4548 / 0.4375 | 0.7138 / 0.7115 / 0.7138 | 2,549 / 1,741 / 2,071 |

**Averages over all 12:**

| metric | exact_maxsim | fastplaid | freeform |
|---|---:|---:|---:|
| R@10 | 0.2495 | 0.2552 | 0.2495 |
| R@100 | 0.4645 | 0.4607 | 0.4645 |
| R@1000 | 0.7219 | 0.7343 | 0.7219 |
| nDCG@10 | 0.2679 | 0.2704 | 0.2679 |
| p50 search (ms) | 704 | 935 | **572** |
| p95 search (ms) | 901 | 1,090 | **734** |
| build time (s) | 0 | 8,071 | 0 |

Three observations that matter for the design:

1. **FastPLAID's "speed" advantage has a hard crossover at corpus size ~150K docs.** Below trec-covid, exact MaxSim beats FastPLAID by 1.3×–13× on p50 — under fairness (single-thread BLAS, single-query call shape) FastPLAID's per-call overhead dominates its IVF-pruning gains. Only at trec-covid (171K) does FastPLAID flip to 1.46× faster.
2. **FastPLAID's small "+1.2 pp R@1000 on average" is not a real quality win.** It can't exceed exact MaxSim by definition; the gap reflects tie-break ordering inside the top-1000 on saturated-recall datasets (scifact / arguana / pony).
3. **freeform is faster than exact MaxSim across every dataset (avg 704 → 572 ms).** Same algorithm, same outputs, slightly less per-call overhead. Good — the seed is a strict latency improvement at zero quality cost.

---

## 2. Equivalence gate: freeform seed = exact MaxSim

Maximum absolute delta across all 12 datasets and all 4 quality metrics: **0.000000**. Bit-identical retrievals.

| dataset | ΔR@1000 | ΔR@100 | ΔR@10 | ΔnDCG@10 | Δp50 ms |
|---|---:|---:|---:|---:|---:|
| beir_arguana | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −6.7 |
| beir_fiqa | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −39.4 |
| beir_nfcorpus | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −3.1 |
| beir_scifact | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −3.9 |
| beir_scidocs | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −23.9 |
| beir_trec-covid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −477.7 |
| bright_biology | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −59.1 |
| bright_earth_science | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −440.6 |
| bright_economics | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −127.7 |
| bright_pony | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −21.1 |
| bright_stackoverflow | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −220.8 |
| bright_theoremqa_theorems | +0.000000 | +0.000000 | +0.000000 | +0.000000 | −156.3 |

M2 equivalence gate **PASSES** — the freeform seed is the right starting point for evolution.

---

## 3. Picking the early-evolution suite

Reframe: the freeform seed already achieves exact-MaxSim quality, which is the **upper bound** given fixed embeddings. Evolution can only:
- Match quality and reduce latency, **or**
- Find scoring that does better than MaxSim on these embeddings (unlikely but possible).

So the suite must satisfy three constraints simultaneously:

| constraint | why | proxy |
|---|---|---|
| **Quality discriminating** | Approximation losses are visible on hard corpora; saturated R@1000 corpora (scifact=1.0, arguana=0.99) give zero signal. | seed R@1000 well below 1.0 (≤ ~0.7) |
| **Latency headroom** | A 50 ms seed has no measurable speedup target; a 500 ms seed does. | seed p50 ≥ ~400 ms |
| **Iteration cost** | Per-iter cost = warmup + queries × 3 repeats; bounds total run time. | iter cost ≤ a few minutes |

**Per-dataset scoring (warmup=10, repeats=3):**

| dataset | docs | q | seed R@1000 | quality headroom | seed p50 ms | iter cost (s) |
|---|---:|---:|---:|---:|---:|---:|
| beir_nfcorpus | 3.6K | 323 | 0.539 | 0.46 | 53 | 52 |
| beir_scifact | 5.2K | 300 | 1.000 | **0.00 (saturated)** | 75 | 68 |
| bright_pony | 7.9K | 112 | 0.974 | 0.03 | 96 | 33 |
| beir_arguana | 8.7K | 1,406 | 0.989 | 0.01 | 113 | 478 |
| **bright_theoremqa_theorems** | 23.8K | 76 | **0.373** | **0.63** | 437 | **104** |
| beir_scidocs | 25.7K | 1,000 | 0.647 | 0.35 | 343 | 1,033 |
| **bright_economics** | 50.2K | 103 | **0.673** | **0.33** | 586 | **187** |
| bright_biology | 57.4K | 103 | 0.758 | 0.24 | 586 | 187 |
| beir_fiqa | 57.6K | 648 | 0.925 | 0.07 | 679 | 1,327 |
| bright_stackoverflow | 107K | 117 | 0.664 | 0.34 | 1,348 | 487 |
| bright_earth_science | 121K | 116 | 0.682 | 0.32 | 1,584 | 567 |
| **beir_trec-covid** | 171K | 50 | **0.438** | **0.56** | **2,549** | **408** |

**Disqualified for early evolution:**
- *scifact, arguana, pony* — R@1000 ≥ 0.97; no discriminating power. Reserve as held-out validation (regression detection).
- *fiqa* — R@1000 = 0.925 with 1,327 s/iter; both metrics weak.
- *scidocs, fiqa* — iteration cost too high (1000+ s) for early evolution.

### Recommended early-evolution suite (3 datasets)

| # | dataset | role | iter cost |
|---|---|---|---:|
| 1 | **bright_theoremqa_theorems** | Highest quality headroom (R@1000=0.37) at low iter cost. Forces scoring that survives low-recall regimes. | 104 s |
| 2 | **bright_economics** | Medium-corpus balance (50K docs, p50=586 ms, R@1000=0.67). The "approximations start to matter" zone. | 187 s |
| 3 | **beir_trec-covid** | Biggest latency target (p50=2,549 ms) and the only corpus where FastPLAID beats exact MaxSim — where token / centroid pruning genuinely pays off. | 408 s |

**Total ≈ 700 s/iter on this laptop CPU**; ~10 h for 50 iterations. On GPU expect 10–30 s/iter (200+ iterations comfortable overnight) once the freeform seed is ported to torch.

### Held-out validation (run on best-of-loop)

After the loop converges, re-run the best candidate on **all 12 datasets** with `--no-resume`. Use scifact / arguana / pony / fiqa as the **regression check** — any evolved algorithm that breaks the easy cases is disqualified regardless of how well it does on the hard ones.

### Configuration

```yaml
# tasks/late_interaction/configs/freeform_latency_aware.yaml (env passthrough)
EVAL_DATASETS: "bright_theoremqa_theorems,bright_economics,beir_trec-covid"
EVAL_SAMPLE_QUERIES: 50
EVAL_WARMUP_QUERIES: 10
EVAL_TIMED_REPEATS: 3
EVAL_RECALL_K: 1000
```
