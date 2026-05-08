# Late-Interaction RankEvolve Plan

**Date:** 2026-05-01 (rewritten from the planning-only draft)
**POC:** Jimmy Nian
**Status:** Phases 1–4 substantially done; remaining work is M1 (fair latency harness + FastPLAID baseline), M2 (freeform seed + latency-aware config + recall floor), M3 (first evolution run + headline plot).

---

## 1. Research question

> Can RankEvolve discover late-interaction retrieval algorithms that improve the **recall@1000 / latency** frontier over a strong external baseline (PyLate FastPLAID with LightOn `LateOn`), starting from an exact MaxSim seed?

Secondary, in priority order:

1. Does the freeform seed reach exact-MaxSim quality on the smoke set within the latency budget?
2. Does evolution find non-trivial, generalizable improvements (held-out BEIR datasets), not per-dataset overfits?
3. Does the latency-aware objective avoid the trivial "fast-but-empty" failure mode?

Everything in this doc exists to answer that question. Cache format zoos, prior-art surveys, and post-v1 large-corpus engineering are out of scope until M3 ships.

---

## 2. What's already done

| Item | Status | Where |
|---|---|---|
| Memmap cache reader/writer + metadata + qrels | DONE | [tasks/late_interaction/embedding_cache.py](../tasks/late_interaction/embedding_cache.py) |
| PyLate `LateOn` encoder CLI; smoke/early/main suites; fp16 storage | DONE | [tasks/late_interaction/encode_embeddings.py](../tasks/late_interaction/encode_embeddings.py) |
| Exact MaxSim score + retriever + deterministic top-k | DONE | [tasks/late_interaction/library.py](../tasks/late_interaction/library.py) |
| Evaluator + worker (recall@1000, nDCG@10, p50/p95/mean latency, build_time, warmup) | DONE | [tasks/late_interaction/evaluator.py](../tasks/late_interaction/evaluator.py), [evaluator_worker.py](../tasks/late_interaction/evaluator_worker.py) |
| `LateInteractionRetriever` Protocol | DONE | [tasks/late_interaction/interface.py](../tasks/late_interaction/interface.py) |
| Cache for all 8 main + early-evolution datasets | DONE | `cache/late_interaction/lightonai__LateOn/` |
| Cache / exact-MaxSim / smoke-evaluator tests | DONE | [tests/tasks/late_interaction/](../tests/tasks/late_interaction/) |
| FastPLAID comparison script (correctness only; latency unfair) | PARTIAL | [tasks/late_interaction/compare_baselines.py](../tasks/late_interaction/compare_baselines.py) |
| **Fair latency harness (device, threading, repeats, fingerprints)** | NOT DONE | M1 |
| **Freeform seed `seeds/freeform.py`** | NOT DONE | M2 |
| **Configs `configs/freeform.yaml`, `configs/freeform_latency_aware.yaml`** | NOT DONE | M2 |
| **Recall floor in evaluator** | NOT DONE | M2 |
| **Deterministic Python approximate baseline** | DROPPED | exact MaxSim alone is the equivalence anchor; FastPLAID is the speed reference |

Reusable framework pieces (don't reinvent):

- Latency-aware objective math in [src/ranking_evolved/evaluation/objective_math.py](../src/ranking_evolved/evaluation/objective_math.py): `inverse_one_plus_ratio` transform, `hard_slowdown_threshold`, per-dataset baseline-relative scoring. The late-interaction YAML can drop `objective:` in directly the way [tasks/bm25/configs/freeform_latency_aware.yaml](../tasks/bm25/configs/freeform_latency_aware.yaml) does.
- Run-directory layout, replay capture, controller loop: nothing task-specific needed.

---

## 3. The objective

The shared objective math is:

```
ratio_d         = candidate_median_query_latency_ms_d / baseline_median_query_latency_ms_d
latency_score_d = 0                          if ratio_d > hard_slowdown_threshold
                  1 / (1 + ratio_d)          otherwise
combined_score  = w_recall * avg_recall@K
                + w_ndcg   * avg_ndcg@10
                + w_latency * mean(latency_score_d)
```

Late-interaction-specific tuning:

- **Weights:** `recall=0.50, ndcg=0.15, latency=0.35`. Recall@1000 matters more for first-stage retrieval; nDCG@10 is noisier on sampled caches.
- **`hard_slowdown_threshold = 8.0`.** The seed (exact MaxSim) is already slow; a candidate that is 4× slower with double the recall is potentially worth keeping. 8× lets us see those candidates while still capping runaway slowdowns.
- **Latency baseline = FastPLAID, not the seed.** Exact MaxSim is 10–100× slower than FastPLAID. If the seed were the baseline, every approximate candidate would trivially win on latency credit and the headline plot would be uninformative. Concretely: M1 emits `tasks/late_interaction/baselines/fastplaid_baseline.{cpu|cuda}.json`; the YAML sets `objective.latency.baseline_source: external` and `baseline_path: tasks/late_interaction/baselines/fastplaid_baseline.${EVAL_DEVICE}.json`. The controller's loader resolves the device-suffixed path and asserts the file's runtime fingerprint matches the current device.
- **Recall floor.** The objective math has no quality floor today; only the latency hard-slowdown is guarded. For late interaction the trivial-collapse risk is real (an algorithm that returns 1000 zero-score docs has zero latency cost). Implement the floor in the late-interaction evaluator (not the framework): if `recall@1000 < 0.5 * baseline_recall@1000`, emit `quality_floor_triggered=1.0` and override `combined_score = recall@1000` (drop the latency credit). Reads `baseline_recall@1000` from the same `fastplaid_baseline.{device}.json`.

The boundary behavior worth knowing: `latency_score = 0.5` when ratio = 1 (parity), and the score drops to 0 above 8× slowdown. With `w_latency = 0.35`, parity is worth +0.175 to `combined_score`. Recall and nDCG must still dominate at the top of the frontier; if real runs show parity earning too much credit we re-tune `w_latency` down.

---

## 4. Latency fairness (the foundation)

We compare the **steady-state, post-warmup, search-only** cost of producing a top-1000 ranking for one query. The current evaluator [evaluator_worker.py:78-85](../tasks/late_interaction/evaluator_worker.py#L78-L85) already times only `retriever.search(one_query, top_k=...)` and excludes warmup queries — the *shape* is right. M1 fills in everything else.

Two orthogonal axes:

1. **Within-run fairness:** both retrievers see identical conditions inside one process (same harness, warmup, top_k, call shape, device, threading regime).
2. **Cross-run / cross-host comparability:** a CPU laptop number cannot be silently compared to a GPU box number. Every result artifact stamps its hardware fingerprint; any code reading a baseline file refuses it on mismatch.

### 4.1 Hardware policy: GPU is first-class

GPU dramatically speeds up exact MaxSim and evolved candidates (5K-doc query: hundreds of ms on a CPU thread → a few ms on GPU; BRIGHT biology with 57K docs: CPU minutes → GPU seconds). Restricting to CPU would make iteration too slow. **Both modes are supported.** The non-negotiable rule: every latency number is tagged with the hardware it was measured on, and the system refuses to mix them.

`RuntimeFingerprint` (computed once per process, embedded in every output):

```json
{
  "device": "cuda" | "cpu",
  "device_name": "NVIDIA RTX 4090" | "Apple M2 Pro" | ...,
  "cuda_version": "12.4" | null,
  "torch_version": "2.5.1",
  "numpy_version": "1.26.4",
  "blas_threads": {"omp": 1, "mkl": 1, "openblas": 1, "veclib": 1} | null,
  "cpu_count": 10,
  "hostname": "..."
}
```

Stamped into every `WorkerResult`, `compare_baselines` output, every `baseline_*.json`, and the run directory's `manifest.json`. Baseline files use a `.{cpu|cuda}` suffix; the loader picks the right one for the active device and hard-errors on mismatch.

### 4.2 Fairness invariants

Each invariant is conditional on device mode where noted.

| # | Invariant | Enforcement |
|---|---|---|
| 1 | **Same harness.** Both retrievers go through `evaluate_cache_dataset` | New `FastPlaidRetriever` adapter implementing the Protocol |
| 2 | **Single device per run, stamped.** Both retrievers run on the same device | `EVAL_DEVICE=cpu\|cuda` (default: `cuda` if available else `cpu`); both retrievers report the same `device` in `WorkerResult` |
| 3 | **Single-thread BLAS *(CPU only)*.** | Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` **before importing numpy/torch**. New `_runtime.py` does this and is imported first. In GPU mode, skipped (irrelevant) but still fingerprinted. |
| 4 | **Single-process / no joblib *(CPU only)*.** FastPLAID forks workers when `num_queries > 10` ([fast_plaid.py:452-459](../.venv/lib/python3.11/site-packages/fast_plaid/search/fast_plaid.py#L452-L459)) | Adapter passes `n_processes=1` |
| 5 | **Per-query call shape.** Both retrievers receive one query at a time | Adapter passes `(1, n_tokens, dim)` to FastPLAID — not batched. (GPU mode: per-call kernel-launch overhead taxes both equally; matches deployment shape we care about.) |
| 6 | **Build excluded.** Index build, mmap open, etc. happen in `build()`, timed under `build_time_ms` | Already enforced by harness; assert adapter does not lazy-load in `search` |
| 7 | **Same `top_k = max(recall_k, ndcg_k) = 1000`** | Already enforced by [evaluator_worker.py:74](../tasks/late_interaction/evaluator_worker.py#L74) |
| 8 | **Sufficient warmup.** First query for exact MaxSim touches the entire 1.19M-token mmap; FastPLAID warms only probed centroids | Default `EVAL_WARMUP_QUERIES=10` for late interaction. On GPU also call `torch.cuda.synchronize()` after warmup before the timed region. |
| 9 | **Per-query repetition.** Single calls have 10–100% variance on laptop, kernel-launch jitter on GPU | Add `EVAL_TIMED_REPEATS` (default 3): inside the timed loop, call `search` N times per measured query, take the median, then aggregate. On GPU surround each call with `torch.cuda.synchronize()` so we time kernel completion, not dispatch. |
| 10 | **GC controlled.** Mid-timing GC adds bimodal tails | `gc.disable()` around the timed region; `gc.collect()` before warmup; re-enable after |
| 11 | **Process pinning / system load.** Best-effort, document-only | Laptop: `caffeinate -dis`. GPU box: no other CUDA processes (`nvidia-smi` check). Documented in YAML, not code-enforced. |
| 12 | **Wrapper overhead counted symmetrically.** PyLate's FastPlaid wrapper opens two SqliteDicts per `search()` call ([pylate/indexes/fast_plaid.py:309-310](../.venv/lib/python3.11/site-packages/pylate/indexes/fast_plaid.py#L309-L310)) | Adapter calls **lower-level** `fast_plaid.search.FastPlaid.search` directly; doc_id lookup happens outside the timed region. Evolved retrievers will likewise have their doc_id lookup outside the timed region. |
| 13 | **No second-encoder cost.** Both retrievers use the cached fp16 mmap | Already enforced by cache design |
| 14 | **Cross-host comparability.** A baseline JSON from a different device must not be silently used | External-baseline loader asserts `loaded_baseline._fingerprint.device == runtime.device`; mismatch → hard error |
| 15 | **Sanity: both retrievers produce usable signal.** On the same device, both retrievers must produce positive latencies and non-trivial recall@1000. **The relative ordering is a measurement output, not an assumption.** Initial CPU smoke on scifact found FastPLAID ~10× *slower* than exact MaxSim because the fairness regime strips its batching/parallelism — that's a genuine finding, not a bug. | M1 test asserts both retrievers complete and produce recall ≥ a basic floor |

---

## 5. Executable plan

Three milestones, each <1 week, each producing one shippable artifact.

### M1 — Fair latency harness + FastPLAID baseline + external baseline loader

**Why first:** every later number depends on this.

Files:

- `tasks/late_interaction/_runtime.py` (new) — `EVAL_DEVICE` resolution, BLAS-thread pinning in CPU mode, `runtime_fingerprint()`. Asserts numpy/torch haven't been imported yet. Imported first by `evaluator.py` and every program module.
- `tasks/late_interaction/programs/exact_maxsim.py` (new) — exposes `LateInteractionRetriever` (alias of `library.ExactMaxSimRetriever`).
- `tasks/late_interaction/programs/fastplaid.py` (new) — exposes `LateInteractionRetriever` (FastPLAID via the lower-level `fast_plaid.search.FastPlaid` API; CPU mode `n_processes=1`; GPU mode wraps `search()` body with `torch.cuda.synchronize()`; doc_id lookup outside `search()`).
- [tasks/late_interaction/evaluator_worker.py](../tasks/late_interaction/evaluator_worker.py) — invariants 9 (`EVAL_TIMED_REPEATS`) and 10 (`gc.disable`/`gc.collect`); stamp `runtime_fingerprint()` into `WorkerResult` and `to_metrics()`.
- [tasks/late_interaction/evaluator.py](../tasks/late_interaction/evaluator.py) — single source of truth. Two entry points sharing one code path:
  - `evaluate(program_path)` — called once per candidate by the controller.
  - `__main__` CLI — `--program PATH --datasets a,b,c [--sample-queries N --warmup-queries N --timed-repeats N]`. Writes `tasks/late_interaction/baselines/<program-stem>.<device>.json` with this schema:
    ```json
    {
      "_fingerprint": { "device": "cuda", "device_name": "...", "...": "..." },
      "_program": ".../seeds/freeform.py",
      "_datasets": ["beir_scifact", "beir_nfcorpus", "beir_arguana"],
      "_average": { "median_query_latency_ms": ..., "recall_at_1000": ..., "...": "..." },
      "beir_scifact": { "median_query_latency_ms": ..., "recall_at_1000": ..., "ndcg_at_10": ..., "build_time_ms": ..., "recall_at_10": ..., "recall_at_100": ... },
      "beir_nfcorpus": { "...": "..." },
      "beir_arguana":  { "...": "..." }
    }
    ```
    Both the latency baseline (controller's external-baseline loader) and the recall floor read the per-dataset blocks; keys starting with `_` are skipped by the loader.
- `tasks/late_interaction/baselines/<program>.cpu.json` and `<program>.cuda.json` — one per program × device, checked in.
- `src/ranking_evolved/core/controller.py` — extend baseline loader to support `objective.latency.baseline_source: external` with `${EVAL_DEVICE}`-interpolated `baseline_path`; assert fingerprint match; new branch alongside `seed` and `disk`. One unit test under `tests/evaluation/`.
- `tests/tasks/late_interaction/test_latency_fairness.py` (new) — invariants 2, 3, 9, 14, 15.

Verification:

```bash
# Pre-evolution sanity comparison: same evaluator, three different programs.
EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \
  --program tasks/late_interaction/programs/exact_maxsim.py \
  --datasets beir_scifact,beir_nfcorpus,beir_arguana \
  --sample-queries 50 --warmup-queries 10 --timed-repeats 3

EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \
  --program tasks/late_interaction/programs/fastplaid.py \
  --datasets beir_scifact,beir_nfcorpus,beir_arguana \
  --sample-queries 50 --warmup-queries 10 --timed-repeats 3

EVAL_DEVICE=cpu uv run python -m tasks.late_interaction.evaluator \
  --program tasks/late_interaction/seeds/freeform.py \
  --datasets beir_scifact,beir_nfcorpus,beir_arguana \
  --sample-queries 50 --warmup-queries 10 --timed-repeats 3
```

Each writes `tasks/late_interaction/baselines/<program-stem>.cpu.json`. Equivalence gate (M2): `freeform.cpu.json` and `exact_maxsim.cpu.json` should match per-dataset metrics within numerical noise. The external-baseline loader unit test confirms it accepts a matching-device baseline and rejects a mismatching-device one.

### M2 — Freeform seed + latency-aware config + recall floor

Files:

- `tasks/late_interaction/seeds/freeform.py` (new) — `class LateInteractionRetriever` calling `library.exact_maxsim_score`. Single file, ~60 lines, no internal class hierarchy.
- [tasks/late_interaction/evaluator.py](../tasks/late_interaction/evaluator.py) — recall-floor wrapper (~10 lines): load `baselines/fastplaid_baseline.{device}.json`, compare `recall@1000` to `0.5 × baseline_recall@1000`, emit `quality_floor_triggered` and override `combined_score = recall@1000` when triggered.
- `tasks/late_interaction/configs/freeform.yaml` (new) — quality-only (`latency.enabled: false`), early-evolution suite (scifact, nfcorpus, arguana), 50 sampled queries.
- `tasks/late_interaction/configs/freeform_latency_aware.yaml` (new) — mirror [tasks/bm25/configs/freeform_latency_aware.yaml](../tasks/bm25/configs/freeform_latency_aware.yaml) with these deltas:
  - `objective.weights`: `recall=0.50, ndcg=0.15, latency=0.35`
  - `objective.latency.hard_slowdown_threshold: 8.0`
  - `objective.latency.baseline_source: external`, `baseline_path: tasks/late_interaction/baselines/fastplaid_baseline.${EVAL_DEVICE}.json`
  - prompt: 5-line ColBERT/PLAID hint block (no XTR/WARP/ScaNN/FAISS detail in the plan; if those help the optimizer they go directly in the prompt)
- New tests:
  - `tests/tasks/late_interaction/test_freeform_seed_equivalence.py` — seed scores match exact MaxSim within 1e-4 on 50 fixed scifact queries
  - `tests/tasks/late_interaction/test_configs_load.py` — both YAMLs parse via the framework's config loader
  - `tests/tasks/late_interaction/test_recall_floor.py` — synthetic case where `recall@1000 = 0.1 × baseline` triggers the floor; `combined_score == 0.1`; latency credit dropped

Verification:

```bash
uv run ranking-evolved run \
  --config tasks/late_interaction/configs/freeform_latency_aware.yaml \
  --replay --max-iterations 3
```

Completes; emits `run.db` and `replay/step_*.json`; the seed eval row shows `combined_score ≈ 0.50*recall_seed + 0.15*ndcg_seed + 0.35*latency_score_seed` with `latency_score_seed = 1/(1 + exact_maxsim_ms/fastplaid_ms)` — a small number, since exact MaxSim is much slower than FastPLAID. That small seed score is exactly what makes the latency-aware objective informative for evolution.

### M3 — First evolution run + headline plot

- Run `freeform_latency_aware.yaml` for 50 iterations on the early suite (scifact, nfcorpus, arguana) on the GPU box.
- Generate one figure: `recall@1000` vs `latency_p50_ms`, scatter of all 50 candidates plus the exact-MaxSim and FastPLAID dots. The headline.
- Write a 1-page `docs/late_interaction_first_results.md`: did any candidate beat FastPLAID on the frontier? What did it do?

Verification: figure exists; this doc links to it; run directory preserved.

---

## 6. Out of scope

- Deterministic Python approximate baseline (formerly §5.2/§6.2 of the original plan). Exact MaxSim alone is the equivalence anchor; FastPLAID is the speed reference.
- Cache format variants beyond fp16 memmap. Revisit only if memory becomes a bottleneck.
- HotpotQA and large-corpus stress tests. Post-M3, if needed.
- Per-prior-art subsections (XTR, WARP, ScaNN, FAISS). Compressed hints can land in the optimizer prompt; they don't belong in a plan doc.
- Quality floor in `objective_math.py`. Lives in the late-interaction evaluator instead — keeps BM25 untouched.

---

## 7. Useful environment variables

```bash
EVAL_DEVICE=cpu|cuda            # default: cuda if available else cpu
EVAL_SAMPLE_QUERIES=50          # measured queries per dataset
EVAL_WARMUP_QUERIES=10          # excluded from latency stats
EVAL_TIMED_REPEATS=3            # repeats per measured query; report median
EVAL_RECALL_K=1000
EVAL_NDCG_K=10
EVAL_CACHE_DIR=cache/late_interaction/lightonai__LateOn
EVAL_DATASETS=beir_scifact,beir_nfcorpus,beir_arguana
```
