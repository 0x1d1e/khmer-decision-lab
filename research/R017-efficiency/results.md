# R017 results

**Status:** Measured. Latency (p50/p95/mean), per-decision cost (ms/row), peak
VRAM (allocated and reserved), load time, and warm-up time for eight existing
checkpoints on `data/processed/massive-km-test-full.jsonl` (2,974 MASSIVE
Khmer intent-choice rows), at batch sizes 1, 4, 8, 16, on one RTX 3050 (6 GB)
under the `isolated-warm-v1` protocol (model loading and one 8-example warm-up
pass are excluded from scored timing and peak-memory measurement). All 32
checkpoint/batch-size configurations completed; none were dropped for OOM and
none were dropped at all. Full per-config records:
`artifacts/runs/R017/measurements/{checkpoint}_bs{batch}.json`; aggregated
summary: `artifacts/runs/R017/summary.json` (produced by
`scripts/summarize_r017.py` from those records).

| Checkpoint | Origin | Seed | BS | p50 ms | p95 ms | mean ms/row | cost p50 ms/row | Peak alloc MiB | Peak reserv MiB | Load ms | Warmup ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R008-mt-s13 | R008 | 13 | 1 | 17.28 | 19.91 | 17.96 | 17.28 | 2865.5 | 3256.0 | 2598 | 1285 |
| R008-mt-s13 | R008 | 13 | 4 | 17.27 | 19.92 | 17.97 | 17.27 | 2860.6 | 3248.0 | 2598 | 1285 |
| R008-mt-s13 | R008 | 13 | 8 | 17.27 | 19.88 | 17.98 | 17.27 | 2859.7 | 3248.0 | 2598 | 1285 |
| R008-mt-s13 | R008 | 13 | 16 | 17.26 | 19.88 | 18.09 | 17.26 | 2859.7 | 3256.0 | 2598 | 1285 |
| R008-sib-s13 | R008 | 13 | 1 | 17.30 | 19.95 | 17.95 | 17.30 | 2865.5 | 3256.0 | 2770 | 1072 |
| R008-sib-s13 | R008 | 13 | 4 | 17.27 | 19.89 | 17.96 | 17.27 | 2860.6 | 3248.0 | 2770 | 1072 |
| R008-sib-s13 | R008 | 13 | 8 | 17.28 | 19.91 | 17.97 | 17.28 | 2859.7 | 3248.0 | 2770 | 1072 |
| R008-sib-s13 | R008 | 13 | 16 | 17.29 | 19.92 | 18.01 | 17.29 | 2859.7 | 3256.0 | 2770 | 1072 |
| R013-all-label-s13 | R013 | 13 | 1 | 17.32 | 19.97 | 18.01 | 17.32 | 2865.5 | 3256.0 | 2685 | 1232 |
| R013-all-label-s13 | R013 | 13 | 4 | 17.30 | 19.97 | 18.02 | 17.30 | 2860.6 | 3248.0 | 2685 | 1232 |
| R013-all-label-s13 | R013 | 13 | 8 | 17.30 | 19.93 | 18.01 | 17.30 | 2859.7 | 3248.0 | 2685 | 1232 |
| R013-all-label-s13 | R013 | 13 | 16 | 17.30 | 19.96 | 17.99 | 17.30 | 2859.7 | 3256.0 | 2685 | 1232 |
| R013-seen-only-s13 | R013 | 13 | 1 | 17.31 | 19.92 | 17.99 | 17.31 | 2865.5 | 3256.0 | 2641 | 1506 |
| R013-seen-only-s13 | R013 | 13 | 4 | 17.31 | 19.95 | 18.00 | 17.31 | 2860.6 | 3248.0 | 2641 | 1506 |
| R013-seen-only-s13 | R013 | 13 | 8 | 17.30 | 19.95 | 18.02 | 17.30 | 2859.7 | 3248.0 | 2641 | 1506 |
| R013-seen-only-s13 | R013 | 13 | 16 | 17.31 | 19.93 | 18.00 | 17.31 | 2859.7 | 3248.0 | 2641 | 1506 |
| full-e2-8bit | R003 | 13 | 1 | 17.33 | 19.96 | 18.22 | 17.33 | 2861.8 | 3240.0 | 5150 | 4301 |
| full-e2-8bit | R003 | 13 | 4 | 17.34 | 19.97 | 18.03 | 17.34 | 2858.3 | 3240.0 | 5150 | 4301 |
| full-e2-8bit | R003 | 13 | 8 | 17.35 | 19.97 | 18.19 | 17.35 | 2857.5 | 3240.0 | 5150 | 4301 |
| full-e2-8bit | R003 | 13 | 16 | 17.34 | 19.98 | 18.03 | 17.34 | 2857.5 | 3240.0 | 5150 | 4301 |
| full-e2-8bit-s14 | R003 | 14 | 1 | 17.35 | 20.01 | 18.03 | 17.35 | 2865.5 | 3256.0 | 2511 | 1096 |
| full-e2-8bit-s14 | R003 | 14 | 4 | 17.34 | 19.99 | 18.04 | 17.34 | 2860.6 | 3248.0 | 2511 | 1096 |
| full-e2-8bit-s14 | R003 | 14 | 8 | 17.34 | 19.98 | 18.04 | 17.34 | 2859.7 | 3248.0 | 2511 | 1096 |
| full-e2-8bit-s14 | R003 | 14 | 16 | 17.34 | 19.99 | 18.04 | 17.34 | 2859.7 | 3256.0 | 2511 | 1096 |
| mt-s13 | R006 | 13 | 1 | 17.36 | 19.99 | 18.04 | 17.36 | 2865.5 | 3256.0 | 2457 | 1208 |
| mt-s13 | R006 | 13 | 4 | 17.34 | 19.98 | 18.05 | 17.34 | 2860.6 | 3248.0 | 2457 | 1208 |
| mt-s13 | R006 | 13 | 8 | 17.35 | 19.98 | 18.05 | 17.35 | 2859.7 | 3248.0 | 2457 | 1208 |
| mt-s13 | R006 | 13 | 16 | 17.34 | 19.96 | 18.02 | 17.34 | 2859.7 | 3256.0 | 2457 | 1208 |
| mt-s14 | R006 | 14 | 1 | 17.32 | 19.93 | 17.98 | 17.32 | 2865.5 | 3256.0 | 2583 | 1219 |
| mt-s14 | R006 | 14 | 4 | 17.29 | 19.93 | 17.99 | 17.29 | 2860.6 | 3248.0 | 2583 | 1219 |
| mt-s14 | R006 | 14 | 8 | 17.28 | 19.88 | 17.98 | 17.28 | 2859.7 | 3248.0 | 2583 | 1219 |
| mt-s14 | R006 | 14 | 16 | 17.27 | 19.89 | 17.98 | 17.27 | 2859.7 | 3256.0 | 2583 | 1219 |

## Interpretation

- **No measurable efficiency difference between arms.** All eight checkpoints
  cluster at ~17.3 ms/row p50 and ~18 ms/row mean across every batch size;
  the R008 specialist (`R008-sib-s13`) and its same-seed multitask comparator
  (`R008-mt-s13`) are statistically indistinguishable (17.27–17.30 vs
  17.26–17.28 ms/row p50), so this study shows no task-specific cost
  difference for the specialist versus multitask arm at inference time.
- **Batch size barely moves latency** (17.26–18.09 ms/row across 1→16) and
  peak allocated VRAM is flat at ~2.86 GiB reserved-headroom ~3.25 GiB: these
  are small models on short inputs, so per-request overhead dominates and no
  batch config came close to OOM on this 6 GB card.
- **Load time is the one real variable:** `full-e2-8bit` (seed 13) loads in
  ~5.2 s with a 4.3 s warm-up versus ~2.5 s load / ~1.1 s warm-up for the
  other checkpoints; same-recipe seed 14 (`full-e2-8bit-s14`) loads fast, so
  the difference is checkpoint-format-specific, not recipe-specific.

## Limitation (read before citing)

Every number here is specific to **one RTX 3050 (6 GB)**, one dataset
(`data/processed/massive-km-test-full.jsonl`, 2,974 rows), and one batch
configuration per run. These results are **not comparable** to historical
latency reports that include cold loading or count a preceding model's peak
VRAM, and they do not generalize beyond this hardware, dataset, and
configuration.

## Reproduction

```sh
uv run --frozen python scripts/measure_r017_efficiency.py --dry-run
uv run --frozen python scripts/measure_r017_efficiency.py   # GPU-serialized internally via /tmp/khmer-gpu.lock
uv run --frozen python scripts/summarize_r017.py
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen pytest -q
```
