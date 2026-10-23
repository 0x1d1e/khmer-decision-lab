# R017 - Efficiency study: accuracy, latency, memory

## Question and hypothesis

How do the existing trained Khmer checkpoints compare on inference efficiency
(latency p50/p95, per-decision cost in ms/row, peak VRAM, and load time) when
run one at a time on the fixed MASSIVE Khmer test set under the repo's
`isolated-warm-v1` protocol? This is measurement only: no training, no
fine-tuning, and no cross-study conclusion. It is expected to be a solid
appendix, not a headline finding.

**Null expectation (exploratory).** All checkpoints are small (~615 MB
`model.safetensors`) and should fit the RTX 3050 (6 GB) at modest batch sizes;
any latency/cost differences come from batch size and architecture, not from
VRAM pressure.

## Data and arms

- **Dataset:** `data/processed/massive-km-test-full.jsonl` (2,974 MASSIVE
  Khmer intent-choice rows), sha256 locked in
  `datasets/manifests/r017-efficiency.json`.
- **Arms (existing checkpoints only, read-only):**
  - `full-e2-8bit` (R003 seed 13) — `artifacts/checkpoints/R003-full-e2-8bit`
  - `full-e2-8bit-s14` (R003 seed 14) — `artifacts/checkpoints/R003-full-e2-8bit-s14`
  - `mt-s13`, `mt-s14` (R006 multitask) — `artifacts/checkpoints/R006-mt-s{seed}`
  - `R008-sib-s13` vs `R008-mt-s13` (R008 specialist vs same-seed multitask,
    to show the task-specific cost difference)
  - `R013-seen-only-s13`, `R013-all-label-s13` (R013 intent-transfer arms, to
    extend coverage)

Exact paths and `model.safetensors` hashes are confirmed against
`scripts/evaluate.py`'s `CHECKPOINTS` dict and `artifacts/checkpoints/`; both
are recorded in `datasets/manifests/r017-efficiency.json`.

## Protocol

`isolated-warm-v1` (canonical implementation: `scripts/evaluate.py`), one
checkpoint at a time, GPU-serialized with an exclusive flock on
`/tmp/khmer-gpu.lock`:

1. Load the model fresh with `laya.Router(..., device="cuda", max_loaded=1,
   preload=False)`, unloading and emptying the cache first.
2. Warm up on the first 8 examples at batch size 8; loading and warm-up are
   excluded from scored timing and peak-memory measurement.
3. Run all 2,974 rows at batch sizes 1, 4, 8, 16 (ascending, fixed config).
4. Record per batch size: latency p50/p95/mean, per-decision cost p50/p95
   (ms/row), peak VRAM allocated and reserved; plus load time and warm-up time
   per checkpoint.
5. Unload, free, and repeat for the next checkpoint.

A batch size that OOMs is dropped with the reason recorded; no larger batch
size is attempted for that checkpoint.

## Outputs

- `scripts/measure_r017_efficiency.py` — the measurement script (`--dry-run`
  lists the plan without loading weights).
- `scripts/summarize_r017.py` — produces `summary.json` and the summary table.
- `artifacts/runs/R017/measurements/{checkpoint}_bs{batch}.json` — per
  (checkpoint, batch size) record.
- `artifacts/runs/R017/summary.json` — aggregated summary.
- `artifacts/runs/R017/preparation.json` — locked scope before and after the run.
- `datasets/manifests/r017-efficiency.json` — locked manifest
  (`protocol_sha256`, implementation fingerprints, checkpoint paths +
  `model.safetensors` hashes, dataset sha256, batch configs, scope).
- `research/R017-efficiency/results.md` — results table.

## Reproduction

```sh
uv run --frozen python scripts/measure_r017_efficiency.py --dry-run
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/measure_r017_efficiency.py'
uv run --frozen python scripts/summarize_r017.py
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen pytest -q
```

## Limitations

Every number below is specific to **one RTX 3050 (6 GB)**, one dataset
(`massive-km-test-full.jsonl`, 2,974 rows), and one batch configuration. These
results are **not comparable** to historical latency reports that include cold
loading or count a preceding model's peak VRAM, and they do not generalize
beyond this hardware, dataset, and configuration.
