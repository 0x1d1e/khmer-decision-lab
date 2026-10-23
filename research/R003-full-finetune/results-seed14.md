# R003 seed 14: results

**Status:** Complete. Training and evaluation finished; the seed-matched paired
comparison is recorded. Machine-readable output:
[`results-seed14.json`](results-seed14.json).

Purpose: R003 had one seed, so every R006 - R003 comparison mixed method with
seed variance. This run adds the second R003 seed with the identical recipe so
the multitask-versus-single-task contrast is seed-matched. It is an extension of
R003, not a new study number, and it tests no new hypothesis about multitask
learning.

## Recipe

Identical to R003 seed 13. The only changes are the seed and the output
directory (`artifacts/checkpoints/R003-full-e2-8bit-s14`); seed 13's checkpoint
was not touched.

- `scripts/train_full_8bit.py` defaults, run as `scripts/train_with_telemetry.py
  data/processed/massive-km-train-full.jsonl
  artifacts/checkpoints/R003-full-e2-8bit-s14 14`
- Full encoder, 2 epochs, soft-ce, micro-batch 1 x grad-accum 32, `max_len`
  512, `head_max_len` 256, encoder lr 2.5e-5, head lr 1e-4, min lr 1e-6,
  weight decay 0.01, grad clip 1.0, option shuffling on `choice`, FP16
  autocast with GradScaler, gradient checkpointing
- `PagedAdamW8bit`, swapped in over `torch.optim.AdamW` by the trainer
- Base pinned to revision `7b928d82` (resolved through `snapshot_download`)
- Laya's own training-file calibration split: `calib_max` 400, `calib_frac`
  0.1, `calib_seed` 20260922. Validation was passed as `eval_data` for logging
  only; no checkpoint selection, no early stopping, no test data.

The scheduler is not gated on optimizer updates, so a GradScaler-skipped update
advances the cosine schedule exactly as it did for seed 13. The instrumentation
in `scripts/train_with_telemetry.py` is passive: an optimizer post-step hook and
a pass-through `CosineAnnealingLR.step`. It observes, and changes no recipe.

Data: `data/processed/massive-km-train-full.jsonl` (11,514 rows, SHA-256
`0d9a37e0...0ab5`), calibration slice 400 rows, 11,114 weight-training items.
Validation `massive-km-validation-full.jsonl` (`35cf7541...64c4`), test
`massive-km-test-full.jsonl` (`a7ba2189...837e`). Test data was opened only for
scoring.

## Run record

Hardware: RTX 3050 Laptop 6 GB, torch 2.14.1+cu130, CUDA 13.0,
`uv.lock` SHA-256 `d688cb82...8fb`.

| | seed 13 (historical) | seed 14 |
|---|---|---|
| Wall clock, launch to checkpoint | 36.6 min | 64.9 min (3,896.9 s) |
| Training-only | not recorded | 63.0 min (3,779.3 s) |
| Optimizer update attempts | not recorded | 696 |
| Successful optimizer updates | not recorded | 692 |
| Skipped (GradScaler inf/NaN) | not recorded | 4 |
| Scheduler steps | 696 (from report) | 696 |
| Peak VRAM, training | not recorded | 3,260 MiB (whole run) |
| Choice temperature | 1.2771 | 1.2318 |
| Epoch mean loss | not recorded | 1.0222, 0.4228 |
| Laya eval, MASSIVE val acc | 83.13% | 83.47% |

Telemetry: `artifacts/checkpoints/R003-full-e2-8bit-s14/training_telemetry.json`.
Optimizer identity is recorded from the constructed object
(`bitsandbytes.optim.adamw.PagedAdamW8bit`), so the 8-bit optimizer claim no
longer rests on patch mechanics alone. The 4 skipped updates confirm the
scheduler-before-optimizer warning R003 flagged as unverified: the schedule
advanced 696 times over 692 real updates, a 0.6% shift.

Both runs took longer than the 36.6 min R003 recorded; seed 14's wall clock is
1.8x seed 13's. Nothing in the recipe changed, and the seed-13 run predates the
`isolated-warm-v1` evaluation protocol and any telemetry, so the difference is
attributed to machine state, not to a recipe difference. It was not
investigated further.

Checkpoints (git-ignored), `model.safetensors` SHA-256:

- seed 13 `artifacts/checkpoints/R003-full-e2-8bit` `b5c682c0...7f93` (unchanged;
  hash verified immediately before and after this run)
- seed 14 `artifacts/checkpoints/R003-full-e2-8bit-s14` `91478920...0cd4`

## Results

`scripts/evaluate.py`, `isolated-warm-v1`, same metric set as seed 13. Seed 13
was re-scored under the current protocol into
`artifacts/runs/massive-km-{split}-s13-isolated/`; it reproduced the historical
accuracies exactly (83.13% validation, 83.69% test). The historical reports in
`artifacts/runs/massive-km-{split}/` are untouched.

MASSIVE accuracy (ECE):

| Model | Validation (n=2033) | Test (n=2974) |
|---|---:|---:|
| R003 s13 | 83.13% (0.019) | 83.69% (0.019) |
| R003 s14 | 83.33% (0.015) | 82.18% (0.018) |
| R006 s13 | 83.82% (0.019) | 84.10% (0.021) |
| R006 s14 | 82.14% (0.023) | 83.69% (0.016) |

Full metrics, test: R003 s13 macro-F1 0.7822, Brier 0.0965, AURC 0.0380,
selective accuracy at 50% coverage 98.59%; R003 s14 macro-F1 0.7728, Brier 0.0965, AURC 0.0386,
selective accuracy at 50% coverage 98.66%. Validation macro-F1 is 0.7946 and 0.7960. Inference
peak VRAM 1,504.5 MiB for both seeds.

### Paired comparisons

Accuracy difference in percentage points, paired group-bootstrap 95% interval.
Groups are the SHA-256 of the NFC-normalized, whitespace-collapsed full state
JSON; the bootstrap resamples groups and keeps every row, so accuracy stays
row-weighted. 10,000 draws, seed 13 — the convention used by
`scripts/summarize_r008.py` and `scripts/summarize_r013.py`. Test has 2,974 rows
in 2,786 groups; validation has 2,033 rows in 1,932 groups.

| Contrast | Test | Validation |
|---|---|---|
| R003 s14 - R003 s13 | -1.51 [-2.53, -0.50] | +0.20 [-1.04, +1.44] |
| R006 s13 - R003 s13 | +0.40 [-0.57, +1.38] | +0.69 [-0.59, +1.96] |
| R006 s14 - R003 s13 | 0.00 [-0.99, +1.00] | -0.98 [-2.29, +0.34] |
| R006 s13 - R003 s14 | +1.92 [+0.93, +2.93] | +0.49 [-0.74, +1.69] |
| R006 s14 - R003 s14 | +1.51 [+0.51, +2.54] | -1.18 [-2.46, +0.05] |
| R006 s14 - R006 s13 | -0.40 [-1.40, +0.60] | -1.67 [-2.93, -0.39] |

## Findings

1. **The two R003 seeds differ materially on test.** Seed 14 is 1.51 pp lower,
   and the interval excludes zero. On validation the two seeds are
   indistinguishable (+0.20). The 1.7 pp validation seed spread R006 reported
   for its own seeds is the same size as this R003 test gap, so a one-point
   MASSIVE difference is not separable from seed noise at this scale.
2. **The multitask-versus-R003 conclusion flips with the R003 reference.** Both
   R006 seeds are statistically above R003 seed 14 on test (+1.92, +1.51, both
   intervals excluding zero) and statistically indistinguishable from R003 seed
   13 (+0.40, 0.00). Neither reading is a multitask effect: R006 s13 and s14
   differ from each other by only -0.40 [-1.40, +0.60], less than either gap to
   an R003 seed.
3. **R006's H1 margin is now stated against a seed-matched baseline.** Against
   R003 seed 14, the seed 14 contrast is +1.51 with a lower bound of +0.51, and
   the seed 13 contrast is +1.92 with a lower bound of +0.93. Both are above
   R006's -2.0 pp non-inferiority threshold, so H1 is unaffected in direction.
   What changes is that H1 no longer rests on the seed-13-only baseline.
4. **Calibration is stable across R003 seeds.** Test ECE 0.0193 versus 0.0178,
   Brier identical at 0.0965. Fitted temperatures differ (1.2771 versus
   1.2318), which is the expected consequence of refitting on the same 400-row
   slice with a different seed.
5. **No overlap inflation.** Test accuracy on utterances absent verbatim from
   train is 82.40% for seed 14 versus 80.89% on seen ones (n=2,545 / 429), the
   same pattern as seed 13 (83.81% / 82.98%).

## Limitations

- Two seeds is still a very small sample for estimating seed variance. The
  intervals above describe these two runs; they do not estimate a seed
  distribution, and no variance component is estimated or removed.
- The seed-13 run predates training telemetry. Its successful/skipped update
  counts, training-only duration and peak VRAM remain unrecorded and are not
  reconstructed here.
- R003 seed 13 was re-scored for protocol parity; its accuracy reproduced
  exactly, but its latency figures differ slightly from the historical report
  because the measurement protocol changed. Accuracies, ECE and Brier are
  unaffected.
- Only MASSIVE was evaluated. The topic tasks are R006's; this run adds no
  single-task News or SIB-200 control, so R006's central multitask-versus-
  specialist question remains untested.
- Validation is reported only. It was not used to select a checkpoint, seed, or
  temperature.

## Reproduction

```
uv run --frozen python scripts/train_with_telemetry.py \
  data/processed/massive-km-train-full.jsonl \
  artifacts/checkpoints/R003-full-e2-8bit-s14 14
for split in validation test; do
  uv run --frozen python scripts/evaluate.py \
    data/processed/massive-km-$split-full.jsonl \
    artifacts/runs/massive-km-$split full-e2-8bit-s14
  uv run --frozen python scripts/evaluate.py \
    data/processed/massive-km-$split-full.jsonl \
    artifacts/runs/massive-km-$split-s13-isolated full-e2-8bit
done
uv run --frozen python scripts/summarize_r003_seed14.py
```
