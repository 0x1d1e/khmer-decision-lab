# R006 - Multi-task fine-tuning

Question: does fine-tuning jointly on MASSIVE, Khmer News and SIB-200 add the
two topic tasks without losing MASSIVE intent accuracy?

## Hypotheses (written before running)

- **H1 no forgetting.** R006 stays within 2 points of R003 on MASSIVE test:
  the lower 95% CI bound of R006 - R003 is above -2.0, for both seeds.
- **H2 in-domain gain.** R006 beats R003 on Khmer News test and SIB-200 test
  (CI excluding 0), for both seeds.
- **H3 calibration.** With the topic tasks in the calibration slice, R006
  test ECE on News and SIB-200 is at most 0.10 (R003: 0.24-0.28), for both
  seeds.

Secondary, descriptive only (seed 13): the R005 suite (MASSIVE candidate
count, option order and option text on all three tasks) compared with R003.

## Protocol

- Data: `scripts/prepare_r006.py` builds `r006-multitask-train.jsonl` from
  MASSIVE train (11,514), Khmer News train (5,136) and SIB-200 train (701) at
  natural sizes (66 / 30 / 4%), shuffled with seed 13. Same questions and
  option texts as R004. 4 News train articles that also appear in News
  validation or test are dropped; SIB-200 train has no overlap. Pinned in
  `datasets/manifests/r006-multitask.json`.
- Training: the unchanged R003 recipe (`scripts/train_full_8bit.py`):
  - full encoder, 2 epochs, soft-ce, micro-batch 1 x 32, `max_len` 512,
    `head_max_len` 256, option shuffling, 8-bit paged AdamW
  - base pinned to revision `7b928d82`
  - seeds 13 and 14
- Laya holds out its 400-row calibration slice from the mixed train file and
  fits temperature on it. MASSIVE validation is passed as `eval_data` for
  logging only. There is no checkpoint selection and no early stopping.
- Evaluation: `scripts/evaluate.py` on MASSIVE, SIB-200 and News validation
  and test, both seeds. The R005 suite is run for seed 13 only.
- Comparisons: R003 (MASSIVE-only, seed 13) and base@512. Paired bootstrap
  95% CI (10k resamples, seed 13).
- Validation is reported, never used for selection.

Smoke runs (48 and 320 rows per task, MASSIVE validation as `eval_data`)
measured training cost and checked memory: 0.12 s per News row and 0.074 s
per MASSIVE row per epoch at micro-batch 1. That predicts about 55 min per
seed. No test data was used.

## Results

Hardware: RTX 3050 6 GB. Run log: `artifacts/logs/r006.log` (git-ignored).

Training:

| | wall clock | choice temperature | laya eval, MASSIVE val acc |
|---|---|---|---|
| R003 (MASSIVE only, s13) | ~37 min | 1.277 | - |
| R006 s13 | ~63 min | 1.262 | 83.72% |
| R006 s14 | ~64 min | 1.255 | 82.19% |

Checkpoints (git-ignored), `model.safetensors` sha256:
`R006-mt-s13` `ae4042ab...2da9`, `R006-mt-s14` `e2ff8c7d...5d92`.

### Primary (test; validation in `research/results.json`)

Accuracy (ECE):

| | MASSIVE (n=2974) | Khmer News (n=1543) | SIB-200 (n=204) |
|---|---|---|---|
| base@512 | 13.32% (0.484) | 52.43% (0.341) | 48.53% (0.360) |
| R003 | 83.69% (0.019) | 54.96% (0.283) | 52.45% (0.236) |
| R006 s13 | 84.10% (0.021) | 92.55% (0.018) | 75.98% (0.117) |
| R006 s14 | 83.69% (0.016) | 93.13% (0.011) | 74.02% (0.077) |

R006 - R003, paired bootstrap 95% CI, points:

| | MASSIVE | Khmer News | SIB-200 |
|---|---|---|---|
| s13 test | +0.40 [-0.57, +1.38] | +37.6 [+35.1, +40.1] | +23.5 [+16.7, +30.4] |
| s14 test | 0.00 [-1.04, +1.01] | +38.2 [+35.6, +40.7] | +21.6 [+14.7, +28.4] |
| s13 val | +0.69 [-0.59, +1.97] | +39.6 [+35.6, +43.6] | +11.1 [+1.0, +22.2] |
| s14 val | -0.98 [-2.26, +0.34] | +39.2 [+35.1, +43.1] | +17.2 [+7.1, +27.3] |

Macro-F1 test, s13 / s14: MASSIVE 0.788 / 0.783 (R003 0.782), News 0.917 /
0.923 (R003 0.492), SIB-200 0.759 / 0.740 (R003 0.492).

Seed spread (s14 - s13): MASSIVE val -1.67 [-2.90, -0.44], test -0.40
[-1.41, +0.57]; News and SIB-200 CIs include 0.

Per-label test recall, R003 -> R006 s13: News `life` 8% -> 80%, `economic`
26% -> 91%, `entertainment` 37% -> 95%; SIB-200 `health` 14% -> 68%,
`travel` 35% -> 78%, `geography` 35% -> 76%. Remaining News errors are
mostly `life` <-> `technology` / `entertainment` and `economic` -> `politic`.

### Verdicts

- **H1 no forgetting: supported.** Test lower bounds -0.57 (s13) and -1.04
  (s14) are above -2.0. On validation the s14 lower bound is -2.26, so the
  margin is not large; H1 is defined on test.
- **H2 in-domain gain: supported.** All four test CIs exclude 0 by a wide
  margin.
- **H3 calibration: not supported as stated.** News ECE is 0.018 / 0.011.
  SIB-200 ECE is 0.077 for s14 but 0.117 for s13, above 0.10. Over the five
  R005 permutations of SIB-200 test, s13 ECE ranges 0.065-0.109, so on 204
  rows the 0.10 threshold is within noise; the pre-registered single-file
  test still fails.

### Secondary: R005 suite, seed 13 (test unless noted)

Candidate count (MASSIVE accuracy):

| | k2 | k5 | k7 | k20 | k60 | k60 long head |
|---|---|---|---|---|---|---|
| R003 | 97.07 | 92.60 | 90.72 | 83.36 | 52.69 | 69.37 |
| R006 s13 | 97.04 | 92.97 | 90.92 | 83.69 | 54.54 | 70.78 |

R006 - R003 CIs include 0 at k2-k20; k60 +1.85 [+0.5, +3.2], k60 long
+1.41 [+0.1, +2.7]. The long head (`max_len` 768, `head_max_len` 576) adds
+16.2 [+14.5, +17.9] for R006, as for R003 (+16.7).

Option order (5 seeded permutations):

| | flip rate base / R003 / R006 | R006 accuracy range |
|---|---|---|
| MASSIVE | 85.0% / 10.4% / 9.6% | 0.40 |
| Khmer News | 12.1% / 10.0% / 1.75% | 0.26 |
| SIB-200 | 18.6% / 16.7% / 12.3% | 4.90 |

R006 flip rate - base: News -10.3 [-12.1, -8.6]; SIB-200 -6.4 [-13.2,
+0.5]. News validation flip rate: 1.82%.

Option text, R006 - short labels:

| | raw label id | English description |
|---|---|---|
| MASSIVE | -2.15 [-3.09, -1.21] (R003 -2.62) | - |
| Khmer News | +0.06 [-0.32, +0.45] | -0.97 [-2.01, +0.06] |
| SIB-200 | -0.49 [-1.47, 0.0] | -2.45 [-5.39, +0.49] |

On SIB-200 validation the description variant is +1.0 [-5.1, +7.1] for R006
versus -7.1 [-13.1, -2.0] for R003.

### Efficiency

Cost per decision is unchanged from R003: MASSIVE ~17.5 ms, News ~23 ms,
SIB-200 ~13 ms (p50, batched). Inference peak VRAM 1.5-1.6 GiB (the 2.5 GiB
for s14 includes the s13 model still resident in the same process).
Training takes ~1.7x R003 wall clock for 1.5x the rows (News articles are
longer).

## Findings

1. Adding the topic tasks to training shows no measurable MASSIVE loss at
   either seed (test diffs +0.4 and 0.0) and turns the two topic tasks
   from near-base to strong: News 55% -> 93%, SIB-200 52% -> 74-76%.
2. With topic rows in training and in the calibration slice, News ECE falls
   from 0.28 to 0.01-0.02 and SIB-200 from 0.24 to 0.08-0.12. Both changed
   at once, so this does not separate the effect of training on the topic
   tasks from that of calibrating on them.
3. Order sensitivity on News is almost gone (flip rate 1.75%). SIB-200 is
   still order-sensitive (12%, accuracy range 4.9 points), consistent with
   only 701 training rows.
4. The k-curve and option-text sensitivities on MASSIVE match R003; R006 is
   slightly better at k60 only.
5. Seed variance on MASSIVE validation (1.7 points) is larger than any
   R006 - R003 MASSIVE difference, so single-seed MASSIVE comparisons at the
   1-point level should not be trusted.

## Limitations

- No single-task News or SIB-200 controls: whether multi-task beats
  task-specific fine-tuning is untested.
- R003 has one seed; the R006 - R003 comparison mixes method and seed
  variance.
- SIB-200 test has 204 rows; its CIs are wide and its ECE is noisy.
- One pooled choice temperature is fit on a 400-row slice of the mixed file.
  Only 20 of those rows are SIB-200 (audit below), so the slice offers little
  task-specific calibration evidence.
- The topic tasks are now in-domain. This experiment says nothing about
  generalization to an unseen Khmer task; that is
  [R007's question](../R007-unseen-transfer/).
- Secondary results are seed 13 only.

## Post-run audit

The hypotheses, verdicts and historical metrics above are unchanged.
R006 demonstrates joint adaptation, not superiority over News/SIB-only
specialists or transfer to tasks absent from training.

The saved training reports use calibration seed `20260922` for both model
seeds. Reconstructing Laya 0.4.1's `read_data`, `items_from_rows` and
`split_calibration` on the manifest-hashed mixed file yields 17,351 usable
items, no skipped items, and 16,951 weight-training items. The 400 held-out
calibration items contain **252 MASSIVE, 128 News and 20 SIB-200** rows.
Both runs therefore share the calibration partition. These counts were
reconstructed after the runs, not logged as per-task telemetry at the time.
They limit calibration interpretation; they do not explain SIB errors by
themselves. Weight-training data and calibration distribution changed
together, so their effects remain unseparated.

The preserved run log contains the scheduler-before-optimizer warning for
both seeds. Laya's AMP path calls `scaler.step(optimizer)` and
`scaler.update()`, then unconditionally calls `scheduler.step()`. A skipped
optimizer update can therefore advance the schedule. Neither skip counts
nor effective updates were recorded; their historical values are unknown.
Before another training experiment, reproduce and correct this upstream
behavior and record actual optimizer identity, successful/skipped updates,
training-only duration, peak allocated VRAM and failures. Do not retrain
R006 merely to fill these records.

The current evaluation script isolates models and separates cold loading
and warm-up from scored inference. Its efficiency numbers follow a new
measurement protocol; rerunning the commands below does not reproduce the
historical cold-load timing/VRAM fields exactly. Historical accuracy reports
and `research/results.json` are preserved.

## Reproduction

```
uv run python scripts/prepare_r006.py
for s in 13 14; do
  uv run python scripts/train_full_8bit.py data/processed/r006-multitask-train.jsonl \
    artifacts/checkpoints/R006-mt-s$s $s
done
bash -c 'for s in validation test; do
  uv run python scripts/evaluate.py data/processed/massive-km-$s-full.jsonl artifacts/runs/massive-km-$s mt-s13,mt-s14
  for t in sib200-km khmer-news; do
    uv run python scripts/evaluate.py data/processed/$t-$s.jsonl artifacts/runs/$t-$s mt-s13,mt-s14
  done
done
for s in validation test; do   # R005 suite, seed 13
  for v in k2 k5 k7 k20 k60 perm0 perm1 perm2 perm3 perm4 raw; do
    uv run python scripts/evaluate.py data/processed/massive-km-$v-$s.jsonl artifacts/runs/massive-km-$v-$s mt-s13
  done
  MAX_LEN=768 HEAD_MAX_LEN=576 uv run python scripts/evaluate.py \
    data/processed/massive-km-k60-$s.jsonl artifacts/runs/massive-km-k60-long-$s mt-s13
  for t in sib200-km khmer-news; do for v in perm0 perm1 perm2 perm3 perm4 raw desc; do
    uv run python scripts/evaluate.py data/processed/$t-$v-$s.jsonl artifacts/runs/$t-$v-$s mt-s13
  done; done
done'
uv run python scripts/summarize_runs.py
```
