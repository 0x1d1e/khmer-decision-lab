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

Pending.

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
