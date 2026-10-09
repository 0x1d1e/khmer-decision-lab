# R008 - Multitask training versus task specialists

## Question and hypothesis

Does the R006-style multitask recipe preserve Khmer News and SIB-200 test accuracy relative to models trained only on the corresponding task, when each arm sees the same target-task training rows for the same number of epochs?

**H1 non-inferiority.** For each task and seed, the multitask model's test accuracy minus the matching specialist's accuracy has a paired, group-bootstrap 95% interval whose lower endpoint is greater than -2 percentage points. All four comparisons must pass to support non-inferiority on both tasks. A failed comparison is inconclusive unless its interval also supports a negative effect; it does not establish equivalence. This is an exploratory benchmark, not a broad claim about multitask learning.

## Data and arms

The source is the manifest-pinned R006 mixed training file, SHA-256 `509efb156bf676ea11882d6af052a4b6447984165929ef88b798df92e81043b3`. R008 reconstructs Laya 0.4.1's seed-20260922 calibration holdout of 400 rows, then trains on the remaining 16,951 rows: 11,262 MASSIVE, 5,008 Khmer News, and 681 SIB-200. The same post-holdout task rows feed all arms:

- **Multitask:** MASSIVE + Khmer News + SIB-200.
- **Khmer News specialist:** Khmer News only.
- **SIB-200 specialist:** SIB-200 only.

The specialist files are exact task-specific subsets of the multitask file. The R006 News preparation removed four train articles overlapping News validation or test. Preparation also refuses any exact normalized train/test state overlap. The News validation set has four exact normalized states also present in test; those validation rows are excluded from calibration. SIB-200 has no such validation/test overlap. Exact normalization uses NFC and collapses whitespace; it does not detect paraphrases or semantic duplicates.

The evaluation sets are the manifest-pinned Khmer News and SIB-200 splits (1,543 and 204 test rows). Do not train on test rows or use test results to select checkpoints, seeds, calibration maps, or settings. R006's historical checkpoints are context only, not a primary comparator, because their training-time calibration and AMP scheduler behavior differ from this protocol.

## Training

Train each arm from the same multilingual base revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, for seeds 13 and 14, in separate serial processes. Use the R006 recipe: full encoder, two epochs, micro-batch 1, gradient accumulation 32, `max_len=512`, `head_max_len=256`, encoder/head learning rates `2.5e-5` / `1e-4`, minimum learning rate `1e-6`, weight decay `0.01`, gradient clipping `1.0`, soft cross-entropy, choice-option shuffling, AMP, gradient checkpointing, and `PagedAdamW8bit`.

Disable Laya's training-file calibration split (`calib_max=0`, `calib_frac=0`) and pass no evaluation file during training. Fit calibration separately from the corresponding task's validation data after training. Match target-task row exposure and epochs, not total optimizer updates. The expected two-epoch update budgets are 1,060 multitask, 314 News-specialist, and 44 SIB-200-specialist updates per seed; record the observed successful/skipped updates and scheduler steps. Differences in total updates are part of this fixed-epoch comparison, not evidence that total optimization was matched.

Use the AMP scheduler guard only in its reviewed scope: standalone serial training with Laya 0.4.1, PyTorch 2.14.1, and bitsandbytes 0.50.2 under the frozen dependency lock. Before training, the locked trainer uses Laya's `dry_run` to require one usable item per exported row, no skipped rows or calibration holdout, and the expected update budget. After training, it checks the returned item and skip counts before recording a conversion audit. Stop if the scheduler-step count differs from successful optimizer updates, update telemetry does not match the configured budget, or any scheduler-before-optimizer warning occurs. Do not run multiple trainings concurrently or change dependencies/workflow without renewed review.

## Calibration and evaluation

Fit a temperature map separately for each checkpoint and task using all usable rows in that task's validation split, excluding validation rows whose normalized state occurs in test. The multitask checkpoints receive one News map and one SIB-200 map each. Each specialist receives only its own task's map. Calibration logits and evaluation predictions both use CUDA BF16 autocast (`LAYA_CUDA_AMP=bf16`); Laya's calibration helper defaults to FP32, so the R008 wrapper explicitly applies the locked autocast policy. The manifest pins this policy, conflicting ambient overrides are rejected, and each calibration map/report records the actual loaded dtype. A forward hook rejects any device or autocast fallback, including Laya's CPU retry after CUDA OOM. The runtime must support CUDA BF16. After a successful run, the trainer writes `training_provenance.json`, binding the arm, task eligibility, seed, training-file hash, locked manifest, conversion audit, telemetry, model weights, and the inference-file inventory. The calibrator requires and verifies that record before loading weights; it rejects a News specialist for SIB-200 calibration and vice versa. Before any test data is opened, the study gate checks each checkpoint's Laya temperature payload, including its calibration shape, config identity, task provenance, and pinned validation source. Use Laya's `Agent.fit_temperatures` and save the map separately from model weights. Do not select calibration settings on test data.

Evaluate each model on the complete corresponding test file. For each task and seed, compare multitask with the same-seed specialist on the same rows. The primary metric is choice accuracy. Report per-seed accuracy, paired differences, interval estimates, class-level recall and calibration metrics; do not pool seeds into one accuracy or treat validation as test.

The primary uncertainty estimate is a paired, uniform-group bootstrap with 10,000 resamples and seed 13. The cluster is the exact normalized input state; draw groups with replacement, retain every row from each draw, and calculate row-weighted accuracy. Use the same group draws for both models in each paired comparison. Report percentile 2.5th and 97.5th endpoints. Apply the -2-point gate to the lower endpoint of every task/seed interval. This procedure accounts for exact duplicate inputs, not unknown semantic dependencies.

## Locked workflow

Preparation writes a pending-lock manifest and data files without loading models or generating predictions:

```sh
uv run --frozen python scripts/prepare_r008.py
```

After the protocol, scripts, tests, and pending manifest have been reviewed, lock them in one commit by changing the manifest status to `locked_exploratory`. Training then requires that locked manifest:

```sh
for seed in 13 14; do
  uv run --frozen python scripts/train_full_8bit.py data/processed/r008-multitask-train.jsonl \
    artifacts/checkpoints/R008-mt-s${seed} ${seed} --no-training-calibration \
    --r008-manifest datasets/manifests/r008-specialist-controls.json
  uv run --frozen python scripts/train_full_8bit.py data/processed/r008-khmer-news-train.jsonl \
    artifacts/checkpoints/R008-news-s${seed} ${seed} --no-training-calibration \
    --r008-manifest datasets/manifests/r008-specialist-controls.json
  uv run --frozen python scripts/train_full_8bit.py data/processed/r008-sib200-km-train.jsonl \
    artifacts/checkpoints/R008-sib-s${seed} ${seed} --no-training-calibration \
    --r008-manifest datasets/manifests/r008-specialist-controls.json
done
```

For every seed, fit News calibration for the multitask and News-specialist checkpoints, and SIB-200 calibration for the multitask and SIB-specialist checkpoints:

```sh
uv run --frozen python scripts/calibrate_r008.py artifacts/checkpoints/R008-mt-s13 khmer-news
uv run --frozen python scripts/calibrate_r008.py artifacts/checkpoints/R008-news-s13 khmer-news
uv run --frozen python scripts/calibrate_r008.py artifacts/checkpoints/R008-mt-s13 sib200-km
uv run --frozen python scripts/calibrate_r008.py artifacts/checkpoints/R008-sib-s13 sib200-km
# Repeat for seed 14.
```

Before opening any test file, `scripts/evaluate_r008.py` verifies all six preregistered checkpoint provenances and all eight task-specific calibration maps. Missing or mismatched sibling artifacts block scoring. It loads the requested checkpoint under the manifest-pinned CUDA BF16 policy and verifies the actual runtime dtype before reading the target test split. Score only after both seeds and all corresponding calibration maps are complete. Calibration and scoring require the exact locked manifest and verified checkpoint provenance. The scorer produces frozen per-checkpoint test reports; `scripts/summarize_r008.py` validates those reports, including inference dtype, and writes `research/R008-specialist-controls/results.json`. Training outputs and reports stay in ignored `artifacts/` paths.

## Interpretation limits

The experiment estimates the effect of adding the other two training tasks under the specified per-task exposure and fixed-epoch recipe. The arms do not have equal aggregate optimizer updates. The SIB-200 test set is small, so its interval may be too wide to establish a two-point non-inferiority margin. Failure to pass the joint criterion is not proof that specialist training is better. Results do not establish transfer to unseen tasks, superiority of multitask training, or generality beyond these datasets and this recipe.
