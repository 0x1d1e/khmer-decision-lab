# R016 - Budget-matched SIB-200 specialist

## Question and hypothesis

Does the R008 SIB-200 specialist's lower SIB-200 accuracy reflect under-training
(44 optimizer updates) rather than a cross-task multitask benefit? R016 trains a
SIB-200 specialist whose optimizer-update budget matches the R006/R008 multitask
arm (~1,060 updates) and compares it, same-seed, against the R008 multitask
checkpoint on the locked 204-row SIB-200 test split.

**Hypothesis (exploratory).** With ~1,060 optimizer updates the specialist's SIB-200
test accuracy closes the gap to same-seed multitask, i.e.
`D_s = acc(SP@budget, s) - acc(MT, s) >= 0` for both seeds.

This is an exploratory measurement only: at n=204 the minimum detectable effect
is ~5.4 pp, so no non-inferiority or superiority claim is supported by this
study. It cannot resolve R008's non-inferiority question.

## Data and arms

- **Training:** `data/processed/r008-sib200-km-train.jsonl` (681 SIB-200 Khmer rows,
  sha256 `ed2ebd5f...57930c3`), manifest-pinned in
  `datasets/manifests/r016-budget-matched-specialist.json`.
- **Test:** `data/processed/sib200-km-test.jsonl` (204 rows, 204 exact normalized
  state groups, sha256 `6d539da9...f057cc`).
- **Arms (seeds 13 and 14):**
  - **Specialist:** SIB-200 only, `artifacts/checkpoints/R016-budget-sib-s{seed}`,
    trained fresh for R016.
  - **Multitask:** R008 multitask comparator,
    `artifacts/checkpoints/R008-mt-s{seed}`, trained under R008 and treated as
    read-only (never overwritten by R016).

Both arms use the same base revision `convaiinnovations/laya@7b928d828b7b0e022f929d9bd2e44165aa270148`
(subfolder `multilingual/`).

## Training

Train each specialist from scratch, one seed at a time, GPU-serialized with an
exclusive flock on `/tmp/khmer-gpu.lock`:

```sh
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 13'
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 14'
```

Recipe: 48 epochs, micro-batch 1, grad accum 32 (effective batch 32), 22 updates
per epoch -> 1,056 optimizer updates per seed (target ~1,060, matching multitask),
`max_len=512`, `head_max_len=256`, encoder_lr `2.5e-5`, head_lr `1e-4`, min_lr
`1e-6`, weight_decay `0.01`, grad_clip `1.0`, loss soft-ce, freeze_encoder false,
shuffle_options choice, AMP true, gradient checkpointing true, `PagedAdamW8bit`.
Laya's training-file calibration is disabled (`calib_max=0`, `calib_frac=0`) and no
evaluation file is passed during training. The trainer's `dry_run` preflight locks
the row count, update budget, and zero skipped/calibration items before training;
telemetry is checked after training against the configured budget.

Outputs per seed under `artifacts/checkpoints/R016-budget-sib-s{seed}/`:
`model.safetensors`, `training_telemetry.json`, `r016_training_report.json`,
`training_provenance.json`. R008's locked checkpoints (`R008-sib-s*`, `R008-mt-s*`)
are never written by this study.

## Evaluation

Score each checkpoint on the full locked SIB-200 test split using
`evaluate.py`'s isolated-warm-v1 protocol: models are unloaded between runs, and
loading plus one 8-example warm-up pass are excluded from scored timing and
peak-memory measurement:

```sh
uv run --frozen python scripts/evaluate_r016_budget_specialist.py
```

Writes `artifacts/runs/R016/{model}.json` for both specialists and both
multitask baselines. Primary metric: choice accuracy. Comparisons are same-seed,
same-rows, paired.

## Summary

`scripts/summarize_r016.py` validates the four test reports, recomputes
`D_s = acc(SP@budget, s) - acc(MT, s)` with paired group-bootstrap 95% CIs
(10,000 draws, seed 13; group = exact normalized input state), and writes
`artifacts/runs/R016/summary.json`.

## Interpretation limits

- The experiment estimates whether matching the multitask arm's optimizer-update
  budget removes the specialist's SIB-200 gap under this recipe, datasets, and two
  seeds. It does not establish superiority, non-inferiority, transfer, or any
  cross-seed generalization.
- At n=204 the gate is untestable: passing R008's -2 pp non-inferiority criterion
  would require a point estimate above roughly +3.4 to +4.4 pp (observed required
  estimates +3.64 and +3.39 pp per seed) against a ~5.4 pp minimum detectable
  effect. Observed `D_s` estimates are negative for both seeds.
- Results are exploratory and conditional on seeds 13 and 14 and this test set.
  See `research/R016-budget-matched-specialist/results.md`.

## Locked workflow

```sh
uv sync --frozen
uv run --frozen ruff check scripts/train_r016_budget_specialist.py scripts/evaluate_r016_budget_specialist.py scripts/summarize_r016.py tests/test_r016.py
uv run --frozen ruff format --check scripts/train_r016_budget_specialist.py scripts/evaluate_r016_budget_specialist.py scripts/summarize_r016.py tests/test_r016.py
uv run --frozen pytest -q
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 13'
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 14'
uv run --frozen python scripts/evaluate_r016_budget_specialist.py
uv run --frozen python scripts/summarize_r016.py
```

Training outputs and reports live under ignored `artifacts/` paths; the study
record is `datasets/manifests/r016-budget-matched-specialist.json`,
`research/R016-budget-matched-specialist/{README.md,results.md}`, and
`artifacts/runs/R016/{summary.json,preparation.json}`.
