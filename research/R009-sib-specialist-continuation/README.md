# R009 - Continue the SIB-200 specialist

**Status: locked exploratory. No R009 training or test scoring has started.**

## Question and hypothesis

R008's two-epoch SIB-200 specialists scored 70.59% and 66.67% on the 204-row test set. Each received only 44 optimizer attempts, with 41 and 42 successful updates. R008's multitask models scored 72.55% and 75.49%. Does continuing each specialist for six more epochs improve its test accuracy?

Primary hypothesis: for both seeds, the continued specialist improves over its matching R008 specialist. Count this as supported only if both seed-specific paired 95% group-bootstrap intervals for the accuracy difference exclude zero in the positive direction. Otherwise report the evidence as inconclusive or unsupported, not as equivalence. The seed-matched multitask model is a secondary comparator.

## Locked design

- Use the exact R008 SIB-200 specialist checkpoint, 681-row training file, and seeds 13 and 14. Continue each checkpoint for six epochs with the same recipe and learning-rate policy. Reset optimizer state; this is a weights-only continuation, not an optimizer-state resume. The 6-epoch stage has 132 expected optimizer attempts per seed.
- Use the 99-row official validation split for pre/post training diagnostics and temperature calibration only. Fix the continuation budget at six epochs; do not select an epoch or checkpoint from validation. Require Laya's input-overlap check to classify all 99 validation items as held out from training. The versioned calibration payload must bind the checkpoint config and all 99 usable validation items; histogram binning and other calibration mechanisms are excluded.
- Before test scoring, require at least one successful optimizer update per seed and changed checkpoint weights; otherwise report a training failure. Score the fixed 204-row official test split only after both seed-specific checkpoints and calibration maps pass provenance checks. R009 training and calibration do not use test rows. Compare paired per-row correctness against the matching R008 specialist and multitask reports. Use 10,000 paired group-bootstrap draws, with normalized full state as the group and seed 13 for bootstrap sampling.
- Report accuracy, macro-F1, per-label recall, macro one-vs-rest AUROC, ECE, top-answer Brier, AURC, and selective accuracy. Preserve test reports and training outputs under ignored `artifacts/` paths.
- Record optimizer attempts, successful and skipped updates, runtime, peak VRAM, hardware, software versions, and failures.

There is no checkpoint selection or early stopping. The intervention is the fixed additional training stage, including a new optimizer and cosine schedule; results do not isolate optimizer reset from extra data exposure. Small test size and two seeds permit only conditional, exploratory conclusions.

## Rationale and limits

The SIB-200 paper reports 701/99/204 Khmer train/validation/test rows. Task-level few-shot validation research finds that validation procedures may not reliably estimate performance on individual tasks. This study therefore fixes the extra budget in advance instead of selecting the best validation epoch. Temperature scaling still uses validation; reserve the test split for final scoring.

- SIB-200: [EACL 2024 paper](https://aclanthology.org/2024.eacl-long.14/)
- Shimabucoro et al., [Evaluating the evaluators: Are validation methods for few-shot learning fit for purpose?](https://www.research.ed.ac.uk/en/publications/evaluating-the-evaluators-are-validation-methods-for-few-shot-lea/), TMLR 2024

R009 tests one continuation recipe on one small classification task. The same SIB-200 test split was scored in R008, and those results motivated this follow-up. R009 is therefore exploratory, not an independent confirmation, and cannot establish population-level improvement, general task superiority, or transfer to unseen tasks.

Before lock, I redundantly hashed the test file and inspected one row to confirm its schema. No other rows were examined, and no R009 predictions, training, calibration, or model selection used test content. The hash matches the locked R008 manifest. This is a recorded setup deviation from the stricter score-stage gate: the R009 scorer still waits for both checkpoints and calibration maps before opening test rows.

## Reproduction

The locked manifest pins the protocol, source data, R008 anchors, code, and environment. After lock, use the commands recorded here:

```sh
uv run --frozen python scripts/train_r009_continuation.py 13
uv run --frozen python scripts/train_r009_continuation.py 14
uv run --frozen python scripts/calibrate_r009.py 13
uv run --frozen python scripts/calibrate_r009.py 14
uv run --frozen python scripts/evaluate_r009.py
```

The summary belongs in [`results.md`](results.md), outside this protocol's fingerprint. Keep this file unchanged after lock.
