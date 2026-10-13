# R011 - OCR-domain temperature calibration

**Status: locked exploratory. Validation OCR preparation, calibration fitting, and R011 scoring have not started.**

## Question and hypothesis

Can calibration on synthetic 12 px OCR text improve the predictive log loss of the fixed R006 MASSIVE intent checkpoints under the same OCR condition? For both R006 seeds (13 and 14), compare the OCR-specific temperature map fitted on eligible validation rows with the checkpoint's existing clean-input temperature map.

The primary outcome is the paired difference in mean test negative log-likelihood (NLL), `OCR-fit minus existing`, on R010's 12 px OCR condition. A lower value favors OCR-specific calibration. Support the hypothesis only if both seed-specific 95% paired group-bootstrap intervals have upper endpoints below zero. An inconclusive interval is not evidence of equivalence. Temperature scaling must leave the predicted class and accuracy unchanged.

## Locked design

- **Models:** frozen R006 multitask checkpoints, seeds 13 and 14, with the exact inference-file inventories and CUDA BF16 policy pinned by R010. Do not update model weights. The existing checkpoint temperature is the baseline.
- **Calibration source:** the 2,033-row official MASSIVE validation split. Group rows by NFKC-normalized, case-folded, whitespace-collapsed utterance. Exclude a whole group if its normalized input occurs in MASSIVE train or test, or if its rows have conflicting intent labels. Keep one source-order row per remaining group. The pre-lock audit found 1,715 eligible groups. Render and transcribe those inputs with the unchanged R010 12 px Noto Sans Khmer / Tesseract recipe. No test labels or predictions may be used to fit calibration. Preparation may read test utterance strings only to identify and exclude exact overlaps.
- **Calibration:** use Laya 0.4.1's `Agent.fit_temperatures` on CUDA BF16 calibration logits, with one-hot targets and its fixed NLL / LBFGS fitting behavior. No held-out tuning, threshold fitting, histogram binning, temperature selection, or validation accuracy reporting. The fit uses 1,715 rows, below Laya's 2,000-example per-option-bucket floor, so the expected map is one choice-type scalar with no option-bucket overrides. Preserve the existing model-specific checkpoint configuration; store calibration maps separately under ignored `artifacts/` paths.
- **Test source:** R010's prepared 12 px OCR test file (2,974 rows), whose source and artifact hashes are pinned. The test has 2,786 normalized groups, 34 conflicting-label groups (129 rows), and 267 exact train-overlap groups (429 rows). The primary analysis excludes every exact train-overlap group, leaving 2,545 rows in 2,519 groups; conflicting-label groups remain and are clustered. Report the full test descriptively as a secondary analysis.
- **Readiness gate:** fit, persist, and verify both seed-specific temperature maps before opening R010 test prediction reports or using test labels. Require each map's exact Laya temperature-only payload, finite bounded values, checkpoint config identity, weight identity, validation source and row count. Reject extra calibration mechanisms or option-bucket overrides. Test scoring aborts if either seed is not ready.
- **Scoring:** after the gate passes, compute raw logits once per seed on the pinned OCR test inputs using the locked CUDA BF16 forward path. Apply both existing and OCR-fitted temperature maps to those same logits. Reproduce R010's baseline class predictions on every row before analyzing differences; abort if they disagree. Compute NLL from unrounded logits and targets. Report accuracy, top-answer Brier, ECE, mean confidence, AURC, selective accuracy at 50% and 80% coverage, and NLL under both maps.
- **Uncertainty:** for the primary no-train-overlap subset, resample 2,519 normalized utterance groups with replacement for 10,000 draws, retain all rows in each sampled group, row-weight NLL differences, and use NumPy seed 13 with percentile 2.5 and 97.5 bounds. Apply the success rule separately to each seed; do not pool seeds.

## Scope and limits

R010's MASSIVE test split has already been evaluated in R001-R010. R011 is an exploratory follow-up, not independent confirmation. Its calibration and test inputs share the same source benchmark, and the OCR condition uses one font, Tesseract version, and rendering recipe. It tests confidence scaling only: it cannot recover decision accuracy lost to OCR transcription errors or establish real-image robustness. Results are conditional on two fixed checkpoints and the selected calibration recipe.

## Reproduction

After locking the manifest, run:

```sh
uv run --frozen pytest -q tests/test_r011_ocr_calibration.py
uv run --frozen python scripts/prepare_r011.py
uv run --frozen python scripts/calibrate_r011.py
uv run --frozen python scripts/evaluate_r011.py
uv run --frozen python scripts/summarize_r011.py
```

Keep this protocol unchanged after lock; record outcomes in `results.md` and `results.json`.
