# R011 results

Status: scored exploratory. The preregistered primary criterion was met for both fixed R006 checkpoints on the reused MASSIVE test split under synthetic 12 px OCR. This is evidence for these checkpoints and this split only. The locked protocol README's status line records the state at lock; this file records the completed run.

## Primary comparison

The primary subset excludes all 429 test rows whose normalized utterance appears in MASSIVE train, leaving 2,545 rows in 2,519 normalized utterance groups. Negative NLL differences favor the OCR-fitted temperature.

| Seed | Existing temperature | OCR-fitted temperature | Existing NLL | OCR-fitted NLL | Difference, 95% paired group-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| 13 | 1.2620 | 1.3756 | 0.8592 | 0.8508 | -0.00840 [-0.01217, -0.00479] |
| 14 | 1.2551 | 1.3801 | 0.8757 | 0.8677 | -0.00802 [-0.01223, -0.00396] |

Both interval upper bounds are below zero, meeting the locked success rule. The intervals resample 2,519 groups with all rows retained and row-weighted differences, using 10,000 draws and NumPy seed 13. Temperature scaling changed no predicted classes. Primary-subset accuracy was 74.81% for seed 13 and 74.11% for seed 14 under both temperatures.

| Seed | Top-answer Brier, existing → OCR fit | ECE, existing → OCR fit | Mean confidence, existing → OCR fit | AURC, existing → OCR fit | Selective accuracy at 50%, existing → OCR fit | Selective accuracy at 80%, existing → OCR fit |
|---|---:|---:|---:|---:|---:|---:|
| 13 | 0.11935 → 0.11752 | 0.03782 → 0.01843 | 0.78564 → 0.76409 | 0.06824 → 0.06805 | 96.47% → 96.62% | 85.90% → 86.05% |
| 14 | 0.12473 → 0.12243 | 0.04606 → 0.02668 | 0.78163 → 0.75623 | 0.07378 → 0.07365 | 96.54% → 96.54% | 84.72% → 84.87% |

On all 2,974 test rows, accuracy remained 74.21% for seed 13 and 73.57% for seed 14. NLL changed from 0.89254 to 0.88425 for seed 13 and from 0.90712 to 0.89963 for seed 14. These full-test results are descriptive; the locked primary analysis excludes train-overlap rows.

## Calibration data and inference alignment

Leakage and ambiguity filtering left 1,715 validation rows from 1,715 distinct normalized groups. The 2,033-row validation split had 1,932 groups; the exclusion union contained 217 groups for conflicting labels, train overlap, or test overlap. Tesseract 12 px transcriptions on the selected rows had 11.58% codepoint CER, 15.81% grapheme CER, 402 exact transcriptions, and no blank transcriptions.

The final maps use one global choice temperature per checkpoint. Model weights were unchanged. Calibration and scoring used R010's Laya evaluation grouping: consecutive rows with identical question schemas, batch size 8, and singleton prediction for singleton groups. All raw class predictions matched the hash-authenticated R010 report on all 2,974 rows for both seeds.

An earlier guarded attempt used adjacent batch-16 inference and found 16/2,974 seed-13 prediction mismatches. It stopped before writing valid evaluation metrics; those temperatures were superseded. The implementation was aligned to R010 batching, temperatures were refit on validation only, and test scoring proceeded only after exact class parity passed. The locked manifest records both pre-score amendments.

After independent review, a fresh calibration run reproduced the recorded temperatures and byte-identical calibration payload hashes. The reviewer independently recomputed the test metrics and bootstrap intervals from saved artifacts. Validation logits are not retained, so the reviewer did not independently refit the temperatures; the clean rerun confirms reproducibility through the locked calibration path, not an independent implementation of the fitter.

## Limits

MASSIVE test data were evaluated in R001-R010, so R011 is not independent confirmation. The study uses two fixed checkpoints and one synthetic OCR recipe with one font and Tesseract configuration. It measures confidence scaling only: temperature calibration cannot recover the decision accuracy lost when OCR changes the input. These results do not establish real-image robustness, transfer to another OCR engine, or deployment value. The primary subset still includes 34 conflicting-label groups spanning 129 rows; the bootstrap clusters those rows by normalized utterance.

The locked protocol and machine-readable results are in [`README.md`](README.md) and [`results.json`](results.json). Large preparation, calibration, and per-row evaluation artifacts remain under ignored `artifacts/runs/R011/` and `data/` paths.
