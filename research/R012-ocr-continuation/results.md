# R012 results

Status: scored exploratory. The locked protocol is in [README.md](README.md); machine-readable metrics and per-label results are in [results.json](results.json).

## Outcome

The OCR-continuation arm improved accuracy on the synthetic 12 px OCR input relative to matched clean-text continuation for both fixed R006 checkpoints. Both seed-specific paired 95% group-bootstrap intervals were above zero. Clean-text accuracy met the preregistered -2 percentage-point preservation threshold for both seeds. The combined rule passed for these two checkpoints and this reused test split; this does not establish equivalence on clean text or general OCR robustness.

| Primary comparison | Seed | Reference accuracy | OCR-continuation accuracy | Difference, percentage points (95% CI) |
|---|---:|---:|---:|---:|
| OCR input: OCR continuation vs clean continuation | 13 | 74.79% | 77.97% | +3.18 [+1.91, +4.52] |
| OCR input: OCR continuation vs clean continuation | 14 | 75.27% | 77.69% | +2.43 [+1.27, +3.59] |
| Clean input: OCR continuation vs frozen R006 | 13 | 84.06% | 83.67% | -0.39 [-1.38, +0.59] |
| Clean input: OCR continuation vs frozen R006 | 14 | 83.71% | 83.55% | -0.16 [-1.18, +0.86] |

The OCR primary subset contains 2,515 rows in 2,490 normalized source-utterance groups. The clean preservation subset contains 2,541 rows in 2,515 groups. Intervals use 10,000 row-weighted paired group-bootstrap draws (NumPy seed 13). The preservation rule is a threshold test, not an equivalence test.

All-row accuracy and macro-F1 are descriptive:

| Input | Seed | Frozen R006 accuracy / macro-F1 | Clean continuation accuracy / macro-F1 | OCR continuation accuracy / macro-F1 |
|---|---:|---:|---:|---:|
| Clean | 13 | 84.10% / 0.7747 | 83.86% / 0.7765 | 83.59% / 0.7743 |
| Synthetic OCR 12 px | 13 | 74.21% / 0.6627 | 73.77% / 0.6506 | 77.67% / 0.7021 |
| Clean | 14 | 83.69% / 0.7703 | 83.52% / 0.7645 | 83.46% / 0.7681 |
| Synthetic OCR 12 px | 14 | 73.57% / 0.6495 | 74.34% / 0.6555 | 77.44% / 0.6966 |

Each all-row result covers 2,974 test rows. The accompanying JSON reports per-label recall, confidence, ECE, top-answer Brier, AURC, and selective accuracy; these metrics are descriptive and were not used for selection.

## Data and training

The OCR training file contains all 11,514 MASSIVE training rows. Its synthetic transcripts have 12.10% codepoint CER and 16.21% grapheme CER; 3,229 rows are exact and 4 transcriptions are blank. The clean and OCR arms used the same source order, initial checkpoint per seed, and locked training recipe. All four runs made 360 optimizer attempts, with 357 successful updates and 3 skipped updates each. No validation-based or checkpoint selection was used. The matching R006 choice temperatures and inference metadata were restored without recalibration. Evaluation used R010-compatible BF16 inference grouping and batch size 8, with authenticated R010 baseline reports. All four outputs passed provenance, changed-weight, telemetry, and inference-file readiness checks before test access.

The test set has 2,786 normalized source-utterance groups and 34 conflicting-label groups (129 rows). The locked overlap rule excluded 271 groups / 433 rows from clean-input analysis and 296 groups / 459 rows from OCR-input analysis. Conflicting-label groups remained clustered in the bootstrap.

## Pre-score failures

Two attempts were rejected before test access. The first clean seed-13 continuation failed readiness because Laya rewrote `tokenizer/tokenizer_config.json`; its checkpoint was discarded. R012 then restored the anchor's exact tokenizer and question metadata and authenticated those files. A later OCR seed-14 command was aborted before a continuation checkpoint or telemetry was written; its partial start copy was removed and the arm was rerun from the locked R006 checkpoint. Failure records are in `artifacts/runs/R012/pre-score-failures/`.

## Scope

R012 reuses the MASSIVE test split previously evaluated in R001-R011, so the result is exploratory, not independent confirmation. OCR consists of synthetic typesetting with one font, renderer, Tesseract version, and configuration. R012 measures the incremental effect of replacing clean continuation text with OCR transcripts for two fixed R006 checkpoints. It does not establish OCR-only training, real-image performance, or generalization to other models or OCR conditions. Exact normalized train-input overlap was excluded, but broader historical exposure and semantic overlap remain possible.

The results are tied to manifest SHA-256 `94b5c7dbcb434bcb1847eeb7f72b66451036c8136112b741830025bc5b8f7430` and protocol SHA-256 `7ae5c0af051b3592ae47cb57778f33039eee5105cac4fdc08238481138683803`.
