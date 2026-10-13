# R010 - Synthetic OCR cascade on MASSIVE Khmer

**Status: locked exploratory. Final R010 OCR preparation and model scoring have not started.**

## Question and hypothesis

How do fixed Khmer intent checkpoints behave when their input utterances are recognized from typeset images by Tesseract? The primary hypothesis is that 12 px synthetic OCR lowers accuracy for both frozen R006 multitask checkpoints (seeds 13 and 14) relative to their clean-text inputs. Support this only if both seed-specific paired 95% group-bootstrap intervals for `ocr_12px - clean` have upper bounds below zero. Otherwise report the primary hypothesis as unsupported or inconclusive; do not claim equivalence or general OCR robustness.

R003 full fine-tuning (seed 13) is a contextual checkpoint, not part of the primary hypothesis. All comparisons use the same test rows, question choices and checkpoint within each clean/OCR pair. No training, checkpoint selection, threshold fitting, or recalibration uses R010 test labels. Each checkpoint retains its previously fitted choice temperature.

## Locked design

- **Data:** `data/processed/massive-km-test-full.jsonl` (2,974 rows) and `data/processed/massive-km-train-full.jsonl` (11,514 rows), pinned with source manifests and hashes in `datasets/manifests/r010-massive-ocr-cascade.json`. MASSIVE test has 2,786 NFKC-normalized, case-folded, whitespace-collapsed utterance groups. Thirty-four groups (129 rows) have conflicting labels. Exact normalized test utterances occur in training for 429 test rows across 267 groups. These rows remain in the evaluation; overlap and non-overlap results are reported separately. Conflicting-label duplicates remain together in bootstrap clusters.
- **Checkpoints:** frozen R003 `full-e2-8bit` and R006 `mt-s13`, `mt-s14`. Checkpoint inference-file inventories, model settings and runtime are pinned in the manifest. Use CUDA BF16 inference; abort on device or precision fallback.
- **Images:** render each distinct raw test utterance in Noto Sans Khmer Regular at 48, 24 and 12 px using Pillow RAQM. Draw black text on white, with 16 px horizontal and 12 px vertical padding, minimum image size 256 by 60 px. Save lossless PNG. Do not resize, compress or augment the image. For repeated identical raw utterances, reuse one render/transcription per size.
- **OCR:** Tesseract `khm+eng`, OEM 1 and PSM 7. Preserve the resulting text apart from trimming leading/trailing whitespace. The executable, language data, font and rendering runtime are fingerprinted in the manifest.
- **Text error:** compute code-point CER and Unicode extended-grapheme-cluster CER after NFC normalization and whitespace collapse. Aggregate edit counts and reference-unit counts before calculating CER. Report exact and blank transcriptions.
- **Analysis:** pair each model's OCR and clean decisions row by row. Resample the 2,786 normalized utterance groups with replacement for 10,000 bootstrap draws using NumPy seed 13; keep all rows from a sampled group and calculate row-weighted accuracy differences. Use percentile bounds at 2.5 and 97.5. Report macro-F1, per-label recall, confidence, ECE, top-answer Brier, AURC and selective accuracy descriptively. Also report exact-train-overlap and exact-vs-changed-transcription strata; these are diagnostics, not causal estimates.

A training-text pilot (100 rows) found non-monotonic CER across rendering conditions: 12 px PNG yielded 11.48% code-point CER, while JPEG and resize variants differed. The final analysis therefore measures OCR error directly rather than treating font size as an error-rate scale. The pilot did not use MASSIVE test text or model predictions.

## Scope and deviations

This is an exploratory synthetic typesetting study, not a real-world OCR benchmark. It reuses the MASSIVE test split scored in R001-R008, so it is not independent confirmation. The split's row/group/label-conflict and train-overlap characteristics were audited before this protocol was locked. No R010 OCR-corrupted model predictions or test-based model selection were made before lock.

Only one font, OCR engine/version, language pair, line-segmentation mode and renderer are tested. Font size is not a calibrated degradation level. R006 seeds share a training task and dataset and are not independent tasks or a population of models. Results cannot establish performance on photographs, scans, other OCR systems, unseen datasets, or deployed user inputs. The previously fitted temperatures were not selected for OCR-corrupted text, so confidence metrics describe the frozen pipeline under input shift.

## Reproduction

The locked manifest binds this protocol, code, dependencies, source data, checkpoints, OCR assets and runtime. Keep this file unchanged after lock; write outcomes in `results.md` and `results.json`.

```sh
uv sync --frozen
uv run --frozen pytest -q tests/test_r010_ocr.py
uv run --frozen python scripts/prepare_r010.py
uv run --frozen python scripts/evaluate_r010.py
uv run --frozen python scripts/summarize_r010.py
```
