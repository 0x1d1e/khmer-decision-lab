# R012 - Matched clean and synthetic-OCR continuation

**Protocol status: locked exploratory.** Execution state and outcomes belong in `results.md`.

## Question and hypotheses

Does a fixed continuation on synthetic 12 px Tesseract transcripts improve the two fixed R006 checkpoints under R010's 12 px OCR condition beyond an otherwise matched clean-text continuation, without materially reducing clean-text accuracy?

- **H1 OCR-domain gain:** for each R006 seed, compare its OCR-trained continuation with its clean-trained continuation on the R010 12 px test input. Support H1 only if both seed-specific paired 95% group-bootstrap intervals for `OCR continuation - clean continuation` accuracy have lower bounds above zero.
- **H2 clean-text preservation:** for each seed, compare its OCR-trained continuation with its frozen R006 checkpoint on clean test input. Call preservation supported only if both paired 95% group-bootstrap lower bounds exceed -2 percentage points. The margin reuses R006's preregistered no-forgetting tolerance; it is a study threshold, not a universal practical margin.

If only one condition passes, report it separately. An interval crossing its threshold is inconclusive, not evidence of equivalence. The combined claim requires both H1 and H2. No result can establish real-image OCR robustness or generalization beyond these fixed models and this reused split.

## Locked design

- **Data and OCR:** use all 11,514 rows of `massive-km-train-full.jsonl`. The clean arm retains each utterance. The OCR arm replaces only the utterance with the transcript from the R010 12 px Noto Sans Khmer / Pillow RAQM / Tesseract `khm+eng`, OEM 1, PSM 7 recipe. Keep source row order, choices, labels, and duplicates; do not filter blanks or rows. Deduplicate OCR work by exact raw utterance, then assign each transcript back to every source row. Abort if Laya skips any row. No validation or test inputs are used to create training data.
- **Models and continuation:** for seeds 13 and 14, start both arms from the matching frozen R006 multitask checkpoint. Each arm gets one epoch over the full 11,514-row training file, no Laya calibration holdout, micro-batch 1, gradient accumulation 32, 360 expected optimizer attempts, full encoder, soft cross-entropy, choice-option shuffling, max lengths 512/256, encoder/head learning rates `2.5e-5` / `1e-4`, minimum learning rate `1e-6`, weight decay `0.01`, gradient clip `1.0`, CUDA AMP, gradient checkpointing, and a fresh 8-bit paged AdamW optimizer and cosine schedule. Match the data order and seed within each seed's clean/OCR pair. Reuse the scheduler guard that advances only after a successful optimizer update. No early stopping, validation-based selection, or test-based selection.
- **Training gate:** all four runs must have exactly 360 optimizer attempts, at least 342 successful updates, changed weights, and matching clean/OCR successful-update counts within each seed (difference no greater than 7). Record skipped updates, optimizer identity, wall time, peak allocated VRAM, runtime, and failures. Do not open test rows or reports unless all four outputs pass source, config, checkpoint, telemetry, and weight-provenance checks.
- **Evaluation:** score all four continuations on R010's existing clean and 12 px OCR test files using Laya's R010-compatible evaluation path, CUDA BF16, consecutive identical question-schema grouping, and batch size 8. Laya resets temperature to identity when training without calibration rows; after each run, restore the matching R006 checkpoint's exact temperature in both saved config files and record that config-only change. Do not fit or tune calibration. Temperature scaling does not change the accuracy outcomes. Authenticate R010's frozen R006 reports before using them as the clean-preservation baseline.
- **Overlap:** after the training gate passes, compare each condition's actual test input text with the union of normalized clean training utterances and normalized OCR-training transcripts. For each condition, exclude an entire normalized source-utterance group if any of its test rows overlaps that union. Use the same eligible rows for both arms and baseline comparisons within a condition. Keep conflicting-label groups, cluster them together, and report their count. Report all test rows descriptively as a secondary analysis.
- **Uncertainty and reporting:** use 10,000 paired group-bootstrap draws with NumPy seed 13, resample normalized source-utterance groups, retain every row from each sampled group, and row-weight accuracy differences. Keep seeds separate. Report accuracy, macro-F1, per-label recall, confidence, ECE, top-answer Brier, AURC, and selective accuracy descriptively. Do not pool seeds or use these test results for another training, calibration, or model-selection decision.

R006's initial weights already reflect clean MASSIVE training. R012 estimates the incremental effect of replacing clean continuation text with synthetic OCR text, not training an OCR-only model. Matching data, initialization, update budget, and recipe controls the continuation stage; text length, OCR errors, and their optimizer effects remain part of the intervention. The exact-input exclusion does not remove all historical exposure to the benchmark.

## Scope and deviations

R010 and R011 already evaluated this MASSIVE test split, and R001-R008 evaluated it in earlier forms. R012 is exploratory and not independent confirmation. OCR is synthetic typesetting with one font, renderer, Tesseract version, and configuration. Two seeds are fixed checkpoints, not a sample of models. No claims about photographs, scans, other OCR engines, unseen tasks, or deployment are supported.

Before test access, record and resolve any implementation or provenance failure without inspecting test rows or predictions. After lock, keep this protocol unchanged; document any necessary pre-score amendment in the manifest and keep outcomes in `results.md` and `results.json`.

## Reproduction

```sh
uv sync --frozen
uv run --frozen pytest -q tests/test_r012_ocr_continuation.py
uv run --frozen python scripts/prepare_r012.py
uv run --frozen python scripts/train_r012_continuation.py 13 clean
uv run --frozen python scripts/train_r012_continuation.py 13 ocr12
uv run --frozen python scripts/train_r012_continuation.py 14 clean
uv run --frozen python scripts/train_r012_continuation.py 14 ocr12
uv run --frozen python scripts/evaluate_r012.py
uv run --frozen python scripts/summarize_r012.py
```
