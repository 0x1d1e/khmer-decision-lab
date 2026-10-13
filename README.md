# Khmer Decision Lab

Reproducible research into lightweight Khmer language understanding
and non-autoregressive decision models.

## Research objectives

1. Establish reproducible Khmer benchmarks.
2. Fine-tune lightweight multilingual models.
3. Measure cross-task generalization.
4. Evaluate robustness to Khmer OCR errors.
5. Optimize accuracy, latency, and memory efficiency.

## Hardware

- NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM
- CachyOS Linux
- PyTorch / CUDA

## Research

| ID | Topic | Status |
|---|---|---|
| [R000](research/R000-environment/) | Environment, MASSIVE Khmer data | done |
| [R001](research/R001-baseline/) | Pretrained Laya Multilingual baseline | done |
| [R002](research/R002-head-finetune/) | Frozen-encoder (head-only) fine-tuning | done |
| [R003](research/R003-full-finetune/) | Full-encoder fine-tuning, 8-bit optimizer | done |
| [R004](research/R004-cross-task/) | Cross-task generalization (SIB-200, Khmer News) | done |
| [R005](research/R005-decision-complexity/) | Decision complexity: candidate count, option order, option text | done |
| [R006](research/R006-multitask/) | Multi-task fine-tuning (MASSIVE + Khmer News + SIB-200) | done |
| [R007](research/R007-unseen-transfer/) | Frozen transfer: derived MKQA choices + ASAG grading | exploratory; protocol locked, primary hypothesis not supported |
| [R008](research/R008-specialist-controls/README.md) ([results](research/R008-specialist-controls/results.md)) | Multitask vs task-specialist controls (News, SIB-200) | exploratory; no multitask loss on News, SIB-200 inconclusive |
| [R009](research/R009-sib-specialist-continuation/README.md) ([results](research/R009-sib-specialist-continuation/results.md)) | Continue R008 SIB-200 specialist training | exploratory; improvement unsupported, results inconclusive or mixed |
| [R010](research/R010-massive-ocr-cascade/README.md) ([results](research/R010-massive-ocr-cascade/results.md)) | Synthetic OCR cascade on MASSIVE intent | exploratory; 12 px OCR lowered accuracy for both fixed R006 seeds |

Machine-readable metrics: `research/results.json`.

## Current evidence

Full-encoder adaptation reaches about 84% on MASSIVE Khmer with 20
candidates. R006 adds Khmer News (93%) and SIB-200 (74-76%) without a
measurable MASSIVE test loss across its two seeds. These are all trained
tasks. R007's primary exploratory result did not support positive transfer
over R003 on its derived MKQA test; ASAG is diagnostic only. R009's fixed
six-epoch specialist continuation did not support improved SIB-200 accuracy:
both seed-specific 95% paired group-bootstrap intervals include zero, and the
test split was reused from R008.
Transfer on other unseen tasks and multitask superiority over task-specific
specialists remain unresolved. R010 found an approximately 10-point accuracy
loss under 12 px synthetic Tesseract OCR for both fixed R006 seeds; this does
not establish real-image OCR performance. See [R007](research/R007-unseen-transfer/),
[R009](research/R009-sib-specialist-continuation/results.md), and
[R010](research/R010-massive-ocr-cascade/results.md) for results. OCR-domain
calibration and real-image evaluation remain open.

Metric definitions: `brier` is the Brier score of top-answer confidence
against correctness, not multiclass Brier. `latency_*` is request wall time;
`cost_per_decision_*` divides batch time by batch size and measures amortized
cost, not individual request latency. Historical reports include cold
loading in timing and may count the preceding model in peak VRAM. New
`scripts/evaluate.py` reports are tagged `isolated-warm-v1`: models unload
before loading the next; loading and warm-up are recorded separately.
Do not compare efficiency across these protocols as if they were identical.

## Reproduction

Datasets and checkpoints are downloaded from Hugging Face and written to the
git-ignored `data/` and `artifacts/` directories.

```
uv sync --frozen
# Checks: uv run ruff check .; uv run pytest -q
# GPU regression tests need the existing local data/checkpoints; otherwise skip.
uv run python scripts/prepare_massive.py           # R000 data
# R002: laya-train command in research/R002-head-finetune/README.md
uv run python scripts/train_full_8bit.py           # R003, about 37 min on RTX 3050
for s in validation test; do                       # R001-R003 eval
  uv run python scripts/evaluate.py data/processed/massive-km-$s-full.jsonl artifacts/runs/massive-km-$s
done
uv run python scripts/prepare_r004.py              # R004 data
for t in sib200-km khmer-news; do for s in validation test; do
  uv run python scripts/evaluate.py data/processed/$t-$s.jsonl artifacts/runs/$t-$s
done; done
# R004 max_len control: see research/R004-cross-task/README.md
# R005: prepare_r005.py + eval loop in research/R005-decision-complexity/README.md
# R006: prepare_r006.py + train/eval loop in research/R006-multitask/README.md
uv run python scripts/prepare_r007.py              # R007 exploratory exports; no scoring
# R007: scoring requires the committed locked manifest; see research/R007-unseen-transfer/README.md
# R010: synthetic OCR cascade; see research/R010-massive-ocr-cascade/README.md
uv run --frozen python scripts/prepare_r010.py
uv run --frozen python scripts/evaluate_r010.py
uv run --frozen python scripts/summarize_r010.py
uv run python scripts/summarize_runs.py            # research/results.json
```
