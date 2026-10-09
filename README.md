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

Machine-readable metrics: `research/results.json`.

## Reproduction

Datasets and checkpoints are downloaded from Hugging Face and written to the
git-ignored `data/` and `artifacts/` directories.

```
uv sync --frozen
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
uv run python scripts/summarize_runs.py            # research/results.json
```
