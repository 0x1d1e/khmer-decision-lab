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
| [R008](research/R008-specialist-controls/README.md) ([results](research/R008-specialist-controls/results.md)) | Multitask vs task-specialist controls (News, SIB-200) | exploratory; no multitask loss on News, SIB-200 non-inferiority untestable at n=204 (post-hoc power note) |
| [R009](research/R009-sib-specialist-continuation/README.md) ([results](research/R009-sib-specialist-continuation/results.md)) | Continue R008 SIB-200 specialist training | exploratory; improvement unsupported, results inconclusive or mixed |
| [R010](research/R010-massive-ocr-cascade/README.md) ([results](research/R010-massive-ocr-cascade/results.md)) | Synthetic OCR cascade on MASSIVE intent | exploratory; 12 px OCR lowered accuracy for both fixed R006 seeds |
| [R011](research/R011-ocr-domain-calibration/README.md) ([results](research/R011-ocr-domain-calibration/results.md)) | Synthetic OCR-domain temperature calibration | exploratory; lower NLL for both fixed seeds, accuracy unchanged |
| [R012](research/R012-ocr-continuation/README.md) ([results](research/R012-ocr-continuation/results.md)) | Matched clean and synthetic-OCR continuation | exploratory; OCR continuation improved 12 px accuracy for both fixed seeds within the clean-text tolerance |
| [R013](research/R013-unseen-intent-transfer/README.md) ([results](research/R013-unseen-intent-transfer/results.md)) | Seen-only adaptation on held-out MASSIVE intent options | exploratory; positive on the fixed selected option set for both seeds |
| [R014](research/R014-sea-nli-khmer-transfer/README.md) ([results](research/R014-sea-nli-khmer-transfer/results.md)) | SEA-NLI Khmer NLI: transfer to an unseen task type | exploratory; primary unsupported and inconclusive |
| [R015](research/R015-document-image-ocr/README.md) ([results](research/R015-document-image-ocr/results.md)) | Document-image OCR: real decree line-crops vs synthetic renders | exploratory; ~+8.6 pp codepoint CER gap, images are PDF re-renders not scans |
| [R016](research/R016-budget-matched-specialist/README.md) ([results](research/R016-budget-matched-specialist/results.md)) | Budget-matched SIB-200 specialist control (~1k updates) | exploratory; memorization at ~1k updates, specialist still loses to multitask |
| [R017](research/R017-efficiency/README.md) ([results](research/R017-efficiency/results.md)) | Isolated-warm-v1 efficiency across existing checkpoints | done; ~17.3 ms p50 across all checkpoints/batch sizes |

Core benchmark metrics: `research/results.json`. Per-study machine-readable outcomes are linked from each results page.

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
specialists remain unresolved. R013's seen-only adaptation improved accuracy on
its fixed 12-intent option set by 25.9 pp and 23.1 pp for seeds 13 and 14; both
paired group-bootstrap intervals were above zero. This supports only the
selected option strings, not broad unseen-intent transfer: earlier runs reused
the source training partition, pretraining exposure is unknown, and intent
names may be opaque. R010 found an approximately 10-point accuracy loss under
12 px synthetic Tesseract OCR for both fixed R006 seeds. R011's
validation-fitted OCR temperature lowered mean NLL by 0.0084 and 0.0080, with
no accuracy change. R012's OCR continuation improved 12 px accuracy over clean
continuation by 3.18 pp and 2.43 pp for the two fixed R006 seeds; both clean
comparisons stayed within the preregistered -2 pp tolerance. R015's paired
document-image control found real decree line-crops worse than matched
synthetic renders by ~+8.6 pp codepoint CER (macro delta +8.62 pp, 95% CI
[7.73, 9.51]; grapheme +12.11 pp, 95% CI [11.31, 12.87] over 4,778 lines /
148 pages / 17 decree groups; real 10.69% vs synthetic 2.07%), but the images
are PDF re-renders of Word-exported decrees, not scans, so capture degradation
is unmeasured and real-image performance is not established. These OCR
findings use the reused MASSIVE test split and synthetic typesetting, so they
are exploratory and do not establish real-image performance.

R014 moves to an unseen task type: the 65 Khmer rows of SEA-NLI, asked as a
fixed three-option NLI choice over 37 premise clusters. It was not supported.
Pooled Khmer accuracy changed by +3.08 pp for seed 13 (95% paired
group-bootstrap CI [-14.93, +20.63]) and +6.15 pp for seed 14 ([-4.48,
+17.39]); neither lower bound exceeds zero, so the result is inconclusive, not
evidence of equivalence or of no transfer. Every arm is dominated by a single
predicted label but a different one (base entailment 0.60, seed 13 neutral
0.72, seed 14 entailment 0.68), the two same-recipe seeds agree on only 26/65
rows, exact McNemar p is 0.87 and 0.39, and Cohen κ against gold is 0.205 to
0.283. The English-text control puts every arm 3.1 to 7.7 pp higher in English
than in Khmer, also with intervals crossing zero. R014 shows no stable NLI
capability and does not close the independent-benchmark gap. The Khmer text is
machine-generated, not naturalistic user input, and the labels are public, so
sealing is procedural only.

R016's budget-matched SIB-200 specialist control trained both seeds to
~1,056 updates matching multitask's ~1,060-update budget; loss collapsed to
0.000 (memorization of 681 rows) and the specialist did not close the gap on
the 204-row test (D_s = -0.49 pp seed 13, -4.90 pp seed 14; both intervals
include zero), so specialist under-training at R008's 44-update budget is not
supported as the explanation for the R008 gap; non-inferiority is untestable
at n=204, exploratory only.

R017's isolated-warm-v1 efficiency measurement covered eight existing
checkpoints at batch sizes 1/4/8/16 (32 configs, no OOMs): all checkpoints
cluster at ~17.3 ms/row p50 (~19.9 ms p95, ~17 ms/row per-decision cost),
batch size barely moves latency, and load time (~2.5–5.2 s) is the one real
variable; the R008 specialist and its same-seed multitask comparator are
indistinguishable, so there is no task-specific cost difference between arms.
Results are specific to one RTX 3050, one dataset, and one config and are not
comparable to historical reports with cold loading or preceding-model VRAM.

See [R007](research/R007-unseen-transfer/),
[R009](research/R009-sib-specialist-continuation/results.md),
[R010](research/R010-massive-ocr-cascade/results.md),
[R011](research/R011-ocr-domain-calibration/results.md),
[R012](research/R012-ocr-continuation/results.md), and
[R013](research/R013-unseen-intent-transfer/results.md), and
[R014](research/R014-sea-nli-khmer-transfer/results.md),
[R015](research/R015-document-image-ocr/results.md),
[R016](research/R016-budget-matched-specialist/results.md),
and [R017](research/R017-efficiency/results.md). Real-image evaluation
remains open.

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
# R011: validation-only OCR temperature calibration; see locked protocol.
uv run --frozen python scripts/prepare_r011.py
uv run --frozen python scripts/calibrate_r011.py
uv run --frozen python scripts/evaluate_r011.py
uv run --frozen python scripts/summarize_r011.py
uv run python scripts/summarize_runs.py            # research/results.json
# R015: document-image OCR on PDF re-renders (not scans); no training commands,
# see research/R015-document-image-ocr/results.md for the harness and results.
# R016: budget-matched SIB-200 specialist (GPU-serialized via /tmp/khmer-gpu.lock)
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 13'
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 14'
uv run --frozen python scripts/evaluate_r016_budget_specialist.py
uv run --frozen python scripts/summarize_r016.py
# R017: isolated-warm-v1 efficiency across existing checkpoints
uv run --frozen python scripts/measure_r017_efficiency.py --dry-run
uv run --frozen python scripts/measure_r017_efficiency.py   # GPU-serialized internally via /tmp/khmer-gpu.lock
uv run --frozen python scripts/summarize_r017.py
```
