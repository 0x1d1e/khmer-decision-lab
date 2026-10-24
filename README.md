# Khmer Decision Lab

Lightweight models that score a fixed set of candidate answers for Khmer text.
The repo holds reproducible experiments. It is not a product, and it is not a released public benchmark.

Hardware: NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM, CachyOS Linux, PyTorch / CUDA.

## Repo map

- `research/R000-environment/` through `research/R017-efficiency/`: each study's protocol and results.
- [`research/results.json`](research/results.json): core benchmark metrics. Per-study machine-readable outcomes are linked from each results page.
- [`research/RESEARCH-STATE.md`](research/RESEARCH-STATE.md): the running position, including intervals not repeated here.
- `scripts/`: preparation, training, evaluation, and summaries.
- `datasets/manifests/`: pinned dataset revisions.
- `data/` and `artifacts/`: gitignored downloads and checkpoints.

## Studies

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

## How to read the results

Intervals and the rest of the qualifications are in [RESEARCH-STATE.md](research/RESEARCH-STATE.md) and the study pages above.

Full-encoder adaptation reaches about 84% on MASSIVE Khmer with 20 candidates. R006 adds Khmer News (93%) and SIB-200 (74-76%) without a measurable MASSIVE test loss across its two seeds. These are trained tasks.

Unseen-task transfer is not established. R007's primary result did not support positive transfer over R003 on its derived MKQA test. ASAG is diagnostic only. R014 was not supported. It is inconclusive: not evidence of equivalence, and not evidence of no transfer. It shows no stable NLI capability and does not close the independent-benchmark gap. The Khmer text is machine-generated, not naturalistic user input. Transfer on other unseen tasks, and multitask superiority over task-specific specialists, remain unresolved.

R013 supports only its fixed selected option set, not broad unseen-intent transfer. Earlier runs reused the source training partition, pretraining exposure is unknown, and intent names may be opaque.

OCR findings are exploratory. They use the reused MASSIVE test split and synthetic typesetting. For R015, the images are PDF re-renders of Word-exported decrees, not scans. Real-image performance is not established. Real-image evaluation remains open.

Efficiency figures are isolated-warm-v1, measured on one RTX 3050. Do not compare them with historical reports that include cold loading, or that count the preceding model in peak VRAM. Do not compare efficiency across these protocols as if they were identical.

`brier` is the Brier score of top-answer confidence against correctness, not multiclass Brier. `latency_*` is request wall time. `cost_per_decision_*` divides batch time by batch size and measures amortized cost, not individual request latency. `scripts/evaluate.py` reports tagged `isolated-warm-v1` unload the model before loading the next one, and record loading and warm-up separately.

## Reproduction

Datasets and checkpoints are downloaded from Hugging Face and written to the git-ignored `data/` and `artifacts/` directories.

```bash
uv sync --frozen
uv run ruff check .
uv run pytest -q
```

GPU regression tests need the existing local data/checkpoints; otherwise skip.

Commands below are the ones that do not appear in a study README. Where a study README already has the run, follow that file. Do not treat a shorter path here as a substitute.

### R000

```bash
uv run python scripts/prepare_massive.py
```

### R001-R003

The R002 `laya-train` command is in [research/R002-head-finetune/README.md](research/R002-head-finetune/README.md). R003 training is `uv run python scripts/train_full_8bit.py` in [research/R003-full-finetune/README.md](research/R003-full-finetune/README.md).

```bash
for s in validation test; do
  uv run python scripts/evaluate.py data/processed/massive-km-$s-full.jsonl artifacts/runs/massive-km-$s
done
```

### R004

The max_len control is in [research/R004-cross-task/README.md](research/R004-cross-task/README.md).

```bash
uv run python scripts/prepare_r004.py
for t in sib200-km khmer-news; do for s in validation test; do
  uv run python scripts/evaluate.py data/processed/$t-$s.jsonl artifacts/runs/$t-$s
done; done
```

### Other studies

- R005: [research/R005-decision-complexity/README.md](research/R005-decision-complexity/README.md)
- R006: [research/R006-multitask/README.md](research/R006-multitask/README.md)
- R007: [research/R007-unseen-transfer/README.md](research/R007-unseen-transfer/README.md). Scoring requires the committed locked manifest.
- R010: [research/R010-massive-ocr-cascade/README.md](research/R010-massive-ocr-cascade/README.md)
- R011: [research/R011-ocr-domain-calibration/README.md](research/R011-ocr-domain-calibration/README.md)
- R015: no training commands. Harness and results: [research/R015-document-image-ocr/results.md](research/R015-document-image-ocr/results.md)
- R016: [research/R016-budget-matched-specialist/README.md](research/R016-budget-matched-specialist/README.md)
- R017: [research/R017-efficiency/README.md](research/R017-efficiency/README.md)
