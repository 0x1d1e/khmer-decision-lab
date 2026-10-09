# R004 - Cross-task generalization

Question: does R003's MASSIVE-Khmer fine-tuning help or hurt on other Khmer
classification tasks it never trained on?

Hypothesis (written before running): R003's encoder adaptation transfers, so
R003 matches or beats the pretrained checkpoint on both tasks. A drop of more
than 2 accuracy points vs base on a task's test split counts as evidence of
forgetting. R002 (head-only) should stay close to base.

## Protocol

Fixed before any results were seen. Evaluation only: no training, tuning,
prompt or label-text iteration, and no temperature refit.

| Task | Source (pinned in `datasets/manifests/`) | Labels | Val | Test |
|---|---|---:|---:|---:|
| SIB-200 Khmer topic | `Davlan/sib200` `khm_Khmr` | 7 | 99 | 204 |
| Khmer News category | `CADT-IDRI/Khmer_News_classification_unsegmented` | 6 | 661 | 1,543 |

- `scripts/prepare_r004.py` exports validation and test. Every row offers
  all labels in one fixed order, identical for every model. Text is used as
  published, including zero-width spaces.
- Models: base `multilingual` (pinned `7b928d82`), R002 head, R003 full,
  each with its own MASSIVE-fit temperature
- `scripts/evaluate.py`: laya-evals defaults, batch size 8. Records peak
  allocated VRAM and weight hashes.
- Primary comparison: test accuracy and macro-F1, R003 vs base. Validation
  is reported as a second independent sample, not used for selection.
- References: chance (1/k) and majority-class accuracy.
- Leakage checks: no R004 text appears in MASSIVE train. Khmer News has 4
  test articles also in validation and 1 duplicate within test (left as
  published).

`mteb/sib200` was rejected: its Khmer validation split is identical to its
test split.

## Results

Test split (validation in `research/results.json`). Paired bootstrap 95% CI
on accuracy difference, 10k resamples, seed 13.

| Task | Model | Acc | Macro-F1 | ECE | Brier | Mean conf |
|---|---|---:|---:|---:|---:|---:|
| SIB-200 (n=204; chance 14.3%, majority 25.0%) | base | 48.53% | 0.455 | 0.360 | 0.363 | 0.84 |
| | R002 | 49.02% | 0.456 | 0.108 | 0.210 | 0.53 |
| | R003 | 52.45% | 0.492 | 0.236 | 0.270 | 0.76 |
| Khmer News (n=1,543; chance 16.7%, majority 29.1%) | base | 49.97% | 0.468 | 0.365 | 0.367 | 0.86 |
| | base@512 (control) | 52.43% | 0.467 | 0.341 | 0.346 | 0.86 |
| | R002 | 51.00% | 0.458 | 0.089 | 0.228 | 0.60 |
| | R003 | 54.96% | 0.492 | 0.283 | 0.301 | 0.83 |

| Comparison | SIB val | SIB test | News val | News test |
|---|---|---|---|---|
| R003 - base | -4.0 [-11.1, +3.0] | +3.9 [-1.5, +9.3] | +6.5 [+3.2, +9.7] | +5.0 [+3.0, +7.1] |
| R003 - base@512 | - | - | +3.0 [+0.6, +5.5] | +2.5 [+1.0, +4.2] |
| R002 - base@512 | - | - | +0.3 [-0.8, +1.4] | -1.4 [-2.3, -0.6] |

## Deviation: context-length control (added after first results)

The base checkpoint config uses `max_len` 1024, while R002/R003 inherited 512
from training. That broke the "identical inputs" premise on Khmer News: 97%
of test articles exceed 400 tokens (median 1,526), so base read up to twice
as much text and ran at twice the per-decision cost (49 vs 23 ms).

Control: `artifacts/checkpoints/control-base-maxlen512` symlinks the base
weights (hash verified `9d628fd9...`) and changes only `max_len` to 512.

```
mkdir -p artifacts/checkpoints/control-base-maxlen512   # weights/encoder/tokenizer -> base snapshot,
# rl_agent_config.json = base config with max_len 512
uv run python scripts/evaluate.py data/processed/khmer-news-test.jsonl \
  artifacts/runs/khmer-news-test control-base-maxlen512
```

SIB-200 needs no control (p90 209 tokens, nothing truncated). Base was more
accurate at 512 than at 1024 on News, so extra context did not help it.

## Findings

- **Khmer News: R003 transfers positively.** It beats base at matched
  context on both splits. The gain is small (+2.5 test) next to its +70 points
  in-domain, and absolute accuracy stays near 55%.
- **SIB-200: inconclusive.** The splits disagree in sign and both CIs include
  0. With n=99/204 the 95% CIs are 10-14 points wide.
- **Forgetting:** no test-split drop beyond the 2-point threshold. Two
  caveats: R003 is -4.0 on SIB validation (n=99, CI [-11.1, +3.0], so not
  resolvable), and R002 shows a small but significant News test drop vs
  base@512 (-1.4), not replicated on validation.

**Verdict: partly supported.** R003 matches or beats base on Khmer News at
matched context. On SIB-200 the data cannot tell either way.
- **Calibration does not transfer.** MASSIVE-fit temperatures leave R003
  overconfident (ECE 0.24-0.28, confidence about 0.8 at about 0.55 accuracy).
  R002 is far better calibrated (0.09-0.14), because head-only training lowered
  its confidence overall, not because it became more accurate. A per-task
  temperature fit on validation would be the fair calibration comparison
  (separate from the joint training/calibration change in R006).
- **Ranking still helps:** R003 has the best AURC on every split (News test
  0.279 vs 0.329 base), so its confidence orders errors better even though
  its level is off.
- **Error analysis (test recall, base -> R003).** Every model over-predicts
  `science/technology` and `politics` (SIB) and `technology`/`politic`
  (News). Recall stays very low for SIB `health` (0.09 -> 0.14) and News
  `life` (0.06 -> 0.08). R003 gains mostly on politics, sports and travel and
  loses on SIB `geography` (0.47 -> 0.35) and News `entertainment`
  (0.44 -> 0.37). This points to label-description and prior effects (R005).
- **Tokenization:** the mmBERT tokenizer averages 1.0 Khmer characters per
  token (5.8 for English), so 512 tokens is about 500 Khmer characters. It
  limits long-document tasks and drives latency; tokenizer/context studies
  remain separate from [R007's transfer evaluation](../R007-unseen-transfer/).

## Efficiency

Inference only, RTX 3050, batch 8. Per-decision cost p50: SIB about 12-13
ms; News 23 ms at max_len 512. Peak allocated VRAM: about 1.6-1.7 GB for the
first model loaded in a process. The about 2.5 GB recorded for R002/R003 includes
the not-yet-freed previous model, so it is not a per-model figure.

## Limitations

- Base training data is not disclosed in the Laya card, so contamination with
  SIB-200 (FLORES-200 text) or Khmer News cannot be ruled out. Every model
  starts from the base, so it would affect all of them.
- SIB-200 is too small to resolve differences of a few points.
- One fixed option order and one hand-written label text per task; the
  sensitivity to both is untested (R005).
- Khmer News: license not stated on the dataset card; articles truncated to
  title + lead; 4 test articles also in validation.
- Single training seed for R002/R003.
