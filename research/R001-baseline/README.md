# R001 - Pretrained baseline

Question: how well does the untouched Laya Multilingual checkpoint choose
Khmer intents among 20 candidates?

Hypothesis: zero-shot accuracy is well above chance (5%) but low, and the
model is overconfident on Khmer.

## Setup

- Model alias `multilingual` = `convaiinnovations/laya`, subfolder
  `multilingual`, Hub revision `7b928d828b7b0e022f929d9bd2e44165aa270148`
  (resolved at download, not pinned by Laya). Encoder `jhu-clsp/mmBERT-base`.
  `model.safetensors` SHA-256 `9d628fd9...f204`.
- Inference: Laya defaults on CUDA (bf16 autocast from checkpoint
  `amp_dtype`). Each row has a distinct candidate set, so every request is a
  single example (`rows_alone` = all rows).

## Results

Full splits, from `uv run python scripts/eval_massive.py {validation,test}`:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 13.63% | 13.32% |
| Macro-F1 | 0.1056 | 0.1060 |
| ECE | 0.4748 | 0.4840 |
| Brier | 0.3845 | 0.3943 |
| Latency p50 / p95 | 17.32 / 19.87 ms | 17.40 / 20.14 ms |

Outcome: hypothesis supported. Accuracy is 2.7x chance, and mean confidence
(0.61) far exceeds accuracy.

`pilot-validation-100.md` is an earlier 100-row validation sample (19.00%)
made with `laya-evals run`. It is a pilot only. Do not compare it with the
full-split numbers.
