# R002 - Frozen-encoder fine-tuning

Question: does training only the Laya decision head on MASSIVE Khmer fix the
baseline?

Hypothesis: head-only training improves calibration, but a frozen encoder
limits Khmer accuracy.

## Run

```
uv run laya-train --data data/processed/massive-km-train-full.jsonl \
  --base multilingual --out artifacts/checkpoints/R002-head --device cuda \
  --loss soft-ce --freeze-encoder --epochs 2 --micro-batch 16 --grad-accum 2 \
  --max-len 512 --head-max-len 256 --shuffle-options --seed 13
```

- Resolved config: `artifacts/checkpoints/R002-head/train_report.json`
  (encoder_lr 2.5e-5, head_lr 1e-4, AdamW, cosine to 1e-6 over 696 updates, Laya default AMP = fp16 autocast on CUDA)
- No `--eval` was given, so `train_report.json` metrics come from the
  in-sample calibration slice, not held-out data.
- Temperature 0.962, fit on the 400-row calibration slice
- Wall clock: about 6 min from launch to checkpoint, including load and
  calibration
- Earlier launches with micro-batch 1, 2 and 8 were replaced by this one. The
  reason was not recorded.
- Checkpoint `model.safetensors` SHA-256 `2541f8ed...a062`

## Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 23.56% | 22.60% |
| Macro-F1 | 0.1561 | 0.1549 |
| ECE | 0.0355 | 0.0203 |
| Brier | 0.1622 | 0.1560 |

Outcome: supported. Calibration became good and accuracy rose about 10
points, but accuracy is still low.
