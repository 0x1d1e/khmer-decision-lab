# R003 — Khmer Full-Encoder Fine-Tuning

## Configuration

- Model: Laya Multilingual (322M)
- GPU: RTX 3050 Laptop, 6 GB
- Dataset: MASSIVE Khmer
- Optimizer: PagedAdamW8bit
- Epochs: 2
- Micro-batch: 1
- Gradient accumulation: 32
- Maximum sequence length: 512
- Precision: FP16 autocast
- Seed: 13

## Results

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Pretrained | 13.63% | 13.32% |
| Head-only, 2 epochs | 23.56% | 22.60% |
| Full encoder, 2 epochs | 83.13% | 83.69% |

**Final test metrics (full encoder):**

- Accuracy: 83.69%
- ECE: 0.0193
- Brier: 0.0965
- AURC: 0.0380
- Median latency: 17.42 ms
- P95 latency: 19.95 ms

## Limitations

- Evaluated with 20 candidate intents per example.
- Cross-task generalization remains unverified for this checkpoint.
- Single training seed.
- Peak training VRAM and total training duration not yet recorded.
- Optimizer substitution and scheduler warning require further investigation.
