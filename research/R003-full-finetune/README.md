# R003 - Full-encoder fine-tuning with an 8-bit optimizer

Question: does adapting the whole encoder fix Khmer intent accuracy, and can
it run on a 6 GB GPU?

Hypothesis: full fine-tuning far exceeds head-only training, and fits in
6 GB once optimizer state is 8-bit.

## Run

`uv run python scripts/train_full_8bit.py`

- Full encoder trainable, 2 epochs, soft-ce, micro-batch 1 x grad-accum 32,
  max_len 512, head_max_len 256, option shuffling, seed 13
- FP16 autocast with GradScaler (`laya/train.py` `_forward`), gradient
  checkpointing
- Optimizer: `bitsandbytes.optim.PagedAdamW8bit`, swapped in with
  `patch.object(torch.optim, "AdamW", ...)`. Verified at source level:
  `laya.train.train_model` calls `torch.optim.AdamW(groups,
  weight_decay=...)`, looked up at call time, so the patch applies. lr comes
  from the param groups, and betas/eps match the AdamW defaults. That the run
  actually used it rests on the patch mechanics plus the fact that it fit in
  6 GB where AdamW did not. No optimizer class or state size was logged.
- Scheduler: cosine, 696 optimizer updates (348 per epoch)
- Validation (`--eval`) was used only for reporting. Epochs were fixed and
  there was no checkpoint selection. Temperature 1.277, fit on the 400-row
  calibration slice from the train file.
- Wall clock: 36.6 min from launch (12:00:26) to checkpoint write
  (12:37:01), including load, calibration and validation passes
- Checkpoint `model.safetensors` SHA-256 `b5c682c0...7f93`

Failed run: the same config through `laya-train` with standard AdamW ran out
of the 6 GB VRAM. The exact error and peak memory were not saved.

## Results

Independent eval via `scripts/evaluate.py`:

| Model | Val acc | Test acc | Test macro-F1 | Test ECE | Test Brier |
|---|---:|---:|---:|---:|---:|
| Pretrained (R001) | 13.63% | 13.32% | 0.1060 | 0.4840 | 0.3943 |
| Head-only (R002) | 23.56% | 22.60% | 0.1549 | 0.0203 | 0.1560 |
| Full encoder (R003) | 83.13% | 83.69% | 0.7822 | 0.0193 | 0.0965 |

R003 test: AURC 0.0380, accuracy at 50% coverage 98.59%, latency p50 17.42 ms,
p95 19.95 ms (batch 1, see R001). Validation macro-F1 is 0.7946. Accuracy on test
utterances not seen verbatim in train: 83.81% (n=2,545), versus 82.98% on seen
ones (n=429), so there is no sign that overlap inflates the result.

Outcome: supported on MASSIVE Khmer with 20 candidates. Do not generalize it
to other Khmer tasks yet (R004).

## Warnings seen during training

- PyTorch: `lr_scheduler.step()` called before `optimizer.step()`. Likely
  cause: GradScaler skips the first update(s) after inf/NaN fp16 gradients,
  while Laya always steps the scheduler. Not verified (no log kept). If so,
  the cosine schedule is shifted by the skipped steps.
- Laya: conflicting schema for question `intent`. Each row has its own
  candidate set, so `questions.json` keeps only the first row's 20
  candidates. Evals pass each row's own candidates (checked by
  `scripts/summarize_runs.py`), so the reported results are unaffected.
  Anyone using `questions.json` at inference would get the wrong label set.

## Not recorded

Peak training VRAM, training-only duration, and throughput. Capture these on
the next training run. Do not retrain just to get them.

Note: `rl_agent_config.json` `training.updates/hours/epochs_completed`
(15987 / 4.97 / 4) are inherited from the base checkpoint, not from this run.
