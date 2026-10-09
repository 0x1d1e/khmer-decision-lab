import torch
from unittest.mock import patch

from bitsandbytes.optim import PagedAdamW8bit
from laya.train import TrainConfig, finetune

DATA = "data/processed/massive-km-train-full.jsonl"
EVAL = "data/processed/massive-km-validation-full.jsonl"
OUTPUT = "artifacts/checkpoints/R003-full-e2-8bit"

config = TrainConfig(
    epochs=2,
    micro_batch=1,
    grad_accum=32,
    max_len=512,
    head_max_len=256,
    loss="soft-ce",
    freeze_encoder=False,
    shuffle_options=("choice",),
    seed=13,
    amp=True,
    gradient_checkpointing=True,
    eval_data=EVAL,
)

# Laya uses torch.optim.AdamW internally.
# Replace it only for this run.
with patch.object(torch.optim, "AdamW", PagedAdamW8bit):
    report = finetune(
        DATA,
        "multilingual",
        OUTPUT,
        config,
        device="cuda",
    )

print(report)
