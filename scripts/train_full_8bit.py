"""Full-encoder fine-tune with an 8-bit optimizer (R003 recipe).

Usage: uv run python scripts/train_full_8bit.py [DATA OUTPUT SEED]

Defaults reproduce R003. The base checkpoint is the pinned multilingual
revision (R003 resolved the same revision through the unpinned alias).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, finetune

DATA = "data/processed/massive-km-train-full.jsonl"
EVAL = "data/processed/massive-km-validation-full.jsonl"
OUTPUT = "artifacts/checkpoints/R003-full-e2-8bit"
SEED = 13
if len(sys.argv) > 1:
    DATA, OUTPUT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])

BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
base = Path(
    snapshot_download(
        "convaiinnovations/laya",
        revision=BASE_REVISION,
        allow_patterns="multilingual/*",
    )
)

config = TrainConfig(
    epochs=2,
    micro_batch=1,
    grad_accum=32,
    max_len=512,
    head_max_len=256,
    loss="soft-ce",
    freeze_encoder=False,
    shuffle_options=("choice",),
    seed=SEED,
    amp=True,
    gradient_checkpointing=True,
    eval_data=EVAL,
)

# Laya uses torch.optim.AdamW internally.
# Replace it only for this run.
with patch.object(torch.optim, "AdamW", PagedAdamW8bit):
    report = finetune(
        DATA,
        str(base / "multilingual"),
        OUTPUT,
        config,
        device="cuda",
    )

print(report)
