"""Budget-matched SIB-200 specialist training (R016).

Trains a SIB-200 specialist from scratch whose optimizer-update budget
matches the R006/R008 multitask arm (~1,060 updates) instead of the R008
specialist's 44-update budget, to test whether specialist under-training
explains its lower SIB-200 accuracy. Same recipe as R008 otherwise; Laya's
training-file calibration split is disabled and no evaluation file is passed
during training.

Usage:
    uv run python scripts/train_r016_budget_specialist.py [SEED]

SEED is 13 or 14 (default 13). Each seed runs in a separate process and must
be GPU-serialized with an exclusive flock on /tmp/khmer-gpu.lock:

    flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 13'
    flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 14'

Outputs artifacts/checkpoints/R016-budget-sib-s{seed}/; R008's locked
artifacts/checkpoints/R008-sib-s{seed} are never touched.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, dry_run, finetune, optimizer_updates
from laya_amp import laya_training_guard

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/r008-sib200-km-train.jsonl"
DATA_SHA256 = "ed2ebd5fbb9e8bdbc78dd9eccaeaa146ee117c1809afc36305036740657930c3"
BASE_MODEL = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
SEEDS = (13, 14)
EPOCHS = 48
MICRO_BATCH = 1
GRAD_ACCUM = 32
MAX_LEN = 512
HEAD_MAX_LEN = 256
TARGET_UPDATES = 1060
STUDY = "R016-budget-matched-specialist"


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", type=int, default=SEEDS[0], choices=SEEDS)
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def output_for(seed):
    return ROOT / f"artifacts/checkpoints/R016-budget-sib-s{seed}"


def main():
    args = parse_args()
    seed = args.seed
    data = Path(args.data)
    output = Path(args.output) if args.output else output_for(seed)
    epochs = args.epochs

    if not data.is_file():
        data = ROOT / args.data
    if sha256(data) != DATA_SHA256:
        raise ValueError(f"R016 training data fingerprint mismatch: {data}")
    rows = sum(
        1 for line in data.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if rows != 681:
        raise ValueError(f"R016 training data has {rows} rows, expected 681")
    if output.resolve() in {
        (ROOT / "artifacts/checkpoints" / f"R008-sib-s{s}").resolve() for s in SEEDS
    }:
        raise ValueError("refusing to write over an R008 locked checkpoint")
    if output.exists():
        raise FileExistsError(f"R016 output already exists: {output}")

    expected_updates = optimizer_updates(rows, MICRO_BATCH, GRAD_ACCUM, epochs)
    updates_per_epoch = expected_updates // epochs
    if expected_updates > TARGET_UPDATES:
        raise ValueError(
            f"{epochs} epochs gives {expected_updates} updates, over the {TARGET_UPDATES} target"
        )
    closer = optimizer_updates(rows, MICRO_BATCH, GRAD_ACCUM, epochs + 1)
    if closer <= TARGET_UPDATES and closer > expected_updates:
        raise ValueError(
            f"{epochs + 1} epochs ({closer} updates) is closer to {TARGET_UPDATES} "
            f"without going over; use {epochs + 1} epochs"
        )

    base = Path(
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            allow_patterns="multilingual/*",
        )
    )

    config = TrainConfig(
        epochs=epochs,
        micro_batch=MICRO_BATCH,
        grad_accum=GRAD_ACCUM,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        loss="soft-ce",
        freeze_encoder=False,
        shuffle_options=("choice",),
        seed=seed,
        amp=True,
        gradient_checkpointing=True,
        calib_max=0,
        calib_frac=0.0,
        calib_seed=20260922,
        eval_data=None,
    )

    preflight = dry_run(str(data), str(base / "multilingual"), config)
    if (
        preflight["rows_read"] != rows
        or preflight["valid_items"] != rows
        or preflight["train_items"] != rows
        or preflight["calibration_items"] != 0
        or preflight["skipped"] != {}
        or preflight["optimizer_updates"] != expected_updates
    ):
        raise ValueError(
            f"R016 Laya preflight differs from the locked budget: {preflight}"
        )

    telemetry = None
    try:
        guard = laya_training_guard()
        with patch.object(torch.optim, "AdamW", PagedAdamW8bit), guard as telemetry:
            report = finetune(
                str(data),
                str(base / "multilingual"),
                str(output),
                config,
                device="cuda",
            )
    finally:
        if telemetry is not None:
            telemetry_payload = telemetry.as_dict()
            telemetry_payload.update(
                {
                    "device": "cuda",
                    "gpu": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            )
            output.mkdir(parents=True, exist_ok=True)
            telemetry_path = output / "training_telemetry.json"
            temporary = telemetry_path.with_suffix(telemetry_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(telemetry_payload, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(telemetry_path)

    if (
        telemetry is None
        or telemetry.failure
        or telemetry.telemetry_cleanup_failure
        or telemetry.optimizer_attempts != expected_updates
        or telemetry.successful_updates < 1
        or telemetry.successful_updates + telemetry.skipped_updates != expected_updates
        or telemetry.scheduler_steps != telemetry.successful_updates
    ):
        raise RuntimeError(
            f"R016 training telemetry does not satisfy the update budget: {telemetry}"
        )

    weights_path = output / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    report_path = output / "r016_training_report.json"
    report_payload = dict(report)
    report_payload["output_dir"] = str(report_payload.get("output_dir", output))
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)

    training_provenance = {
        "schema": "r016-training-provenance/1",
        "study": STUDY,
        "seed": seed,
        "checkpoint_directory": str(output.relative_to(ROOT)),
        "training_file": str(data.relative_to(ROOT)),
        "training_file_sha256": DATA_SHA256,
        "training_rows": rows,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "epochs": epochs,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "encoder_lr": 2.5e-5,
        "head_lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "loss": "soft-ce",
        "freeze_encoder": False,
        "shuffle_options": ["choice"],
        "amp": True,
        "gradient_checkpointing": True,
        "optimizer": "bitsandbytes.optim.PagedAdamW8bit",
        "calib_max": 0,
        "calib_frac": 0.0,
        "eval_data": None,
        "target_optimizer_updates": TARGET_UPDATES,
        "expected_optimizer_updates": expected_updates,
        "updates_per_epoch": updates_per_epoch,
        "optimizer_attempts": telemetry.optimizer_attempts,
        "successful_updates": telemetry.successful_updates,
        "skipped_updates": telemetry.skipped_updates,
        "scheduler_steps": telemetry.scheduler_steps,
        "training_seconds": telemetry.training_seconds,
        "peak_allocated_bytes": telemetry.peak_allocated_bytes,
        "weights_sha256": sha256(weights_path),
        "training_telemetry_sha256": sha256(telemetry_path),
        "training_report_sha256": sha256(report_path),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "guard_sha256": sha256(ROOT / "scripts/laya_amp.py"),
        "finished_at": telemetry_payload["finished_at"],
    }
    provenance_path = output / "training_provenance.json"
    temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(training_provenance, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(provenance_path)

    print(json.dumps(training_provenance, indent=2))


if __name__ == "__main__":
    main()
