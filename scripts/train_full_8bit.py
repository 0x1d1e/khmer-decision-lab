"""Full-encoder fine-tune with an 8-bit optimizer (R003 recipe).

Usage: uv run python scripts/train_full_8bit.py [DATA OUTPUT SEED]
       [--no-training-calibration] [--r008-manifest MANIFEST]

Defaults reproduce R003. The base checkpoint is the pinned multilingual
revision (R003 resolved the same revision through the unpinned alias). R008
requires a locked manifest and disables Laya's training-file calibration split.
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, dry_run, finetune
from laya_amp import laya_training_guard
from r008_identity import (
    validate_training_conversion,
    verify_locked_manifest,
    write_training_conversion_audit,
    write_training_provenance,
)

DEFAULT_DATA = "data/processed/massive-km-train-full.jsonl"
EVAL = "data/processed/massive-km-validation-full.jsonl"
DEFAULT_OUTPUT = "artifacts/checkpoints/R003-full-e2-8bit"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("seed", nargs="?", type=int, default=13)
    parser.add_argument("--no-training-calibration", action="store_true")
    parser.add_argument("--r008-manifest", type=Path)
    return parser.parse_args()


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_r008_manifest(args):
    manifest_path = args.r008_manifest
    if manifest_path is None:
        return None
    if not args.no_training_calibration:
        raise ValueError(
            "R008 runs must disable Laya's training-file calibration split"
        )
    manifest, manifest_hash = verify_locked_manifest(manifest_path)
    if manifest.get("recipe", {}).get("base_revision") != BASE_REVISION:
        raise ValueError("R008 manifest base revision does not match this trainer")

    selected_arm = None
    for name, arm in manifest.get("training_split", {}).get("arms", {}).items():
        if (
            Path(args.data).resolve()
            == (Path(__file__).resolve().parents[1] / arm["file"]).resolve()
        ):
            selected_arm = name
            if sha256(args.data) != arm["sha256"]:
                raise ValueError("R008 training data fingerprint mismatch")
            break
    if selected_arm is None:
        raise ValueError("training data is not an arm in the locked R008 manifest")

    prefixes = {
        "multitask": "R008-mt",
        "khmer-news-specialist": "R008-news",
        "sib200-specialist": "R008-sib",
    }
    expected_output = f"artifacts/checkpoints/{prefixes[selected_arm]}-s{args.seed}"
    if args.output != expected_output:
        raise ValueError(f"R008 output must be {expected_output}")
    if args.seed not in manifest.get("recipe", {}).get("seeds", []):
        raise ValueError("seed is not preregistered in the R008 manifest")
    return selected_arm, manifest_hash, manifest


args = parse_args()
r008_run = verify_r008_manifest(args)
DATA, OUTPUT, SEED = args.data, args.output, args.seed
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
    encoder_lr=2.5e-5,
    head_lr=1e-4,
    min_lr=1e-6,
    weight_decay=0.01,
    grad_clip=1.0,
    loss="soft-ce",
    freeze_encoder=False,
    shuffle_options=("choice",),
    seed=SEED,
    amp=True,
    gradient_checkpointing=True,
    calib_max=0 if args.no_training_calibration else 400,
    calib_frac=0.0 if args.no_training_calibration else 0.1,
    calib_seed=20260922,
    eval_data=None if args.no_training_calibration else EVAL,
)

preflight = None
if r008_run is not None:
    training_arm, _manifest_hash, r008_manifest = r008_run
    preflight = dry_run(DATA, str(base / "multilingual"), config)
    validate_training_conversion(preflight, training_arm, r008_manifest)


def write_telemetry(telemetry):
    output = Path(OUTPUT) / "training_telemetry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = telemetry.as_dict()
    payload.update(
        {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    )
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


telemetry = None
try:
    # Keep the reviewed scheduler patch scoped to R008. Legacy R003 calls retain
    # the behavior they had before this guard was added.
    guard = laya_training_guard() if r008_run is not None else nullcontext(None)
    with patch.object(torch.optim, "AdamW", PagedAdamW8bit), guard as telemetry:
        report = finetune(
            DATA,
            str(base / "multilingual"),
            OUTPUT,
            config,
            device="cuda",
        )
        if r008_run is not None:
            training_arm, _manifest_hash, _manifest = r008_run
            write_training_conversion_audit(
                OUTPUT,
                training_arm,
                SEED,
                DATA,
                args.r008_manifest,
                preflight,
                report,
            )
finally:
    if telemetry is not None:
        try:
            write_telemetry(telemetry)
        except Exception as exc:
            if telemetry.failure is None:
                raise
            print(f"failed to write training telemetry: {exc}", file=sys.stderr)

if r008_run is not None:
    training_arm, _manifest_hash, _manifest = r008_run
    write_training_provenance(OUTPUT, training_arm, SEED, DATA, args.r008_manifest)

print(report)
