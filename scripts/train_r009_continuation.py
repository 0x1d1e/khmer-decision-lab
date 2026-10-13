"""Continue one locked R008 SIB-200 specialist for R009."""

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from laya.train import TrainConfig, dry_run, finetune, optimizer_updates
from laya_amp import laya_training_guard
from r008_identity import checkpoint_inference_inventory
from r009_identity import (
    ROOT,
    SEEDS,
    sha256,
    verify_locked_manifest,
    verify_r008_anchor,
    write_json,
)


def isolated_checkpoint(source, destination):
    """Hard-link immutable checkpoint files; copy the tokenizer config Laya may patch."""
    if destination.exists():
        raise FileExistsError(
            f"isolated start checkpoint already exists: {destination}"
        )
    shutil.copytree(source, destination, copy_function=os.link)
    source_config = source / "tokenizer/tokenizer_config.json"
    destination_config = destination / "tokenizer/tokenizer_config.json"
    if not source_config.is_file() or destination_config.is_symlink():
        raise ValueError("R008 tokenizer configuration is missing or unsafe")
    destination_config.unlink()
    shutil.copy2(source_config, destination_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=SEEDS)
    args = parser.parse_args()

    manifest, r009_hash, r008_manifest, r008_hash = verify_locked_manifest()
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]
        or torch.version.cuda != manifest["environment"]["cuda"]
    ):
        raise RuntimeError("R009 requires the locked RTX 3050 CUDA/BF16 environment")
    anchor, source_provenance, _ = verify_r008_anchor(
        args.seed, "specialist", manifest, r008_manifest, r008_hash
    )
    source_checkpoint = ROOT / anchor["checkpoint"]
    source_inventory = checkpoint_inference_inventory(source_checkpoint)
    data = ROOT / manifest["source"]["training_file"]
    validation = ROOT / manifest["source"]["validation_file"]
    output_rel = f"artifacts/checkpoints/R009-sib-s{args.seed}-cont"
    output = ROOT / output_rel
    start = ROOT / f"artifacts/checkpoints/R009-start-s{args.seed}"
    if output.exists():
        raise FileExistsError(f"R009 output already exists: {output}")
    if sha256(data) != manifest["source"]["training_sha256"]:
        raise ValueError("R009 training data fingerprint mismatch")
    if sha256(validation) != manifest["source"]["validation_sha256"]:
        raise ValueError("R009 validation data fingerprint mismatch")

    isolated_checkpoint(source_checkpoint, start)
    recipe = manifest["recipe"]
    config = TrainConfig(
        epochs=recipe["continuation_epochs"],
        micro_batch=recipe["micro_batch"],
        grad_accum=recipe["grad_accum"],
        max_len=recipe["max_len"],
        head_max_len=recipe["head_max_len"],
        encoder_lr=recipe["encoder_lr"],
        head_lr=recipe["head_lr"],
        min_lr=recipe["min_lr"],
        weight_decay=recipe["weight_decay"],
        grad_clip=recipe["grad_clip"],
        loss=recipe["loss"],
        freeze_encoder=recipe["freeze_encoder"],
        shuffle_options=tuple(recipe["shuffle_options"]),
        seed=args.seed,
        amp=recipe["amp"],
        gradient_checkpointing=recipe["gradient_checkpointing"],
        calib_max=0,
        calib_frac=0.0,
        calib_seed=recipe["calib_seed"],
        eval_data=str(validation),
    )
    preflight = dry_run(str(data), str(start), config)
    expected_updates = optimizer_updates(
        manifest["source"]["training_rows"],
        recipe["micro_batch"],
        recipe["grad_accum"],
        recipe["continuation_epochs"],
    )
    expected_preflight = {
        "rows_read": manifest["source"]["training_rows"],
        "valid_items": manifest["source"]["training_rows"],
        "train_items": manifest["source"]["training_rows"],
        "calibration_items": 0,
        "optimizer_updates": recipe["expected_optimizer_attempts_per_seed"],
        "eval_rows_read": manifest["source"]["validation_rows"],
        "eval_valid_items": manifest["source"]["validation_rows"],
        "eval_skipped": {},
    }
    if expected_updates != recipe["expected_optimizer_attempts_per_seed"] or any(
        preflight.get(key) != value for key, value in expected_preflight.items()
    ):
        raise ValueError(
            f"R009 Laya preflight differs from the locked budget: {preflight}"
        )

    telemetry = None
    try:
        with (
            patch.object(torch.optim, "AdamW", PagedAdamW8bit),
            laya_training_guard() as run_telemetry,
        ):
            telemetry = run_telemetry
            report = finetune(str(data), str(start), str(output), config, device="cuda")
    finally:
        if telemetry is not None:
            telemetry_payload = telemetry.as_dict()
            telemetry_payload.update(
                {
                    "device": "cuda",
                    "gpu": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
            )
            write_json(output / "training_telemetry.json", telemetry_payload)

    if telemetry is None or telemetry.failure or telemetry.telemetry_cleanup_failure:
        raise RuntimeError("R009 training telemetry reports a failure")
    if (
        telemetry.optimizer_attempts != expected_updates
        or telemetry.successful_updates < 1
        or telemetry.successful_updates + telemetry.skipped_updates != expected_updates
        or telemetry.scheduler_steps != telemetry.successful_updates
    ):
        raise RuntimeError("R009 optimizer telemetry does not match the locked budget")
    if checkpoint_inference_inventory(source_checkpoint) != source_inventory:
        raise RuntimeError("R009 training modified its R008 source checkpoint")

    report_path = output / "r009_training_report.json"
    write_json(report_path, report)
    weights_path = output / "model.safetensors"
    telemetry_path = output / "training_telemetry.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if sha256(weights_path) == anchor["weights_sha256"]:
        raise RuntimeError("R009 training produced unchanged checkpoint weights")
    training_provenance = {
        "schema": "r009-training-provenance/1",
        "study": "R009-sib-specialist-continuation",
        "seed": args.seed,
        "checkpoint_directory": output_rel,
        "source_checkpoint": anchor["checkpoint"],
        "source_weights_sha256": anchor["weights_sha256"],
        "source_training_provenance_sha256": anchor["training_provenance_sha256"],
        "source_training_telemetry_sha256": anchor["training_telemetry_sha256"],
        "source_training_arm": source_provenance["training_arm"],
        "training_file": manifest["source"]["training_file"],
        "training_file_sha256": manifest["source"]["training_sha256"],
        "training_rows": manifest["source"]["training_rows"],
        "validation_file": manifest["source"]["validation_file"],
        "validation_file_sha256": manifest["source"]["validation_sha256"],
        "validation_rows": manifest["source"]["validation_rows"],
        "validation_used_for_checkpoint_selection": False,
        "r008_manifest_sha256": r008_hash,
        "r009_manifest_sha256": r009_hash,
        "base_revision": manifest["recipe"]["base_revision"],
        "continuation_epochs": recipe["continuation_epochs"],
        "optimizer_reset": recipe["optimizer_reset"],
        "optimizer_attempts": telemetry.optimizer_attempts,
        "successful_updates": telemetry.successful_updates,
        "skipped_updates": telemetry.skipped_updates,
        "weights_sha256": sha256(weights_path),
        "inference_files_sha256": checkpoint_inference_inventory(output),
        "training_telemetry_sha256": sha256(telemetry_path),
        "training_report_sha256": sha256(report_path),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "identity_sha256": sha256(ROOT / "scripts/r009_identity.py"),
        "training_config": {
            **asdict(config),
            "eval_data": manifest["source"]["validation_file"],
        },
    }
    write_json(output / "training_provenance.json", training_provenance)
    print(json.dumps(training_provenance, indent=2))


if __name__ == "__main__":
    main()
