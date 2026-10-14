"""Continue one fixed R006 seed on clean or synthetic 12 px OCR text."""

import argparse
import json
import os
import shutil
from dataclasses import asdict
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from laya.train import TrainConfig, dry_run, finetune, optimizer_updates
from laya_amp import laya_training_guard
from r008_identity import checkpoint_inference_inventory
from r012_identity import (
    ARTIFACT_DIR,
    PREPARED_TRAIN,
    ROOT,
    SEEDS,
    read_json,
    sha256,
    training_output_path,
    verify_locked_manifest,
    verify_preparation,
    write_json,
)


def isolated_checkpoint(source, destination):
    """Hard-link immutable files and give Laya a private tokenizer config to rewrite."""
    if destination.exists():
        raise FileExistsError(f"R012 start checkpoint already exists: {destination}")
    shutil.copytree(source, destination, copy_function=os.link)
    source_config = source / "tokenizer/tokenizer_config.json"
    destination_config = destination / "tokenizer/tokenizer_config.json"
    if not source_config.is_file() or destination_config.is_symlink():
        raise ValueError("R012 source tokenizer configuration is missing or unsafe")
    destination_config.unlink()
    shutil.copy2(source_config, destination_config)


def restore_temperature(checkpoint, source_config):
    temperature = source_config.get("temperature")
    if (
        not isinstance(temperature, list)
        or len(temperature) != 3
        or any(not isinstance(value, (int, float)) for value in temperature)
    ):
        raise ValueError("R012 source checkpoint has an invalid temperature vector")
    files = (
        checkpoint / "rl_agent_config.json",
        checkpoint / "checkpoint_latest/rl_agent_config.json",
    )
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"R012 output config missing or unsafe: {path}")
        config = read_json(path)
        config["temperature"] = temperature
        if "temperature_by_options" in source_config:
            config["temperature_by_options"] = source_config["temperature_by_options"]
        else:
            config.pop("temperature_by_options", None)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return {
        "laya_training_report_temperature": None,
        "restored_temperature": temperature,
        "changed_model_weights": False,
        "files": ["rl_agent_config.json", "checkpoint_latest/rl_agent_config.json"],
    }


def restore_inference_metadata(checkpoint, source_checkpoint, source_checkpoint_rel):
    """Restore Laya-rewritten inference metadata from the matching R006 anchor."""
    restorations = {}
    for relative in ("questions.json", "tokenizer/tokenizer_config.json"):
        source = source_checkpoint / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"R012 source inference file is missing or unsafe: {source}"
            )
        source_hash = sha256(source)
        targets = (relative, f"checkpoint_latest/{relative}")
        contents = source.read_bytes()
        for target_rel in targets:
            target = checkpoint / target_rel
            if target.is_symlink() or not target.is_file():
                raise ValueError(
                    f"R012 output inference file is missing or unsafe: {target}"
                )
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(contents)
            temporary.replace(target)
        if any(sha256(checkpoint / target) != source_hash for target in targets):
            raise ValueError(f"R012 inference metadata restoration failed: {relative}")
        restorations[relative] = {"sha256": source_hash, "paths": list(targets)}
    return {
        "source_checkpoint": source_checkpoint_rel,
        "changed_model_weights": False,
        "files": restorations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("arm", choices=("clean", "ocr12"))
    args = parser.parse_args()

    manifest, manifest_hash, _r010, _r010_hash = verify_locked_manifest(
        test_access=False
    )
    preparation = verify_preparation(manifest, manifest_hash)
    if (
        torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]
        or torch.version.cuda != manifest["environment"]["torch_cuda"]
    ):
        raise RuntimeError("R012 training device differs from the locked environment")

    model_name = f"mt-s{args.seed}"
    anchor = manifest["models"][model_name]
    source_checkpoint = ROOT / anchor["checkpoint"]
    source_inventory = checkpoint_inference_inventory(source_checkpoint)
    if source_inventory != anchor["inference_files_sha256"]:
        raise ValueError(f"R012 source checkpoint identity mismatch: {model_name}")
    source_config = read_json(source_checkpoint / "rl_agent_config.json")
    data_rel = (
        manifest["source"]["train_file"] if args.arm == "clean" else PREPARED_TRAIN
    )
    data_path = ROOT / data_rel
    expected_data_sha = (
        manifest["source"]["train_sha256"]
        if args.arm == "clean"
        else preparation["outputs_sha256"][PREPARED_TRAIN]
    )
    if sha256(data_path) != expected_data_sha:
        raise ValueError(f"R012 {args.arm} training data changed")

    output_rel = training_output_path(args.seed, args.arm)
    output = ROOT / output_rel
    start = ROOT / f"artifacts/checkpoints/R012-start-{args.arm}-s{args.seed}"
    if output.exists():
        raise FileExistsError(f"R012 continuation output already exists: {output}")
    isolated_checkpoint(source_checkpoint, start)

    recipe = manifest["recipe"]
    config = TrainConfig(
        epochs=recipe["epochs"],
        micro_batch=recipe["micro_batch"],
        grad_accum=recipe["grad_accum"],
        max_len=anchor["max_len"],
        head_max_len=anchor["head_max_len"],
        encoder_lr=recipe["encoder_lr"],
        head_lr=recipe["head_lr"],
        min_lr=recipe["min_lr"],
        weight_decay=recipe["weight_decay"],
        grad_clip=recipe["grad_clip"],
        loss=recipe["loss"],
        freeze_encoder=False,
        shuffle_options=tuple(recipe["shuffle_options"]),
        seed=args.seed,
        amp=True,
        gradient_checkpointing=True,
        calib_max=0,
        calib_frac=0.0,
        calib_seed=recipe["calib_seed"],
        eval_data=None,
    )
    preflight = dry_run(str(data_path), str(start), config)
    expected_updates = optimizer_updates(
        manifest["source"]["training_rows"],
        recipe["micro_batch"],
        recipe["grad_accum"],
        recipe["epochs"],
    )
    expected_preflight = {
        "rows_read": manifest["source"]["training_rows"],
        "valid_items": manifest["source"]["training_rows"],
        "train_items": manifest["source"]["training_rows"],
        "calibration_items": 0,
        "optimizer_updates": recipe["expected_optimizer_attempts"],
        "skipped": {},
    }
    if expected_updates != recipe["expected_optimizer_attempts"] or any(
        preflight.get(key) != value for key, value in expected_preflight.items()
    ):
        raise ValueError(f"R012 Laya preflight differs from the lock: {preflight}")

    telemetry = None
    report = None
    try:
        with (
            patch.object(torch.optim, "AdamW", PagedAdamW8bit),
            laya_training_guard() as run_telemetry,
        ):
            telemetry = run_telemetry
            report = finetune(
                str(data_path), str(start), str(output), config, device="cuda"
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
                }
            )
            write_json(output / "training_telemetry.json", telemetry_payload)

    if report is None or telemetry is None:
        raise RuntimeError("R012 training did not produce a report and telemetry")
    report_path = output / "r012_training_report.json"
    write_json(report_path, report)
    restoration = restore_temperature(output, source_config)
    restoration["laya_training_report_temperature"] = report.get("temperature")
    inference_metadata_restoration = restore_inference_metadata(
        output, source_checkpoint, anchor["checkpoint"]
    )

    if telemetry.failure or telemetry.telemetry_cleanup_failure:
        raise RuntimeError("R012 training telemetry reports a failure")
    if (
        telemetry.optimizer_attempts != expected_updates
        or telemetry.successful_updates < recipe["minimum_successful_updates"]
        or telemetry.successful_updates + telemetry.skipped_updates != expected_updates
        or telemetry.scheduler_steps != telemetry.successful_updates
    ):
        raise RuntimeError("R012 optimizer telemetry differs from the locked budget")
    if checkpoint_inference_inventory(source_checkpoint) != source_inventory:
        raise RuntimeError("R012 training modified its frozen R006 source checkpoint")

    weights_path = output / "model.safetensors"
    telemetry_path = output / "training_telemetry.json"
    if (
        not weights_path.is_file()
        or sha256(weights_path) == anchor["inference_files_sha256"]["model.safetensors"]
    ):
        raise RuntimeError("R012 training did not produce changed checkpoint weights")
    provenance = {
        "schema": "r012-training-provenance/1",
        "study": manifest["study"],
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint_directory": output_rel,
        "source_checkpoint": anchor["checkpoint"],
        "source_weights_sha256": anchor["inference_files_sha256"]["model.safetensors"],
        "source_inference_files_sha256": source_inventory,
        "training_file": data_rel,
        "training_file_sha256": expected_data_sha,
        "training_rows": manifest["source"]["training_rows"],
        "r006_manifest_sha256": manifest["source"]["r006_manifest_sha256"],
        "r010_manifest_sha256": manifest["source"]["r010_manifest_sha256"],
        "r012_manifest_sha256": manifest_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "validation_used": False,
        "checkpoint_selection": False,
        "optimizer_reset": True,
        "base_revision": recipe["base_revision"],
        "optimizer_attempts": telemetry.optimizer_attempts,
        "successful_updates": telemetry.successful_updates,
        "skipped_updates": telemetry.skipped_updates,
        "weights_sha256": sha256(weights_path),
        "inference_files_sha256": checkpoint_inference_inventory(output),
        "training_telemetry_sha256": sha256(telemetry_path),
        "training_report_sha256": sha256(report_path),
        "checkpoint_latest_config_sha256": sha256(
            output / "checkpoint_latest/rl_agent_config.json"
        ),
        "temperature_restoration": restoration,
        "inference_metadata_restoration": inference_metadata_restoration,
        "trainer_sha256": manifest["implementation_sha256"][
            "scripts/train_r012_continuation.py"
        ],
        "identity_sha256": manifest["implementation_sha256"][
            "scripts/r012_identity.py"
        ],
        "training_config": {
            **asdict(config),
            "eval_data": None,
        },
    }
    write_json(output / "training_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
