"""Train one locked R013 arm from the pinned multilingual base."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from bitsandbytes.optim import PagedAdamW8bit
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, dry_run, finetune
from laya_amp import laya_training_guard
from r008_identity import checkpoint_inference_inventory
from r013_identity import (
    ARMS,
    BASE_REPO,
    BASE_REVISION,
    BASE_SUBDIR,
    MANIFEST,
    OUTPUTS,
    RECIPE,
    ROOT,
    SEEDS,
    TRAIN_FILES,
    normalized_base_checkpoint_inventory,
    read_json,
    sha256,
    verify_cuda_environment,
    verify_locked_manifest,
    verify_preparation,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("arm", choices=ARMS)
    return parser.parse_args()


def _base_provenance(inventory):
    return {
        "repo": BASE_REPO,
        "revision": BASE_REVISION,
        "subdir": BASE_SUBDIR,
        "inference_files_sha256": inventory,
    }


def make_config(seed):
    return TrainConfig(
        epochs=RECIPE["epochs"],
        micro_batch=RECIPE["micro_batch"],
        grad_accum=RECIPE["grad_accum"],
        max_len=RECIPE["max_len"],
        head_max_len=RECIPE["head_max_len"],
        encoder_lr=RECIPE["encoder_lr"],
        head_lr=RECIPE["head_lr"],
        min_lr=RECIPE["min_lr"],
        weight_decay=RECIPE["weight_decay"],
        grad_clip=RECIPE["grad_clip"],
        loss=RECIPE["loss"],
        freeze_encoder=RECIPE["freeze_encoder"],
        shuffle_options=tuple(RECIPE["shuffle_options"]),
        seed=seed,
        amp=RECIPE["amp"],
        gradient_checkpointing=RECIPE["gradient_checkpointing"],
        calib_max=RECIPE["calib_max"],
        calib_frac=RECIPE["calib_frac"],
        eval_data=RECIPE["eval_data"],
    )


def train(seed, arm):
    manifest, manifest_sha256 = verify_locked_manifest(MANIFEST)
    verify_cuda_environment(manifest)
    preparation = verify_preparation(manifest, manifest_sha256, test_access=False)
    output = ROOT / OUTPUTS[(seed, arm)]
    if output.exists():
        raise FileExistsError(f"R013 output already exists: {output}")

    base_snapshot = Path(
        snapshot_download(
            BASE_REPO,
            revision=BASE_REVISION,
            allow_patterns=f"{BASE_SUBDIR}/*",
        )
    )
    base = base_snapshot / BASE_SUBDIR
    # Validate cache links before Laya's compatibility fix can replace a symlink.
    base_inventory = normalized_base_checkpoint_inventory(base)
    if base_inventory != manifest["base"]["inference_files_sha256"]:
        raise ValueError("R013 multilingual base checkpoint fingerprint mismatch")

    config = make_config(seed)
    training_file = ROOT / TRAIN_FILES[arm]
    preflight = dry_run(str(training_file), str(base), config)
    if (
        preflight.get("rows_read") != RECIPE["training_rows_per_arm"]
        or preflight.get("valid_items") != RECIPE["training_rows_per_arm"]
        or preflight.get("train_items") != RECIPE["training_rows_per_arm"]
        or preflight.get("calibration_items") != 0
        or preflight.get("skipped")
        or preflight.get("optimizer_updates") != RECIPE["expected_optimizer_attempts"]
    ):
        raise ValueError(f"R013 training dry-run failed its locked budget: {preflight}")

    telemetry = None
    try:
        with (
            patch.object(torch.optim, "AdamW", PagedAdamW8bit),
            laya_training_guard() as telemetry,
        ):
            report = finetune(
                str(training_file),
                str(base),
                str(output),
                config,
                device="cuda",
            )
    finally:
        if telemetry is not None:
            payload = telemetry.as_dict()
            payload.update(
                {
                    "device": "cuda",
                    "gpu": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
            )
            write_json(output / "training_telemetry.json", payload)

    output_inventory = checkpoint_inference_inventory(output)
    run_recipe = {**RECIPE, "seed": seed, "arm": arm}
    provenance = {
        "study": manifest["study"],
        "manifest_sha256": manifest_sha256,
        "preparation_sha256": sha256(ROOT / "artifacts/runs/R013/preparation.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "source_sha256": manifest["source"]["sha256"],
        "preparer_sha256": manifest["implementation_sha256"]["scripts/prepare_r013.py"],
        "seed": seed,
        "arm": arm,
        "recipe": run_recipe,
        "training_file": TRAIN_FILES[arm],
        "training_file_sha256": preparation["outputs_sha256"][TRAIN_FILES[arm]],
        "training_rows": preflight["train_items"],
        "dry_run": preflight,
        "base": _base_provenance(base_inventory),
        "output_inventory": output_inventory,
        "training_telemetry_sha256": sha256(output / "training_telemetry.json"),
        "train_report_sha256": sha256(output / "train_report.json"),
    }
    write_json(output / "training_provenance.json", provenance)
    print(
        json.dumps(
            {
                "seed": seed,
                "arm": arm,
                "output": str(output),
                "telemetry": read_json(output / "training_telemetry.json"),
                "train_items": report["train_items"],
                "temperature": report["temperature"],
            },
            indent=2,
        )
    )
    return provenance


def main():
    args = parse_args()
    train(args.seed, args.arm)


if __name__ == "__main__":
    main()
