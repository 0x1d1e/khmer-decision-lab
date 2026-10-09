"""Fit and persist one R008 validation temperature map.

Usage: uv run python scripts/calibrate_r008.py CHECKPOINT TASK
TASK is khmer-news or sib200-km. The locked manifest selects the exact
validation calibration file; this command never reads a test file.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from laya.train import calibration_records, items_from_rows, read_data
from r008_identity import (
    ROOT,
    load_r008_agent,
    verify_checkpoint_training_provenance,
    verify_locked_manifest,
)

MANIFEST = Path("datasets/manifests/r008-specialist-controls.json")
TASKS = ("khmer-news", "sib200-km")
MAX_LEN = 512
HEAD_MAX_LEN = 256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest_path):
    return verify_locked_manifest(manifest_path)


def calibration_records_bf16(agent, items):
    """Generate calibration logits under the locked inference precision."""
    if (
        agent.device.type != "cuda"
        or agent.dtype != torch.bfloat16
        or not agent.amp_enabled
    ):
        raise ValueError("R008 validation calibration requires CUDA BF16 inference")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return calibration_records(
            agent.model,
            agent.tok,
            items,
            torch.device("cuda"),
            MAX_LEN,
            HEAD_MAX_LEN,
            parallel=agent.parallel_options,
        )


def fit_calibration(checkpoint, task, manifest_path=MANIFEST, output=None):
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    manifest, manifest_hash = verify_manifest(manifest_path)
    provenance = verify_checkpoint_training_provenance(
        checkpoint, task, manifest, manifest_hash
    )
    task_manifest = manifest["validation_calibration"][task]
    validation_path = ROOT / task_manifest["calibration_file"]
    if (
        task_manifest["calibration_file"]
        != f"data/processed/r008-{task}-calibration.jsonl"
        or task_manifest["test_file"] != f"data/processed/{task}-test.jsonl"
    ):
        raise ValueError("R008 manifest target mapping is inconsistent")
    if sha256(validation_path) != task_manifest["calibration_sha256"]:
        raise ValueError("R008 validation calibration file fingerprint mismatch")

    checkpoint = Path(checkpoint).resolve()
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha256 = sha256(weights)

    agent, inference_dtype = load_r008_agent(checkpoint, manifest)
    if (
        agent.cfg.get("max_len") != MAX_LEN
        or agent.cfg.get("head_max_len") != HEAD_MAX_LEN
    ):
        raise ValueError("checkpoint token budgets differ from the locked R008 recipe")
    rows = read_data(str(validation_path))
    if len(rows) != task_manifest["calibration_rows"]:
        raise ValueError("R008 validation calibration row count mismatch")
    items, skipped = items_from_rows(
        agent.tok,
        rows,
        MAX_LEN,
        HEAD_MAX_LEN,
        label_smoothing=0.0,
    )
    if skipped or len(items) != len(rows):
        raise ValueError(f"R008 calibration inputs were skipped: {skipped!r}")
    records = calibration_records_bf16(agent, items)
    if (
        agent.device.type != "cuda"
        or str(agent.dtype).removeprefix("torch.") != inference_dtype
        or not agent.amp_enabled
    ):
        raise ValueError("R008 inference runtime changed precision during calibration")
    fitted = agent.fit_temperatures(records)

    output = Path(output) if output else checkpoint / f"calibration-{task}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    agent.save_calibration(str(temporary))
    payload = json.loads(temporary.read_text(encoding="utf-8"))
    payload["r008"] = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": weights_sha256,
        "training_arm": provenance["training_arm"],
        "training_provenance_sha256": sha256(checkpoint / "training_provenance.json"),
        "validation_file": task_manifest["calibration_file"],
        "validation_sha256": task_manifest["calibration_sha256"],
        "validation_rows": len(rows),
        "usable_validation_items": len(items),
        "inference_device": agent.device.type,
        "inference_dtype": inference_dtype,
        "locked_manifest_sha256": manifest_hash,
    }
    final_temporary = output.with_suffix(output.suffix + ".new")
    final_temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(final_temporary, output)
    temporary.unlink(missing_ok=True)
    return output, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output, fitted = fit_calibration(
        args.checkpoint, args.task, args.manifest, args.output
    )
    print(json.dumps({"calibration": str(output), "fit": fitted}, indent=2))


if __name__ == "__main__":
    main()
