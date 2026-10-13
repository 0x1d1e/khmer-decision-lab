"""Fit R009 temperature maps on the fixed SIB-200 validation split only."""

import argparse
import json
from pathlib import Path

import torch
from calibrate_r008 import calibration_records_bf16
from laya.train import items_from_rows, read_data
from r008_identity import ROOT, checkpoint_inference_inventory, load_r008_agent
from r009_identity import (
    SEEDS,
    read_json,
    sha256,
    verify_locked_manifest,
    verify_r009_training,
    write_json,
)


def record_laya_tokenizer_normalization(checkpoint, provenance):
    """Bind Laya's deterministic tokenizer compatibility rewrite to provenance."""
    checkpoint = Path(checkpoint)
    before = provenance.get("inference_files_sha256")
    after = checkpoint_inference_inventory(checkpoint)
    if after == before:
        return provenance
    if not isinstance(before, dict) or set(before) != set(after):
        raise ValueError(
            "R009 inference inventory changed beyond tokenizer normalization"
        )

    relative = "tokenizer/tokenizer_config.json"
    changed = {key for key in before if before[key] != after[key]}
    original_path = checkpoint / "checkpoint_latest" / relative
    if changed != {relative} or sha256(original_path) != before.get(relative):
        raise ValueError(
            "R009 inference inventory changed beyond tokenizer normalization"
        )

    original = read_json(original_path)
    expected = dict(original)
    if expected.get("tokenizer_class") in (None, "TokenizersBackend"):
        expected["tokenizer_class"] = "PreTrainedTokenizerFast"
        expected.pop("backend", None)
        expected.pop("is_local", None)
    extra = expected.get("extra_special_tokens")
    if isinstance(extra, list):
        expected["extra_special_tokens"] = {
            f"extra_{index}": token for index, token in enumerate(extra)
        }
    if read_json(checkpoint / relative) != expected:
        raise ValueError(
            "R009 tokenizer config differs from Laya's compatibility rewrite"
        )

    provenance["inference_files_sha256"] = after
    provenance["post_training_inference_normalization"] = {
        "stage": "calibration model load before temperature fitting",
        "library": "Laya 0.4.1",
        "file": relative,
        "before_sha256": before[relative],
        "after_sha256": after[relative],
        "operation": "Laya _fix_tokenizer_config compatibility rewrite",
        "model_weights_changed": False,
    }
    write_json(checkpoint / "training_provenance.json", provenance)
    return provenance


def calibrate(seed):
    manifest, r009_hash, r008_manifest, r008_hash = verify_locked_manifest()
    checkpoint, provenance, _ = verify_r009_training(
        seed, manifest, r009_hash, r008_manifest, r008_hash
    )
    validation_path = ROOT / manifest["source"]["validation_file"]
    if sha256(validation_path) != manifest["source"]["validation_sha256"]:
        raise ValueError("R009 validation file fingerprint mismatch")

    agent, inference_dtype = load_r008_agent(checkpoint, r008_manifest)
    provenance = record_laya_tokenizer_normalization(checkpoint, provenance)
    if (
        torch.version.cuda != manifest["environment"]["cuda"]
        or torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]
    ):
        raise RuntimeError("R009 calibration requires the locked GPU/CUDA environment")
    if (
        agent.cfg.get("max_len") != manifest["recipe"]["max_len"]
        or agent.cfg.get("head_max_len") != manifest["recipe"]["head_max_len"]
    ):
        raise ValueError("R009 checkpoint token budgets differ from the locked recipe")
    rows = read_data(str(validation_path))
    if len(rows) != manifest["source"]["validation_rows"]:
        raise ValueError("R009 validation row count mismatch")
    items, skipped = items_from_rows(
        agent.tok,
        rows,
        manifest["recipe"]["max_len"],
        manifest["recipe"]["head_max_len"],
        label_smoothing=0.0,
    )
    if skipped or len(items) != len(rows):
        raise ValueError(f"R009 validation inputs were skipped: {skipped!r}")
    records = calibration_records_bf16(agent, items)
    fitted = agent.fit_temperatures(records)

    output = checkpoint / "calibration-sib200-km.json"
    temporary = output.with_suffix(".json.tmp")
    agent.save_calibration(str(temporary))
    payload = read_json(temporary)
    payload["r009"] = {
        "study": "R009-sib-specialist-continuation",
        "seed": seed,
        "checkpoint": f"artifacts/checkpoints/R009-sib-s{seed}-cont",
        "checkpoint_weights_sha256": provenance["weights_sha256"],
        "training_provenance_sha256": sha256(checkpoint / "training_provenance.json"),
        "validation_file": manifest["source"]["validation_file"],
        "validation_sha256": manifest["source"]["validation_sha256"],
        "validation_rows": len(rows),
        "usable_validation_items": len(items),
        "inference_device": agent.device.type,
        "inference_dtype": inference_dtype,
        "r009_manifest_sha256": r009_hash,
        "r008_manifest_sha256": r008_hash,
    }
    write_json(output, payload)
    temporary.unlink(missing_ok=True)
    return output, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    output, fitted = calibrate(args.seed)
    print(json.dumps({"calibration": str(output), "fit": fitted}, indent=2))


if __name__ == "__main__":
    main()
