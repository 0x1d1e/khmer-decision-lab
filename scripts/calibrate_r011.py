"""Fit OCR-specific temperatures for both locked R006 checkpoints."""

import json

import numpy as np
import torch
from laya.train import items_from_rows, read_data
from r008_identity import (
    INFERENCE_POLICY,
    checkpoint_inference_inventory,
    load_r008_agent,
)
from r011_identity import (
    ARTIFACT_DIR,
    CALIBRATION_DATA,
    ROOT,
    read_json,
    sha256,
    verify_locked_manifest,
    verify_preparation,
    write_json,
)
from r011_inference import BATCH_SIZE, GROUPING, capture_laya_logits

MAX_LEN = 512
HEAD_MAX_LEN = 256


def calibrate_all():
    manifest, manifest_hash, _r010, _r010_hash = verify_locked_manifest()
    preparation = verify_preparation(manifest, manifest_hash)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R011 calibration requires CUDA BF16")
    if torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]:
        raise RuntimeError("R011 calibration GPU differs from the locked environment")

    data_path = ROOT / CALIBRATION_DATA
    if sha256(data_path) != preparation["outputs_sha256"][CALIBRATION_DATA]:
        raise ValueError("R011 OCR calibration data changed")
    rows = read_data(str(data_path))
    expected_rows = manifest["recipe"]["expected_calibration_rows"]
    if len(rows) != expected_rows:
        raise ValueError("R011 OCR calibration row count mismatch")

    ready = {}
    for model_name in ("mt-s13", "mt-s14"):
        model = manifest["models"][model_name]
        checkpoint = (ROOT / model["checkpoint"]).resolve()
        before_inventory = checkpoint_inference_inventory(checkpoint)
        if before_inventory != model["inference_files_sha256"]:
            raise ValueError(
                f"R011 checkpoint changed before calibration: {model_name}"
            )
        agent, inference_dtype = load_r008_agent(
            checkpoint, {"inference": INFERENCE_POLICY}
        )
        if (
            agent.device.type != "cuda"
            or agent.dtype != torch.bfloat16
            or not agent.amp_enabled
            or str(agent.dtype).removeprefix("torch.") != inference_dtype
            or agent.cfg.get("max_len") != MAX_LEN
            or agent.cfg.get("head_max_len") != HEAD_MAX_LEN
            or agent.temperature != model["choice_temperature"]
            or agent.temperature_by_options
            or agent.lang_temperatures
            or getattr(agent, "binning_map", None) is not None
        ):
            raise ValueError(
                f"R011 checkpoint inference settings mismatch: {model_name}"
            )
        items, skipped = items_from_rows(
            agent.tok,
            rows,
            MAX_LEN,
            HEAD_MAX_LEN,
            label_smoothing=0.0,
        )
        if skipped or len(items) != len(rows):
            raise ValueError(f"R011 validation rows were skipped: {skipped!r}")
        if any(item["qtype"] != 0 or item["k"] != 20 for item in items):
            raise ValueError("R011 calibration rows must be 20-option intent choices")

        existing_temperature = list(agent.temperature)
        logits, cases = capture_laya_logits(agent, rows, model_name)
        if len(cases) != len(items):
            raise ValueError(f"R011 calibration logits are incomplete: {model_name}")
        records = []
        for index, (row, item, vector, case) in enumerate(
            zip(rows, items, logits, cases, strict=True)
        ):
            labels = list(row["questions"]["intent"]["criteria"])
            target = np.asarray(item["target"], dtype=np.float64)
            if (
                item["qtype"] != 0
                or item["k"] != 20
                or vector.shape != (20,)
                or target.shape != (20,)
                or not np.isfinite(vector).all()
                or np.count_nonzero(target) != 1
                or labels[int(target.argmax())] != row["expected"]["intent"]
                or labels[int(vector.argmax())] != case.get("answer", {}).get("choice")
            ):
                raise ValueError(
                    f"R011 calibration row integrity mismatch: {model_name}/{index}"
                )
            records.append((item["qtype"], vector, item["target"], item["k"]))
        fitted = agent.fit_temperatures(records)
        if fitted["temperature_by_options"] != {}:
            raise ValueError(
                f"R011 fit unexpectedly produced option-bucket temperatures: {model_name}"
            )
        if not torch.isfinite(torch.tensor(fitted["temperature"])).all():
            raise ValueError(f"R011 fitted temperature is non-finite: {model_name}")

        if checkpoint_inference_inventory(checkpoint) != before_inventory:
            raise ValueError(f"R011 calibration changed checkpoint files: {model_name}")
        calibration_path = ARTIFACT_DIR / "calibration" / f"{model_name}.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save_calibration(str(calibration_path))
        payload = read_json(calibration_path)
        if (
            payload.get("temperature") != fitted["temperature"]
            or payload.get("temperature_by_options") != {}
        ):
            raise ValueError(f"R011 saved calibration differs from fit: {model_name}")

        metadata = {
            "schema": "r011-temperature-provenance/1",
            "study": manifest["study"],
            "seed": model["seed"],
            "checkpoint": model["checkpoint"],
            "checkpoint_weights_sha256": before_inventory["model.safetensors"],
            "inference_files_sha256": before_inventory,
            "validation_file": manifest["source"]["validation_file"],
            "validation_sha256": manifest["source"]["validation_sha256"],
            "calibration_file": CALIBRATION_DATA,
            "calibration_file_sha256": preparation["outputs_sha256"][CALIBRATION_DATA],
            "calibration_rows": len(rows),
            "calibration_groups": len(rows),
            "inference_batch_size": BATCH_SIZE,
            "inference_grouping": GROUPING,
            "inference_device": agent.device.type,
            "inference_dtype": inference_dtype,
            "existing_temperature": existing_temperature,
            "fitted_temperature": fitted["temperature"],
            "temperature_by_options": fitted["temperature_by_options"],
            "items_by_bucket": fitted["n_by_bucket"],
            "r011_manifest_sha256": manifest_hash,
            "calibration_sha256": sha256(calibration_path),
            "calibrator_sha256": manifest["implementation_sha256"][
                "scripts/calibrate_r011.py"
            ],
            "identity_sha256": manifest["implementation_sha256"][
                "scripts/r011_identity.py"
            ],
            "model_weights_changed": False,
        }
        write_json(
            ARTIFACT_DIR / "calibration" / f"{model_name}.provenance.json",
            metadata,
        )
        ready[model_name] = {
            "calibration": str(calibration_path.relative_to(ROOT)),
            "calibration_sha256": metadata["calibration_sha256"],
            "existing_temperature": existing_temperature,
            "fitted_temperature": fitted["temperature"],
            "calibration_rows": len(rows),
        }
        del agent
        torch.cuda.empty_cache()

    print(json.dumps(ready, indent=2))
    return ready


def main():
    calibrate_all()


if __name__ == "__main__":
    main()
