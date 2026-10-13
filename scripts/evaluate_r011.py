"""Score both locked R006 checkpoints under clean and OCR-domain temperatures."""

import json
from pathlib import Path

import numpy as np
import torch
from laya.train import items_from_rows, read_data
from r008_identity import (
    INFERENCE_POLICY,
    checkpoint_inference_inventory,
    load_r008_agent,
)
from r010_identity import verify_prepared as verify_r010_prepared
from r011_identity import (
    ARTIFACT_DIR,
    CALIBRATION_DATA,
    ROOT,
    read_json,
    read_jsonl,
    sha256,
    verify_calibration_readiness,
    verify_locked_manifest,
    write_json,
)
from r011_inference import BATCH_SIZE, GROUPING, capture_laya_logits

MAX_LEN = 512
HEAD_MAX_LEN = 256
CONDITION = "ocr_12px"


def _tag(row, prefix):
    matches = [
        tag[len(prefix) :] for tag in row.get("tags", []) if tag.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"R011 row must contain one {prefix} tag")
    return matches[0]


def _labels(row):
    criteria = row["questions"]["intent"]["criteria"]
    if isinstance(criteria, dict):
        return list(criteria)
    if isinstance(criteria, list):
        return list(criteria)
    raise ValueError("R011 intent criteria must be an ordered mapping or list")


def _validate_baseline_cases(test_rows, cases, model_name):
    if len(test_rows) != len(cases):
        raise ValueError(f"R011 baseline row count mismatch: {model_name}")
    for index, (row, case) in enumerate(zip(test_rows, cases, strict=True)):
        labels = _labels(row)
        if (
            _tag(case, "r010-row:") != _tag(row, "r010-row:")
            or _tag(case, "r010-group:") != _tag(row, "r010-group:")
            or case.get("expected") != row["expected"]["intent"]
            or case.get("model") != model_name
            or case.get("qid") != "intent"
            or case.get("correct")
            != (case.get("answer", {}).get("choice") == case.get("expected"))
            or case.get("expected") not in labels
            or case.get("answer", {}).get("choice") not in labels
        ):
            raise ValueError(
                f"R011 baseline row alignment mismatch: {model_name}/{index}"
            )


def _baseline_report(manifest, r010, r010_hash, preparation, model_name):
    source = manifest["source"]
    record = source["baseline_test_reports"][model_name]
    relative = Path(record["report"])
    expected_relative = (
        Path("artifacts/runs/R010/massive-km") / CONDITION / f"{model_name}.json"
    )
    if relative != expected_relative:
        raise ValueError(f"R011 baseline report path mismatch: {model_name}")
    path = ROOT / relative
    if path.is_symlink() or sha256(path) != record["sha256"]:
        raise ValueError(f"R011 baseline report hash mismatch: {model_name}")
    report = read_json(path)
    checkpoint = r010["models"][model_name]
    expected_config = {
        "study": r010["study"],
        "condition": CONDITION,
        "manifest_sha256": r010_hash,
        "source_test_file": r010["source"]["test_file"],
        "source_test_sha256": r010["source"]["test_sha256"],
        "dataset": preparation["outputs"][CONDITION]["file"],
        "dataset_sha256": preparation["outputs"][CONDITION]["sha256"],
        "preparation_sha256": sha256(ROOT / "artifacts/runs/R010/preparation.json"),
        "preparer_sha256": r010["implementation_sha256"]["scripts/prepare_r010.py"],
        "model": model_name,
        "checkpoint": checkpoint["checkpoint"],
        "weights_sha256": checkpoint["inference_files_sha256"]["model.safetensors"],
        "inference_files_sha256": checkpoint["inference_files_sha256"],
        "max_len": checkpoint["max_len"],
        "head_max_len": checkpoint["head_max_len"],
        "device": INFERENCE_POLICY["device"],
        "autocast_dtype": INFERENCE_POLICY["autocast_dtype"],
        "laya_cuda_amp": INFERENCE_POLICY["laya_cuda_amp"],
        "gpu": r010["environment"]["gpu"],
        "torch_version": r010["environment"]["torch_version"],
        "cuda_version": r010["environment"]["torch_cuda"],
    }
    config = report.get("config", {})
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"R011 baseline report configuration mismatch: {model_name}")
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != source["test_rows"]
        or report.get("overall", {}).get("choice_accuracy") != record["choice_accuracy"]
    ):
        raise ValueError(f"R011 baseline report summary mismatch: {model_name}")
    return report, record


def evaluate():
    manifest, manifest_hash, r010, r010_hash = verify_locked_manifest()
    ready = verify_calibration_readiness(manifest, manifest_hash)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R011 evaluation requires the locked CUDA BF16 runtime")
    if torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]:
        raise RuntimeError("R011 evaluation GPU differs from the locked environment")

    r010_preparation = verify_r010_prepared(r010, r010_hash)
    test_output = r010_preparation["outputs"].get(CONDITION)
    source = manifest["source"]
    if (
        test_output is None
        or test_output["file"] != source["r010_test_ocr_file"]
        or test_output["sha256"] != source["r010_test_ocr_sha256"]
    ):
        raise ValueError("R011 test input differs from the locked R010 OCR artifact")
    test_rows = read_jsonl(ROOT / test_output["file"])
    if len(test_rows) != source["test_rows"]:
        raise ValueError("R011 test input row count mismatch")

    records = {}
    for model_name in ("mt-s13", "mt-s14"):
        model = manifest["models"][model_name]
        r010_model = r010["models"][model_name]
        report, report_record = _baseline_report(
            manifest, r010, r010_hash, r010_preparation, model_name
        )
        cases = report["cases"]
        _validate_baseline_cases(test_rows, cases, model_name)

        checkpoint = (ROOT / model["checkpoint"]).resolve()
        if (
            checkpoint_inference_inventory(checkpoint)
            != model["inference_files_sha256"]
        ):
            raise ValueError(f"R011 checkpoint fingerprint mismatch: {model_name}")
        agent, inference_dtype = load_r008_agent(
            checkpoint, {"inference": INFERENCE_POLICY}
        )
        if (
            agent.device.type != "cuda"
            or agent.dtype != torch.bfloat16
            or not agent.amp_enabled
            or inference_dtype != "bfloat16"
            or agent.cfg.get("max_len") != r010_model["max_len"]
            or agent.cfg.get("head_max_len") != r010_model["head_max_len"]
            or agent.temperature != model["choice_temperature"]
            or agent.temperature_by_options
            or agent.lang_temperatures
            or getattr(agent, "binning_map", None) is not None
        ):
            raise ValueError(f"R011 inference settings mismatch: {model_name}")

        items, skipped = items_from_rows(
            agent.tok,
            read_data(str(ROOT / test_output["file"])),
            MAX_LEN,
            HEAD_MAX_LEN,
            label_smoothing=0.0,
        )
        if skipped or len(items) != len(test_rows):
            raise ValueError(f"R011 test rows were skipped: {skipped!r}")
        if any(item["qtype"] != 0 or item["k"] != 20 for item in items):
            raise ValueError("R011 test rows must be 20-option intent choices")

        logits, inference_cases = capture_laya_logits(agent, test_rows, model_name)
        if len(inference_cases) != len(items):
            raise ValueError(f"R011 test logits are incomplete: {model_name}")
        if (
            checkpoint_inference_inventory(checkpoint)
            != model["inference_files_sha256"]
        ):
            raise ValueError(f"R011 checkpoint changed during evaluation: {model_name}")

        baseline_temperature = ready[model_name]["metadata"]["existing_temperature"]
        ocr_temperature = ready[model_name]["calibration"]["temperature"]
        row_records = []
        for index, (row, case, item, vector, inference_case) in enumerate(
            zip(test_rows, cases, items, logits, inference_cases, strict=True)
        ):
            if item["qtype"] != 0 or item["k"] != 20 or vector.shape != (20,):
                raise ValueError(f"R011 logit shape mismatch: {model_name}/{index}")
            label_keys = _labels(row)
            expected_label = row["expected"]["intent"]
            target_index = label_keys.index(expected_label)
            target_vector = np.asarray(item["target"], dtype=np.float64)
            if (
                target_vector.shape != (20,)
                or not np.isclose(target_vector.sum(), 1.0)
                or int(target_vector.argmax()) != target_index
                or np.count_nonzero(target_vector) != 1
            ):
                raise ValueError(f"R011 gold target mismatch: {model_name}/{index}")
            logit_vector = np.asarray(vector, dtype=np.float64)
            if not np.isfinite(logit_vector).all():
                raise ValueError(f"R011 non-finite test logits: {model_name}/{index}")
            prediction = label_keys[int(logit_vector.argmax())]
            baseline_prediction = case["answer"]["choice"]
            if (
                prediction != baseline_prediction
                or inference_case.get("answer", {}).get("choice") != baseline_prediction
            ):
                raise ValueError(
                    f"R011 raw baseline prediction mismatch: {model_name}/{index}"
                )
            overlap_tag = _tag(row, "r010-train-overlap:")
            if overlap_tag not in ("0", "1"):
                raise ValueError(
                    f"R011 train-overlap tag is invalid: {model_name}/{index}"
                )
            row_records.append(
                {
                    "row_index": index,
                    "row_id": _tag(row, "r010-row:"),
                    "group_id": _tag(row, "r010-group:"),
                    "train_overlap": overlap_tag == "1",
                    "expected": expected_label,
                    "baseline_prediction": baseline_prediction,
                    "target_index": target_index,
                    "logits": logit_vector.tolist(),
                }
            )

        records[model_name] = {
            "baseline_report": report_record["report"],
            "baseline_report_sha256": report_record["sha256"],
            "baseline_accuracy": report_record["choice_accuracy"],
            "existing_temperature": baseline_temperature,
            "ocr_temperature": ocr_temperature,
            "calibration_sha256": ready[model_name]["metadata"]["calibration_sha256"],
            "rows": row_records,
        }
        del agent
        torch.cuda.empty_cache()

    output = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "r010_preparation_sha256": sha256(
            ROOT / "artifacts/runs/R010/preparation.json"
        ),
        "calibration_data_sha256": sha256(ROOT / CALIBRATION_DATA),
        "inference_batch_size": BATCH_SIZE,
        "inference_grouping": GROUPING,
        "models": records,
    }
    output_path = ARTIFACT_DIR / "evaluation.json"
    write_json(output_path, output)
    print(
        json.dumps(
            {
                model_name: {
                    "rows": len(result["rows"]),
                    "baseline_accuracy": result["baseline_accuracy"],
                    "existing_temperature": result["existing_temperature"],
                    "ocr_temperature": result["ocr_temperature"],
                }
                for model_name, result in records.items()
            },
            indent=2,
        )
    )
    return output


def main():
    evaluate()


if __name__ == "__main__":
    main()
