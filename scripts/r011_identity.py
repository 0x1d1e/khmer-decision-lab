"""Source, calibration, and test-access checks for R011."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from r008_identity import ROOT, checkpoint_inference_inventory
from r010_identity import (
    MANIFEST as R010_MANIFEST,
)
from r010_identity import (
    cer_counts,
    normalize_input,
)
from r010_identity import (
    verify_locked_manifest as verify_r010_locked_manifest,
)

MANIFEST = ROOT / "datasets/manifests/r011-ocr-domain-calibration.json"
PROTOCOL = ROOT / "research/R011-ocr-domain-calibration/README.md"
ARTIFACT_DIR = ROOT / "artifacts/runs/R011"
CALIBRATION_DATA = "data/processed/r011-massive-km-validation-ocr_12px.jsonl"
TRANSCRIPT_ARTIFACT = "artifacts/runs/R011/validation-ocr-transcripts.jsonl"
SEEDS = (13, 14)
IMPLEMENTATION_FILES = (
    "scripts/r008_identity.py",
    "scripts/r010_identity.py",
    "scripts/prepare_r010.py",
    "scripts/summarize_r010.py",
    "scripts/r011_identity.py",
    "scripts/prepare_r011.py",
    "scripts/calibrate_r011.py",
    "scripts/evaluate_r011.py",
    "scripts/r011_inference.py",
    "scripts/summarize_r011.py",
    "tests/test_r011_ocr_calibration.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _input_keys(path):
    return {
        normalize_input(row["state"]["utterance"])
        for row in read_jsonl(path)
        if row.get("state", {}).get("utterance")
    }


def calibration_selection(validation_rows, train_inputs, test_inputs):
    """Return one validation row per clean, unambiguous, non-overlapping input group."""
    groups = {}
    for index, row in enumerate(validation_rows):
        key = normalize_input(row["state"]["utterance"])
        if not key:
            raise ValueError(
                f"R011 validation row {index} has an empty normalized input"
            )
        group = groups.setdefault(key, {"indices": [], "labels": set()})
        group["indices"].append(index)
        group["labels"].add(row["expected"]["intent"])

    conflicting = {key for key, group in groups.items() if len(group["labels"]) > 1}
    train_overlap = {key for key in groups if key in train_inputs}
    test_overlap = {key for key in groups if key in test_inputs}
    excluded = conflicting | train_overlap | test_overlap
    selected = [
        (groups[key]["indices"][0], validation_rows[groups[key]["indices"][0]])
        for key in groups
        if key not in excluded
    ]
    audit = {
        "validation_rows": len(validation_rows),
        "validation_normalized_groups": len(groups),
        "validation_conflicting_label_groups": len(conflicting),
        "validation_conflicting_label_rows": sum(
            len(groups[key]["indices"]) for key in conflicting
        ),
        "validation_train_overlap_groups": len(train_overlap),
        "validation_train_overlap_rows": sum(
            len(groups[key]["indices"]) for key in train_overlap
        ),
        "validation_test_overlap_groups": len(test_overlap),
        "validation_test_overlap_rows": sum(
            len(groups[key]["indices"]) for key in test_overlap
        ),
        "validation_excluded_group_union": len(excluded),
        "eligible_calibration_groups": len(selected),
        "eligible_calibration_rows": len(selected),
    }
    return selected, audit


def verify_locked_manifest(path=MANIFEST):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = read_json(path)
    if (
        manifest.get("study") != "R011-ocr-domain-calibration"
        or manifest.get("status") != "locked_exploratory"
    ):
        raise ValueError("R011 requires a locked exploratory manifest")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("R011 protocol fingerprint mismatch")
    for relative in IMPLEMENTATION_FILES:
        if manifest.get("implementation_sha256", {}).get(relative) != sha256(
            ROOT / relative
        ):
            raise ValueError(f"R011 implementation fingerprint mismatch: {relative}")

    r010, r010_hash = verify_r010_locked_manifest(R010_MANIFEST)
    source = manifest["source"]
    if (
        source["r010_manifest_sha256"] != r010_hash
        or source["r010_protocol_sha256"] != r010["protocol_sha256"]
    ):
        raise ValueError("R011 R010 lock identity mismatch")
    for relative, key in (
        (source["dataset_manifest"], "dataset_manifest_sha256"),
        (source["r006_manifest"], "r006_manifest_sha256"),
        (source["validation_file"], "validation_sha256"),
        (source["r010_results_markdown"], "r010_results_markdown_sha256"),
        (source["r010_results_json"], "r010_results_json_sha256"),
    ):
        if sha256(ROOT / relative) != source[key]:
            raise ValueError(f"R011 source fingerprint mismatch: {relative}")
    if manifest.get("environment") != r010.get("environment"):
        raise ValueError("R011 environment differs from the R010 locked runtime")

    for model_name, model in manifest["models"].items():
        if model_name not in ("mt-s13", "mt-s14"):
            raise ValueError(f"unexpected R011 model: {model_name}")
        r010_model = r010["models"].get(model_name)
        if (
            r010_model is None
            or model["checkpoint"] != r010_model["checkpoint"]
            or model["seed"] != r010_model["seed"]
            or model["choice_temperature"] != r010_model["choice_temperature"]
            or model["inference_files_sha256"] != r010_model["inference_files_sha256"]
        ):
            raise ValueError(f"R011 model anchor differs from R010: {model_name}")
        checkpoint = ROOT / model["checkpoint"]
        if (
            checkpoint_inference_inventory(checkpoint)
            != model["inference_files_sha256"]
        ):
            raise ValueError(f"R011 checkpoint fingerprint mismatch: {model_name}")

    if sha256(ROOT / source["train_file"]) != source["train_sha256"]:
        raise ValueError("R011 training-source fingerprint mismatch")
    if sha256(ROOT / source["test_file"]) != source["test_sha256"]:
        raise ValueError("R011 test-source fingerprint mismatch")
    if (
        source["train_file"] != r010["source"]["train_file"]
        or source["train_sha256"] != r010["source"]["train_sha256"]
        or source["test_file"] != r010["source"]["test_file"]
        or source["test_sha256"] != r010["source"]["test_sha256"]
    ):
        raise ValueError("R011 MASSIVE source split differs from the R010 anchor")

    for relative, key in (
        (source["r010_preparation_artifact"], "r010_preparation_sha256"),
        (source["r010_evaluation_artifact"], "r010_evaluation_sha256"),
        (source["r010_transcripts_artifact"], "r010_transcripts_sha256"),
    ):
        if sha256(ROOT / relative) != source[key]:
            raise ValueError(f"R011 R010 artifact fingerprint mismatch: {relative}")
    if set(manifest["models"]) != {"mt-s13", "mt-s14"}:
        raise ValueError("R011 requires both fixed R006 seed checkpoints")
    audit = source["audit"]
    recipe = manifest["recipe"]
    evaluation = manifest["evaluation"]
    if (
        recipe["ocr_condition"] != "R010 ocr_12px"
        or recipe["render_font_size_px"] != 12
        or recipe["expected_calibration_rows"] != audit["eligible_calibration_rows"]
        or recipe["expected_temperature_by_options"] != {}
        or recipe["temperature_method"]
        != "Laya 0.4.1 Agent.fit_temperatures; NLL plus LBFGS on log temperature"
        or recipe["calibration_batch_size"] != 8
        or recipe["evaluation_batch_size"] != 8
        or recipe["model_weights_changed"] is not False
        or recipe["test_predictions_used_for_fit"] is not False
        or recipe["checkpoint_selection"] is not False
        or recipe["histogram_binning"] is not False
        or recipe["abstention_threshold_fit"] is not False
        or recipe["inference"]
        != {
            "device": "cuda",
            "autocast_dtype": "bfloat16",
            "laya_cuda_amp": "bf16",
        }
        or evaluation["bootstrap_groups"] != audit["primary_test_groups"]
        or evaluation["bootstrap_samples"] != 10_000
        or evaluation["bootstrap_seed"] != 13
        or evaluation["interval_percentiles"] != [2.5, 97.5]
        or evaluation["primary_metric"] != "mean choice negative log-likelihood"
        or evaluation["primary_comparison"]
        != "OCR-validation temperature map minus checkpoint's existing temperature map"
        or evaluation["bootstrap_weighting"]
        != "resample groups with replacement; retain every row; row-weight metric differences"
        or evaluation["group_key"]
        != "SHA-256 of NFKC-normalized, case-folded, whitespace-collapsed utterance"
        or evaluation["secondary_subset"] != f"all {source['test_rows']} test rows"
        or evaluation["inference_grouping"]
        != "Laya evals.evaluate: consecutive identical question schemas, singleton predict, batch_size=8"
    ):
        raise ValueError("R011 locked recipe or evaluation differs from the protocol")
    r010_audit = r010["source"]["audit"]
    if any(
        audit[target] != r010_audit[anchor]
        for target, anchor in (
            ("test_rows", "test_rows"),
            ("test_normalized_groups", "normalized_test_groups"),
            ("test_conflicting_label_groups", "conflicting_label_groups"),
            ("test_conflicting_label_rows", "conflicting_label_rows"),
            ("test_train_overlap_groups", "exact_train_overlap_groups"),
            ("test_train_overlap_rows", "exact_train_overlap_rows"),
        )
    ):
        raise ValueError("R011 test audit differs from the R010 source audit")
    return manifest, sha256(path), r010, r010_hash


def verify_preparation(manifest, manifest_hash):
    source = manifest["source"]
    artifact_path = ARTIFACT_DIR / "preparation.json"
    preparation = read_json(artifact_path)
    expected_identity = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "preparer_sha256": manifest["implementation_sha256"]["scripts/prepare_r011.py"],
        "source_validation_sha256": source["validation_sha256"],
        "source_train_sha256": source["train_sha256"],
        "source_test_sha256": source["test_sha256"],
        "calibration_file": CALIBRATION_DATA,
        "transcript_artifact": TRANSCRIPT_ARTIFACT,
    }
    if any(preparation.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("R011 preparation identity mismatch")

    validation_rows = read_jsonl(ROOT / source["validation_file"])
    train_inputs = _input_keys(ROOT / source["train_file"])
    test_inputs = _input_keys(ROOT / source["test_file"])
    selected, audit = calibration_selection(validation_rows, train_inputs, test_inputs)
    if (
        len(validation_rows) != source["validation_rows"]
        or audit != {key: source["audit"][key] for key in audit}
        or preparation.get("validation_audit") != audit
    ):
        raise ValueError("R011 validation audit differs from the locked source audit")

    calibration_path = ROOT / CALIBRATION_DATA
    transcript_path = ROOT / TRANSCRIPT_ARTIFACT
    expected_files = {
        CALIBRATION_DATA: sha256(calibration_path),
        TRANSCRIPT_ARTIFACT: sha256(transcript_path),
    }
    if preparation.get("outputs_sha256") != expected_files:
        raise ValueError("R011 preparation file inventory mismatch")
    prepared_rows = read_jsonl(calibration_path)
    transcript_rows = read_jsonl(transcript_path)
    if len(prepared_rows) != len(selected) or len(transcript_rows) != len(selected):
        raise ValueError("R011 calibration row count mismatch")

    for (source_index, original), prepared, record in zip(
        selected, prepared_rows, transcript_rows, strict=True
    ):
        group_key = normalize_input(original["state"]["utterance"])
        group_id = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
        source_row_id = hashlib.sha256(
            f"{source['validation_sha256']}\0{source_index}\0".encode()
            + json.dumps(
                original, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        expected_record = {
            "source_row_index": source_index,
            "source_row_id": source_row_id,
            "group_id": group_id,
            "reference": original["state"]["utterance"],
            "train_overlap": False,
            "test_overlap": False,
            "conflicting_label": False,
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError("R011 validation transcript row identity mismatch")
        row = copy.deepcopy(original)
        row["state"]["utterance"] = record.get("text")
        row["tags"] = list(original.get("tags", [])) + [
            f"r011-row:{source_row_id}",
            f"r011-group:{group_id}",
        ]
        if prepared != row:
            raise ValueError("R011 prepared validation row differs from its source")
        if (
            not isinstance(record.get("text"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("image_sha256", ""))
            or record.get("cer") != cer_counts(record["reference"], record["text"])
        ):
            raise ValueError("R011 validation transcript fields are invalid")
    return preparation


def verify_calibration_readiness(manifest, manifest_hash):
    """Authenticate both validation-only maps before opening test reports or labels."""
    from laya.calibrate import CALIBRATION_VERSION, TEMP_MAX, TEMP_MIN

    prep = verify_preparation(manifest, manifest_hash)
    calibration_data_sha256 = prep["outputs_sha256"][CALIBRATION_DATA]
    ready = {}
    for model_name in ("mt-s13", "mt-s14"):
        model = manifest["models"][model_name]
        base = ARTIFACT_DIR / "calibration" / model_name
        calibration_path = base.with_suffix(".json")
        provenance_path = base.with_suffix(".provenance.json")
        if any(
            path.is_symlink() or not path.is_file()
            for path in (calibration_path, provenance_path)
        ):
            raise ValueError(
                f"R011 calibration output is missing or unsafe: {model_name}"
            )
        payload = read_json(calibration_path)
        metadata = read_json(provenance_path)
        if set(payload) != {
            "version",
            "temperature",
            "temperature_by_options",
            "model_id_or_path",
            "subfolder",
            "config",
        }:
            raise ValueError(
                f"R011 calibration payload has unexpected fields: {model_name}"
            )
        if payload.get("version") != CALIBRATION_VERSION:
            raise ValueError(f"R011 calibration version mismatch: {model_name}")
        temperatures = payload.get("temperature")
        if (
            not isinstance(temperatures, list)
            or len(temperatures) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not TEMP_MIN <= value <= TEMP_MAX
                for value in temperatures
            )
            or payload.get("temperature_by_options") != {}
        ):
            raise ValueError(f"R011 calibration temperatures are invalid: {model_name}")
        checkpoint = ROOT / model["checkpoint"]
        config = read_json(checkpoint / "rl_agent_config.json")
        expected_config = {
            key: value
            for key, value in config.items()
            if key not in ("temperature", "temperature_by_options")
        }
        if (
            payload.get("model_id_or_path") != str(checkpoint.resolve())
            or payload.get("subfolder") is not None
            or payload.get("config") != expected_config
        ):
            raise ValueError(
                f"R011 calibration checkpoint identity mismatch: {model_name}"
            )

        expected_rows = manifest["recipe"]["expected_calibration_rows"]
        expected_metadata = {
            "schema": "r011-temperature-provenance/1",
            "study": manifest["study"],
            "seed": model["seed"],
            "checkpoint": model["checkpoint"],
            "checkpoint_weights_sha256": model["inference_files_sha256"][
                "model.safetensors"
            ],
            "inference_files_sha256": model["inference_files_sha256"],
            "validation_file": manifest["source"]["validation_file"],
            "validation_sha256": manifest["source"]["validation_sha256"],
            "calibration_file": CALIBRATION_DATA,
            "calibration_file_sha256": calibration_data_sha256,
            "calibration_rows": expected_rows,
            "calibration_groups": expected_rows,
            "inference_batch_size": 8,
            "inference_grouping": manifest["evaluation"]["inference_grouping"],
            "inference_device": "cuda",
            "inference_dtype": "bfloat16",
            "existing_temperature": model["choice_temperature"],
            "fitted_temperature": temperatures,
            "temperature_by_options": {},
            "items_by_bucket": {"choice:11+": expected_rows},
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
        if set(metadata) != set(expected_metadata) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise ValueError(f"R011 calibration provenance mismatch: {model_name}")
        ready[model_name] = {
            "calibration": payload,
            "metadata": metadata,
            "calibration_path": calibration_path,
        }
    return ready
