"""Integrity checks shared by the R009 training, calibration, and scoring stages."""

import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

from r008_identity import (
    ROOT,
    checkpoint_inference_inventory,
    verify_calibration_artifact,
    verify_checkpoint_training_provenance,
)
from r008_identity import verify_locked_manifest as verify_r008_manifest

MANIFEST = ROOT / "datasets/manifests/r009-sib-specialist-continuation.json"
PROTOCOL = ROOT / "research/R009-sib-specialist-continuation/README.md"
IMPLEMENTATION_FILES = (
    "scripts/r009_identity.py",
    "scripts/train_r009_continuation.py",
    "scripts/calibrate_r009.py",
    "scripts/evaluate_r009.py",
    "tests/test_r009_evaluation.py",
)
SEEDS = (13, 14)


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


def verify_locked_manifest(path=MANIFEST):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = read_json(path)
    if manifest.get("study") != "R009-sib-specialist-continuation":
        raise ValueError("unexpected R009 manifest")
    if manifest.get("status") != "locked_exploratory":
        raise ValueError("R009 requires a locked exploratory manifest")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("R009 protocol fingerprint mismatch")
    implementation = manifest.get("implementation_sha256", {})
    for relative in IMPLEMENTATION_FILES:
        if implementation.get(relative) != sha256(ROOT / relative):
            raise ValueError(f"R009 implementation fingerprint mismatch: {relative}")

    source = manifest["source"]
    r008_manifest, r008_hash = verify_r008_manifest(ROOT / source["r008_manifest"])
    if r008_hash != source["r008_manifest_sha256"]:
        raise ValueError("R009 R008-manifest fingerprint mismatch")
    if sha256(ROOT / "uv.lock") != manifest["environment"]["uv_lock_sha256"]:
        raise ValueError("R009 frozen dependency lock fingerprint mismatch")
    if platform.python_version() != manifest["environment"]["python"]:
        raise ValueError("R009 Python version differs from the locked environment")
    for package in ("laya", "torch", "bitsandbytes", "transformers"):
        if version(package) != manifest["environment"][package]:
            raise ValueError(
                f"R009 {package} version differs from the locked environment"
            )

    for key, path_key in (
        ("training_sha256", "training_file"),
        ("validation_sha256", "validation_file"),
    ):
        if sha256(ROOT / source[path_key]) != source[key]:
            raise ValueError(f"R009 {path_key} fingerprint mismatch")
    if sha256(ROOT / source["dataset_manifest"]) != source["dataset_manifest_sha256"]:
        raise ValueError("R009 source dataset manifest fingerprint mismatch")
    if r008_manifest["recipe"]["base_revision"] != manifest["recipe"]["base_revision"]:
        raise ValueError("R009 base revision differs from the R008 anchor recipe")
    return manifest, sha256(path), r008_manifest, r008_hash


def verify_r008_anchor(
    seed, arm_name, manifest, r008_manifest, r008_hash, check_report=False
):
    if seed not in SEEDS or arm_name not in ("specialist", "multitask"):
        raise ValueError("unknown R008 anchor seed or arm")
    anchor = manifest["source"]["r008_anchors"][str(seed)][arm_name]
    checkpoint = ROOT / anchor["checkpoint"]
    training_arm = "sib200-specialist" if arm_name == "specialist" else "multitask"
    provenance = verify_checkpoint_training_provenance(
        checkpoint, "sib200-km", r008_manifest, r008_hash
    )
    if provenance["training_arm"] != training_arm or provenance["seed"] != seed:
        raise ValueError(
            f"R008 {arm_name} checkpoint identity mismatch for seed {seed}"
        )
    if sha256(checkpoint / "model.safetensors") != anchor["weights_sha256"]:
        raise ValueError(f"R008 {arm_name} weight fingerprint mismatch for seed {seed}")
    if (
        sha256(checkpoint / "training_provenance.json")
        != anchor["training_provenance_sha256"]
    ):
        raise ValueError(
            f"R008 {arm_name} provenance fingerprint mismatch for seed {seed}"
        )
    if (
        sha256(checkpoint / "training_telemetry.json")
        != anchor["training_telemetry_sha256"]
    ):
        raise ValueError(
            f"R008 {arm_name} telemetry fingerprint mismatch for seed {seed}"
        )
    verify_calibration_artifact(
        checkpoint, "sib200-km", provenance, r008_manifest, r008_hash
    )
    if sha256(ROOT / anchor["calibration"]) != anchor["calibration_sha256"]:
        raise ValueError(
            f"R008 {arm_name} calibration fingerprint mismatch for seed {seed}"
        )

    report = (
        verify_anchor_report(seed, arm_name, anchor, manifest) if check_report else None
    )
    return anchor, provenance, report


def verify_anchor_report(seed, arm_name, anchor, manifest):
    report_path = ROOT / anchor["test_report"]
    if sha256(report_path) != anchor["test_report_sha256"]:
        raise ValueError(
            f"R008 {arm_name} test report fingerprint mismatch for seed {seed}"
        )
    report = read_json(report_path)
    config = report.get("config", {})
    expected_model = (
        f"sib200-specialist-s{seed}"
        if arm_name == "specialist"
        else f"multitask-s{seed}"
    )
    expected = {
        "study": "R008-specialist-controls",
        "task": "sib200-km",
        "model": expected_model,
        "seed": seed,
        "dataset": manifest["source"]["test_file"],
        "dataset_sha256": manifest["source"]["test_sha256"],
        "weights_sha256": anchor["weights_sha256"],
        "calibration_sha256": anchor["calibration_sha256"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"R008 {arm_name} test report identity mismatch for seed {seed}"
        )
    if len(report.get("cases", [])) != manifest["source"]["test_rows"]:
        raise ValueError(f"R008 {arm_name} case count mismatch for seed {seed}")
    if (
        abs(
            report.get("overall", {}).get("choice_accuracy", -1)
            - anchor["test_accuracy"]
        )
        > 1e-12
    ):
        raise ValueError(f"R008 {arm_name} reported accuracy mismatch for seed {seed}")
    return report


def verify_r009_training(seed, manifest, r009_hash, r008_manifest, r008_hash):
    if seed not in SEEDS:
        raise ValueError("R009 seed is not preregistered")
    output_rel = f"artifacts/checkpoints/R009-sib-s{seed}-cont"
    checkpoint = ROOT / output_rel
    provenance_path = checkpoint / "training_provenance.json"
    telemetry_path = checkpoint / "training_telemetry.json"
    weights_path = checkpoint / "model.safetensors"
    report_path = checkpoint / "r009_training_report.json"
    for path in (provenance_path, telemetry_path, weights_path, report_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"R009 training output is missing or unsafe: {path}")
    provenance = read_json(provenance_path)
    telemetry = read_json(telemetry_path)
    anchor, anchor_provenance, _ = verify_r008_anchor(
        seed, "specialist", manifest, r008_manifest, r008_hash
    )
    training_report = read_json(report_path)
    diagnostics = training_report.get("train_report", {})
    expected_eval_source = str(ROOT / manifest["source"]["validation_file"])
    if (
        diagnostics.get("eval_mode") != "held_out"
        or diagnostics.get("is_held_out") is not True
        or diagnostics.get("eval_source") != expected_eval_source
        or diagnostics.get("eval_items") != manifest["source"]["validation_rows"]
    ):
        raise ValueError(
            f"R009 validation diagnostics are not held out for seed {seed}"
        )
    expected = {
        "schema": "r009-training-provenance/1",
        "study": "R009-sib-specialist-continuation",
        "seed": seed,
        "checkpoint_directory": output_rel,
        "source_checkpoint": anchor["checkpoint"],
        "source_weights_sha256": anchor["weights_sha256"],
        "source_training_provenance_sha256": anchor["training_provenance_sha256"],
        "training_file": manifest["source"]["training_file"],
        "training_file_sha256": manifest["source"]["training_sha256"],
        "validation_file": manifest["source"]["validation_file"],
        "validation_file_sha256": manifest["source"]["validation_sha256"],
        "r008_manifest_sha256": r008_hash,
        "r009_manifest_sha256": r009_hash,
        "continuation_epochs": manifest["recipe"]["continuation_epochs"],
        "optimizer_reset": manifest["recipe"]["optimizer_reset"],
        "optimizer_attempts": manifest["recipe"][
            "expected_optimizer_attempts_per_seed"
        ],
        "validation_used_for_checkpoint_selection": False,
        "weights_sha256": sha256(weights_path),
        "inference_files_sha256": checkpoint_inference_inventory(checkpoint),
        "training_telemetry_sha256": sha256(telemetry_path),
        "training_report_sha256": sha256(report_path),
        "trainer_sha256": manifest["implementation_sha256"][
            "scripts/train_r009_continuation.py"
        ],
        "identity_sha256": manifest["implementation_sha256"][
            "scripts/r009_identity.py"
        ],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"R009 training provenance mismatch for seed {seed}")
    config = provenance.get("training_config", {})
    expected_config = {
        "epochs": manifest["recipe"]["continuation_epochs"],
        "micro_batch": manifest["recipe"]["micro_batch"],
        "grad_accum": manifest["recipe"]["grad_accum"],
        "encoder_lr": manifest["recipe"]["encoder_lr"],
        "head_lr": manifest["recipe"]["head_lr"],
        "min_lr": manifest["recipe"]["min_lr"],
        "weight_decay": manifest["recipe"]["weight_decay"],
        "grad_clip": manifest["recipe"]["grad_clip"],
        "loss": manifest["recipe"]["loss"],
        "freeze_encoder": manifest["recipe"]["freeze_encoder"],
        "shuffle_options": manifest["recipe"]["shuffle_options"],
        "seed": seed,
        "max_len": manifest["recipe"]["max_len"],
        "head_max_len": manifest["recipe"]["head_max_len"],
        "amp": manifest["recipe"]["amp"],
        "gradient_checkpointing": manifest["recipe"]["gradient_checkpointing"],
        "calib_max": 0,
        "calib_frac": 0.0,
        "calib_seed": manifest["recipe"]["calib_seed"],
        "eval_data": manifest["source"]["validation_file"],
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"R009 training config mismatch for seed {seed}")
    expected_attempts = manifest["recipe"]["expected_optimizer_attempts_per_seed"]
    if (
        telemetry.get("optimizer") != manifest["recipe"]["optimizer"]
        or telemetry.get("optimizer_attempts") != expected_attempts
        or telemetry.get("successful_updates", 0) < 1
        or telemetry.get("successful_updates", 0) + telemetry.get("skipped_updates", 0)
        != expected_attempts
        or telemetry.get("scheduler_steps") != telemetry.get("successful_updates")
        or telemetry.get("failure")
        or telemetry.get("telemetry_cleanup_failure")
        or telemetry.get("device") != "cuda"
        or telemetry.get("gpu") != manifest["environment"]["gpu"]
        or not str(telemetry.get("torch_version", "")).startswith(
            manifest["environment"]["torch"]
        )
        or telemetry.get("cuda_version") != manifest["environment"]["cuda"]
        or not telemetry.get("peak_allocated_bytes")
    ):
        raise ValueError(f"R009 optimizer telemetry invalid for seed {seed}")
    if anchor_provenance["weights_sha256"] != provenance["source_weights_sha256"]:
        raise ValueError("R009 continuation source weight identity mismatch")
    if provenance["weights_sha256"] == anchor["weights_sha256"]:
        raise ValueError(f"R009 continuation weights did not change for seed {seed}")
    return checkpoint, provenance, telemetry


def validate_r009_calibration_payload(
    payload, seed, checkpoint, provenance, manifest, r009_hash, r008_hash
):
    """Require a temperature-only Laya map bound to this checkpoint and validation."""
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "temperature",
        "temperature_by_options",
        "model_id_or_path",
        "subfolder",
        "config",
        "r009",
    }:
        raise ValueError("R009 calibration has an unexpected Laya payload shape")

    from laya.calibrate import CALIBRATION_VERSION
    from laya.common import TEMP_MAX, TEMP_MIN

    version = payload["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != CALIBRATION_VERSION
    ):
        raise ValueError("R009 calibration has an unsupported Laya version")

    def validate_temperature(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"R009 calibration has invalid {name}")
        try:
            temperature = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"R009 calibration has invalid {name}") from exc
        if not math.isfinite(temperature) or not TEMP_MIN <= temperature <= TEMP_MAX:
            raise ValueError(f"R009 calibration has invalid {name}")

    temperatures = payload["temperature"]
    if not isinstance(temperatures, list) or len(temperatures) != 3:
        raise ValueError("R009 calibration must contain three temperatures")
    for index, temperature in enumerate(temperatures):
        validate_temperature(f"temperature[{index}]", temperature)

    temperatures_by_options = payload["temperature_by_options"]
    if not isinstance(temperatures_by_options, dict):
        raise TypeError("R009 option temperatures must be an object")
    for bucket, temperature in temperatures_by_options.items():
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("R009 calibration has an invalid option bucket")
        validate_temperature(f"temperature_by_options[{bucket!r}]", temperature)

    config = read_json(checkpoint / "rl_agent_config.json")
    if not isinstance(config, dict):
        raise TypeError("R009 checkpoint config must be an object")
    config_identity = {
        key: value
        for key, value in config.items()
        if key not in ("temperature", "temperature_by_options")
    }
    if (
        payload["model_id_or_path"] != str(checkpoint.resolve())
        or payload["subfolder"] is not None
        or payload["config"] != config_identity
    ):
        raise ValueError("R009 Laya calibration identity differs from its checkpoint")

    metadata = payload["r009"]
    expected_metadata = {
        "study": "R009-sib-specialist-continuation",
        "seed": seed,
        "checkpoint": f"artifacts/checkpoints/R009-sib-s{seed}-cont",
        "checkpoint_weights_sha256": provenance["weights_sha256"],
        "training_provenance_sha256": sha256(checkpoint / "training_provenance.json"),
        "validation_file": manifest["source"]["validation_file"],
        "validation_sha256": manifest["source"]["validation_sha256"],
        "validation_rows": manifest["source"]["validation_rows"],
        "usable_validation_items": manifest["source"]["validation_rows"],
        "inference_device": manifest["inference"]["device"],
        "inference_dtype": manifest["inference"]["autocast_dtype"],
        "r009_manifest_sha256": r009_hash,
        "r008_manifest_sha256": r008_hash,
    }
    if not isinstance(metadata, dict) or metadata != expected_metadata:
        raise ValueError(
            "R009 calibration is not bound to this checkpoint, validation, and lock"
        )


def verify_r009_calibration(seed, manifest, r009_hash, r008_manifest, r008_hash):
    checkpoint, provenance, _ = verify_r009_training(
        seed, manifest, r009_hash, r008_manifest, r008_hash
    )
    path = checkpoint / "calibration-sib200-km.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"R009 calibration is missing or unsafe for seed {seed}")
    payload = read_json(path)
    validate_r009_calibration_payload(
        payload, seed, checkpoint, provenance, manifest, r009_hash, r008_hash
    )
    return path, sha256(path), payload


def verify_score_readiness(manifest, r009_hash, r008_manifest, r008_hash):
    """Check every model and baseline artifact without opening the test JSONL."""
    ready = {}
    for seed in SEEDS:
        anchors = {}
        for arm in ("specialist", "multitask"):
            anchor, provenance, _ = verify_r008_anchor(
                seed, arm, manifest, r008_manifest, r008_hash
            )
            anchors[arm] = {"anchor": anchor, "provenance": provenance}
        checkpoint, training, telemetry = verify_r009_training(
            seed, manifest, r009_hash, r008_manifest, r008_hash
        )
        calibration = verify_r009_calibration(
            seed, manifest, r009_hash, r008_manifest, r008_hash
        )
        ready[seed] = {
            "checkpoint": checkpoint,
            "training": training,
            "telemetry": telemetry,
            "calibration": calibration,
            "anchors": anchors,
        }

    # Read the already-scored R008 cases only after every R009 checkpoint and
    # calibration artifact has passed its provenance checks.
    for seed in SEEDS:
        for arm in ("specialist", "multitask"):
            anchor = ready[seed]["anchors"][arm]["anchor"]
            ready[seed]["anchors"][arm]["report"] = verify_anchor_report(
                seed, arm, anchor, manifest
            )
    return ready


def verify_test_file(manifest):
    """Call only after verify_score_readiness has succeeded."""
    test_path = ROOT / manifest["source"]["test_file"]
    if sha256(test_path) != manifest["source"]["test_sha256"]:
        raise ValueError("R009 test file fingerprint mismatch")
    return test_path
