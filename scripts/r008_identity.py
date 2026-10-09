"""Shared identities and lock checks for the R008 study scripts."""

import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r008-specialist-controls.json"
PROTOCOL = ROOT / "research/R008-specialist-controls/README.md"
PREPARER = ROOT / "scripts/prepare_r008.py"
IMPLEMENTATION_FILES = (
    "scripts/train_full_8bit.py",
    "scripts/laya_amp.py",
    "scripts/calibrate_r008.py",
    "scripts/evaluate_r008.py",
    "scripts/summarize_r008.py",
    "scripts/r008_identity.py",
)
ARM_TASKS = {
    "multitask": ("khmer-news", "sib200-km"),
    "khmer-news-specialist": ("khmer-news",),
    "sib200-specialist": ("sib200-km",),
}
ARM_OUTPUT_PREFIXES = {
    "multitask": "R008-mt",
    "khmer-news-specialist": "R008-news",
    "sib200-specialist": "R008-sib",
}
ARM_UPDATE_KEYS = {
    "multitask": "multitask",
    "khmer-news-specialist": "khmer-news-specialist",
    "sib200-specialist": "sib200-specialist",
}
PREREGISTERED_SEEDS = (13, 14)
PROVENANCE_FILENAME = "training_provenance.json"
TELEMETRY_FILENAME = "training_telemetry.json"
CONVERSION_AUDIT_FILENAME = "training_conversion_audit.json"
INFERENCE_POLICY = {
    "device": "cuda",
    "autocast_dtype": "bfloat16",
    "laya_cuda_amp": "bf16",
}
INFERENCE_ROOT_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "questions.json",
)
INFERENCE_DIRECTORIES = ("encoder", "tokenizer")


def sha256(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_laya_cuda_amp_override(override=None):
    if override is None:
        override = os.environ.get("LAYA_CUDA_AMP")
    if override and override.lower() not in ("bf16", "bfloat16"):
        raise ValueError(
            "R008 requires BF16 inference; conflicting LAYA_CUDA_AMP override"
        )


def validate_inference_policy(manifest):
    if manifest.get("inference") != INFERENCE_POLICY:
        raise ValueError(
            "R008 inference precision policy differs from the locked protocol"
        )
    validate_laya_cuda_amp_override()


def install_r008_forward_guard(agent):
    """Reject any Laya forward that would leave the locked CUDA BF16 path."""
    import torch

    def check_forward_policy(_module, _args, _kwargs):
        valid = (
            agent.device.type == INFERENCE_POLICY["device"]
            and agent.dtype == torch.bfloat16
            and agent.amp_enabled
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
        if not valid:
            reason = getattr(agent, "last_fallback_reason", None)
            detail = f"; preceding Laya failure: {reason}" if reason else ""
            raise RuntimeError(
                "R008 requires CUDA BF16 autocast for every model forward; "
                f"refusing a precision or device fallback{detail}"
            )

    agent.model.register_forward_pre_hook(check_forward_policy, with_kwargs=True)


def load_r008_agent(checkpoint, manifest, calibration=None):
    """Load Laya with the locked precision, rejecting unsupported runtime changes."""
    validate_inference_policy(manifest)
    from unittest.mock import patch

    import laya
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("R008 inference requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("R008 inference requires CUDA BF16 support")
    kwargs = {"device": INFERENCE_POLICY["device"]}
    if calibration is not None:
        kwargs["calibration"] = str(calibration)
    with patch.dict(os.environ, {"LAYA_CUDA_AMP": INFERENCE_POLICY["laya_cuda_amp"]}):
        agent = laya.load(str(checkpoint), **kwargs)
    if (
        agent.device.type != INFERENCE_POLICY["device"]
        or agent.dtype != torch.bfloat16
        or not agent.amp_enabled
    ):
        raise ValueError("Laya did not load with the locked CUDA BF16 inference policy")
    install_r008_forward_guard(agent)
    return agent, INFERENCE_POLICY["autocast_dtype"]


def checkpoint_inference_inventory(checkpoint):
    """Hash every file that can affect loading or default R008 inference."""
    checkpoint = Path(checkpoint).resolve()
    inventory = {}
    for relative in INFERENCE_ROOT_FILES:
        path = checkpoint / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"checkpoint inference file is missing or unsafe: {relative}"
            )
        inventory[relative] = sha256(path)

    for relative in INFERENCE_DIRECTORIES:
        directory = checkpoint / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                f"checkpoint inference directory is missing or unsafe: {relative}"
            )
        found = False
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"checkpoint inference file is a symlink: {path}")
            if path.is_file():
                found = True
                inventory[path.relative_to(checkpoint).as_posix()] = sha256(path)
        if not found:
            raise ValueError(f"checkpoint inference directory is empty: {relative}")
    return dict(sorted(inventory.items()))


def validate_training_conversion(summary, training_arm, manifest):
    """Require one usable Laya item per exported training row and no holdout."""
    if training_arm not in manifest.get("training_split", {}).get("arms", {}):
        raise ValueError(
            f"R008 training arm is absent from the manifest: {training_arm}"
        )
    expected_rows = manifest["training_split"]["arms"][training_arm]["rows"]
    expected_updates = manifest["recipe"]["expected_optimizer_updates_per_seed"][
        ARM_UPDATE_KEYS[training_arm]
    ]
    if (
        summary.get("rows_read") != expected_rows
        or summary.get("valid_items") != expected_rows
        or summary.get("train_items") != expected_rows
        or summary.get("calibration_items") != 0
        or summary.get("optimizer_updates") != expected_updates
        or summary.get("skipped") != {}
    ):
        raise ValueError(
            "R008 Laya conversion differs from the locked row/item/update budget: "
            f"{summary!r}"
        )
    return expected_rows, expected_updates


def verify_locked_manifest(path=MANIFEST):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "locked_exploratory":
        raise ValueError("R008 requires a locked exploratory manifest")
    validate_inference_policy(manifest)
    if manifest.get("preparer_sha256") != sha256(PREPARER):
        raise ValueError("R008 preparer fingerprint mismatch")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("R008 protocol fingerprint mismatch")
    implementation = manifest.get("implementation_sha256", {})
    for relative in IMPLEMENTATION_FILES:
        if implementation.get(relative) != sha256(relative):
            raise ValueError(f"R008 implementation fingerprint mismatch: {relative}")
    environment = manifest.get("environment", {})
    if environment.get("uv_lock_sha256") != sha256("uv.lock"):
        raise ValueError("R008 frozen dependency lock fingerprint mismatch")
    for package in ("laya", "torch", "bitsandbytes"):
        from importlib.metadata import version

        if environment.get(package) != version(package):
            raise ValueError(
                f"R008 {package} version differs from the locked environment"
            )
    return manifest, sha256(path)


def write_training_conversion_audit(
    checkpoint,
    training_arm,
    seed,
    data_path,
    manifest_path,
    preflight,
    training_result,
):
    """Persist Laya's tokenizer-conversion counts and verify the post-run summary."""
    checkpoint = Path(checkpoint).resolve()
    manifest, manifest_hash = verify_locked_manifest(manifest_path)
    if training_arm not in ARM_TASKS:
        raise ValueError(f"unknown R008 training arm: {training_arm!r}")
    if seed not in manifest["recipe"]["seeds"]:
        raise ValueError(f"R008 seed is not preregistered: {seed}")
    expected_rows, expected_updates = validate_training_conversion(
        preflight, training_arm, manifest
    )

    arm_manifest = manifest["training_split"]["arms"][training_arm]
    expected_data = (ROOT / arm_manifest["file"]).resolve()
    data_path = Path(data_path).resolve()
    if data_path != expected_data or sha256(data_path) != arm_manifest["sha256"]:
        raise ValueError("R008 conversion audit data differs from the locked arm")
    expected_checkpoint = (
        ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[training_arm]}-s{seed}"
    ).resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError(f"R008 checkpoint path must be {expected_checkpoint}")

    observed = {
        "train_items": training_result.get("train_items"),
        "calibration_items": training_result.get("calibration_items"),
        "skipped": training_result.get("skipped"),
    }
    if observed != {
        "train_items": expected_rows,
        "calibration_items": 0,
        "skipped": {},
    }:
        raise ValueError(
            f"R008 post-training conversion differs from preflight: {observed!r}"
        )

    payload = {
        "schema": "r008-training-conversion-audit/1",
        "training_arm": training_arm,
        "seed": seed,
        "data_file": str(data_path.relative_to(ROOT)),
        "training_file_sha256": arm_manifest["sha256"],
        "locked_manifest_sha256": manifest_hash,
        "expected_rows_and_items": expected_rows,
        "expected_optimizer_updates": expected_updates,
        "preflight": preflight,
        "observed": observed,
    }
    output = checkpoint / CONVERSION_AUDIT_FILENAME
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def verify_training_conversion_audit(
    checkpoint, training_arm, seed, data_path, manifest, manifest_hash
):
    checkpoint = Path(checkpoint).resolve()
    audit_path = checkpoint / CONVERSION_AUDIT_FILENAME
    if not audit_path.is_file():
        raise ValueError("checkpoint has no R008 training conversion audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_rows, expected_updates = validate_training_conversion(
        audit.get("preflight", {}), training_arm, manifest
    )
    arm_manifest = manifest["training_split"]["arms"][training_arm]
    expected_data = (ROOT / arm_manifest["file"]).resolve()
    expected_checkpoint = (
        ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[training_arm]}-s{seed}"
    ).resolve()
    expected_observed = {
        "train_items": expected_rows,
        "calibration_items": 0,
        "skipped": {},
    }
    if (
        checkpoint != expected_checkpoint
        or Path(data_path).resolve() != expected_data
        or audit.get("schema") != "r008-training-conversion-audit/1"
        or audit.get("training_arm") != training_arm
        or audit.get("seed") != seed
        or audit.get("data_file") != str(expected_data.relative_to(ROOT))
        or audit.get("training_file_sha256") != arm_manifest["sha256"]
        or sha256(expected_data) != arm_manifest["sha256"]
        or audit.get("locked_manifest_sha256") != manifest_hash
        or audit.get("expected_rows_and_items") != expected_rows
        or audit.get("expected_optimizer_updates") != expected_updates
        or audit.get("observed") != expected_observed
    ):
        raise ValueError("R008 training conversion audit does not match the locked arm")
    return audit


def write_training_provenance(
    checkpoint, training_arm, seed, data_path, manifest_path=MANIFEST
):
    """Record the locked arm and exact weights only after a successful guarded run."""
    checkpoint = Path(checkpoint).resolve()
    manifest, manifest_hash = verify_locked_manifest(manifest_path)
    if training_arm not in ARM_TASKS:
        raise ValueError(f"unknown R008 training arm: {training_arm!r}")
    if seed not in manifest["recipe"]["seeds"]:
        raise ValueError(f"R008 seed is not preregistered: {seed}")

    arm = manifest["training_split"]["arms"][training_arm]
    expected_data = (ROOT / arm["file"]).resolve()
    data_path = Path(data_path).resolve()
    if data_path != expected_data or sha256(data_path) != arm["sha256"]:
        raise ValueError("R008 training data does not match the locked arm")

    expected_output = (
        ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[training_arm]}-s{seed}"
    )
    if checkpoint != expected_output.resolve():
        raise ValueError(f"R008 checkpoint path must be {expected_output}")

    weights_path = checkpoint / "model.safetensors"
    telemetry_path = checkpoint / TELEMETRY_FILENAME
    if not weights_path.is_file() or not telemetry_path.is_file():
        raise FileNotFoundError("R008 weights and training telemetry must both exist")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    update_key = ARM_UPDATE_KEYS[training_arm]
    expected_updates = manifest["recipe"]["expected_optimizer_updates_per_seed"][
        update_key
    ]
    if telemetry.get("failure") or telemetry.get("telemetry_cleanup_failure"):
        raise ValueError("R008 training telemetry records a failure")
    if (
        telemetry.get("optimizer_attempts") != expected_updates
        or telemetry.get("successful_updates", 0) + telemetry.get("skipped_updates", 0)
        != expected_updates
        or telemetry.get("scheduler_steps") != telemetry.get("successful_updates")
    ):
        raise ValueError(
            "R008 training telemetry does not satisfy the locked update budget"
        )

    verify_training_conversion_audit(
        checkpoint, training_arm, seed, data_path, manifest, manifest_hash
    )
    inference_files = checkpoint_inference_inventory(checkpoint)

    payload = {
        "schema": "r008-training-provenance/1",
        "study": "R008-specialist-controls",
        "training_arm": training_arm,
        "calibration_tasks": list(ARM_TASKS[training_arm]),
        "seed": seed,
        "checkpoint_directory": str(checkpoint),
        "weights_sha256": sha256(weights_path),
        "inference_files_sha256": inference_files,
        "training_conversion_audit_sha256": sha256(
            checkpoint / CONVERSION_AUDIT_FILENAME
        ),
        "training_file": str(data_path.relative_to(ROOT)),
        "training_file_sha256": arm["sha256"],
        "locked_manifest_sha256": manifest_hash,
        "base_revision": manifest["recipe"]["base_revision"],
        "trainer_sha256": manifest["implementation_sha256"][
            "scripts/train_full_8bit.py"
        ],
        "amp_guard_sha256": manifest["implementation_sha256"]["scripts/laya_amp.py"],
        "training_telemetry_sha256": sha256(telemetry_path),
        "optimizer_updates": {
            "attempts": telemetry["optimizer_attempts"],
            "successful": telemetry["successful_updates"],
            "skipped": telemetry["skipped_updates"],
            "scheduler_steps": telemetry["scheduler_steps"],
        },
    }
    output = checkpoint / PROVENANCE_FILENAME
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def verify_checkpoint_training_provenance(checkpoint, task, manifest, manifest_hash):
    """Require a matching, integrity-checked R008 training record for calibration."""
    if task not in ("khmer-news", "sib200-km"):
        raise ValueError(f"unknown R008 calibration task: {task!r}")
    checkpoint = Path(checkpoint).resolve()
    provenance_path = checkpoint / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise ValueError("checkpoint has no R008 training provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    arm = provenance.get("training_arm")
    if arm not in ARM_TASKS or task not in ARM_TASKS[arm]:
        raise ValueError(
            "checkpoint training arm is incompatible with the calibration task"
        )
    if provenance.get("schema") != "r008-training-provenance/1":
        raise ValueError("unsupported R008 training provenance schema")
    if provenance.get("study") != "R008-specialist-controls":
        raise ValueError("checkpoint provenance belongs to a different study")
    if provenance.get("calibration_tasks") != list(ARM_TASKS[arm]):
        raise ValueError("checkpoint provenance task mapping is inconsistent")
    if provenance.get("checkpoint_directory") != str(checkpoint):
        raise ValueError("checkpoint path differs from its R008 training provenance")
    if provenance.get("locked_manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint was trained under a different R008 manifest")

    seed = provenance.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in manifest["recipe"]["seeds"]
    ):
        raise ValueError(
            "checkpoint seed is not preregistered in the locked R008 manifest"
        )
    expected_checkpoint = (
        ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[arm]}-s{seed}"
    ).resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError("checkpoint path does not match its training arm and seed")
    arms = manifest["training_split"]["arms"]
    if arm not in arms:
        raise ValueError("checkpoint arm is absent from the locked R008 manifest")
    arm_manifest = arms[arm]
    training_file = (ROOT / arm_manifest["file"]).resolve()
    if (
        provenance.get("training_file") != str(training_file.relative_to(ROOT))
        or provenance.get("training_file_sha256") != arm_manifest["sha256"]
        or sha256(training_file) != arm_manifest["sha256"]
    ):
        raise ValueError("checkpoint training file differs from the locked R008 arm")
    if provenance.get("base_revision") != manifest["recipe"]["base_revision"]:
        raise ValueError("checkpoint base revision differs from the locked R008 recipe")
    implementation = manifest["implementation_sha256"]
    if (
        provenance.get("trainer_sha256") != implementation["scripts/train_full_8bit.py"]
        or provenance.get("amp_guard_sha256") != implementation["scripts/laya_amp.py"]
    ):
        raise ValueError(
            "checkpoint training implementation differs from the locked R008 run"
        )

    weights_path = checkpoint / "model.safetensors"
    telemetry_path = checkpoint / TELEMETRY_FILENAME
    if not weights_path.is_file() or provenance.get("weights_sha256") != sha256(
        weights_path
    ):
        raise ValueError("checkpoint weights differ from the R008 training provenance")
    if not telemetry_path.is_file() or provenance.get(
        "training_telemetry_sha256"
    ) != sha256(telemetry_path):
        raise ValueError(
            "checkpoint telemetry differs from the R008 training provenance"
        )
    verify_training_conversion_audit(
        checkpoint, arm, seed, training_file, manifest, manifest_hash
    )
    if provenance.get("training_conversion_audit_sha256") != sha256(
        checkpoint / CONVERSION_AUDIT_FILENAME
    ):
        raise ValueError("checkpoint conversion audit differs from R008 provenance")
    if provenance.get("inference_files_sha256") != checkpoint_inference_inventory(
        checkpoint
    ):
        raise ValueError(
            "inference-defining checkpoint files differ from R008 provenance"
        )
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    update_key = ARM_UPDATE_KEYS[arm]
    expected_updates = manifest["recipe"]["expected_optimizer_updates_per_seed"][
        update_key
    ]
    updates = provenance.get("optimizer_updates", {})
    if (
        updates.get("attempts") != expected_updates
        or updates.get("successful", 0) + updates.get("skipped", 0) != expected_updates
        or updates.get("scheduler_steps") != updates.get("successful")
        or updates.get("attempts") != telemetry.get("optimizer_attempts")
        or updates.get("successful") != telemetry.get("successful_updates")
        or updates.get("skipped") != telemetry.get("skipped_updates")
        or updates.get("scheduler_steps") != telemetry.get("scheduler_steps")
        or telemetry.get("failure")
        or telemetry.get("telemetry_cleanup_failure")
    ):
        raise ValueError(
            "checkpoint update telemetry does not satisfy the locked R008 budget"
        )
    return provenance


def verify_calibration_artifact(checkpoint, task, provenance, manifest, manifest_hash):
    """Verify a temperature-only Laya payload and its pinned validation identity."""
    if task not in ARM_TASKS[provenance["training_arm"]]:
        raise ValueError(
            "checkpoint training arm is incompatible with the calibration task"
        )
    task_manifest = manifest["validation_calibration"][task]
    expected_validation = f"data/processed/r008-{task}-calibration.jsonl"
    expected_test = f"data/processed/{task}-test.jsonl"
    if (
        task_manifest.get("calibration_file") != expected_validation
        or task_manifest.get("test_file") != expected_test
    ):
        raise ValueError("R008 task-to-validation/test mapping is inconsistent")
    if sha256(expected_validation) != task_manifest.get("calibration_sha256"):
        raise ValueError("R008 validation calibration file fingerprint mismatch")

    checkpoint = Path(checkpoint).resolve()
    calibration_path = checkpoint / f"calibration-{task}.json"
    if calibration_path.is_symlink() or not calibration_path.is_file():
        raise ValueError("R008 calibration artifact is missing or unsafe")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_fields = {
        "version",
        "temperature",
        "temperature_by_options",
        "model_id_or_path",
        "subfolder",
        "config",
        "r008",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(
            "R008 calibration artifact has an unexpected Laya payload shape"
        )

    from laya.calibrate import CALIBRATION_VERSION
    from laya.common import TEMP_MAX, TEMP_MIN

    if (
        isinstance(payload["version"], bool)
        or not isinstance(payload["version"], int)
        or payload["version"] != CALIBRATION_VERSION
    ):
        raise ValueError("R008 calibration artifact has an unsupported Laya version")

    def validate_temperature(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"R008 calibration artifact has invalid {name}")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"R008 calibration artifact has invalid {name}") from exc
        if not math.isfinite(value) or not TEMP_MIN <= value <= TEMP_MAX:
            raise ValueError(f"R008 calibration artifact has invalid {name}")

    temperatures = payload["temperature"]
    if not isinstance(temperatures, list) or len(temperatures) != 3:
        raise ValueError("R008 calibration artifact must contain three temperatures")
    for index, value in enumerate(temperatures):
        validate_temperature(f"temperature[{index}]", value)

    temperatures_by_options = payload["temperature_by_options"]
    if not isinstance(temperatures_by_options, dict):
        raise TypeError("R008 calibration option temperatures must be an object")
    for bucket, value in temperatures_by_options.items():
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("R008 calibration artifact has an invalid option bucket")
        validate_temperature(f"temperature_by_options[{bucket!r}]", value)

    config_path = checkpoint / "rl_agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("R008 checkpoint config must be an object")
    config_identity = {
        key: value
        for key, value in config.items()
        if key not in ("temperature", "temperature_by_options")
    }
    if (
        payload["model_id_or_path"] != str(checkpoint)
        or payload["subfolder"] is not None
        or payload["config"] != config_identity
    ):
        raise ValueError("R008 Laya calibration identity differs from its checkpoint")

    metadata = payload["r008"]
    expected = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": provenance["weights_sha256"],
        "training_arm": provenance["training_arm"],
        "training_provenance_sha256": sha256(checkpoint / PROVENANCE_FILENAME),
        "validation_file": expected_validation,
        "validation_sha256": task_manifest["calibration_sha256"],
        "validation_rows": task_manifest["calibration_rows"],
        "usable_validation_items": task_manifest["calibration_rows"],
        "inference_device": INFERENCE_POLICY["device"],
        "inference_dtype": INFERENCE_POLICY["autocast_dtype"],
        "locked_manifest_sha256": manifest_hash,
    }
    if not isinstance(metadata, dict) or metadata != expected:
        raise ValueError(
            "calibration map is not bound to this checkpoint, arm, task, validation, and lock"
        )
    return calibration_path, sha256(calibration_path)


def verify_study_readiness(manifest, manifest_hash):
    """Verify all preregistered checkpoints and calibration maps before test access."""
    if manifest.get("recipe", {}).get("seeds") != list(PREREGISTERED_SEEDS):
        raise ValueError("R008 manifest seeds differ from the preregistered study")
    if set(manifest.get("training_split", {}).get("arms", {})) != set(ARM_TASKS):
        raise ValueError(
            "R008 manifest training arms differ from the preregistered study"
        )
    if set(manifest.get("validation_calibration", {})) != {"khmer-news", "sib200-km"}:
        raise ValueError("R008 manifest task maps differ from the preregistered study")

    ready = {}
    for seed in PREREGISTERED_SEEDS:
        for arm, tasks in ARM_TASKS.items():
            checkpoint = (
                ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[arm]}-s{seed}"
            ).resolve()
            provenance = verify_checkpoint_training_provenance(
                checkpoint, tasks[0], manifest, manifest_hash
            )
            if provenance.get("training_arm") != arm or provenance.get("seed") != seed:
                raise ValueError("R008 sibling checkpoint provenance is misbound")
            calibrations = {
                task: verify_calibration_artifact(
                    checkpoint, task, provenance, manifest, manifest_hash
                )
                for task in tasks
            }
            ready[(arm, seed)] = {
                "checkpoint": checkpoint,
                "provenance": provenance,
                "calibrations": calibrations,
            }
    return ready


def normalize_value(value):
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, dict):
        return {key: normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


def normalized_state_key(row):
    return json.dumps(
        normalize_value(row["state"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def state_group_id(row):
    return hashlib.sha256(normalized_state_key(row).encode("utf-8")).hexdigest()


def row_identity_tag(row):
    """Bind one report case to the full exported row, including its gold and options."""
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "r008:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
