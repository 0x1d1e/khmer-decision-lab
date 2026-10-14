"""Identity, preparation, and pre-test readiness checks for R012."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from r008_identity import checkpoint_inference_inventory
from r010_identity import (
    IMPLEMENTATION_FILES as R010_IMPLEMENTATION_FILES,
)
from r010_identity import collect_environment as collect_r010_environment
from r010_identity import (
    verify_locked_manifest as verify_r010_locked_manifest,
)
from r010_identity import (
    verify_prepared as verify_r010_prepared,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r012-ocr-continuation.json"
PROTOCOL = ROOT / "research/R012-ocr-continuation/README.md"
ARTIFACT_DIR = ROOT / "artifacts/runs/R012"
PREPARED_TRAIN = "data/processed/r012-massive-km-train-ocr_12px.jsonl"
TRAIN_TRANSCRIPTS = "artifacts/runs/R012/train-ocr-transcripts.jsonl"
SEEDS = (13, 14)
ARMS = ("clean", "ocr12")
CONDITIONS = ("clean", "ocr_12px")
IMPLEMENTATION_FILES = (
    "scripts/r008_identity.py",
    "scripts/laya_amp.py",
    "scripts/r010_identity.py",
    "scripts/prepare_r010.py",
    "scripts/r012_identity.py",
    "scripts/prepare_r012.py",
    "scripts/train_r012_continuation.py",
    "scripts/evaluate_r012.py",
    "scripts/summarize_r012.py",
    "tests/test_r012_ocr_continuation.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def train_row_id(train_sha256, index, row):
    encoded = f"{train_sha256}\0{index}\0{canonical_json(row)}".encode()
    return hashlib.sha256(encoded).hexdigest()


def tag_value(row, prefix):
    matches = [
        str(tag)[len(prefix) :]
        for tag in row.get("tags", [])
        if str(tag).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"row must contain exactly one {prefix} tag")
    return matches[0]


def intent_labels(row):
    criteria = row["questions"]["intent"]["criteria"]
    if isinstance(criteria, dict):
        return list(criteria)
    if isinstance(criteria, list):
        return list(criteria)
    raise ValueError("R012 intent criteria must be a list or ordered mapping")


def validate_report_cases(cases, rows, model_name):
    if len(cases) != len(rows):
        raise ValueError(f"R012 row count mismatch for {model_name}")
    for index, (case, row) in enumerate(zip(cases, rows, strict=True)):
        if not isinstance(case, dict) or not isinstance(row, dict):
            raise TypeError(f"R012 report row is not an object: {model_name}/{index}")
        gold = row["expected"]["intent"]
        labels = intent_labels(row)
        answer = case.get("answer")
        probabilities = (
            answer.get("probabilities") if isinstance(answer, dict) else None
        )
        if (
            case.get("model") != model_name
            or case.get("qid") != "intent"
            or case.get("language") != row.get("language")
            or case.get("expected") != gold
            or case.get("tags") != row.get("tags", [])
            or len(labels) != 20
            or len(set(labels)) != len(labels)
            or gold not in labels
            or not isinstance(answer, dict)
            or answer.get("type") != "choice"
            or answer.get("choice") not in labels
            or type(case.get("correct")) is not bool
            or case["correct"] != (answer.get("choice") == gold)
            or not isinstance(probabilities, dict)
            or set(probabilities) != set(labels)
        ):
            raise ValueError(
                f"R012 report row alignment mismatch: {model_name}/{index}"
            )
        values = list(probabilities.values())
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or value > 1
                for value in values
            )
            or abs(sum(values) - 1.0) > 0.01
        ):
            raise ValueError(
                f"R012 report probabilities are invalid: {model_name}/{index}"
            )


def _verify_r010_lock_without_test(manifest, source):
    r010_path = ROOT / source["r010_manifest"]
    if sha256(r010_path) != source["r010_manifest_sha256"]:
        raise ValueError("R012 R010 manifest fingerprint mismatch")
    r010 = read_json(r010_path)
    if (
        r010.get("study") != "R010-massive-ocr-cascade"
        or r010.get("status") != "locked_exploratory"
        or r010.get("protocol_sha256")
        != sha256(ROOT / "research/R010-massive-ocr-cascade/README.md")
    ):
        raise ValueError("R012 R010 lock identity mismatch")
    for relative in R010_IMPLEMENTATION_FILES:
        if r010.get("implementation_sha256", {}).get(relative) != sha256(
            ROOT / relative
        ):
            raise ValueError(
                f"R012 R010 implementation fingerprint mismatch: {relative}"
            )

    r010_source = r010["source"]
    for key, expected in (
        ("train_file", source["train_file"]),
        ("train_sha256", source["train_sha256"]),
        ("test_file", source["test_file"]),
        ("test_sha256", source["test_sha256"]),
        ("r006_manifest", source["r006_manifest"]),
        ("r006_manifest_sha256", source["r006_manifest_sha256"]),
    ):
        if r010_source.get(key) != expected:
            raise ValueError(f"R012 source differs from the R010 anchor: {key}")

    for name, model in r010["models"].items():
        if name not in ("mt-s13", "mt-s14"):
            continue
        checkpoint = ROOT / model["checkpoint"]
        if (
            checkpoint_inference_inventory(checkpoint)
            != model["inference_files_sha256"]
        ):
            raise ValueError(f"R012 R006 checkpoint fingerprint mismatch: {name}")
    return r010, sha256(r010_path)


def _check_recipe(manifest, r010):
    recipe = manifest["recipe"]
    source = manifest["source"]
    base_revision = r010["models"]["mt-s13"]["base_revision"]
    expected_recipe = {
        "base_revision": base_revision,
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 32,
        "encoder_lr": 0.000025,
        "head_lr": 0.0001,
        "min_lr": 0.000001,
        "weight_decay": 0.01,
        "optimizer_reset": True,
        "grad_clip": 1.0,
        "loss": "soft-ce",
        "freeze_encoder": False,
        "shuffle_options": ["choice"],
        "amp": True,
        "gradient_checkpointing": True,
        "optimizer": "bitsandbytes.optim.adamw.PagedAdamW8bit",
        "calib_max": 0,
        "calib_frac": 0.0,
        "calib_seed": 20260922,
        "expected_optimizer_attempts": 360,
        "minimum_successful_updates": 342,
        "maximum_paired_update_difference": 7,
        "evaluation_batch_size": 8,
        "inference": {
            "device": "cuda",
            "autocast_dtype": "bfloat16",
            "laya_cuda_amp": "bf16",
        },
        "inference_grouping": "Laya evals.evaluate: consecutive identical question schemas, singleton predict, batch_size=8",
        "training_data_conditions": {
            "clean": "original MASSIVE train utterance",
            "ocr12": "R010 12px Tesseract transcript",
        },
    }
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("R012 recipe differs from the locked protocol")
    expected_evaluation = {
        "primary_metric": "choice_accuracy",
        "ocr_domain_gain_condition": "ocr_12px",
        "ocr_domain_gain_comparison": "ocr12 continuation minus clean continuation",
        "ocr_domain_gain_ci_lower_bound_gt_percentage_points": 0,
        "clean_text_preservation_comparison": "ocr12 continuation minus frozen R006 checkpoint",
        "clean_text_preservation_condition": "clean",
        "clean_text_preservation_ci_lower_bound_gt_percentage_points": -2,
        "group_key": "r010-group tag for NFKC-normalized, casefolded, whitespace-collapsed source utterance",
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 13,
        "bootstrap_method": "resample groups with replacement; retain all rows; row-weight paired accuracy difference",
        "overlap_policy": "exclude whole source-utterance groups whose clean or OCR test input matches normalized clean or OCR training input",
    }
    if manifest.get("evaluation") != expected_evaluation:
        raise ValueError("R012 evaluation differs from the locked protocol")
    if set(manifest.get("models", {})) != {"mt-s13", "mt-s14"}:
        raise ValueError("R012 requires exactly the two locked R006 seed anchors")
    expected_rows = {
        "training_rows": 11_514,
        "test_rows": r010["source"]["test_rows"],
    }
    if any(source.get(key) != value for key, value in expected_rows.items()):
        raise ValueError("R012 source row counts differ from the locked splits")
    updates = (
        math.ceil(
            math.ceil(source["training_rows"] / recipe["micro_batch"])
            / recipe["grad_accum"]
        )
        * recipe["epochs"]
    )
    if updates != recipe["expected_optimizer_attempts"]:
        raise ValueError("R012 optimizer-attempt count does not follow its data budget")
    if r010.get("recipe", {}).get("inference") != recipe["inference"]:
        raise ValueError("R012 inference precision differs from R010")
    for name, model in manifest["models"].items():
        anchor = r010["models"].get(name)
        if (
            anchor is None
            or model.get("checkpoint") != anchor.get("checkpoint")
            or model.get("seed") != anchor.get("seed")
            or model.get("base_revision") != anchor.get("base_revision")
            or model.get("choice_temperature") != anchor.get("choice_temperature")
            or model.get("max_len") != anchor.get("max_len")
            or model.get("head_max_len") != anchor.get("head_max_len")
            or model.get("inference_files_sha256")
            != anchor.get("inference_files_sha256")
        ):
            raise ValueError(f"R012 checkpoint anchor differs from R010: {name}")


def verify_locked_manifest(path=MANIFEST, *, test_access=False):
    """Verify lock and training sources; defer every test read until readiness passes."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = read_json(path)
    if (
        manifest.get("study") != "R012-ocr-continuation"
        or manifest.get("status") != "locked_exploratory"
    ):
        raise ValueError("R012 requires a locked exploratory manifest")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("R012 protocol fingerprint mismatch")
    if set(manifest.get("implementation_sha256", {})) != set(IMPLEMENTATION_FILES):
        raise ValueError("R012 implementation inventory differs from the locked set")
    for relative in IMPLEMENTATION_FILES:
        if manifest.get("implementation_sha256", {}).get(relative) != sha256(
            ROOT / relative
        ):
            raise ValueError(f"R012 implementation fingerprint mismatch: {relative}")

    import platform
    from importlib.metadata import version

    import torch

    environment = manifest["environment"]
    current_environment = collect_r010_environment()
    current_environment["bitsandbytes"] = version("bitsandbytes")
    if current_environment != environment:
        raise ValueError("R012 runtime environment differs from the locked environment")
    if sha256(ROOT / "uv.lock") != environment["uv_lock_sha256"]:
        raise ValueError("R012 frozen dependency lock fingerprint mismatch")
    if platform.python_version() != environment["python"]:
        raise ValueError("R012 Python version differs from the lock")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R012 requires the locked CUDA BF16 environment")

    source = manifest["source"]
    for relative, expected in (
        (source["massive_manifest"], source["massive_manifest_sha256"]),
        (source["r006_manifest"], source["r006_manifest_sha256"]),
        (source["train_file"], source["train_sha256"]),
    ):
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"R012 source fingerprint mismatch: {relative}")
    r010, r010_hash = _verify_r010_lock_without_test(manifest, source)
    if r010_hash != source["r010_manifest_sha256"]:
        raise ValueError("R012 R010 anchor hash mismatch")
    r010_environment = dict(manifest["environment"])
    r010_environment.pop("bitsandbytes")
    if r010_environment != r010["environment"]:
        raise ValueError("R012 runtime differs from R010 apart from bitsandbytes")
    _check_recipe(manifest, r010)

    if test_access:
        verified_r010, verified_hash = verify_r010_locked_manifest(
            ROOT / source["r010_manifest"]
        )
        if verified_hash != source["r010_manifest_sha256"]:
            raise ValueError("R012 R010 test-stage identity mismatch")
        preparation = verify_r010_prepared(verified_r010, verified_hash)
        if (
            sha256(ROOT / source["r010_preparation_artifact"])
            != source["r010_preparation_sha256"]
        ):
            raise ValueError("R012 R010 preparation artifact fingerprint mismatch")
        if (
            sha256(ROOT / source["r010_evaluation_artifact"])
            != source["r010_evaluation_sha256"]
        ):
            raise ValueError("R012 R010 evaluation artifact fingerprint mismatch")
        for condition, output_key in (("clean", "clean"), ("ocr_12px", "ocr_12px")):
            expected = source["test_outputs"][output_key]
            actual = preparation["outputs"].get(condition)
            if (
                actual is None
                or actual["file"] != expected["file"]
                or actual["sha256"] != expected["sha256"]
            ):
                raise ValueError(
                    f"R012 R010 test input differs from its lock: {condition}"
                )

    return manifest, sha256(path), r010, r010_hash


def verify_preparation(manifest, manifest_hash):
    source = manifest["source"]
    artifact_path = ARTIFACT_DIR / "preparation.json"
    preparation = read_json(artifact_path)
    identity = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "preparer_sha256": manifest["implementation_sha256"]["scripts/prepare_r012.py"],
        "source_train_sha256": source["train_sha256"],
        "training_rows": source["training_rows"],
        "prepared_train_file": PREPARED_TRAIN,
        "transcript_artifact": TRAIN_TRANSCRIPTS,
    }
    if any(preparation.get(key) != value for key, value in identity.items()):
        raise ValueError("R012 training preparation identity mismatch")
    train_rows = read_jsonl(ROOT / source["train_file"])
    prepared_rows = read_jsonl(ROOT / PREPARED_TRAIN)
    transcript_rows = read_jsonl(ROOT / TRAIN_TRANSCRIPTS)
    if not (
        len(train_rows)
        == len(prepared_rows)
        == len(transcript_rows)
        == source["training_rows"]
    ):
        raise ValueError("R012 training preparation row count mismatch")
    expected_files = {
        PREPARED_TRAIN: sha256(ROOT / PREPARED_TRAIN),
        TRAIN_TRANSCRIPTS: sha256(ROOT / TRAIN_TRANSCRIPTS),
    }
    if preparation.get("outputs_sha256") != expected_files:
        raise ValueError("R012 training preparation file inventory mismatch")

    from r010_identity import cer_counts

    image_hashes = set()
    transcript_by_raw = {}
    codepoint_edits = grapheme_edits = reference_codepoints = reference_graphemes = 0
    exact_rows = blank_rows = 0
    for index, (original, prepared, record) in enumerate(
        zip(train_rows, prepared_rows, transcript_rows, strict=True)
    ):
        reference = original.get("state", {}).get("utterance")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"R012 training row {index} has no source utterance")
        row_id = train_row_id(source["train_sha256"], index, original)
        if (
            record.get("row_index") != index
            or record.get("row_id") != row_id
            or record.get("reference") != reference
            or not isinstance(record.get("text"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("image_sha256", ""))
        ):
            raise ValueError(
                f"R012 training transcript identity mismatch at row {index}"
            )
        cer = cer_counts(reference, record["text"])
        if record.get("cer") != cer:
            raise ValueError(f"R012 training transcript CER mismatch at row {index}")
        expected = copy.deepcopy(original)
        expected["state"]["utterance"] = record["text"]
        if prepared != expected:
            raise ValueError(
                f"R012 OCR training row changed fields beyond utterance: {index}"
            )
        if (
            reference in transcript_by_raw
            and transcript_by_raw[reference] != record["text"]
        ):
            raise ValueError(
                "R012 duplicate raw utterance received inconsistent transcripts"
            )
        transcript_by_raw[reference] = record["text"]
        image_hashes.add(record["image_sha256"])
        codepoint_edits += cer["codepoint_edits"]
        grapheme_edits += cer["grapheme_edits"]
        reference_codepoints += cer["reference_codepoints"]
        reference_graphemes += cer["reference_graphemes"]
        exact_rows += bool(cer["exact"])
        blank_rows += not record["text"]

    expected_cer = {
        "rows": len(train_rows),
        "unique_raw_utterances": len(transcript_by_raw),
        "codepoint_edits": codepoint_edits,
        "reference_codepoints": reference_codepoints,
        "codepoint_cer": codepoint_edits / reference_codepoints,
        "grapheme_edits": grapheme_edits,
        "reference_graphemes": reference_graphemes,
        "grapheme_cer": grapheme_edits / reference_graphemes,
        "exact_rows": exact_rows,
        "blank_rows": blank_rows,
    }
    if preparation.get("ocr_metrics") != expected_cer:
        raise ValueError("R012 training OCR metrics do not recompute")
    return preparation


def training_output_path(seed, arm):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("unknown R012 seed or training arm")
    return f"artifacts/checkpoints/R012-{arm}-s{seed}-cont"


def verify_inference_metadata_restoration(
    checkpoint, anchor_inventory, provenance, source_checkpoint
):
    """Require restored question and tokenizer metadata to match the R006 anchor."""
    expected = {
        "source_checkpoint": source_checkpoint,
        "changed_model_weights": False,
        "files": {},
    }
    for relative in ("questions.json", "tokenizer/tokenizer_config.json"):
        source_hash = anchor_inventory[relative]
        targets = (relative, f"checkpoint_latest/{relative}")
        expected["files"][relative] = {
            "sha256": source_hash,
            "paths": list(targets),
        }
    if provenance.get("inference_metadata_restoration") != expected:
        raise ValueError("R012 inference-metadata restoration provenance mismatch")
    for relative, record in expected["files"].items():
        for target in record["paths"]:
            path = checkpoint / target
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(
                    f"R012 inference metadata differs from its anchor: {target}"
                )
    return expected


def verify_training_run(seed, arm, manifest, manifest_hash, preparation):
    recipe = manifest["recipe"]
    source = manifest["source"]
    model_name = f"mt-s{seed}"
    model = manifest["models"][model_name]
    output_rel = training_output_path(seed, arm)
    checkpoint = ROOT / output_rel
    paths = {
        "provenance": checkpoint / "training_provenance.json",
        "telemetry": checkpoint / "training_telemetry.json",
        "report": checkpoint / "r012_training_report.json",
        "weights": checkpoint / "model.safetensors",
    }
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"R012 training artifact missing or unsafe: {path}")
    provenance = read_json(paths["provenance"])
    telemetry = read_json(paths["telemetry"])
    report = read_json(paths["report"])
    anchor_checkpoint = ROOT / model["checkpoint"]
    anchor_inventory = checkpoint_inference_inventory(anchor_checkpoint)
    if anchor_inventory != model["inference_files_sha256"]:
        raise ValueError(f"R012 anchor checkpoint changed: {model_name}")
    training_file = source["train_file"] if arm == "clean" else PREPARED_TRAIN
    training_file_sha = (
        source["train_sha256"]
        if arm == "clean"
        else preparation["outputs_sha256"][PREPARED_TRAIN]
    )
    expected = {
        "schema": "r012-training-provenance/1",
        "study": manifest["study"],
        "arm": arm,
        "seed": seed,
        "checkpoint_directory": output_rel,
        "source_checkpoint": model["checkpoint"],
        "source_weights_sha256": model["inference_files_sha256"]["model.safetensors"],
        "source_inference_files_sha256": model["inference_files_sha256"],
        "training_file": training_file,
        "training_file_sha256": training_file_sha,
        "training_rows": source["training_rows"],
        "r006_manifest_sha256": source["r006_manifest_sha256"],
        "r010_manifest_sha256": source["r010_manifest_sha256"],
        "r012_manifest_sha256": manifest_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "validation_used": False,
        "checkpoint_selection": False,
        "weights_sha256": sha256(paths["weights"]),
        "inference_files_sha256": checkpoint_inference_inventory(checkpoint),
        "training_telemetry_sha256": sha256(paths["telemetry"]),
        "training_report_sha256": sha256(paths["report"]),
        "trainer_sha256": manifest["implementation_sha256"][
            "scripts/train_r012_continuation.py"
        ],
        "identity_sha256": manifest["implementation_sha256"][
            "scripts/r012_identity.py"
        ],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"R012 training provenance mismatch: {model_name}/{arm}")
    if provenance.get("base_revision") != recipe["base_revision"]:
        raise ValueError(f"R012 base revision mismatch: {model_name}/{arm}")
    if provenance.get("optimizer_reset") is not True:
        raise ValueError(f"R012 optimizer was not reset: {model_name}/{arm}")
    expected_config = {
        "epochs": recipe["epochs"],
        "micro_batch": recipe["micro_batch"],
        "grad_accum": recipe["grad_accum"],
        "encoder_lr": recipe["encoder_lr"],
        "head_lr": recipe["head_lr"],
        "min_lr": recipe["min_lr"],
        "weight_decay": recipe["weight_decay"],
        "grad_clip": recipe["grad_clip"],
        "loss": recipe["loss"],
        "freeze_encoder": False,
        "shuffle_options": recipe["shuffle_options"],
        "seed": seed,
        "max_len": model["max_len"],
        "head_max_len": model["head_max_len"],
        "amp": True,
        "gradient_checkpointing": True,
        "calib_max": 0,
        "calib_frac": 0.0,
        "eval_data": None,
    }
    config = provenance.get("training_config", {})
    if any(
        type(config.get(key)) is not type(value) or config.get(key) != value
        for key, value in expected_config.items()
    ):
        raise ValueError(f"R012 training config mismatch: {model_name}/{arm}")
    actual_inventory = checkpoint_inference_inventory(checkpoint)
    if provenance.get("inference_files_sha256") != actual_inventory:
        raise ValueError(f"R012 checkpoint inventory mismatch: {model_name}/{arm}")
    if set(actual_inventory) != set(anchor_inventory):
        raise ValueError(f"R012 inference file set changed: {model_name}/{arm}")
    verify_inference_metadata_restoration(
        checkpoint, anchor_inventory, provenance, model["checkpoint"]
    )
    stable_files = set(anchor_inventory) - {"model.safetensors", "rl_agent_config.json"}
    if any(actual_inventory[key] != anchor_inventory[key] for key in stable_files):
        raise ValueError(f"R012 non-weight inference files changed: {model_name}/{arm}")
    if (
        actual_inventory["model.safetensors"]
        == model["inference_files_sha256"]["model.safetensors"]
    ):
        raise ValueError(
            f"R012 continuation weights did not change: {model_name}/{arm}"
        )

    expected_attempts = recipe["expected_optimizer_attempts"]
    expected_optimizer = recipe["optimizer"]
    if (
        type(telemetry.get("optimizer_attempts")) is not int
        or telemetry["optimizer"] != expected_optimizer
        or telemetry["optimizer_attempts"] != expected_attempts
        or type(telemetry.get("successful_updates")) is not int
        or telemetry["successful_updates"] < recipe["minimum_successful_updates"]
        or type(telemetry.get("skipped_updates")) is not int
        or telemetry["successful_updates"] + telemetry["skipped_updates"]
        != expected_attempts
        or telemetry.get("scheduler_steps") != telemetry.get("successful_updates")
        or telemetry.get("failure") is not None
        or telemetry.get("telemetry_cleanup_failure") is not None
        or not isinstance(telemetry.get("training_seconds"), (int, float))
        or isinstance(telemetry.get("training_seconds"), bool)
        or not math.isfinite(telemetry["training_seconds"])
        or telemetry["training_seconds"] <= 0
        or type(telemetry.get("peak_allocated_bytes")) is not int
        or telemetry["peak_allocated_bytes"] <= 0
        or any(
            provenance.get(key) != telemetry.get(key)
            for key in ("optimizer_attempts", "successful_updates", "skipped_updates")
        )
    ):
        raise ValueError(f"R012 optimizer telemetry invalid: {model_name}/{arm}")
    if (
        report.get("train_items") != source["training_rows"]
        or report.get("calibration_items") != 0
        or report.get("skipped") != {}
    ):
        raise ValueError(f"R012 Laya training report invalid: {model_name}/{arm}")

    parent_cfg = read_json(anchor_checkpoint / "rl_agent_config.json")
    output_cfg = read_json(checkpoint / "rl_agent_config.json")
    if (
        output_cfg.get("temperature") != parent_cfg.get("temperature")
        or output_cfg.get("temperature_by_options", {})
        != parent_cfg.get("temperature_by_options", {})
        or output_cfg.get("max_len") != model["max_len"]
        or output_cfg.get("head_max_len") != model["head_max_len"]
        or output_cfg.get("amp_dtype") != "bf16"
    ):
        raise ValueError(
            f"R012 inference config differs from the anchor: {model_name}/{arm}"
        )
    latest_cfg = read_json(checkpoint / "checkpoint_latest/rl_agent_config.json")
    if latest_cfg.get("temperature") != parent_cfg.get("temperature") or latest_cfg.get(
        "temperature_by_options", {}
    ) != parent_cfg.get("temperature_by_options", {}):
        raise ValueError(
            f"R012 checkpoint_latest calibration differs: {model_name}/{arm}"
        )
    restoration = provenance.get("temperature_restoration")
    if restoration != {
        "laya_training_report_temperature": report.get("temperature"),
        "restored_temperature": parent_cfg.get("temperature"),
        "changed_model_weights": False,
        "files": ["rl_agent_config.json", "checkpoint_latest/rl_agent_config.json"],
    }:
        raise ValueError(
            f"R012 temperature restoration record mismatch: {model_name}/{arm}"
        )
    return {
        "seed": seed,
        "arm": arm,
        "checkpoint": output_rel,
        "weights_sha256": actual_inventory["model.safetensors"],
        "provenance_sha256": sha256(paths["provenance"]),
        "training_telemetry_sha256": sha256(paths["telemetry"]),
        "optimizer_attempts": telemetry["optimizer_attempts"],
        "successful_updates": telemetry["successful_updates"],
        "skipped_updates": telemetry["skipped_updates"],
        "training_seconds": telemetry["training_seconds"],
        "peak_allocated_bytes": telemetry["peak_allocated_bytes"],
        "inference_files_sha256": actual_inventory,
    }


def verify_training_readiness(manifest, manifest_hash):
    """Authenticate all four training runs without opening evaluation files."""
    preparation = verify_preparation(manifest, manifest_hash)
    ready = {
        f"{arm}-s{seed}": verify_training_run(
            seed, arm, manifest, manifest_hash, preparation
        )
        for seed in SEEDS
        for arm in ARMS
    }
    limit = manifest["recipe"]["maximum_paired_update_difference"]
    for seed in SEEDS:
        clean = ready[f"clean-s{seed}"]["successful_updates"]
        ocr = ready[f"ocr12-s{seed}"]["successful_updates"]
        if abs(clean - ocr) > limit:
            raise ValueError(f"R012 successful updates are not matched for seed {seed}")
    return ready, preparation


def verify_r010_baselines(manifest, r010, r010_hash, preparation):
    source = manifest["source"]
    evaluation_path = ROOT / source["r010_evaluation_artifact"]
    evaluation = read_json(evaluation_path)
    if (
        evaluation.get("study") != r010["study"]
        or evaluation.get("manifest_sha256") != r010_hash
        or evaluation.get("preparation_sha256")
        != sha256(ROOT / source["r010_preparation_artifact"])
    ):
        raise ValueError("R012 frozen R010 evaluation identity mismatch")

    expected_keys = {
        (condition, model) for condition in CONDITIONS for model in ("mt-s13", "mt-s14")
    }
    records = [
        item
        for item in evaluation.get("reports", [])
        if (item.get("condition"), item.get("model")) in expected_keys
    ]
    record_keys = [(item.get("condition"), item.get("model")) for item in records]
    baseline_reports = source["baseline_reports"]
    if (
        set(baseline_reports)
        != {f"{condition}/{model}" for condition, model in expected_keys}
        or len(record_keys) != len(expected_keys)
        or set(record_keys) != expected_keys
    ):
        raise ValueError(
            "R012 frozen R010 evaluation has incomplete or duplicate baselines"
        )
    records_by_key = {(item["condition"], item["model"]): item for item in records}

    outputs = {}
    for condition, model_name in sorted(expected_keys):
        key = (condition, model_name)
        record = records_by_key[key]
        locked = baseline_reports[f"{condition}/{model_name}"]
        if any(record.get(field) != value for field, value in locked.items()):
            raise ValueError(f"R012 frozen R010 baseline pointer differs: {key}")
        path = ROOT / locked["report"]
        if path.is_symlink() or sha256(path) != locked["report_sha256"]:
            raise ValueError(f"R012 frozen R010 baseline report changed: {path}")
        report = read_json(path)
        model = r010["models"][model_name]
        output = preparation["outputs"][condition]
        expected_config = {
            "study": r010["study"],
            "condition": condition,
            "manifest_sha256": r010_hash,
            "source_test_file": source["test_file"],
            "source_test_sha256": source["test_sha256"],
            "dataset": output["file"],
            "dataset_sha256": output["sha256"],
            "preparation_sha256": source["r010_preparation_sha256"],
            "preparer_sha256": r010["implementation_sha256"]["scripts/prepare_r010.py"],
            "model": model_name,
            "checkpoint": model["checkpoint"],
            "weights_sha256": model["inference_files_sha256"]["model.safetensors"],
            "inference_files_sha256": model["inference_files_sha256"],
            "max_len": model["max_len"],
            "head_max_len": model["head_max_len"],
            "device": manifest["recipe"]["inference"]["device"],
            "autocast_dtype": manifest["recipe"]["inference"]["autocast_dtype"],
            "laya_cuda_amp": manifest["recipe"]["inference"]["laya_cuda_amp"],
            "gpu": manifest["environment"]["gpu"],
            "torch_version": manifest["environment"]["torch_version"],
            "cuda_version": manifest["environment"]["torch_cuda"],
        }
        config = report.get("config", {})
        if any(config.get(field) != value for field, value in expected_config.items()):
            raise ValueError(f"R012 R010 baseline report identity mismatch: {key}")
        cases = report.get("cases", [])
        rows = read_jsonl(ROOT / output["file"])
        validate_report_cases(cases, rows, model_name)
        accuracy = sum(case["correct"] for case in cases) / len(cases)
        if (
            len(cases) != source["test_rows"]
            or record.get("rows") != len(cases)
            or record.get("choice_accuracy") != report["overall"]["choice_accuracy"]
            or abs(accuracy - report["overall"]["choice_accuracy"]) > 1e-12
        ):
            raise ValueError(f"R012 R010 baseline metrics do not recompute: {key}")
        outputs[f"{condition}/{model_name}"] = report
    return outputs
