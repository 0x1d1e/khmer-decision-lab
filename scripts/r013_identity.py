"""Locked identities and readiness checks for R013 unseen-intent transfer."""

import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Any

from r008_identity import checkpoint_inference_inventory

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r013-unseen-intent-transfer.json"
PROTOCOL = ROOT / "research/R013-unseen-intent-transfer/README.md"
ARTIFACT_DIR = ROOT / "artifacts/runs/R013"
TOKENIZER_RECOVERY_FILE = "artifacts/runs/R013/tokenizer-config-recovery.json"
SOURCE_FILE = "data/processed/massive-km-train-full.jsonl"
PREPARATION_FILE = "artifacts/runs/R013/preparation.json"
SEEDS = (13, 14)
ARMS = ("seen-only", "all-label")
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
BASE_REPO = "convaiinnovations/laya"
BASE_SUBDIR = "multilingual"
PACKAGE_VERSIONS = (
    "laya",
    "torch",
    "bitsandbytes",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "numpy",
    "safetensors",
    "datasets",
)
HELDOUT_INTENTS = (
    "datetime_convert",
    "email_addcontact",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lightup",
    "news_query",
    "play_game",
    "play_podcasts",
    "qa_definition",
    "qa_stock",
    "recommendation_movies",
    "takeaway_order",
)
TRAIN_FILES = {
    "seen-only": "data/processed/r013-unseen-intent-seen-only-train.jsonl",
    "all-label": "data/processed/r013-unseen-intent-all-label-train.jsonl",
}
SPLIT_FILES = {
    "validation-unseen": "data/processed/r013-unseen-intent-validation-unseen.jsonl",
    "validation-seen": "data/processed/r013-unseen-intent-validation-seen.jsonl",
    "test-unseen": "data/processed/r013-unseen-intent-test-unseen.jsonl",
    "test-seen": "data/processed/r013-unseen-intent-test-seen.jsonl",
}
OUTPUTS = {
    (seed, arm): f"artifacts/checkpoints/R013-{arm}-s{seed}"
    for seed in SEEDS
    for arm in ARMS
}
IMPLEMENTATION_FILES = (
    "scripts/r008_identity.py",
    "scripts/laya_amp.py",
    "scripts/r013_identity.py",
    "scripts/prepare_r013.py",
    "scripts/train_r013.py",
    "scripts/evaluate_r013.py",
    "scripts/r013_inference.py",
    "scripts/summarize_r013.py",
    "tests/test_r013_unseen_intent.py",
)
RECIPE: dict[str, Any] = {
    "base_revision": BASE_REVISION,
    "seeds": list(SEEDS),
    "arms": list(ARMS),
    "epochs": 2,
    "micro_batch": 1,
    "grad_accum": 32,
    "max_len": 512,
    "head_max_len": 256,
    "encoder_lr": 0.000025,
    "head_lr": 0.0001,
    "min_lr": 0.000001,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "loss": "soft-ce",
    "freeze_encoder": False,
    "shuffle_options": ["choice"],
    "amp": True,
    "gradient_checkpointing": True,
    "calib_max": 0,
    "calib_frac": 0.0,
    "eval_data": None,
    "optimizer": "bitsandbytes.optim.adamw.PagedAdamW8bit",
    "training_rows_per_arm": 6012,
    "expected_optimizer_attempts": 376,
    "minimum_successful_updates": 358,
    "maximum_paired_update_difference": 5,
    "inference": {
        "device": "cuda",
        "autocast_dtype": "bfloat16",
        "laya_cuda_amp": "bf16",
        "batch_size": 8,
    },
}
EXPECTED_SPLIT = {
    "source_rows": 11514,
    "distinct_normalized_groups": 10320,
    "same_label_duplicate_groups": 330,
    "conflicting_label_groups": 127,
    "conflicting_label_rows": 789,
    "eligible_groups": 10193,
    "eligible_labels": 53,
    "train_groups": 7139,
    "validation_groups": 1014,
    "test_groups": 2040,
    "heldout_test_groups": 320,
    "known_test_groups": 1720,
    "seen_only_train_groups": 6012,
    "validation_unseen_groups": 158,
    "validation_seen_groups": 856,
}
EVALUATION = {
    "primary_metric": "top-1 choice accuracy",
    "primary_comparison": "seen-only minus frozen multilingual base on test-unseen",
    "support_rule": "both seed-specific paired group-bootstrap 95% lower bounds > 0",
    "bootstrap_draws": 10000,
    "bootstrap_seed": 13,
    "bootstrap_interval": "percentile; numpy.quantile(method='linear') at 2.5 and 97.5",
    "bootstrap_weighting": "resample normalized utterance groups; retain all rows; row-weight difference",
    "generalization_scope": "the fixed selected 12-intent set, not a population of unseen intents",
    "reported_metrics": [
        "choice_accuracy",
        "macro_f1",
        "per_intent_recall",
        "macro_ovr_auc",
        "mean_multiclass_nll",
        "categorical_brier",
        "top_answer_brier",
        "ece_15_equal_width_bins",
        "mean_answer_confidence",
        "aurc",
        "selective_accuracy@50",
        "selective_accuracy@80",
    ],
    "known_test_auroc": "per-intent OvR AUROC uses only rows whose 20-choice candidate set contains that label",
}
INFERENCE_LIMITS = {"max_len": 512, "head_max_len": 256}
INFERENCE_BATCH_SIZE = 1
RUNTIME_TOKENIZER_CONFIG_SHA256 = (
    "f2ff584a8f78ac9f3b6fd0afcf9ab310e41c2d445ef628f9ab4b0d594e992b7a"
)
TOKENIZER_CONFIG = "tokenizer/tokenizer_config.json"
AMENDMENT: dict[str, Any] = {
    "reason": "After the locked scoring run completed, report validation was amended to accept artifacts bound to the immediate predecessor lock and validate the recorded singleton inference call shape. This corrects post-score lineage and batch-metadata checks only. No model inference was rerun; reports, logits, test rows, metrics, bootstrap method, estimand, and support thresholds are unchanged.",
    "predecessor_manifest_file": "datasets/manifests/r013-unseen-intent-transfer-pre-reporting-fix.json",
    "predecessor_manifest_sha256": "93725d7d914b0c10f678cc649b3a57c348ae3fffeb7b94efdaf1e2ad97ee5d41",
    "predecessor_protocol_sha256": "04044d5ae4e2cb34f39c1d84b448e7e6d05da205a21b41117a45db84892599f3",
    "predecessor_implementation_sha256": {
        "scripts/evaluate_r013.py": "4c02d4c60ed29d398d67cd3b438e0f2a47a3ad61370666c7e1470446e6762065",
        "scripts/laya_amp.py": "9c1eb076971201a502f8c6327bce09cfda92f4733b4383b0c5caa888f34972f6",
        "scripts/prepare_r013.py": "d5d388c6db4192a377b5301758ae9a279fe8c89a97e03b72e18cf9f0d99a0a4a",
        "scripts/r008_identity.py": "61f23f4eff8f7cef7b8218abf3e997d4219b78a773c6c078fbf3515f16f8ad11",
        "scripts/r013_identity.py": "9ffa057c60cbd6c97a5a63707157f124ae909fddeaa86d8cbe7777445f5208fb",
        "scripts/r013_inference.py": "1a6d123db9e95af761ae1b07363fb275a39dd9b143d3cd7fbe047719a5348720",
        "scripts/summarize_r013.py": "193732bdf9e7a03b7533956cc1416a5ebe3076a9e99526621f207f5bfc8915ed",
        "scripts/train_r013.py": "9af1db8cfa971b18c325c68a5c3974cab7a1202d6e4d2823df8fd6004ef05987",
        "tests/test_r013_unseen_intent.py": "4cafebdcb369cd533e2aedb062cc20d34c0192c2bc8a217dff08db67d2415636",
    },
    "effective_inference_batch_size": 1,
    "runtime_tokenizer_config_sha256": RUNTIME_TOKENIZER_CONFIG_SHA256,
}
TRAINING_CRITICAL_FILES = (
    "scripts/r008_identity.py",
    "scripts/laya_amp.py",
    "scripts/prepare_r013.py",
    "scripts/train_r013.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_input(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _tag(row, prefix):
    matches = [
        tag[len(prefix) :] for tag in row.get("tags", []) if tag.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"R013 row must have one {prefix} tag")
    return matches[0]


def _base_checkpoint_files(checkpoint):
    checkpoint = Path(checkpoint)
    repo_root = checkpoint.parents[2]
    if (
        checkpoint.is_symlink()
        or not repo_root.name.startswith("models--")
        or repo_root.parent.name != "hub"
        or not checkpoint.resolve().is_relative_to(repo_root.resolve())
    ):
        raise ValueError(
            "R013 base checkpoint is not inside a Hugging Face cache snapshot"
        )
    cache_root = repo_root.parent.resolve()
    allowed_roots = (repo_root.resolve(), (cache_root / "blobs").resolve())

    def is_allowed_cache_path(path):
        resolved = path.resolve()
        return any(
            resolved == root or resolved.is_relative_to(root) for root in allowed_roots
        )

    files = {}
    for relative in ("model.safetensors", "rl_agent_config.json"):
        path = checkpoint / relative
        if not path.is_file():
            raise ValueError(f"R013 base inference file is missing: {relative}")
        if path.is_symlink() and not is_allowed_cache_path(path):
            raise ValueError(
                f"R013 base symlink escapes its Hugging Face cache: {relative}"
            )
        files[relative] = path

    questions = checkpoint / "questions.json"
    if questions.is_symlink() and not is_allowed_cache_path(questions):
        raise ValueError(
            "R013 base symlink escapes its Hugging Face cache: questions.json"
        )
    if questions.exists() or questions.is_symlink():
        if not questions.is_file():
            raise ValueError("R013 base questions.json is not a regular file")
        files["questions.json"] = questions
    for relative in ("encoder", "tokenizer"):
        directory = checkpoint / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                f"R013 base inference directory is missing or unsafe: {relative}"
            )
        found = False
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() and not is_allowed_cache_path(path):
                raise ValueError(
                    f"R013 base symlink escapes its Hugging Face cache: {path}"
                )
            if path.is_file():
                found = True
                files[path.relative_to(checkpoint).as_posix()] = path
        if not found:
            raise ValueError(f"R013 base inference directory is empty: {relative}")
    return files


def validate_base_checkpoint_cache_paths(checkpoint):
    """Reject snapshot paths that escape the Hugging Face cache before normalization."""
    _base_checkpoint_files(checkpoint)


def base_checkpoint_inventory(checkpoint):
    """Hash a pinned Hugging Face snapshot and its cache-local blob links."""
    files = _base_checkpoint_files(checkpoint)
    return {relative: sha256(path) for relative, path in sorted(files.items())}


def normalized_base_checkpoint_inventory(checkpoint):
    """Validate snapshot links, apply Laya's tokenizer fix, then hash inference files."""
    validate_base_checkpoint_cache_paths(checkpoint)
    from laya.agent import _fix_tokenizer_config

    _fix_tokenizer_config(str(checkpoint))
    return base_checkpoint_inventory(checkpoint)


def row_identity(source_sha256, index, row):
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value = f"{source_sha256}\0{index}\0{payload}".encode()
    return hashlib.sha256(value).hexdigest()


def group_identity(normalized_text):
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _manifest_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _verified_manifest_lineage(manifest, manifest_sha256):
    lineage = {}
    current = manifest
    current_sha256 = manifest_sha256
    while True:
        if current_sha256 in lineage:
            raise ValueError("R013 manifest amendment lineage contains a cycle")
        implementation = current.get("implementation_sha256")
        if not isinstance(implementation, dict) or set(implementation) != set(
            IMPLEMENTATION_FILES
        ):
            raise ValueError("R013 manifest implementation inventory is invalid")
        lineage[current_sha256] = {
            "protocol_sha256": current.get("protocol_sha256"),
            "implementation_sha256": implementation,
        }
        amendment = current.get("amendment")
        if amendment is None:
            return lineage
        predecessor_path = _manifest_path(
            amendment.get("predecessor_manifest_file", "")
        )
        if (
            not predecessor_path.is_relative_to(ROOT)
            or predecessor_path.is_symlink()
            or not predecessor_path.is_file()
        ):
            raise ValueError("R013 predecessor manifest is missing or unsafe")
        predecessor = read_json(predecessor_path)
        predecessor_sha256 = sha256(predecessor_path)
        predecessor_implementation = amendment.get("predecessor_implementation_sha256")
        if (
            predecessor_sha256 != amendment.get("predecessor_manifest_sha256")
            or predecessor.get("study") != current.get("study")
            or predecessor.get("status") != "locked_exploratory"
            or predecessor.get("protocol_sha256")
            != amendment.get("predecessor_protocol_sha256")
            or predecessor.get("implementation_sha256") != predecessor_implementation
            or predecessor.get("recipe") != current.get("recipe")
            or predecessor.get("evaluation") != current.get("evaluation")
            or predecessor.get("base") != current.get("base")
            or predecessor.get("environment") != current.get("environment")
            or predecessor.get("source") != current.get("source")
            or predecessor.get("split") != current.get("split")
            or not isinstance(predecessor_implementation, dict)
            or set(predecessor_implementation) != set(IMPLEMENTATION_FILES)
            or any(
                predecessor_implementation[relative] != implementation[relative]
                for relative in TRAINING_CRITICAL_FILES
            )
        ):
            raise ValueError("R013 amendment changed training-critical identity")
        current = predecessor
        current_sha256 = predecessor_sha256


def verify_locked_manifest(path=MANIFEST):
    manifest_path = _manifest_path(path)
    manifest = read_json(manifest_path)
    if (
        manifest.get("study") != "R013-unseen-intent-transfer"
        or manifest.get("status") != "locked_exploratory"
        or manifest.get("protocol_sha256") != sha256(PROTOCOL)
        or manifest.get("evaluation") != EVALUATION
    ):
        raise ValueError("R013 protocol lock identity mismatch")
    implementation = manifest.get("implementation_sha256")
    if implementation != {
        relative: sha256(ROOT / relative) for relative in IMPLEMENTATION_FILES
    }:
        raise ValueError("R013 implementation fingerprint mismatch")
    if manifest.get("recipe") != RECIPE:
        raise ValueError("R013 training recipe differs from the lock")
    if (
        manifest.get("inference_limits") != INFERENCE_LIMITS
        or manifest.get("inference_batch_size") != INFERENCE_BATCH_SIZE
        or manifest.get("runtime_tokenizer_config_sha256")
        != RUNTIME_TOKENIZER_CONFIG_SHA256
        or manifest.get("amendment") != AMENDMENT
    ):
        raise ValueError("R013 inference amendment identity mismatch")
    _verified_manifest_lineage(manifest, sha256(manifest_path))
    split = manifest.get("split")
    if (
        not isinstance(split, dict)
        or split.get("heldout_intents") != list(HELDOUT_INTENTS)
        or split.get("expected_counts") != EXPECTED_SPLIT
    ):
        raise ValueError("R013 split identity differs from the lock")

    source = manifest.get("source", {})
    source_path = ROOT / SOURCE_FILE
    if (
        source.get("file") != SOURCE_FILE
        or source.get("sha256") != sha256(source_path)
        or source.get("rows") != EXPECTED_SPLIT["source_rows"]
        or source.get("massive_manifest_sha256")
        != sha256(ROOT / "datasets/manifests/massive-km.json")
    ):
        raise ValueError("R013 source data identity mismatch")
    massive_manifest = read_json(ROOT / "datasets/manifests/massive-km.json")
    if (
        massive_manifest.get("dataset") != "mteb/amazon_massive_intent"
        or massive_manifest.get("config") != "km"
        or massive_manifest.get("revision") != source.get("revision")
    ):
        raise ValueError("R013 MASSIVE source revision mismatch")

    base = manifest.get("base", {})
    if (
        base.get("repo") != BASE_REPO
        or base.get("revision") != BASE_REVISION
        or base.get("subdir") != BASE_SUBDIR
    ):
        raise ValueError("R013 multilingual base revision mismatch")
    base_inventory = base.get("inference_files_sha256")
    if not isinstance(base_inventory, dict) or not base_inventory:
        raise ValueError("R013 multilingual base inventory is missing")

    environment = manifest.get("environment", {})
    if (
        environment.get("python") != platform.python_version()
        or environment.get("uv_lock_sha256") != sha256(ROOT / "uv.lock")
        or environment.get("packages")
        != {package: version(package) for package in PACKAGE_VERSIONS}
    ):
        raise ValueError("R013 locked software environment differs from this runtime")
    return manifest, sha256(manifest_path)


def verify_cuda_environment(manifest):
    import torch

    environment = manifest["environment"]
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_name(0) != environment["gpu"]
        or list(torch.cuda.get_device_capability(0)) != environment["gpu_capability"]
        or torch.cuda.get_device_properties(0).total_memory // 2**20
        != environment["gpu_memory_mib"]
        or torch.__version__ != environment["torch_build"]
        or torch.version.cuda != environment["torch_cuda"]
    ):
        raise RuntimeError(
            "R013 CUDA, BF16, GPU, or PyTorch runtime differs from the lock"
        )


def _artifact_lock_identity(
    manifest, artifact_manifest_sha256, current_manifest_sha256
):
    lineage = _verified_manifest_lineage(manifest, current_manifest_sha256)
    try:
        return lineage[artifact_manifest_sha256]
    except KeyError as exc:
        raise ValueError("R013 artifact is not bound to this lock lineage") from exc


def verify_preparation(manifest, manifest_sha256, *, test_access=False):
    preparation_path = ROOT / PREPARATION_FILE
    if preparation_path.is_symlink() or not preparation_path.is_file():
        raise ValueError("R013 preparation artifact is missing or unsafe")
    preparation = read_json(preparation_path)
    preparation_lock = _artifact_lock_identity(
        manifest, preparation.get("manifest_sha256"), manifest_sha256
    )
    if (
        preparation.get("study") != manifest["study"]
        or preparation.get("preparer_sha256")
        != preparation_lock["implementation_sha256"]["scripts/prepare_r013.py"]
        or preparation.get("source_sha256") != manifest["source"]["sha256"]
        or preparation.get("heldout_intents") != list(HELDOUT_INTENTS)
        or preparation.get("split_counts") != EXPECTED_SPLIT
    ):
        raise ValueError("R013 preparation provenance mismatch")
    outputs = preparation.get("outputs_sha256")
    expected_files = {*TRAIN_FILES.values(), *SPLIT_FILES.values()}
    if not isinstance(outputs, dict) or set(outputs) != expected_files:
        raise TypeError("R013 preparation output inventory is missing or unexpected")
    required = dict(TRAIN_FILES)
    if test_access:
        required.update(SPLIT_FILES)
    for relative in required.values():
        path = ROOT / relative
        expected_hash = outputs.get(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(expected_hash, str)
            or sha256(path) != expected_hash
        ):
            raise ValueError(f"R013 prepared file fingerprint mismatch: {relative}")
    training_counts = preparation.get("training_rows")
    if training_counts != {arm: RECIPE["training_rows_per_arm"] for arm in ARMS}:
        raise ValueError("R013 training arm row counts do not match the locked budget")
    source_labels = {
        row.get("expected", {}).get("intent") for row in read_jsonl(ROOT / SOURCE_FILE)
    }
    seen_labels = preparation.get("seen_intents")
    distractors = preparation.get("heldout_test_distractors")
    if (
        len(source_labels) != 60
        or any(not isinstance(label, str) for label in source_labels)
        or not isinstance(seen_labels, list)
        or len(seen_labels) != 48
        or any(not isinstance(label, str) for label in seen_labels)
        or len(set(seen_labels)) != 48
        or set(seen_labels) != source_labels - set(HELDOUT_INTENTS)
        or not isinstance(distractors, list)
        or len(distractors) != 8
        or any(not isinstance(label, str) for label in distractors)
        or len(set(distractors)) != 8
        or not set(distractors) <= set(seen_labels)
    ):
        raise ValueError(
            "R013 prepared label pools differ from the locked source split"
        )
    for arm, relative in TRAIN_FILES.items():
        rows = read_jsonl(ROOT / relative)
        expected_labels = set(seen_labels) if arm == "seen-only" else source_labels
        group_ids = set()
        source_ids = set()
        for index, row in enumerate(rows):
            question = row.get("questions", {}).get("intent", {})
            criteria = question.get("criteria")
            gold = row.get("expected", {}).get("intent")
            if (
                not isinstance(criteria, dict)
                or len(criteria) != 20
                or not set(criteria) <= expected_labels
                or gold not in expected_labels
                or gold not in criteria
                or question.get("type") != "choice"
                or question.get("instructions")
                != "What is the user's intent in this utterance?"
                or any(criteria[label] != label.replace("_", " ") for label in criteria)
            ):
                raise ValueError(
                    f"R013 training row violates {arm} option semantics: {index}"
                )
            group = _tag(row, "r013-group:")
            source_id = _tag(row, "r013-row:")
            if group in group_ids or source_id in source_ids:
                raise ValueError(f"R013 {arm} training rows repeat an input group")
            group_ids.add(group)
            source_ids.add(source_id)
        if len(rows) != RECIPE["training_rows_per_arm"]:
            raise ValueError(f"R013 {arm} training row count differs from the lock")
        if arm == "seen-only" and any(
            row["expected"]["intent"] in HELDOUT_INTENTS
            or set(row["questions"]["intent"]["criteria"]) & set(HELDOUT_INTENTS)
            for row in rows
        ):
            raise ValueError("R013 held-out label leaked into seen-only training")
        if (
            arm == "all-label"
            and {row["expected"]["intent"] for row in rows} != source_labels
        ):
            raise ValueError("R013 all-label training arm omits a source intent")
    expected_validation = {
        "unseen": EXPECTED_SPLIT["validation_unseen_groups"],
        "seen": EXPECTED_SPLIT["validation_seen_groups"],
    }
    expected_test = {
        "unseen": EXPECTED_SPLIT["heldout_test_groups"],
        "seen": EXPECTED_SPLIT["known_test_groups"],
    }
    if (
        preparation.get("validation_rows") != expected_validation
        or preparation.get("test_rows") != expected_test
    ):
        raise ValueError(
            "R013 prepared validation or test row counts differ from the lock"
        )
    if test_access:
        expected_rows = {
            "validation-unseen": expected_validation["unseen"],
            "validation-seen": expected_validation["seen"],
            "test-unseen": expected_test["unseen"],
            "test-seen": expected_test["seen"],
        }
        for name, relative in SPLIT_FILES.items():
            rows = read_jsonl(ROOT / relative)
            if len(rows) != expected_rows[name]:
                raise ValueError(f"R013 prepared split row count mismatch: {name}")
    return preparation


def output_directory(seed, arm):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("R013 seed or training arm is not preregistered")
    return ROOT / OUTPUTS[(seed, arm)]


def _identity_temperature(value):
    values = value if isinstance(value, list) else [value]
    return bool(values) and all(
        not isinstance(item, bool)
        and isinstance(item, (int, float))
        and math.isclose(float(item), 1.0, rel_tol=0.0, abs_tol=1e-8)
        for item in values
    )


def verify_training_run(manifest, manifest_sha256, preparation, seed, arm):
    output = output_directory(seed, arm)
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"R013 output directory is missing or unsafe: {arm}/s{seed}")
    provenance = read_json(output / "training_provenance.json")
    telemetry = read_json(output / "training_telemetry.json")
    report = read_json(output / "train_report.json")
    artifact_lock = _artifact_lock_identity(
        manifest, provenance.get("manifest_sha256"), manifest_sha256
    )
    expected_train_file = TRAIN_FILES[arm]
    expected_inventory = checkpoint_inference_inventory(output)
    expected_recipe = {**RECIPE, "seed": seed, "arm": arm}
    if (
        provenance.get("study") != manifest["study"]
        or provenance.get("preparation_sha256") != sha256(ROOT / PREPARATION_FILE)
        or provenance.get("protocol_sha256") != artifact_lock["protocol_sha256"]
        or provenance.get("implementation_sha256")
        != artifact_lock["implementation_sha256"]
        or provenance.get("seed") != seed
        or provenance.get("arm") != arm
        or provenance.get("recipe") != expected_recipe
        or provenance.get("training_file") != expected_train_file
        or provenance.get("training_file_sha256")
        != preparation["outputs_sha256"][expected_train_file]
        or provenance.get("training_rows") != RECIPE["training_rows_per_arm"]
        or provenance.get("output_inventory") != expected_inventory
        or provenance.get("training_telemetry_sha256")
        != sha256(output / "training_telemetry.json")
        or provenance.get("train_report_sha256") != sha256(output / "train_report.json")
    ):
        raise ValueError(f"R013 training provenance mismatch: {arm}/s{seed}")

    attempts = RECIPE["expected_optimizer_attempts"]
    minimum = RECIPE["minimum_successful_updates"]
    if (
        telemetry.get("optimizer") != RECIPE["optimizer"]
        or telemetry.get("optimizer_attempts") != attempts
        or not isinstance(telemetry.get("successful_updates"), int)
        or telemetry["successful_updates"] < minimum
        or telemetry.get("skipped_updates")
        != attempts - telemetry["successful_updates"]
        or telemetry["successful_updates"] > attempts
        or telemetry.get("scheduler_steps") != telemetry["successful_updates"]
        or telemetry.get("failure") is not None
        or telemetry.get("telemetry_cleanup_failure") is not None
        or telemetry.get("device") != "cuda"
        or telemetry.get("gpu") != manifest["environment"].get("gpu")
        or telemetry.get("torch_version") != manifest["environment"].get("torch_build")
        or telemetry.get("cuda_version") != manifest["environment"].get("torch_cuda")
        or not isinstance(telemetry.get("peak_allocated_bytes"), int)
        or isinstance(telemetry.get("peak_allocated_bytes"), bool)
        or telemetry["peak_allocated_bytes"] <= 0
    ):
        raise ValueError(f"R013 training telemetry failed readiness: {arm}/s{seed}")

    report_training = report.get("training")
    expected_training = {
        "epochs": RECIPE["epochs"],
        "micro_batch": RECIPE["micro_batch"],
        "grad_accum": RECIPE["grad_accum"],
        "encoder_lr": RECIPE["encoder_lr"],
        "head_lr": RECIPE["head_lr"],
        "min_lr": RECIPE["min_lr"],
        "weight_decay": RECIPE["weight_decay"],
        "grad_clip": RECIPE["grad_clip"],
        "loss": RECIPE["loss"],
        "freeze_encoder": RECIPE["freeze_encoder"],
        "shuffle_options": RECIPE["shuffle_options"],
        "seed": seed,
        "max_len": RECIPE["max_len"],
        "head_max_len": RECIPE["head_max_len"],
        "amp": RECIPE["amp"],
        "gradient_checkpointing": RECIPE["gradient_checkpointing"],
        "calib_max": 0,
        "calib_frac": 0.0,
        "eval_data": None,
    }
    calibration = report.get("calibration")
    expected_calibration_types = {"choice", "score", "noul"}
    no_calibration = (
        isinstance(calibration, dict)
        and set(calibration) == expected_calibration_types
        and all(
            isinstance(calibration[qtype], dict)
            and calibration[qtype].get("items") == 0
            and calibration[qtype].get("temperature") == 1.0
            and not calibration[qtype].get("issues")
            for qtype in expected_calibration_types
        )
    )
    if (
        report.get("eval_source") is not None
        or report.get("eval_items") != 0
        or report.get("eval_mode") is not None
        or not no_calibration
        or not isinstance(report_training, dict)
        or any(
            report_training.get(key) != value
            for key, value in expected_training.items()
        )
    ):
        raise ValueError(
            f"R013 training report differs from the locked recipe: {arm}/s{seed}"
        )

    config = read_json(output / "rl_agent_config.json")
    latest_config = read_json(output / "checkpoint_latest/rl_agent_config.json")
    for checkpoint_config in (config, latest_config):
        temperature_by_options = checkpoint_config.get("temperature_by_options")
        if not _identity_temperature(
            checkpoint_config.get("temperature")
        ) or temperature_by_options not in (None, {}, []):
            raise ValueError(f"R013 output calibration is not identity: {arm}/s{seed}")
    base_inventory = manifest["base"]["inference_files_sha256"]
    expected_base = {
        "repo": BASE_REPO,
        "revision": BASE_REVISION,
        "subdir": BASE_SUBDIR,
        "inference_files_sha256": base_inventory,
    }
    if provenance.get("base") != expected_base:
        raise ValueError(f"R013 source checkpoint provenance mismatch: {arm}/s{seed}")
    base_weight = base_inventory.get("model.safetensors")
    output_weight = expected_inventory.get("model.safetensors")
    if not base_weight or output_weight == base_weight:
        raise ValueError(f"R013 weights did not change: {arm}/s{seed}")

    latest = output / "checkpoint_latest"
    if latest.is_symlink() or not latest.is_dir():
        raise ValueError(f"R013 latest checkpoint is missing or unsafe: {arm}/s{seed}")
    latest_inventory = checkpoint_inference_inventory(latest)
    if latest_inventory != expected_inventory:
        raise ValueError(
            f"R013 latest checkpoint differs from the final output: {arm}/s{seed}"
        )
    return {"telemetry": telemetry, "inventory": expected_inventory}


def verify_all_training_runs(manifest, manifest_sha256, preparation):
    runs = {}
    for seed in SEEDS:
        for arm in ARMS:
            runs[(seed, arm)] = verify_training_run(
                manifest, manifest_sha256, preparation, seed, arm
            )
        difference = abs(
            runs[(seed, "seen-only")]["telemetry"]["successful_updates"]
            - runs[(seed, "all-label")]["telemetry"]["successful_updates"]
        )
        if difference > RECIPE["maximum_paired_update_difference"]:
            raise ValueError(f"R013 paired update counts differ for seed {seed}")
    return runs


def restore_known_laya_tokenizer_rewrite(checkpoint, expected_inventory):
    """Restore only the pinned Laya rewrite from matching checkpoint_latest metadata."""
    checkpoint = Path(checkpoint)
    latest = checkpoint / "checkpoint_latest"
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or latest.is_symlink()
        or not latest.is_dir()
    ):
        raise ValueError("R013 tokenizer metadata recovery path is missing or unsafe")
    if (
        not isinstance(expected_inventory, dict)
        or TOKENIZER_CONFIG not in expected_inventory
    ):
        raise ValueError("R013 tokenizer metadata recovery inventory is invalid")

    expected_hash = expected_inventory[TOKENIZER_CONFIG]
    latest_inventory = checkpoint_inference_inventory(latest)
    if latest_inventory != expected_inventory:
        raise ValueError("R013 latest checkpoint differs from authenticated inventory")
    root_inventory = checkpoint_inference_inventory(checkpoint)
    if set(root_inventory) != set(expected_inventory) or any(
        root_inventory[relative] != expected_inventory[relative]
        for relative in expected_inventory
        if relative != TOKENIZER_CONFIG
    ):
        raise ValueError("R013 output differs beyond the tokenizer metadata rewrite")
    before_hash = root_inventory[TOKENIZER_CONFIG]
    if before_hash not in (expected_hash, RUNTIME_TOKENIZER_CONFIG_SHA256):
        raise ValueError("R013 tokenizer metadata differs from known locked states")

    restored = before_hash != expected_hash
    if restored:
        source = latest / TOKENIZER_CONFIG
        target = checkpoint / TOKENIZER_CONFIG
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.r013-", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    after_inventory = checkpoint_inference_inventory(checkpoint)
    if after_inventory != expected_inventory:
        raise ValueError("R013 tokenizer metadata restoration failed inventory check")
    return {
        "checkpoint": str(checkpoint),
        "source": "checkpoint_latest/tokenizer/tokenizer_config.json",
        "before_sha256": before_hash,
        "source_sha256": latest_inventory[TOKENIZER_CONFIG],
        "after_sha256": after_inventory[TOKENIZER_CONFIG],
        "restored": restored,
    }


def restore_training_tokenizer_metadata(manifest, manifest_sha256):
    """Repair known inference-time metadata rewrites without opening prepared splits."""
    events = []
    for seed in SEEDS:
        for arm in ARMS:
            output = output_directory(seed, arm)
            provenance_path = output / "training_provenance.json"
            if provenance_path.is_symlink() or not provenance_path.is_file():
                raise ValueError(
                    f"R013 training provenance is missing or unsafe: {arm}/s{seed}"
                )
            provenance = read_json(provenance_path)
            artifact_lock = _artifact_lock_identity(
                manifest, provenance.get("manifest_sha256"), manifest_sha256
            )
            if (
                provenance.get("study") != manifest["study"]
                or provenance.get("seed") != seed
                or provenance.get("arm") != arm
                or provenance.get("protocol_sha256") != artifact_lock["protocol_sha256"]
                or provenance.get("implementation_sha256")
                != artifact_lock["implementation_sha256"]
            ):
                raise ValueError(f"R013 recovery provenance mismatch: {arm}/s{seed}")
            event = restore_known_laya_tokenizer_rewrite(
                output, provenance.get("output_inventory")
            )
            event.update({"seed": seed, "arm": arm, "phase": "pre-readiness"})
            events.append(event)
    return events


def load_test_rows_after_readiness(manifest, manifest_sha256):
    preparation = verify_preparation(manifest, manifest_sha256, test_access=False)
    runs = verify_all_training_runs(manifest, manifest_sha256, preparation)
    # Test file hashes and rows are accessed only after all four training outputs pass.
    verify_preparation(manifest, manifest_sha256, test_access=True)
    return (
        preparation,
        runs,
        {name: read_jsonl(ROOT / relative) for name, relative in SPLIT_FILES.items()},
    )
