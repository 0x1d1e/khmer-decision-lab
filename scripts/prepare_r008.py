"""Prepare matched R008 training arms and task-specific validation files.

The training rows reproduce Laya 0.4.1's historical 400-row R006 holdout.
No model is loaded and no predictions are made.
"""

import importlib.metadata
import json
import math
import random
from collections import Counter
from pathlib import Path

from r008_identity import IMPLEMENTATION_FILES, normalized_state_key, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
MANIFESTS = ROOT / "datasets/manifests"
OUTPUT_MANIFEST = MANIFESTS / "r008-specialist-controls.json"
PROTOCOL = ROOT / "research/R008-specialist-controls/README.md"
SOURCE_TRAIN = DATA / "r006-multitask-train.jsonl"
SOURCE_MANIFEST = MANIFESTS / "r006-multitask.json"
CALIBRATION_SEED = 20260922
CALIBRATION_MAX = 400
CALIBRATION_FRACTION = 0.1
EXPECTED_SOURCE_SHA256 = (
    "509efb156bf676ea11882d6af052a4b6447984165929ef88b798df92e81043b3"
)
EXPECTED_SOURCE_COUNTS = {"massive-km": 11514, "khmer-news": 5136, "sib200-km": 701}
EXPECTED_TRAIN_COUNTS = {"massive-km": 11262, "khmer-news": 5008, "sib200-km": 681}
TASK_FILES = {"khmer-news": "khmer-news", "sib200-km": "sib200-km"}
EXPECTED_SPLITS = {
    "khmer-news": {"validation": 661, "test": 1543},
    "sib200-km": {"validation": 99, "test": 204},
}


def task_for_row(row):
    questions = row.get("questions", {})
    if set(questions) == {"intent"}:
        return "massive-km"
    if set(questions) != {"topic"}:
        raise ValueError("R006 row has an unexpected question schema")
    labels = set(questions["topic"].get("criteria", {}))
    if labels == {
        "economic",
        "entertainment",
        "life",
        "politic",
        "sport",
        "technology",
    }:
        return "khmer-news"
    if labels == {
        "entertainment",
        "geography",
        "health",
        "politics",
        "science/technology",
        "sports",
        "travel",
    }:
        return "sib200-km"
    raise ValueError("R006 topic row has an unrecognized label schema")


def calibration_indices(
    row_count,
    seed=CALIBRATION_SEED,
    calib_max=CALIBRATION_MAX,
    calib_fraction=CALIBRATION_FRACTION,
):
    """Match Laya split_calibration's seeded index selection exactly."""
    count = min(calib_max, int(row_count * calib_fraction))
    order = list(range(row_count))
    random.Random(seed).shuffle(order)
    return set(order[:count])


def split_training_rows(rows):
    if not rows:
        raise ValueError("R006 mixed-training file is empty")
    tasks = [task_for_row(row) for row in rows]
    source_counts = Counter(tasks)
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"R006 task row counts changed: {dict(source_counts)!r}")

    held = calibration_indices(len(rows))
    training = [row for index, row in enumerate(rows) if index not in held]
    held_counts = Counter(tasks[index] for index in held)
    training_counts = Counter(task_for_row(row) for row in training)
    expected_held = {
        task: EXPECTED_SOURCE_COUNTS[task] - EXPECTED_TRAIN_COUNTS[task]
        for task in EXPECTED_SOURCE_COUNTS
    }
    if dict(held_counts) != expected_held:
        raise ValueError(
            f"Laya calibration holdout composition changed: {dict(held_counts)!r}"
        )
    if dict(training_counts) != EXPECTED_TRAIN_COUNTS:
        raise ValueError(
            f"R008 post-calibration training counts changed: {dict(training_counts)!r}"
        )
    return training, held_counts


def validation_calibration_rows(validation_rows, test_rows):
    test_states = {normalized_state_key(row) for row in test_rows}
    calibration_rows = [
        row for row in validation_rows if normalized_state_key(row) not in test_states
    ]
    excluded = len(validation_rows) - len(calibration_rows)
    return calibration_rows, excluded, len(test_states)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip() and not line.lstrip().startswith("#")
        ]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(training, held_counts, arm_paths, validation_details):
    task_counts = {
        name: dict(Counter(task_for_row(row) for row in rows))
        for name, (_, rows) in arm_paths.items()
    }
    return {
        "study": "R008-specialist-controls",
        "status": "prepared_exploratory_pending_protocol_lock",
        "protocol_sha256": sha256(PROTOCOL),
        "preparer_sha256": sha256("scripts/prepare_r008.py"),
        "implementation_sha256": {path: sha256(path) for path in IMPLEMENTATION_FILES},
        "source": {
            "r006_manifest": "datasets/manifests/r006-multitask.json",
            "r006_manifest_sha256": sha256(SOURCE_MANIFEST),
            "mixed_training_file": "data/processed/r006-multitask-train.jsonl",
            "mixed_training_sha256": sha256(SOURCE_TRAIN),
            "expected_mixed_training_sha256": EXPECTED_SOURCE_SHA256,
            "mixed_training_rows": sum(EXPECTED_SOURCE_COUNTS.values()),
            "task_rows": EXPECTED_SOURCE_COUNTS,
        },
        "training_split": {
            "method": "Laya 0.4.1 split_calibration index selection reproduced before training",
            "historical_calibration_seed": CALIBRATION_SEED,
            "historical_calib_max": CALIBRATION_MAX,
            "historical_calib_fraction": CALIBRATION_FRACTION,
            "removed_rows_by_task": dict(held_counts),
            "removed_rows_total": sum(held_counts.values()),
            "laya_training_calibration_disabled": True,
            "arms": {
                name: {
                    "file": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "rows": len(rows),
                    "task_rows": task_counts[name],
                }
                for name, (path, rows) in arm_paths.items()
            },
            "target_task_train_test_exact_state_overlap_rows": {
                task: validation_details[task]["train_test_exact_state_overlap_rows"]
                for task in TASK_FILES
            },
        },
        "validation_calibration": validation_details,
        "recipe": {
            "base_model": "convaiinnovations/laya/multilingual",
            "base_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
            "epochs": 2,
            "micro_batch": 1,
            "grad_accum": 32,
            "encoder_lr": 2.5e-5,
            "head_lr": 1e-4,
            "min_lr": 1e-6,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "loss": "soft-ce",
            "freeze_encoder": False,
            "max_len": 512,
            "head_max_len": 256,
            "shuffle_options": ["choice"],
            "amp": True,
            "gradient_checkpointing": True,
            "optimizer": "bitsandbytes.optim.PagedAdamW8bit",
            "expected_optimizer_updates_per_seed": {
                "multitask": math.ceil(16951 / 32) * 2,
                "khmer-news-specialist": math.ceil(5008 / 32) * 2,
                "sib200-specialist": math.ceil(681 / 32) * 2,
            },
            "seeds": [13, 14],
            "matched_on": "per-task row exposure and epochs, not aggregate optimizer updates",
        },
        "environment": {
            "uv_lock_sha256": sha256("uv.lock"),
            "laya": importlib.metadata.version("laya"),
            "torch": importlib.metadata.version("torch"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        },
        "inference": {
            "device": "cuda",
            "autocast_dtype": "bfloat16",
            "laya_cuda_amp": "bf16",
        },
        "amp_guard_scope": "Controlled standalone serial R008 training only, with the frozen dependency lock; see scripts/laya_amp.py and its safety review.",
        "evaluation": {
            "test_reports": "artifacts/runs/R008/{task}/{arm}-s{seed}.json",
            "test_task_rows": {
                task: EXPECTED_SPLITS[task]["test"] for task in TASK_FILES
            },
            "grouping": "sha256 of NFC-normalized, whitespace-collapsed complete state JSON",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 13,
            "interval_percentiles": [2.5, 97.5],
            "noninferiority_margin_percentage_points": -2.0,
        },
    }


def prepare():
    if OUTPUT_MANIFEST.exists():
        existing = json.loads(OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        if existing.get("status") == "locked_exploratory":
            raise RuntimeError("refusing to overwrite a locked R008 manifest")

    if sha256(SOURCE_TRAIN) != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "R006 mixed-training file hash differs from the audited source"
        )
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        source_manifest["sha256"]["data/processed/r006-multitask-train.jsonl"]
        != EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("R006 manifest does not pin the expected mixed-training file")

    rows = read_jsonl(SOURCE_TRAIN)
    training_rows, held_counts = split_training_rows(rows)
    arms = {
        "multitask": (DATA / "r008-multitask-train.jsonl", training_rows),
        "khmer-news-specialist": (
            DATA / "r008-khmer-news-train.jsonl",
            [row for row in training_rows if task_for_row(row) == "khmer-news"],
        ),
        "sib200-specialist": (
            DATA / "r008-sib200-km-train.jsonl",
            [row for row in training_rows if task_for_row(row) == "sib200-km"],
        ),
    }
    for path, arm_rows in arms.values():
        write_jsonl(path, arm_rows)

    validation_details = {}
    for task, basename in TASK_FILES.items():
        source_manifest_path = MANIFESTS / f"{basename}.json"
        source_task_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        if source_task_manifest.get("splits") != EXPECTED_SPLITS[task]:
            raise ValueError(f"{task} split sizes differ from its dataset manifest")

        validation_path = DATA / f"{task}-validation.jsonl"
        test_path = DATA / f"{task}-test.jsonl"
        validation_rows = read_jsonl(validation_path)
        test_rows = read_jsonl(test_path)
        if (
            len(validation_rows) != EXPECTED_SPLITS[task]["validation"]
            or len(test_rows) != EXPECTED_SPLITS[task]["test"]
        ):
            raise ValueError(f"{task} processed split rows differ from its manifest")

        train_rows = [row for row in training_rows if task_for_row(row) == task]
        test_states = {normalized_state_key(row) for row in test_rows}
        train_overlap = sum(
            normalized_state_key(row) in test_states for row in train_rows
        )
        if train_overlap:
            raise ValueError(
                f"{task} training rows overlap test input states: {train_overlap}"
            )

        calibration_rows, excluded, test_groups = validation_calibration_rows(
            validation_rows, test_rows
        )
        if not calibration_rows:
            raise ValueError(
                f"{task} has no validation rows after test-state exclusions"
            )
        calibration_path = DATA / f"r008-{task}-calibration.jsonl"
        write_jsonl(calibration_path, calibration_rows)
        validation_details[task] = {
            "dataset_manifest": str(source_manifest_path.relative_to(ROOT)),
            "dataset_manifest_sha256": sha256(source_manifest_path),
            "dataset": source_task_manifest["dataset"],
            "revision": source_task_manifest["revision"],
            "source_validation_file": str(validation_path.relative_to(ROOT)),
            "source_validation_sha256": sha256(validation_path),
            "source_validation_rows": len(validation_rows),
            "calibration_file": str(calibration_path.relative_to(ROOT)),
            "calibration_sha256": sha256(calibration_path),
            "calibration_rows": len(calibration_rows),
            "test_file": str(test_path.relative_to(ROOT)),
            "test_sha256": sha256(test_path),
            "test_rows": len(test_rows),
            "test_exact_normalized_state_groups": test_groups,
            "validation_rows_excluded_for_exact_test_state_overlap": excluded,
            "train_test_exact_state_overlap_rows": train_overlap,
        }

    arm_payload = {name: (path, arm_rows) for name, (path, arm_rows) in arms.items()}
    manifest = build_manifest(
        training_rows, held_counts, arm_payload, validation_details
    )
    write_manifest(manifest)
    return manifest


def write_manifest(payload):
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = prepare()
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_MANIFEST.relative_to(ROOT)),
                "status": result["status"],
                "training": result["training_split"]["arms"],
                "validation_calibration": result["validation_calibration"],
            },
            indent=2,
        )
    )
