"""Validate R011 inference artifacts and summarize paired calibration outcomes."""

import json
from collections import defaultdict
from typing import Any, cast

import numpy as np
from evaluate_r011 import CONDITION, _baseline_report, _labels, _tag
from laya import evals
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

DRAWS = 10_000
BOOTSTRAP_SEED = 13
MODELS = ("mt-s13", "mt-s14")


def _required_metric(value, name):
    if value is None:
        raise ValueError(f"R011 metric is unavailable: {name}")
    return float(value)


def _nll(logits, targets, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    maxes = scaled.max(axis=1, keepdims=True)
    log_normalizers = maxes[:, 0] + np.log(np.exp(scaled - maxes).sum(axis=1))
    return log_normalizers - scaled[np.arange(len(scaled)), targets]


def _metrics(rows, temperature):
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    scaled = logits / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(
        [row["baseline_prediction"] == row["expected"] for row in rows],
        dtype=bool,
    )
    if not np.array_equal(predicted, logits.argmax(axis=1)):
        raise ValueError("R011 temperature scaling changed a predicted class")
    if not np.array_equal(predicted == targets, correct):
        raise ValueError("R011 stored correctness differs from raw logits and labels")
    return {
        "rows": len(rows),
        "mean_choice_nll": float(_nll(logits, targets, temperature).mean()),
        "choice_accuracy": float(correct.mean()),
        "top_answer_brier": _required_metric(
            evals.brier(confidence.tolist(), correct.tolist()), "top_answer_brier"
        ),
        "ece_15_equal_width_bins": _required_metric(
            evals.ece(confidence.tolist(), correct.tolist(), bins=15), "ece"
        ),
        "mean_answer_confidence": float(confidence.mean()),
        "aurc": _required_metric(
            evals.aurc(confidence.tolist(), correct.tolist()), "aurc"
        ),
        "selective_accuracy@50": _required_metric(
            evals.selective_accuracy(confidence.tolist(), correct.tolist(), 0.5),
            "selective_accuracy@50",
        ),
        "selective_accuracy@80": _required_metric(
            evals.selective_accuracy(confidence.tolist(), correct.tolist(), 0.8),
            "selective_accuracy@80",
        ),
    }


def _paired_group_bootstrap(deltas, group_ids):
    if len(deltas) != len(group_ids) or not deltas:
        raise ValueError(
            "R011 paired bootstrap arrays have different lengths or are empty"
        )
    by_group = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        by_group[group_id].append(index)
    groups = sorted(by_group)
    sizes = np.asarray([len(by_group[group]) for group in groups], dtype=np.int64)
    sums = np.asarray(
        [sum(deltas[index] for index in by_group[group]) for group in groups],
        dtype=np.float64,
    )
    point = float(sum(deltas) / len(deltas))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for start in range(0, DRAWS, 100):
        batch_size = min(100, DRAWS - start)
        selected = rng.integers(0, len(groups), size=(batch_size, len(groups)))
        counts = np.stack(
            [np.bincount(sample, minlength=len(groups)) for sample in selected]
        )
        samples.extend((counts @ sums) / (counts @ sizes))
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "mean_nll_difference_ocr_fit_minus_existing": point,
        "paired_group_bootstrap_95_percent_interval": [float(lower), float(upper)],
        "groups": len(groups),
        "draws": DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def _validate_evaluation(manifest, manifest_hash, r010, r010_hash, ready, preparation):
    evaluation_path = ARTIFACT_DIR / "evaluation.json"
    evaluation: dict[str, Any] = read_json(evaluation_path)
    expected_identity = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "r010_preparation_sha256": sha256(
            ROOT / "artifacts/runs/R010/preparation.json"
        ),
        "calibration_data_sha256": preparation["outputs_sha256"][CALIBRATION_DATA],
        "inference_batch_size": manifest["recipe"]["evaluation_batch_size"],
        "inference_grouping": manifest["evaluation"]["inference_grouping"],
    }
    if any(evaluation.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("R011 evaluation identity differs from locked inputs")
    if set(evaluation.get("models", {})) != set(MODELS):
        raise ValueError("R011 evaluation does not contain both locked seeds")

    r010_preparation = verify_r010_prepared(r010, r010_hash)
    test_relative = r010_preparation["outputs"][CONDITION]["file"]
    test_rows = read_jsonl(ROOT / test_relative)
    if len(test_rows) != manifest["source"]["test_rows"]:
        raise ValueError("R011 test source row count mismatch")
    expected_case_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for model_name in MODELS:
        report, report_record = _baseline_report(
            manifest, r010, r010_hash, r010_preparation, model_name
        )
        stored: dict[str, Any] = evaluation["models"][model_name]
        expected_model_identity = {
            "baseline_report": report_record["report"],
            "baseline_report_sha256": report_record["sha256"],
            "baseline_accuracy": report_record["choice_accuracy"],
            "existing_temperature": ready[model_name]["metadata"][
                "existing_temperature"
            ],
            "ocr_temperature": ready[model_name]["calibration"]["temperature"],
            "calibration_sha256": ready[model_name]["metadata"]["calibration_sha256"],
        }
        if any(
            stored.get(key) != value for key, value in expected_model_identity.items()
        ):
            raise ValueError(
                f"R011 per-model evaluation identity mismatch: {model_name}"
            )
        cases = report["cases"]
        raw_records = stored.get("rows")
        if (
            not isinstance(raw_records, list)
            or len(raw_records) != len(test_rows)
            or any(not isinstance(item, dict) for item in raw_records)
        ):
            raise ValueError(f"R011 scored row count mismatch: {model_name}")
        records = cast(list[dict[str, Any]], raw_records)
        for index, (row, case, result) in enumerate(
            zip(test_rows, cases, records, strict=True)
        ):
            labels = _labels(row)
            logits = result.get("logits")
            expected = row["expected"]["intent"]
            target_index = result.get("target_index")
            expected_target_index = labels.index(expected)
            overlap = _tag(row, "r010-train-overlap:") == "1"
            if (
                result.get("row_index") != index
                or result.get("row_id") != _tag(row, "r010-row:")
                or result.get("group_id") != _tag(row, "r010-group:")
                or result.get("train_overlap") != overlap
                or result.get("expected") != expected
                or isinstance(target_index, bool)
                or not isinstance(target_index, int)
                or target_index != expected_target_index
                or result.get("baseline_prediction")
                != case.get("answer", {}).get("choice")
                or result.get("baseline_prediction") not in labels
                or result.get("expected") not in labels
                or case.get("expected") != expected
                or case.get("model") != model_name
                or case.get("qid") != "intent"
                or not isinstance(logits, list)
                or len(logits) != 20
                or not np.isfinite(np.asarray(logits, dtype=np.float64)).all()
                or labels[int(np.argmax(logits))] != result.get("baseline_prediction")
            ):
                raise ValueError(
                    f"R011 scored row integrity mismatch: {model_name}/{index}"
                )
        expected_case_data[model_name] = (report, records)
    return test_rows, expected_case_data


def summarize():
    manifest, manifest_hash, r010, r010_hash = verify_locked_manifest()
    ready = verify_calibration_readiness(manifest, manifest_hash)
    preparation = read_json(ARTIFACT_DIR / "preparation.json")
    expected_preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "preparer_sha256": manifest["implementation_sha256"]["scripts/prepare_r011.py"],
        "source_validation_sha256": manifest["source"]["validation_sha256"],
        "source_train_sha256": manifest["source"]["train_sha256"],
        "source_test_sha256": manifest["source"]["test_sha256"],
    }
    if any(
        preparation.get(key) != value for key, value in expected_preparation.items()
    ):
        raise ValueError("R011 preparation provenance differs from locked inputs")

    test_rows, validated = _validate_evaluation(
        manifest, manifest_hash, r010, r010_hash, ready, preparation
    )
    output_models = {}
    intervals = {}
    for model_name in MODELS:
        report, rows = validated[model_name]
        old_temperature = ready[model_name]["metadata"]["existing_temperature"][0]
        ocr_temperature = ready[model_name]["calibration"]["temperature"][0]
        if old_temperature <= 0 or ocr_temperature <= 0:
            raise ValueError(f"R011 temperature is invalid: {model_name}")
        train_overlap_rows = [row for row in rows if row["train_overlap"]]
        primary_rows = [row for row in rows if not row["train_overlap"]]
        primary_groups = {row["group_id"] for row in primary_rows}
        expected_audit = manifest["source"]["audit"]
        if (
            len(primary_rows) != expected_audit["primary_test_rows"]
            or len(primary_groups) != expected_audit["primary_test_groups"]
            or len(train_overlap_rows) != expected_audit["test_train_overlap_rows"]
        ):
            raise ValueError("R011 primary test subset differs from its locked audit")

        baseline_full = _metrics(rows, old_temperature)
        ocr_full = _metrics(rows, ocr_temperature)
        baseline_primary = _metrics(primary_rows, old_temperature)
        ocr_primary = _metrics(primary_rows, ocr_temperature)
        if baseline_full["choice_accuracy"] != ocr_full["choice_accuracy"]:
            raise ValueError(
                f"R011 accuracy changed under temperature scaling: {model_name}"
            )
        if baseline_full["choice_accuracy"] != report["overall"]["choice_accuracy"]:
            raise ValueError(f"R011 baseline accuracy differs from R010: {model_name}")

        baseline_nll = _nll(
            [row["logits"] for row in primary_rows],
            [row["target_index"] for row in primary_rows],
            old_temperature,
        )
        ocr_nll = _nll(
            [row["logits"] for row in primary_rows],
            [row["target_index"] for row in primary_rows],
            ocr_temperature,
        )
        paired = _paired_group_bootstrap(
            (ocr_nll - baseline_nll).tolist(),
            [row["group_id"] for row in primary_rows],
        )
        intervals[model_name] = paired["paired_group_bootstrap_95_percent_interval"]
        output_models[model_name] = {
            "baseline_test_accuracy": baseline_full["choice_accuracy"],
            "accuracy_invariant": True,
            "accuracy_changed_rows": 0,
            "train_overlap_rows_excluded_from_primary": len(train_overlap_rows),
            "primary_subset": {
                "rows": len(primary_rows),
                "groups": len(primary_groups),
                "existing_temperature": baseline_primary,
                "ocr_fitted_temperature": ocr_primary,
                "nll_difference_ocr_fit_minus_existing": paired,
            },
            "all_test_rows_secondary": {
                "rows": len(rows),
                "existing_temperature": baseline_full,
                "ocr_fitted_temperature": ocr_full,
            },
        }

    pass_rule = all(intervals[name][1] < 0 for name in MODELS)
    results = {
        "study": manifest["study"],
        "status": "scored_exploratory",
        "manifest_sha256": manifest_hash,
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "evaluation_sha256": sha256(ARTIFACT_DIR / "evaluation.json"),
        "calibration_data_sha256": preparation["outputs_sha256"][CALIBRATION_DATA],
        "validation_audit": preparation["validation_audit"],
        "calibration_ocr": preparation["cer"],
        "inference_batch_size": manifest["recipe"]["evaluation_batch_size"],
        "inference_grouping": manifest["evaluation"]["inference_grouping"],
        "test_audit": {
            "rows": len(test_rows),
            "normalized_groups": manifest["source"]["audit"]["test_normalized_groups"],
            "conflicting_label_groups": manifest["source"]["audit"][
                "test_conflicting_label_groups"
            ],
            "train_overlap_rows": manifest["source"]["audit"][
                "test_train_overlap_rows"
            ],
            "primary_rows": manifest["source"]["audit"]["primary_test_rows"],
            "primary_groups": manifest["source"]["audit"]["primary_test_groups"],
        },
        "primary_hypothesis_supported": pass_rule,
        "models": output_models,
    }
    write_json(ARTIFACT_DIR / "results.json", results)
    print(json.dumps(results, indent=2))
    return results


def main():
    summarize()


if __name__ == "__main__":
    main()
