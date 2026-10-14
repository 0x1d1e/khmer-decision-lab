"""Validate and summarize R012 paired continuation results."""

import json
from collections import defaultdict

import numpy as np
from r010_identity import normalize_input
from r012_identity import (
    ARMS,
    ARTIFACT_DIR,
    CONDITIONS,
    ROOT,
    SEEDS,
    intent_labels,
    read_json,
    read_jsonl,
    sha256,
    tag_value,
    training_output_path,
    validate_report_cases,
    verify_locked_manifest,
    verify_r010_baselines,
    verify_r010_prepared,
    verify_training_readiness,
    write_json,
)

DRAWS = 10_000
BOOTSTRAP_SEED = 13
MODEL_NAMES = {13: "mt-s13", 14: "mt-s14"}


def _metric(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"R012 report metric is unavailable or invalid: {name}")
    return float(value)


def _classification_metrics(cases, selected, labels):
    if not selected:
        raise ValueError("R012 cannot calculate metrics on an empty test subset")
    gold = [cases[index]["expected"] for index in selected]
    predicted = [cases[index]["answer"]["choice"] for index in selected]
    correct = [
        prediction == expected
        for prediction, expected in zip(predicted, gold, strict=True)
    ]
    per_label = {}
    f1_values = []
    for label in labels:
        true_positive = sum(
            p == label and y == label for p, y in zip(predicted, gold, strict=True)
        )
        false_positive = sum(
            p == label and y != label for p, y in zip(predicted, gold, strict=True)
        )
        false_negative = sum(
            p != label and y == label for p, y in zip(predicted, gold, strict=True)
        )
        support = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_label[label] = {"support": support, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "rows": len(selected),
        "choice_accuracy": sum(correct) / len(correct),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
        "correct": correct,
    }


def _group_bootstrap(reference, comparison, group_ids):
    if not reference or not (len(reference) == len(comparison) == len(group_ids)):
        raise ValueError("R012 paired bootstrap inputs are empty or misaligned")
    by_group = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                "R012 bootstrap group identifiers must be non-empty strings"
            )
        by_group[group_id].append(index)
    groups = sorted(by_group)
    sizes = np.asarray([len(by_group[group]) for group in groups], dtype=np.int64)
    sums = np.asarray(
        [
            sum(comparison[i] - reference[i] for i in by_group[group])
            for group in groups
        ],
        dtype=np.float64,
    )
    point = (sum(comparison) - sum(reference)) / len(reference) * 100
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for start in range(0, DRAWS, 100):
        batch_size = min(100, DRAWS - start)
        sampled = rng.integers(0, len(groups), size=(batch_size, len(groups)))
        counts = np.stack(
            [np.bincount(sample, minlength=len(groups)) for sample in sampled]
        )
        denominators = counts @ sizes
        numerators = counts @ sums
        samples.extend(numerators / denominators * 100)
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "difference_percentage_points": float(point),
        "paired_group_bootstrap_95_percent_interval_percentage_points": [
            float(lower),
            float(upper),
        ],
        "groups": len(groups),
        "draws": DRAWS,
        "seed": BOOTSTRAP_SEED,
        "weighting": "row-weighted within resampled normalized-source-utterance groups",
    }


def _test_eligibility(manifest, preparation, rows_by_condition):
    train_rows = read_jsonl(ROOT / manifest["source"]["train_file"])
    clean_train = {normalize_input(row["state"]["utterance"]) for row in train_rows}
    transcript_rows = read_jsonl(ROOT / preparation["transcript_artifact"])
    ocr_train = {normalize_input(row["text"]) for row in transcript_rows}
    train_inputs = clean_train | ocr_train

    output = {}
    for condition, rows in rows_by_condition.items():
        group_sizes = defaultdict(int)
        group_labels = defaultdict(set)
        group_overlaps = defaultdict(bool)
        group_ids = []
        for row in rows:
            group_id = tag_value(row, "r010-group:")
            group_ids.append(group_id)
            group_sizes[group_id] += 1
            group_labels[group_id].add(row["expected"]["intent"])
            actual_input = normalize_input(row["state"]["utterance"])
            group_overlaps[group_id] |= actual_input in train_inputs
        excluded_groups = {
            group_id for group_id, overlaps in group_overlaps.items() if overlaps
        }
        selected = [
            index
            for index, group_id in enumerate(group_ids)
            if group_id not in excluded_groups
        ]
        conflicting_groups = {
            group_id for group_id, labels in group_labels.items() if len(labels) > 1
        }
        output[condition] = {
            "all_rows": len(rows),
            "all_groups": len(group_sizes),
            "conflicting_label_groups": len(conflicting_groups),
            "conflicting_label_rows": sum(
                group_sizes[group] for group in conflicting_groups
            ),
            "excluded_train_overlap_groups": len(excluded_groups),
            "excluded_train_overlap_rows": sum(
                group_sizes[group] for group in excluded_groups
            ),
            "eligible_rows": len(selected),
            "eligible_groups": len(group_sizes) - len(excluded_groups),
            "eligible_conflicting_label_groups": len(
                conflicting_groups - excluded_groups
            ),
            "selected_indices": selected,
            "group_ids": group_ids,
        }
    return output


def _validate_r012_evaluation(
    manifest,
    manifest_hash,
    r010_hash,
    ready,
    preparation,
    r010_preparation,
    rows_by_condition,
):
    evaluation_path = ARTIFACT_DIR / "evaluation.json"
    evaluation = read_json(evaluation_path)
    expected_identity = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "training_ready": {
            key: value["provenance_sha256"] for key, value in ready.items()
        },
    }
    if any(evaluation.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("R012 evaluation identity differs from locked inputs")

    expected_keys = {
        (arm, MODEL_NAMES[seed], condition)
        for arm in ARMS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    records = evaluation.get("reports", [])
    record_keys = [
        (item.get("arm"), item.get("model"), item.get("condition")) for item in records
    ]
    if len(record_keys) != len(expected_keys) or set(record_keys) != expected_keys:
        raise ValueError(
            "R012 evaluation must contain each arm, seed, and condition once"
        )

    output = {}
    records_by_key = {
        (item["arm"], item["model"], item["condition"]): item for item in records
    }
    for arm, model_name, condition in sorted(expected_keys):
        seed = 13 if model_name == "mt-s13" else 14
        key = (arm, model_name, condition)
        record = records_by_key[key]
        expected_path = (
            (ARTIFACT_DIR / arm / model_name / f"{condition}.json")
            .relative_to(ROOT)
            .as_posix()
        )
        if record.get("report") != expected_path:
            raise ValueError(f"R012 evaluation report path mismatch: {key}")
        report_path = ROOT / expected_path
        if report_path.is_symlink() or sha256(report_path) != record.get(
            "report_sha256"
        ):
            raise ValueError(f"R012 evaluation report hash mismatch: {key}")
        report = read_json(report_path)
        checkpoint_rel = training_output_path(seed, arm)
        model = manifest["models"][model_name]
        data = r010_preparation["outputs"][condition]
        training = ready[f"{arm}-s{seed}"]
        expected_config = {
            "study": manifest["study"],
            "arm": arm,
            "seed": seed,
            "condition": condition,
            "manifest_sha256": manifest_hash,
            "r010_manifest_sha256": r010_hash,
            "source_test_sha256": manifest["source"]["test_sha256"],
            "dataset": data["file"],
            "dataset_sha256": data["sha256"],
            "r012_training_provenance_sha256": training["provenance_sha256"],
            "checkpoint": checkpoint_rel,
            "weights_sha256": training["weights_sha256"],
            "inference_files_sha256": training["inference_files_sha256"],
            "max_len": model["max_len"],
            "head_max_len": model["head_max_len"],
            "choice_temperature": model["choice_temperature"],
            "device": manifest["recipe"]["inference"]["device"],
            "autocast_dtype": manifest["recipe"]["inference"]["autocast_dtype"],
            "laya_cuda_amp": manifest["recipe"]["inference"]["laya_cuda_amp"],
            "inference_grouping": manifest["recipe"]["inference_grouping"],
            "evaluation_batch_size": manifest["recipe"]["evaluation_batch_size"],
            "gpu": manifest["environment"]["gpu"],
            "torch_version": manifest["environment"]["torch_version"],
            "cuda_version": manifest["environment"]["torch_cuda"],
        }
        config = report.get("config", {})
        if any(config.get(field) != value for field, value in expected_config.items()):
            raise ValueError(f"R012 evaluation report config mismatch: {key}")
        rows = rows_by_condition[condition]
        cases = report.get("cases", [])
        validate_report_cases(cases, rows, model_name)
        overall_accuracy = _metric(
            report.get("overall", {}).get("choice_accuracy"), "choice_accuracy"
        )
        accuracy = sum(case["correct"] for case in cases) / len(cases)
        if (
            len(cases) != manifest["source"]["test_rows"]
            or record.get("rows") != len(cases)
            or record.get("choice_accuracy") != overall_accuracy
            or abs(accuracy - overall_accuracy) > 1e-12
        ):
            raise ValueError(f"R012 evaluation report metrics do not recompute: {key}")
        output[key] = report
    return evaluation, output


def _all_metrics(report, selected_indices, labels):
    cases = report["cases"]
    classification = _classification_metrics(cases, selected_indices, labels)
    if len(selected_indices) == len(cases):
        accuracy = _metric(report["overall"].get("choice_accuracy"), "choice_accuracy")
        if abs(classification["choice_accuracy"] - accuracy) > 1e-12:
            raise ValueError("R012 report accuracy differs from validated rows")
    metrics = {key: value for key, value in classification.items() if key != "correct"}
    if len(selected_indices) == len(cases):
        metric_names = {
            "mean_confidence": "mean_confidence",
            "ece": "ece",
            "top_answer_brier": "brier",
            "aurc": "aurc",
            "selective_accuracy@50": "selective_accuracy@50",
            "selective_accuracy@80": "selective_accuracy@80",
        }
        for output_name, report_name in metric_names.items():
            metrics[output_name] = _metric(
                report["overall"].get(report_name), report_name
            )
    return metrics


def summarize():
    manifest, manifest_hash, _r010, _r010_hash = verify_locked_manifest(
        test_access=False
    )
    ready, preparation = verify_training_readiness(manifest, manifest_hash)

    manifest, manifest_hash, r010, r010_hash = verify_locked_manifest(test_access=True)
    ready, preparation = verify_training_readiness(manifest, manifest_hash)
    r010_preparation = verify_r010_prepared(r010, r010_hash)
    baselines = verify_r010_baselines(manifest, r010, r010_hash, r010_preparation)
    rows_by_condition = {
        condition: read_jsonl(ROOT / r010_preparation["outputs"][condition]["file"])
        for condition in CONDITIONS
    }
    audit = _test_eligibility(manifest, preparation, rows_by_condition)
    _evaluation, reports = _validate_r012_evaluation(
        manifest,
        manifest_hash,
        r010_hash,
        ready,
        preparation,
        r010_preparation,
        rows_by_condition,
    )
    labels = sorted(
        {
            label
            for rows in rows_by_condition.values()
            for row in rows
            for label in intent_labels(row)
        }
    )

    metric_results = {"all_test_rows": {}, "eligible_rows": {}}
    primary = {"ocr_domain_gain": {}, "clean_text_preservation": {}}
    for condition in CONDITIONS:
        selected = audit[condition]["selected_indices"]
        metric_results["all_test_rows"][condition] = {}
        metric_results["eligible_rows"][condition] = {}
        for seed in SEEDS:
            model_name = MODEL_NAMES[seed]
            metric_results["all_test_rows"][condition][f"frozen_r006_s{seed}"] = (
                _all_metrics(
                    baselines[f"{condition}/{model_name}"],
                    list(range(manifest["source"]["test_rows"])),
                    labels,
                )
            )
            for arm in ARMS:
                report = reports[(arm, model_name, condition)]
                all_metrics = _all_metrics(
                    report, list(range(manifest["source"]["test_rows"])), labels
                )
                eligible_metrics = _all_metrics(report, selected, labels)
                metric_results["all_test_rows"][condition][f"{arm}_s{seed}"] = (
                    all_metrics
                )
                metric_results["eligible_rows"][condition].setdefault(
                    f"frozen_r006_s{seed}",
                    _all_metrics(
                        baselines[f"{condition}/{model_name}"], selected, labels
                    ),
                )
                metric_results["eligible_rows"][condition][f"{arm}_s{seed}"] = (
                    eligible_metrics
                )

    for seed in SEEDS:
        model_name = MODEL_NAMES[seed]
        ocr_selected = audit["ocr_12px"]["selected_indices"]
        ocr_groups = [audit["ocr_12px"]["group_ids"][i] for i in ocr_selected]
        clean_selected = audit["clean"]["selected_indices"]
        clean_groups = [audit["clean"]["group_ids"][i] for i in clean_selected]
        clean_arm_ocr = reports[("clean", model_name, "ocr_12px")]["cases"]
        ocr_arm_ocr = reports[("ocr12", model_name, "ocr_12px")]["cases"]
        primary["ocr_domain_gain"][f"seed_{seed}"] = _group_bootstrap(
            [clean_arm_ocr[index]["correct"] for index in ocr_selected],
            [ocr_arm_ocr[index]["correct"] for index in ocr_selected],
            ocr_groups,
        )
        frozen_clean = baselines[f"clean/{model_name}"]["cases"]
        ocr_arm_clean = reports[("ocr12", model_name, "clean")]["cases"]
        primary["clean_text_preservation"][f"seed_{seed}"] = _group_bootstrap(
            [frozen_clean[index]["correct"] for index in clean_selected],
            [ocr_arm_clean[index]["correct"] for index in clean_selected],
            clean_groups,
        )

    gain_supported = all(
        primary["ocr_domain_gain"][f"seed_{seed}"][
            "paired_group_bootstrap_95_percent_interval_percentage_points"
        ][0]
        > 0
        for seed in SEEDS
    )
    preservation_supported = all(
        primary["clean_text_preservation"][f"seed_{seed}"][
            "paired_group_bootstrap_95_percent_interval_percentage_points"
        ][0]
        > -2
        for seed in SEEDS
    )
    primary.update(
        {
            "ocr_domain_gain_supported": gain_supported,
            "clean_text_preservation_supported": preservation_supported,
            "combined_claim_supported": gain_supported and preservation_supported,
            "ocr_domain_gain_rule": "both seed-specific 95% paired grouped-bootstrap lower bounds exceed 0 pp",
            "clean_text_preservation_rule": "both seed-specific 95% paired grouped-bootstrap lower bounds exceed -2 pp",
            "bootstrap_draws": DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
    )

    training = {
        key: {
            field: value[field]
            for field in (
                "seed",
                "arm",
                "checkpoint",
                "weights_sha256",
                "provenance_sha256",
                "optimizer_attempts",
                "successful_updates",
                "skipped_updates",
                "training_seconds",
                "peak_allocated_bytes",
            )
        }
        for key, value in ready.items()
    }
    result = {
        "study": manifest["study"],
        "status": "scored_exploratory",
        "manifest_sha256": manifest_hash,
        "protocol_sha256": manifest["protocol_sha256"],
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "evaluation_sha256": sha256(ARTIFACT_DIR / "evaluation.json"),
        "training": training,
        "test": {
            "file": manifest["source"]["test_file"],
            "sha256": manifest["source"]["test_sha256"],
            "rows": manifest["source"]["test_rows"],
            "normalized_source_groups": manifest["source"]["audit"][
                "normalized_test_groups"
            ],
            "conflicting_label_groups": manifest["source"]["audit"][
                "conflicting_label_groups"
            ],
            "conflicting_label_rows": manifest["source"]["audit"][
                "conflicting_label_rows"
            ],
            "eligibility_by_condition": {
                condition: {
                    key: value
                    for key, value in condition_audit.items()
                    if key != "selected_indices" and key != "group_ids"
                }
                for condition, condition_audit in audit.items()
            },
            "overlap_policy": "exclude whole normalized source-utterance groups with any clean or OCR test input overlapping the union of clean and OCR training inputs",
        },
        "primary": primary,
        "metrics": metric_results,
        "labels": labels,
        "limitations": [
            "R012 reuses the MASSIVE test split scored by R001-R011; results are exploratory and not independent confirmation.",
            "OCR inputs are synthetic typeset Tesseract transcripts from one font, renderer, engine version, and configuration.",
            "R012 estimates the incremental effect of replacing clean continuation text with OCR transcripts on two fixed R006 checkpoints, not OCR-only training or general OCR robustness.",
            "The two seed checkpoints are fixed study anchors, not a representative sample of models.",
            "The train-overlap exclusion removes exact normalized input matches but not broader historical exposure or semantic overlap.",
        ],
    }
    results_path = ROOT / "research/R012-ocr-continuation/results.json"
    write_json(results_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {results_path}")
    return result


def main():
    summarize()


if __name__ == "__main__":
    main()
