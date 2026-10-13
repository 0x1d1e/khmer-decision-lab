"""Summarize paired R010 OCR robustness results from frozen Laya reports."""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from r008_identity import INFERENCE_POLICY
from r010_identity import (
    ARTIFACT_DIR,
    CONDITIONS,
    MODEL_PATHS,
    ROOT,
    read_json,
    sha256,
    verify_locked_manifest,
    verify_prepared,
    write_json,
)

DRAWS = 10_000
BOOTSTRAP_SEED = 13


def _tag(case, prefix):
    matches = [
        tag[len(prefix) :] for tag in case.get("tags", []) if tag.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"report case must contain one {prefix} tag")
    return matches[0]


def _classification_metrics(cases, labels):
    gold = [case["expected"] for case in cases]
    predicted = [case["answer"]["choice"] for case in cases]
    label_set = set(labels)
    if any(value not in label_set for value in gold + predicted):
        raise ValueError("R010 report contains a label outside the locked choice set")
    correct = [prediction == expected for prediction, expected in zip(predicted, gold)]
    per_label = {}
    f1_values = []
    for label in labels:
        true_positive = sum(p == label and y == label for p, y in zip(predicted, gold))
        false_positive = sum(p == label and y != label for p, y in zip(predicted, gold))
        false_negative = sum(p != label and y == label for p, y in zip(predicted, gold))
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
        "n": len(cases),
        "choice_accuracy": sum(correct) / len(correct),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
        "correct": correct,
    }


def _paired_group_bootstrap(reference, comparison, group_ids):
    if not (len(reference) == len(comparison) == len(group_ids)):
        raise ValueError("paired R010 arrays have different lengths")
    by_group = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        by_group[group_id].append(index)
    groups = sorted(by_group)
    sizes = np.asarray([len(by_group[group]) for group in groups], dtype=np.int64)
    deltas = np.asarray(
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
        numerators = counts @ deltas
        samples.extend(numerators / denominators * 100)
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "difference_percentage_points": point,
        "paired_group_bootstrap_95_percent_interval_percentage_points": [
            float(lower),
            float(upper),
        ],
        "groups": len(groups),
        "draws": DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def _ocr_metrics(records, condition):
    values = [record["ocr"][condition] for record in records]
    codepoints = sum(value["reference_codepoints"] for value in values)
    grapheme_count = sum(value["reference_graphemes"] for value in values)
    return {
        "rows": len(values),
        "codepoint_edits": sum(value["codepoint_edits"] for value in values),
        "reference_codepoints": codepoints,
        "codepoint_cer": sum(value["codepoint_edits"] for value in values) / codepoints,
        "grapheme_edits": sum(value["grapheme_edits"] for value in values),
        "reference_graphemes": grapheme_count,
        "grapheme_cer": sum(value["grapheme_edits"] for value in values)
        / grapheme_count,
        "exact_rows": sum(value["exact"] for value in values),
        "non_exact_rows": sum(not value["exact"] for value in values),
        "blank_transcription_rows": sum(not value["text"] for value in values),
    }


@lru_cache(maxsize=len(CONDITIONS))
def _prepared_rows(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _report_cases(manifest, condition, model, preparation, evaluation):
    record = next(
        item
        for item in evaluation["reports"]
        if item["condition"] == condition and item["model"] == model
    )
    expected_report = (
        Path("artifacts/runs/R010/massive-km") / condition / f"{model}.json"
    )
    if record.get("report") != expected_report.as_posix():
        raise ValueError(f"R010 report path mismatch: {condition}/{model}")
    report_path = ROOT / expected_report
    if report_path.is_symlink() or sha256(report_path) != record["report_sha256"]:
        raise ValueError(f"R010 report changed: {record['report']}")
    report = read_json(report_path)
    config = report.get("config", {})
    checkpoint = manifest["models"][model]
    expected_config = {
        "study": manifest["study"],
        "condition": condition,
        "manifest_sha256": evaluation["manifest_sha256"],
        "source_test_file": manifest["source"]["test_file"],
        "source_test_sha256": manifest["source"]["test_sha256"],
        "dataset": preparation["outputs"][condition]["file"],
        "dataset_sha256": preparation["outputs"][condition]["sha256"],
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "preparer_sha256": manifest["implementation_sha256"]["scripts/prepare_r010.py"],
        "model": model,
        "checkpoint": checkpoint["checkpoint"],
        "weights_sha256": checkpoint["inference_files_sha256"]["model.safetensors"],
        "inference_files_sha256": checkpoint["inference_files_sha256"],
        "max_len": checkpoint["max_len"],
        "head_max_len": checkpoint["head_max_len"],
        "device": INFERENCE_POLICY["device"],
        "autocast_dtype": INFERENCE_POLICY["autocast_dtype"],
        "laya_cuda_amp": INFERENCE_POLICY["laya_cuda_amp"],
        "gpu": manifest["environment"]["gpu"],
        "torch_version": manifest["environment"]["torch_version"],
        "cuda_version": manifest["environment"]["torch_cuda"],
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"R010 report identity mismatch: {condition}/{model}")
    cases = report.get("cases", [])
    if (
        len(cases) != manifest["source"]["test_rows"]
        or record.get("rows") != len(cases)
        or record.get("choice_accuracy") != report["overall"]["choice_accuracy"]
    ):
        raise ValueError(
            f"R010 report row count or accuracy mismatch: {condition}/{model}"
        )
    prepared_rows = _prepared_rows(preparation["outputs"][condition]["file"])
    for index, (case, prepared) in enumerate(zip(cases, prepared_rows)):
        if (
            _tag(case, "r010-row:") != _tag(prepared, "r010-row:")
            or _tag(case, "r010-group:") != _tag(prepared, "r010-group:")
            or case.get("expected") != prepared.get("expected", {}).get(case.get("qid"))
            or case.get("language") != prepared.get("language")
            or case.get("qid") != "intent"
            or case.get("model") != model
            or case.get("correct") != (case["answer"]["choice"] == case["expected"])
        ):
            raise ValueError(
                f"R010 case alignment or correctness mismatch: {condition}/{model}/{index}"
            )
    return report, cases


def main():
    manifest, manifest_sha256 = verify_locked_manifest()
    preparation = verify_prepared(manifest, manifest_sha256)
    evaluation_path = ARTIFACT_DIR / "evaluation.json"
    evaluation = read_json(evaluation_path)
    if (
        evaluation.get("study") != manifest["study"]
        or evaluation.get("manifest_sha256") != manifest_sha256
        or evaluation.get("preparation_sha256")
        != sha256(ARTIFACT_DIR / "preparation.json")
    ):
        raise ValueError("R010 evaluation provenance does not match locked inputs")

    expected_reports = {
        (condition, model) for condition in CONDITIONS for model in MODEL_PATHS
    }
    actual_reports = [
        (item.get("condition"), item.get("model"))
        for item in evaluation.get("reports", [])
    ]
    if (
        len(actual_reports) != len(expected_reports)
        or set(actual_reports) != expected_reports
    ):
        raise ValueError("R010 evaluation does not contain exactly one report per run")

    source_rows = [
        json.loads(line)
        for line in (ROOT / manifest["source"]["test_file"]).read_text().splitlines()
    ]
    labels = sorted(
        {
            label
            for row in source_rows
            for label in row["questions"]["intent"]["criteria"]
        }
    )
    ocr_records = [
        json.loads(line)
        for line in (ARTIFACT_DIR / "ocr-transcripts.jsonl").read_text().splitlines()
    ]
    group_ids = [record["group_id"] for record in ocr_records]
    overlaps = [record["train_overlap"] for record in ocr_records]
    metrics = {}
    correct_by_model_condition = {}

    for model in MODEL_PATHS:
        metrics[model] = {}
        correct_by_model_condition[model] = {}
        for condition in CONDITIONS:
            report, cases = _report_cases(
                manifest, condition, model, preparation, evaluation
            )
            report_group_ids = [_tag(case, "r010-group:") for case in cases]
            if report_group_ids != group_ids:
                raise ValueError(f"R010 group alignment mismatch: {condition}/{model}")
            values = _classification_metrics(cases, labels)
            if (
                abs(values["choice_accuracy"] - report["overall"]["choice_accuracy"])
                > 1e-12
            ):
                raise ValueError(
                    f"R010 accuracy does not match Laya: {condition}/{model}"
                )
            metrics[model][condition] = {
                key: value for key, value in values.items() if key != "correct"
            }
            for key in (
                "mean_confidence",
                "ece",
                "brier",
                "aurc",
                "selective_accuracy@50",
                "selective_accuracy@80",
            ):
                metrics[model][condition][key] = report["overall"].get(key)
            correct_by_model_condition[model][condition] = values["correct"]

    ocr = {
        condition: _ocr_metrics(ocr_records, condition)
        for condition in CONDITIONS
        if condition != "clean"
    }
    paired = {}
    for model in MODEL_PATHS:
        paired[model] = {}
        clean = correct_by_model_condition[model]["clean"]
        for condition in CONDITIONS:
            if condition == "clean":
                continue
            paired[model][condition] = _paired_group_bootstrap(
                clean, correct_by_model_condition[model][condition], group_ids
            )

    strata = {model: {} for model in MODEL_PATHS}
    ocr_exactness = {model: {} for model in MODEL_PATHS}
    for model in MODEL_PATHS:
        for condition in CONDITIONS:
            correct = correct_by_model_condition[model][condition]
            strata[model][condition] = {}
            for overlap_value, name in (
                (True, "exact_train_overlap"),
                (False, "no_exact_train_overlap"),
            ):
                selected = [
                    i for i, value in enumerate(overlaps) if value is overlap_value
                ]
                strata[model][condition][name] = {
                    "n": len(selected),
                    "choice_accuracy": sum(correct[i] for i in selected)
                    / len(selected),
                }
            if condition != "clean":
                exact = [record["ocr"][condition]["exact"] for record in ocr_records]
                exactness = {}
                for exact_value, name in (
                    (True, "exact_transcription"),
                    (False, "ocr_changed_text"),
                ):
                    selected = [
                        i for i, value in enumerate(exact) if value is exact_value
                    ]
                    exactness[name] = {
                        "n": len(selected),
                        "choice_accuracy": sum(correct[i] for i in selected)
                        / len(selected)
                        if selected
                        else None,
                    }
                ocr_exactness[model][condition] = exactness

    primary = {
        "comparison": "R006 mt-s13 and mt-s14: ocr_12px minus clean accuracy",
        "success_rule": "both paired group-bootstrap 95% intervals have upper bounds below zero",
        "per_seed": {
            model: paired[model]["ocr_12px"] for model in ("mt-s13", "mt-s14")
        },
    }
    primary["decision"] = (
        "supported for these fixed checkpoints and this test split"
        if all(
            primary["per_seed"][model][
                "paired_group_bootstrap_95_percent_interval_percentage_points"
            ][1]
            < 0
            for model in primary["per_seed"]
        )
        else "not supported; results are inconclusive or mixed"
    )

    result = {
        "study": manifest["study"],
        "status": "scored_exploratory",
        "manifest_sha256": manifest_sha256,
        "test": {
            "file": manifest["source"]["test_file"],
            "sha256": manifest["source"]["test_sha256"],
            "rows": manifest["source"]["test_rows"],
            "unique_normalized_utterance_groups": len(set(group_ids)),
            "conflicting_label_groups": manifest["source"]["audit"][
                "conflicting_label_groups"
            ],
            "conflicting_label_rows": manifest["source"]["audit"][
                "conflicting_label_rows"
            ],
            "exact_train_overlap_groups": manifest["source"]["audit"][
                "exact_train_overlap_groups"
            ],
            "exact_train_overlap_rows": sum(overlaps),
            "labels": len(labels),
            "previously_scored_in": ["R001-R008"],
        },
        "ocr": {
            "engine": manifest["ocr"],
            "conditions": ocr,
        },
        "primary_hypothesis": primary,
        "models": metrics,
        "paired_accuracy_differences": paired,
        "train_overlap_strata": strata,
        "accuracy_by_ocr_exactness": ocr_exactness,
        "limitations": [
            "The official MASSIVE test split was scored in earlier experiments; this is a reused exploratory evaluation, not independent confirmation.",
            "The test inputs are synthetic typeset images rendered from the task text, not camera photographs or scanned documents.",
            "Only one Tesseract version, font, language configuration and line-segmentation mode are tested.",
            "Exact normalized utterances overlap MASSIVE training for some test rows; both overlap strata are reported.",
            "The two R006 checkpoints share a training recipe and dataset and do not represent independent tasks or a population of models.",
        ],
    }
    output = ROOT / "research/R010-massive-ocr-cascade/results.json"
    write_json(output, result)
    print(f"Wrote {output}")
    print(f"Primary decision: {primary['decision']}")
    for condition, values in ocr.items():
        print(
            f"{condition}: codepoint CER={values['codepoint_cer']:.4f}, "
            f"grapheme CER={values['grapheme_cer']:.4f}, "
            f"exact={values['exact_rows']}/{values['rows']}"
        )
    for model in MODEL_PATHS:
        print(
            f"{model}: "
            + ", ".join(
                f"{condition}={metrics[model][condition]['choice_accuracy']:.4f}"
                for condition in CONDITIONS
            )
        )


if __name__ == "__main__":
    main()
