"""Validate locked R008 reports and summarize paired specialist comparisons.

Usage: uv run --frozen python scripts/summarize_r008.py
This script reads frozen test reports and exports only. It does not train,
calibrate, or generate predictions.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from laya import evals
from r008_identity import (
    ARM_OUTPUT_PREFIXES,
    ARM_TASKS,
    row_identity_tag,
    sha256,
    state_group_id,
    verify_calibration_artifact,
    verify_checkpoint_training_provenance,
    verify_locked_manifest,
    verify_study_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r008-specialist-controls.json"
OUTPUT = ROOT / "research/R008-specialist-controls/results.json"
TASKS = ("khmer-news", "sib200-km")
SEEDS = (13, 14)
METRICS = (
    "choice_accuracy",
    "mean_confidence",
    "ece",
    "brier",
    "aurc",
    "selective_accuracy@50",
    "selective_accuracy@80",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_decision_p50_ms",
    "cost_per_decision_p95_ms",
)
RECOMPUTED_METRICS = METRICS[:7]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_metrics(confidence, correct):
    return {
        "choice_accuracy": sum(correct) / len(correct),
        "mean_confidence": sum(confidence) / len(confidence),
        "ece": evals.ece(confidence, correct, bins=15),
        "brier": evals.brier(confidence, correct),
        "aurc": evals.aurc(confidence, correct),
        "selective_accuracy@50": evals.selective_accuracy(confidence, correct, 0.5),
        "selective_accuracy@80": evals.selective_accuracy(confidence, correct, 0.8),
    }


def check_report_cases(report, model, rows):
    """Validate case alignment and recompute outcome/calibration metrics."""
    cases = report.get("cases", [])
    if not rows or len(cases) != len(rows):
        raise ValueError(f"R008 report row count mismatch: {model}")

    confidence, correct, golds, predictions = [], [], [], []
    for case, row in zip(cases, rows):
        question_ids = list(row["questions"])
        if len(question_ids) != 1:
            raise ValueError("R008 test row must have exactly one question")
        qid = question_ids[0]
        expected = row.get("expected", {})
        if not isinstance(expected, dict) or set(expected) != {qid}:
            raise ValueError(f"R008 test row must label only its question: {model}")
        gold = expected[qid]
        answer = case.get("answer", {})
        probabilities = answer.get("probabilities", {})
        options = row["questions"][qid]["criteria"]
        choice = answer.get("choice")
        case_confidence = case.get("confidence")
        identity = row_identity_tag(row)
        expected_tags = [str(tag) for tag in row.get("tags") or ()] + [identity]
        if (
            gold not in options
            or case.get("qid") != qid
            or case.get("model") != model
            or case.get("expected") != gold
            or case.get("tags") != expected_tags
            or answer.get("type") != "choice"
            or set(probabilities) != set(options)
            or choice not in probabilities
            or case.get("correct") != (choice == gold)
            or case_confidence != answer.get("answer_confidence")
            or not isinstance(case_confidence, (int, float))
            or not math.isfinite(case_confidence)
            or not 0 <= case_confidence <= 1
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in probabilities.values()
            )
            or abs(math.fsum(probabilities.values()) - 1.0)
            > len(probabilities) * 0.00005 + 1e-12
            or max(probabilities.values()) - probabilities[choice] > 0.0001 + 1e-12
            or abs(case_confidence - max(probabilities.values())) > 0.0001 + 1e-12
        ):
            raise ValueError(f"R008 report case alignment/answer mismatch: {model}")
        confidence.append(float(case_confidence))
        correct.append(choice == gold)
        golds.append(gold)
        predictions.append(choice)

    computed = expected_metrics(confidence, correct)
    overall = report.get("overall", {})
    for key, value in computed.items():
        if (
            value is None
            or key not in overall
            or not math.isclose(overall[key], value, abs_tol=1e-10)
        ):
            raise ValueError(f"R008 report metric mismatch ({key}): {model}")
    return confidence, correct, golds, predictions


def paired_group_bootstrap(
    multitask_correct, specialist_correct, groups, samples=10_000, seed=13
):
    """Paired, uniform-group draws with row-weighted accuracy differences."""
    matrix = np.asarray([multitask_correct, specialist_correct], dtype=float).T
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 2
        or matrix.shape[0] != len(groups)
        or matrix.shape[0] == 0
        or samples < 1
    ):
        raise ValueError("Expected paired nonempty correctness vectors and groups")
    unique = sorted(set(groups))
    if not unique:
        raise ValueError("R008 bootstrap requires at least one group")
    lookup = {group: index for index, group in enumerate(unique)}
    row_groups = np.asarray([lookup[group] for group in groups])
    sizes = np.bincount(row_groups, minlength=len(unique))
    totals = np.zeros((len(unique), 2), dtype=float)
    np.add.at(totals, row_groups, matrix)

    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0, samples, 256):
        batch_size = min(256, samples - start)
        draws = rng.integers(0, len(unique), (batch_size, len(unique)))
        sampled_totals = totals[draws].sum(axis=1)
        sampled_rows = sizes[draws].sum(axis=1)
        accuracies = sampled_totals / sampled_rows[:, None]
        differences.append(accuracies[:, 0] - accuracies[:, 1])
    return np.concatenate(differences)


def recall_by_label(golds, predictions, labels):
    counts = Counter(golds)
    hits = Counter(
        gold for gold, prediction in zip(golds, predictions) if gold == prediction
    )
    return {
        label: hits[label] / counts[label] if counts[label] else None
        for label in labels
    }


def verify_report(
    report,
    model,
    task,
    arm,
    seed,
    task_data,
    provenance,
    calibration_hash,
    manifest_hash,
    manifest,
    rows,
):
    if arm not in ARM_TASKS or task not in ARM_TASKS[arm]:
        raise ValueError(f"R008 report arm is incompatible with task: {arm}/{task}")
    config = report.get("config", {})
    expected = {
        "study": "R008-specialist-controls",
        "task": task,
        "model": model,
        "training_arm": arm,
        "seed": seed,
        "dataset": task_data["test_file"],
        "dataset_sha256": task_data["test_sha256"],
        "locked_manifest_sha256": manifest_hash,
        "weights_sha256": provenance["weights_sha256"],
        "calibration_sha256": calibration_hash,
        "training_provenance_sha256": sha256(
            ROOT
            / "artifacts/checkpoints"
            / f"{ARM_OUTPUT_PREFIXES[arm]}-s{seed}"
            / "training_provenance.json"
        ),
        "training_file_sha256": provenance["training_file_sha256"],
        "base_revision": manifest["recipe"]["base_revision"],
        "max_len": manifest["recipe"]["max_len"],
        "head_max_len": manifest["recipe"]["head_max_len"],
        "device": manifest["inference"]["device"],
        "inference_dtype": manifest["inference"]["autocast_dtype"],
        "laya_version": manifest["environment"]["laya"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"R008 frozen report identity mismatch: {model}/{task}")
    if len(rows) != task_data["test_rows"]:
        raise ValueError(f"R008 locked test row count mismatch: {task}")
    confidence, correct, golds, predictions = check_report_cases(report, model, rows)
    labels = sorted(
        {
            label
            for row in rows
            for label in row["questions"][next(iter(row["questions"]))]["criteria"]
        }
    )
    metrics = {key: report["overall"].get(key) for key in METRICS}
    if any(metrics[key] is None for key in METRICS):
        raise ValueError(f"R008 report is missing required metrics: {model}/{task}")
    return {
        "metrics": metrics,
        "recall_by_label": recall_by_label(golds, predictions, labels),
        "rows": len(rows),
        "groups": len({state_group_id(row) for row in rows}),
        "report_sha256": None,
        "_confidence": confidence,
        "_correct": correct,
    }


def summarize_task(task, manifest, manifest_hash):
    task_data = manifest["validation_calibration"][task]
    expected_test_file = f"data/processed/{task}-test.jsonl"
    if task_data.get("test_file") != expected_test_file:
        raise ValueError(f"R008 test-task mapping is inconsistent: {task}")
    test_path = ROOT / expected_test_file
    if sha256(test_path) != task_data["test_sha256"]:
        raise ValueError(f"R008 test file fingerprint mismatch: {task}")
    rows = read_jsonl(test_path)
    groups = [state_group_id(row) for row in rows]
    if len(rows) != task_data["test_rows"]:
        raise ValueError(f"R008 test row count differs from manifest: {task}")

    specialist_arm = {
        "khmer-news": "khmer-news-specialist",
        "sib200-km": "sib200-specialist",
    }[task]
    arms = ("multitask", specialist_arm)
    model_results = {}
    paired = {}
    for seed in SEEDS:
        for arm in arms:
            model = f"{arm}-s{seed}"
            checkpoint = (
                ROOT / "artifacts/checkpoints" / f"{ARM_OUTPUT_PREFIXES[arm]}-s{seed}"
            )
            provenance = verify_checkpoint_training_provenance(
                checkpoint, task, manifest, manifest_hash
            )
            calibration_path, calibration_hash = verify_calibration_artifact(
                checkpoint, task, provenance, manifest, manifest_hash
            )
            report_path = ROOT / manifest["evaluation"]["test_reports"].format(
                task=task, arm=arm, seed=seed
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = verify_report(
                report,
                model,
                task,
                arm,
                seed,
                task_data,
                provenance,
                calibration_hash,
                manifest_hash,
                manifest,
                rows,
            )
            result["report"] = str(report_path.relative_to(ROOT))
            result["report_sha256"] = sha256(report_path)
            result["calibration"] = str(calibration_path.relative_to(ROOT))
            result["calibration_sha256"] = calibration_hash
            model_results[model] = result

        multitask_model = f"multitask-s{seed}"
        specialist_model = f"{specialist_arm}-s{seed}"
        mt = model_results[multitask_model]
        specialist = model_results[specialist_model]
        samples = manifest["evaluation"]["bootstrap_samples"]
        bootstrap = paired_group_bootstrap(
            mt["_correct"],
            specialist["_correct"],
            groups,
            samples=samples,
            seed=manifest["evaluation"]["bootstrap_seed"],
        )
        difference_pp = (
            100 * (sum(mt["_correct"]) - sum(specialist["_correct"])) / len(rows)
        )
        low, high = (100 * np.quantile(bootstrap, [0.025, 0.975])).tolist()
        margin = manifest["evaluation"]["noninferiority_margin_percentage_points"]
        paired[str(seed)] = {
            "multitask_model": multitask_model,
            "specialist_model": specialist_model,
            "difference_percentage_points": difference_pp,
            "paired_group_bootstrap_ci95_percentage_points": [low, high],
            "noninferiority_pass": low > margin,
            "negative_difference_supported": high < 0,
            "bootstrap_samples": samples,
            "bootstrap_seed": manifest["evaluation"]["bootstrap_seed"],
            "group_count": len(set(groups)),
        }

    for result in model_results.values():
        del result["_confidence"]
        del result["_correct"]
    return {
        "test_file": expected_test_file,
        "test_sha256": task_data["test_sha256"],
        "test_rows": len(rows),
        "test_groups": len(set(groups)),
        "models": model_results,
        "paired_comparisons": paired,
        "all_seeds_pass_noninferiority": all(
            paired[str(seed)]["noninferiority_pass"] for seed in SEEDS
        ),
    }


def summarize(manifest_path=MANIFEST, output_path=OUTPUT):
    manifest, manifest_hash = verify_locked_manifest(manifest_path)
    if manifest.get("study") != "R008-specialist-controls":
        raise ValueError("Unexpected R008 study manifest")
    verify_study_readiness(manifest, manifest_hash)
    results = {
        "study": "R008-specialist-controls",
        "status": "exploratory_results",
        "manifest_sha256": manifest_hash,
        "analysis": {
            "primary_metric": "choice_accuracy",
            "comparison": "paired multitask minus matching specialist, same seed and rows",
            "bootstrap": "uniform exact-normalized-state groups; row-weighted accuracy difference",
            "samples": manifest["evaluation"]["bootstrap_samples"],
            "seed": manifest["evaluation"]["bootstrap_seed"],
            "interval": "percentile 2.5/97.5",
            "noninferiority_margin_percentage_points": manifest["evaluation"][
                "noninferiority_margin_percentage_points"
            ],
            "interpretation": (
                "Exploratory intervals are conditional on the two preregistered seeds; "
                "they do not establish population-level equivalence or confirmatory non-inferiority."
            ),
        },
        "tasks": {
            task: summarize_task(task, manifest, manifest_hash) for task in TASKS
        },
    }
    results["joint_noninferiority_supported"] = all(
        results["tasks"][task]["all_seeds_pass_noninferiority"] for task in TASKS
    )
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return results


def main():
    results = summarize()
    print(
        json.dumps(
            {
                task: {
                    "paired_comparisons": result["paired_comparisons"],
                    "all_seeds_pass_noninferiority": result[
                        "all_seeds_pass_noninferiority"
                    ],
                }
                for task, result in results["tasks"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
