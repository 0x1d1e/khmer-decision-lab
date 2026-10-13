"""Score both locked R009 continuations after all artifacts pass the test gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict

import laya
import numpy as np
import torch
from laya import evals
from r008_identity import load_r008_agent, row_identity_tag, state_group_id
from r009_identity import (
    ROOT,
    SEEDS,
    read_json,
    sha256,
    verify_locked_manifest,
    verify_score_readiness,
    verify_test_file,
    write_json,
)
from summarize_r008 import paired_group_bootstrap


class AgentRunner:
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name

    def _check_model(self, model):
        if model is not None and model != self.model_name:
            raise ValueError(f"runner serves {self.model_name!r}, not {model!r}")

    def predict(self, state, questions, model=None, min_confidence=None):
        self._check_model(model)
        kwargs = {} if min_confidence is None else {"min_confidence": min_confidence}
        return self.agent.predict(state, questions, **kwargs)

    def predict_batch(
        self,
        states,
        questions,
        model=None,
        batch_size=None,
        sort_by_length=False,
        min_confidence=None,
    ):
        self._check_model(model)
        kwargs = {"batch_size": batch_size, "sort_by_length": sort_by_length}
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        return self.agent.predict_batch(states, questions, **kwargs)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_tag(row):
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "r009:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_test_rows_after_readiness(manifest, r009_hash, r008_manifest, r008_hash):
    ready = verify_score_readiness(manifest, r009_hash, r008_manifest, r008_hash)
    test_path = verify_test_file(manifest)
    return ready, test_path, read_jsonl(test_path)


def bind_rows(dataset, rows, model_name):
    if len(dataset.examples) != len(rows):
        raise ValueError("R009 dataset and raw row counts differ")
    for example, row in zip(dataset.examples, rows):
        if example.model not in (None, model_name):
            raise ValueError("R009 dataset row names a different model")
        if any(str(tag).startswith("r009:") for tag in example.tags):
            raise ValueError("R009 dataset already has an identity tag")
        example.model = model_name
        example.tags = (*example.tags, row_tag(row))


def score_agent(agent, dataset, model_name, config):
    if any(example.model != model_name for example in dataset.examples):
        raise ValueError("R009 dataset rows must be bound to this model")
    return evals.evaluate(
        AgentRunner(agent, model_name),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=8,
        config=config,
    )


def check_cases(report, rows, model_name, tag_prefix):
    cases = report.get("cases", [])
    if not rows or len(cases) != len(rows):
        raise ValueError(f"R009 case count mismatch for {model_name}")
    confidences, correct, golds, predictions, probability_rows = [], [], [], [], []
    for case, row in zip(cases, rows):
        questions = row.get("questions", {})
        expected_map = row.get("expected", {})
        if len(questions) != 1 or set(expected_map) != set(questions):
            raise ValueError("R009 SIB-200 row must have one labeled question")
        qid = next(iter(questions))
        gold = expected_map[qid]
        answer = case.get("answer", {})
        probabilities = answer.get("probabilities", {})
        options = questions[qid]["criteria"]
        choice = answer.get("choice")
        tags = list(row.get("tags") or ()) + [tag_prefix(row)]
        confidence = case.get("confidence")
        if (
            case.get("qid") != qid
            or case.get("model") != model_name
            or case.get("expected") != gold
            or case.get("tags") != tags
            or answer.get("type") != "choice"
            or gold not in options
            or set(probabilities) != set(options)
            or choice not in options
            or case.get("correct") != (choice == gold)
            or confidence != answer.get("answer_confidence")
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in probabilities.values()
            )
            or abs(math.fsum(probabilities.values()) - 1.0)
            > len(probabilities) * 0.00005 + 1e-12
            or max(probabilities.values()) - probabilities[choice] > 0.0001 + 1e-12
            or abs(confidence - max(probabilities.values())) > 0.0001 + 1e-12
        ):
            raise ValueError(f"R009 case identity or answer mismatch for {model_name}")
        confidences.append(float(confidence))
        correct.append(choice == gold)
        golds.append(gold)
        predictions.append(choice)
        probability_rows.append(probabilities)
    if abs(sum(correct) / len(correct) - report["overall"]["choice_accuracy"]) > 1e-12:
        raise ValueError(f"R009 accuracy does not recompute for {model_name}")
    return confidences, correct, golds, predictions, probability_rows


def macro_ovr_auc(golds, probability_rows, labels):
    aucs = []
    for label in labels:
        positives = [
            p[label] for gold, p in zip(golds, probability_rows) if gold == label
        ]
        negatives = [
            p[label] for gold, p in zip(golds, probability_rows) if gold != label
        ]
        if not positives or not negatives:
            raise ValueError(f"cannot calculate one-vs-rest AUROC for {label}")
        wins = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        aucs.append(wins / (len(positives) * len(negatives)))
    return sum(aucs) / len(aucs)


def classification_metrics(
    report, confidences, correct, golds, predictions, probabilities, labels
):
    recalls = {}
    f1s = []
    for label in labels:
        tp = sum(
            gold == label and pred == label for gold, pred in zip(golds, predictions)
        )
        actual = sum(gold == label for gold in golds)
        predicted = sum(pred == label for pred in predictions)
        recall = tp / actual if actual else None
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else 0.0
        )
        recalls[label] = recall
        f1s.append(f1)
    computed = {
        "choice_accuracy": sum(correct) / len(correct),
        "mean_confidence": sum(confidences) / len(confidences),
        "ece": evals.ece(confidences, correct, bins=15),
        "brier": evals.brier(confidences, correct),
        "aurc": evals.aurc(confidences, correct),
        "selective_accuracy@50": evals.selective_accuracy(confidences, correct, 0.5),
        "selective_accuracy@80": evals.selective_accuracy(confidences, correct, 0.8),
    }
    overall = report["overall"]
    for key, value in computed.items():
        if value is None or not math.isclose(
            overall.get(key, float("nan")), value, abs_tol=1e-10
        ):
            raise ValueError(f"R009 reported {key} does not recompute")
    return {
        **computed,
        "macro_f1": sum(f1s) / len(f1s),
        "per_label_recall": recalls,
        "macro_ovr_auc": macro_ovr_auc(golds, probabilities, labels),
    }


class ComparisonResult(TypedDict):
    difference_percentage_points: float
    paired_group_bootstrap_95_percent_interval_percentage_points: list[float]


def comparison(
    new_correct, baseline_correct, groups, samples, seed
) -> ComparisonResult:
    distribution = paired_group_bootstrap(
        new_correct, baseline_correct, groups, samples=samples, seed=seed
    )
    return {
        "difference_percentage_points": 100
        * (sum(new_correct) - sum(baseline_correct))
        / len(new_correct),
        "paired_group_bootstrap_95_percent_interval_percentage_points": [
            float(100 * np.percentile(distribution, 2.5)),
            float(100 * np.percentile(distribution, 97.5)),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest, r009_hash, r008_manifest, r008_hash = verify_locked_manifest()
    # Readiness covers both new checkpoints/calibration maps and every R008 baseline.
    ready, test_path, rows = read_test_rows_after_readiness(
        manifest, r009_hash, r008_manifest, r008_hash
    )
    if len(rows) != manifest["source"]["test_rows"]:
        raise ValueError("R009 test row count differs from the locked manifest")
    labels = read_json(ROOT / manifest["source"]["dataset_manifest"])["labels"]
    r009_reports = {}
    r009_metrics = {}
    r009_correct = {}

    for seed in SEEDS:
        entry = ready[seed]
        checkpoint = entry["checkpoint"]
        calibration_path, calibration_hash, _ = entry["calibration"]
        model_name = f"r009-sib-specialist-cont-s{seed}"
        model_dataset = evals.Dataset.from_jsonl(str(test_path))
        bind_rows(model_dataset, rows, model_name)
        agent, inference_dtype = load_r008_agent(
            checkpoint, r008_manifest, calibration=calibration_path
        )
        if (
            agent.cfg.get("max_len") != manifest["recipe"]["max_len"]
            or agent.cfg.get("head_max_len") != manifest["recipe"]["head_max_len"]
            or agent.device.type != manifest["inference"]["device"]
            or str(agent.dtype).removeprefix("torch.") != inference_dtype
            or not agent.amp_enabled
            or torch.version.cuda != manifest["environment"]["cuda"]
            or torch.cuda.get_device_name(0) != manifest["environment"]["gpu"]
        ):
            raise ValueError("R009 inference runtime differs from the locked policy")
        config = {
            "study": "R009-sib-specialist-continuation",
            "task": "sib200-km",
            "model": model_name,
            "training_arm": "continued-sib200-specialist",
            "seed": seed,
            "dataset": manifest["source"]["test_file"],
            "dataset_sha256": manifest["source"]["test_sha256"],
            "r009_manifest_sha256": r009_hash,
            "r008_manifest_sha256": r008_hash,
            "weights_sha256": entry["training"]["weights_sha256"],
            "calibration_sha256": calibration_hash,
            "training_provenance_sha256": sha256(
                checkpoint / "training_provenance.json"
            ),
            "training_file_sha256": manifest["source"]["training_sha256"],
            "base_revision": manifest["recipe"]["base_revision"],
            "max_len": manifest["recipe"]["max_len"],
            "head_max_len": manifest["recipe"]["head_max_len"],
            "device": agent.device.type,
            "inference_dtype": inference_dtype,
            "gpu": torch.cuda.get_device_name(0),
            "laya_version": laya.__version__,
        }
        report = score_agent(agent, model_dataset, model_name, config).to_json()
        confidence, correct, golds, predictions, probabilities = check_cases(
            report, rows, model_name, row_tag
        )
        r009_correct[seed] = correct
        r009_metrics[seed] = classification_metrics(
            report, confidence, correct, golds, predictions, probabilities, labels
        )
        output = ROOT / manifest["evaluation"]["test_reports"].format(seed=seed)
        write_json(output, report)
        r009_reports[seed] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
            "overall": r009_metrics[seed],
        }
        del report, agent, model_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    groups = [state_group_id(row) for row in rows]
    if len(set(groups)) != manifest["source"]["test_exact_state_groups"]:
        raise ValueError("R009 test state-group count differs from the locked manifest")
    per_seed = []
    primary_intervals: list[list[float]] = []
    for seed in SEEDS:
        anchor_specialist = ready[seed]["anchors"]["specialist"]["report"]
        anchor_multitask = ready[seed]["anchors"]["multitask"]["report"]
        specialist_name = f"sib200-specialist-s{seed}"
        multitask_name = f"multitask-s{seed}"
        specialist_values = check_cases(
            anchor_specialist, rows, specialist_name, row_identity_tag
        )
        multitask_values = check_cases(
            anchor_multitask, rows, multitask_name, row_identity_tag
        )
        (
            specialist_confidence,
            specialist_correct,
            specialist_golds,
            specialist_predictions,
            specialist_probabilities,
        ) = specialist_values
        (
            multitask_confidence,
            multitask_correct,
            multitask_golds,
            multitask_predictions,
            multitask_probabilities,
        ) = multitask_values
        specialist_metrics = classification_metrics(
            anchor_specialist,
            specialist_confidence,
            specialist_correct,
            specialist_golds,
            specialist_predictions,
            specialist_probabilities,
            labels,
        )
        multitask_metrics = classification_metrics(
            anchor_multitask,
            multitask_confidence,
            multitask_correct,
            multitask_golds,
            multitask_predictions,
            multitask_probabilities,
            labels,
        )
        specialist_comparison = comparison(
            r009_correct[seed],
            specialist_correct,
            groups,
            manifest["evaluation"]["bootstrap_samples"],
            manifest["evaluation"]["bootstrap_seed"],
        )
        multitask_comparison = comparison(
            r009_correct[seed],
            multitask_correct,
            groups,
            manifest["evaluation"]["bootstrap_samples"],
            manifest["evaluation"]["bootstrap_seed"],
        )
        primary_intervals.append(
            specialist_comparison[
                "paired_group_bootstrap_95_percent_interval_percentage_points"
            ]
        )
        per_seed.append(
            {
                "seed": seed,
                "continued_metrics": r009_metrics[seed],
                "r008_specialist_metrics": specialist_metrics,
                "r008_multitask_metrics": multitask_metrics,
                "continued_minus_r008_specialist": specialist_comparison,
                "continued_minus_r008_multitask": multitask_comparison,
                "training_updates": {
                    "attempts": ready[seed]["telemetry"]["optimizer_attempts"],
                    "successful": ready[seed]["telemetry"]["successful_updates"],
                    "skipped": ready[seed]["telemetry"]["skipped_updates"],
                    "seconds": ready[seed]["telemetry"]["training_seconds"],
                    "peak_allocated_bytes": ready[seed]["telemetry"][
                        "peak_allocated_bytes"
                    ],
                },
                "test_report": r009_reports[seed],
            }
        )

    if all(interval[0] > 0 for interval in primary_intervals):
        decision = "supported conditionally on these two seeds and this test set"
    elif all(interval[1] < 0 for interval in primary_intervals):
        decision = "evidence of lower accuracy conditionally on these two seeds and this test set"
    else:
        decision = "not supported; results are inconclusive or mixed"

    results = {
        "study": "R009-sib-specialist-continuation",
        "status": "scored_exploratory",
        "manifest_sha256": r009_hash,
        "r008_manifest_sha256": r008_hash,
        "test": {
            "file": manifest["source"]["test_file"],
            "sha256": manifest["source"]["test_sha256"],
            "rows": len(rows),
            "unique_state_groups": len(set(groups)),
        },
        "primary_hypothesis": {
            "comparison": manifest["evaluation"]["primary_comparison"],
            "success_rule": manifest["evaluation"]["primary_success"],
            "decision": decision,
            "per_seed": per_seed,
        },
        "interpretation_limit": "Two seed-matched continuations on one 204-row task test support only a conditional exploratory conclusion; no population-level or cross-task claim is warranted.",
    }
    output = ROOT / manifest["evaluation"]["summary"]
    write_json(output, results)
    print(json.dumps({"results": str(output), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
