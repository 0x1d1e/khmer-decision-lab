"""Validate R013 inference artifacts and summarize its locked comparisons."""

from datetime import UTC, datetime

import numpy as np
from evaluate_r013 import CONDITIONS, MODELS, _tags, _validate_test_rows, _verify_cases
from laya import evals
from r013_identity import (
    AMENDMENT,
    ARTIFACT_DIR,
    EVALUATION,
    INFERENCE_BATCH_SIZE,
    INFERENCE_LIMITS,
    OUTPUTS,
    ROOT,
    SEEDS,
    SPLIT_FILES,
    _artifact_lock_identity,
    load_test_rows_after_readiness,
    read_json,
    sha256,
    verify_locked_manifest,
    write_json,
)

RESULTS_JSON = ROOT / "research/R013-unseen-intent-transfer/results.json"
RESULTS_MD = ROOT / "research/R013-unseen-intent-transfer/results.md"


def _softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _binary_auc(targets, scores):
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        mean_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += mean_rank * int(sorted_targets[start:end].sum())
        start = end
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def compute_metrics(rows, cases, logits):
    """Compute row-weighted metrics from the captured, identity-temperature logits."""
    probabilities = _softmax(logits)
    gold = [row["expected"]["intent"] for row in rows]
    predictions = [case["answer"]["choice"] for case in cases]
    correct = np.asarray(
        [
            prediction == label
            for prediction, label in zip(predictions, gold, strict=True)
        ]
    )
    labels = sorted(set(gold))
    candidate_labels = sorted(
        {label for row in rows for label in row["questions"]["intent"]["criteria"]}
    )
    if probabilities.shape != (len(rows), 20) or not len(labels):
        raise ValueError(
            "R013 metrics received an invalid logits matrix or empty label set"
        )

    per_intent_recall = {}
    per_intent_f1 = {}
    for label in labels:
        true_positive = sum(
            y == label and p == label for y, p in zip(gold, predictions, strict=True)
        )
        false_positive = sum(
            y != label and p == label for y, p in zip(gold, predictions, strict=True)
        )
        false_negative = sum(
            y == label and p != label for y, p in zip(gold, predictions, strict=True)
        )
        support = sum(y == label for y in gold)
        per_intent_recall[label] = true_positive / support
        denominator = 2 * true_positive + false_positive + false_negative
        per_intent_f1[label] = 2 * true_positive / denominator if denominator else 0.0

    auc_by_intent = {}
    for label in labels:
        mask = np.asarray(
            [label in row["questions"]["intent"]["criteria"] for row in rows],
            dtype=bool,
        )
        auc_by_intent[label] = _binary_auc(
            np.asarray([gold[index] == label for index in np.flatnonzero(mask)]),
            probabilities[
                mask,
                [
                    list(row["questions"]["intent"]["criteria"]).index(label)
                    for row, keep in zip(rows, mask, strict=True)
                    if keep
                ],
            ],
        )
    available_auc = [value for value in auc_by_intent.values() if value is not None]
    if len(available_auc) != len(labels):
        raise ValueError("R013 macro AUROC is undefined for one or more gold intents")

    target_probabilities = np.asarray(
        [
            probabilities[
                index, list(row["questions"]["intent"]["criteria"]).index(label)
            ]
            for index, (row, label) in enumerate(zip(rows, gold, strict=True))
        ],
        dtype=np.float64,
    )
    confidences = probabilities.max(axis=1)
    one_hot = np.zeros_like(probabilities)
    for index, (row, label) in enumerate(zip(rows, gold, strict=True)):
        option_index = list(row["questions"]["intent"]["criteria"]).index(label)
        one_hot[index, option_index] = 1.0

    ece_value = evals.ece(confidences.tolist(), correct.tolist(), bins=15)
    aurc_value = evals.aurc(confidences.tolist(), correct.tolist())
    selective_50 = evals.selective_accuracy(confidences.tolist(), correct.tolist(), 0.5)
    selective_80 = evals.selective_accuracy(confidences.tolist(), correct.tolist(), 0.8)
    if (
        ece_value is None
        or aurc_value is None
        or selective_50 is None
        or selective_80 is None
    ):
        raise ValueError(
            "R013 confidence metrics are undefined for non-empty test rows"
        )

    return {
        "rows": len(rows),
        "groups": len({_tags(row, "r013-group:") for row in rows}),
        "choice_accuracy": float(correct.mean()),
        "per_intent_recall": per_intent_recall,
        "macro_f1": float(np.mean(list(per_intent_f1.values()))),
        "per_intent_f1": per_intent_f1,
        "macro_ovr_auc": float(np.mean(available_auc)),
        "per_intent_ovr_auc": auc_by_intent,
        "mean_multiclass_nll": float(
            -np.log(np.clip(target_probabilities, 1e-15, 1.0)).mean()
        ),
        "categorical_brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "top_answer_brier": float(
            np.square(confidences - correct.astype(float)).mean()
        ),
        "ece_15_equal_width_bins": float(ece_value),
        "mean_answer_confidence": float(confidences.mean()),
        "aurc": float(aurc_value),
        "selective_accuracy@50": float(selective_50),
        "selective_accuracy@80": float(selective_80),
        "gold_intents": labels,
        "candidate_intents": candidate_labels,
    }


def paired_group_bootstrap(reference_cases, candidate_cases, draws=10000, seed=13):
    if len(reference_cases) != len(candidate_cases) or not reference_cases:
        raise ValueError("R013 paired bootstrap requires aligned non-empty reports")
    grouped_differences = {}
    for reference, candidate in zip(reference_cases, candidate_cases, strict=True):
        if reference.get("expected") != candidate.get("expected") or reference.get(
            "tags"
        ) != candidate.get("tags"):
            raise ValueError("R013 paired bootstrap reports do not share aligned rows")
        group = _tags(reference, "r013-group:")
        grouped_differences.setdefault(group, []).append(
            float(bool(candidate.get("correct")))
            - float(bool(reference.get("correct")))
        )

    group_ids = sorted(grouped_differences)
    if not group_ids:
        raise ValueError("R013 paired bootstrap has no source groups")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        differences = [
            value for group in sampled for value in grouped_differences[str(group)]
        ]
        estimates[draw] = 100.0 * float(np.mean(differences))
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    point = 100.0 * np.mean(
        [value for values in grouped_differences.values() for value in values]
    )
    return {
        "difference_pp": float(point),
        "ci95_pp": [float(lower), float(upper)],
        "draws": draws,
        "seed": seed,
        "groups": len(group_ids),
        "rows": sum(len(values) for values in grouped_differences.values()),
        "quantile_method": "numpy.quantile(method='linear')",
    }


def _expected_report_paths():
    return {
        (condition, model): ARTIFACT_DIR / "evaluation" / condition / f"{model}.json"
        for condition in CONDITIONS
        for model in MODELS
    }


def _verify_report_inference_config(config, key):
    timing = config.get("timing")
    if (
        config.get("device") != "cuda"
        or config.get("autocast_dtype") != "bfloat16"
        or config.get("max_len") != INFERENCE_LIMITS["max_len"]
        or config.get("head_max_len") != INFERENCE_LIMITS["head_max_len"]
        or config.get("laya_cuda_amp") != "bf16"
        or config.get("batch_size") != INFERENCE_BATCH_SIZE
        or not isinstance(timing, dict)
        or timing.get("batch_size") != INFERENCE_BATCH_SIZE
        or timing.get("max_chunk") != 1
    ):
        raise ValueError(f"R013 report inference configuration mismatch: {key}")


def _load_and_validate_reports(
    manifest, manifest_sha256, preparation, training_runs, prepared_rows
):
    index_path = ARTIFACT_DIR / "evaluation.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("R013 evaluation index is missing or unsafe")
    index = read_json(index_path)
    evaluation_manifest_sha256 = index.get("manifest_sha256")
    if (
        index.get("study") != manifest["study"]
        or evaluation_manifest_sha256 != AMENDMENT["predecessor_manifest_sha256"]
        or index.get("preparation_sha256") != sha256(ARTIFACT_DIR / "preparation.json")
    ):
        raise ValueError("R013 evaluation index provenance mismatch")
    evaluation_lock = _artifact_lock_identity(
        manifest, evaluation_manifest_sha256, manifest_sha256
    )
    expected_paths = _expected_report_paths()
    entries = index.get("reports")
    if not isinstance(entries, list) or len(entries) != len(expected_paths):
        raise ValueError("R013 evaluation index does not contain all required reports")
    entry_by_key = {}
    for entry in entries:
        key = (entry.get("condition"), entry.get("model"))
        if key not in expected_paths or key in entry_by_key:
            raise ValueError(
                "R013 evaluation index has an unexpected or repeated report"
            )
        entry_by_key[key] = entry

    payloads = {}
    for key, path in expected_paths.items():
        entry = entry_by_key[key]
        relative = str(path.relative_to(ROOT))
        if (
            entry.get("report") != relative
            or path.is_symlink()
            or not path.is_file()
            or entry.get("report_sha256") != sha256(path)
        ):
            raise ValueError(f"R013 evaluation report inventory mismatch: {key}")
        payload = read_json(path)
        config = payload.get("config", {})
        row_file = SPLIT_FILES[key[0]]
        if key[1] == "base":
            checkpoint = config.get("checkpoint")
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError("R013 base report has no checkpoint identity")
            expected_inventory = manifest["base"]["inference_files_sha256"]
            expected_provenance_sha256 = None
        else:
            arm, seed_text = key[1].rsplit("-s", 1)
            seed = int(seed_text)
            run_key = (seed, arm)
            if seed not in SEEDS or run_key not in training_runs:
                raise ValueError(f"R013 report names an unregistered model: {key[1]}")
            checkpoint = str((ROOT / OUTPUTS[run_key]).resolve())
            expected_inventory = training_runs[run_key]["inventory"]
            expected_provenance_sha256 = sha256(
                ROOT / OUTPUTS[run_key] / "training_provenance.json"
            )
        if (
            config.get("study") != manifest["study"]
            or config.get("condition") != key[0]
            or config.get("manifest_sha256") != evaluation_manifest_sha256
            or config.get("preparation_sha256")
            != sha256(ARTIFACT_DIR / "preparation.json")
            or config.get("prepared_file") != row_file
            or config.get("prepared_file_sha256")
            != preparation["outputs_sha256"][row_file]
            or config.get("model") != key[1]
            or config.get("checkpoint") != checkpoint
            or config.get("inference_files_sha256") != expected_inventory
            or config.get("training_provenance_sha256") != expected_provenance_sha256
            or not isinstance(payload.get("cases"), list)
            or not isinstance(payload.get("r013_logits"), list)
        ):
            raise ValueError(f"R013 report configuration mismatch: {key}")
        _verify_report_inference_config(config, key)
        rows = prepared_rows[key[0]]
        if _validate_test_rows(rows, key[0], preparation) != len(rows):
            raise ValueError(f"R013 prepared groups failed validation: {key[0]}")
        _verify_cases(payload["cases"], rows, key[1], payload["r013_logits"])
        computed = compute_metrics(rows, payload["cases"], payload["r013_logits"])
        accuracy = payload.get("overall", {}).get("choice_accuracy")
        if (
            accuracy is None
            or abs(float(accuracy) - computed["choice_accuracy"]) > 1e-12
        ):
            raise ValueError(f"R013 report aggregate accuracy mismatch: {key}")
        payloads[key] = {"payload": payload, "metrics": computed, "path": relative}
    return (
        payloads,
        sha256(index_path),
        evaluation_manifest_sha256,
        evaluation_lock["protocol_sha256"],
    )


def _format_metrics(value):
    return f"{value:.4f}"


def _render_markdown(result):
    lines = [
        "# R013 results",
        "",
        "Exploratory open-vocabulary transfer to one fixed set of held-out MASSIVE intent option names.",
        "",
        f"Manifest SHA-256: `{result['manifest_sha256']}`",
        f"Protocol SHA-256: `{result['protocol_sha256']}`",
        f"Evaluation manifest SHA-256: `{result['evaluation_manifest_sha256']}`",
        f"Evaluation protocol SHA-256: `{result['evaluation_protocol_sha256']}`",
        f"Preparation SHA-256: `{result['preparation_sha256']}`",
        f"Evaluation SHA-256: `{result['evaluation_sha256']}`",
        "",
        "## Execution record",
        "",
        "The frozen base's saved `max_len=1024` caused the first gated scoring attempt to stop before warmup or a model forward; no prediction or outcome was produced. The inference-only amendment applied `max_len=512` and `head_max_len=256` to every checkpoint. A later probability-validation failure left seven partial reports, preserved as non-results under `artifacts/runs/R013/pre-score-failures/partial-scoring-ac704/`. Final scoring completed under the exact-probability amendment. A post-scoring reporting amendment corrected predecessor-lock and singleton-call validation only; it did not rerun inference or change reports, logits, metrics, bootstrap computation, or support thresholds. Preparation and training artifacts retain their predecessor-lock provenance.",
        "",
        "## Primary comparison",
        "",
        "Seen-only adaptation versus the frozen multilingual base on identical held-out-intent rows.",
        "",
        "| Seed | Rows | Base accuracy | Seen-only accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = result["primary"]["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {row['rows']} | {_format_metrics(row['base_accuracy'])} | "
            f"{_format_metrics(row['seen_only_accuracy'])} | {row['difference_pp']:+.2f} | "
            f"[{row['ci95_pp'][0]:+.2f}, {row['ci95_pp'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            f"Primary support rule met: **{result['primary']['supported']}**. Both seed-specific lower bounds must exceed zero.",
            "",
            "## All-label control",
            "",
            "The matched-budget all-label versus seen-only comparison is descriptive, not a pure causal effect because the training data composition differs.",
            "",
            "| Seed | Seen-only accuracy | All-label accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        row = result["all_label_control"]["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {_format_metrics(row['seen_only_accuracy'])} | "
            f"{_format_metrics(row['all_label_accuracy'])} | {row['difference_pp']:+.2f} | "
            f"[{row['ci95_pp'][0]:+.2f}, {row['ci95_pp'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Descriptive metrics",
            "",
            "Accuracy, macro-F1, macro OvR AUROC, multiclass NLL, categorical Brier, top-answer Brier, ECE, AURC, and selective accuracy are in `results.json`; values remain separated by condition and model.",
            "",
            "The study tests transfer to these exact English option strings, not a population of unseen intents. MASSIVE training data were used in earlier project checkpoints; foundation-model pretraining exposure is unknown. Short intent names may be opaque, only one intent split and two fixed seeds were used, and the dataset is not natural service traffic.",
            "",
            "Intervals resample normalized source-utterance groups and retain all rows. Because the preparation keeps one representative row per normalized group, each test group currently contributes one row.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize():
    manifest, manifest_sha256 = verify_locked_manifest()
    preparation, training_runs, prepared_rows = load_test_rows_after_readiness(
        manifest, manifest_sha256
    )
    (
        payloads,
        evaluation_sha256,
        evaluation_manifest_sha256,
        evaluation_protocol_sha256,
    ) = _load_and_validate_reports(
        manifest, manifest_sha256, preparation, training_runs, prepared_rows
    )

    metrics = {
        condition: {model: payloads[(condition, model)]["metrics"] for model in MODELS}
        for condition in CONDITIONS
    }
    primary = {"per_seed": {}, "supported": True}
    all_label_control = {"per_seed": {}}
    for seed in SEEDS:
        base_cases = payloads[("test-unseen", "base")]["payload"]["cases"]
        seen_cases = payloads[("test-unseen", f"seen-only-s{seed}")]["payload"]["cases"]
        seen_result = paired_group_bootstrap(
            base_cases,
            seen_cases,
            draws=EVALUATION["bootstrap_draws"],
            seed=EVALUATION["bootstrap_seed"],
        )
        base_accuracy = metrics["test-unseen"]["base"]["choice_accuracy"]
        seen_accuracy = metrics["test-unseen"][f"seen-only-s{seed}"]["choice_accuracy"]
        primary["per_seed"][str(seed)] = {
            **seen_result,
            "base_accuracy": base_accuracy,
            "seen_only_accuracy": seen_accuracy,
            "supported": seen_result["ci95_pp"][0] > 0.0,
        }
        primary["supported"] &= seen_result["ci95_pp"][0] > 0.0

        all_cases = payloads[("test-unseen", f"all-label-s{seed}")]["payload"]["cases"]
        all_result = paired_group_bootstrap(
            seen_cases,
            all_cases,
            draws=EVALUATION["bootstrap_draws"],
            seed=EVALUATION["bootstrap_seed"],
        )
        all_label_control["per_seed"][str(seed)] = {
            **all_result,
            "seen_only_accuracy": seen_accuracy,
            "all_label_accuracy": metrics["test-unseen"][f"all-label-s{seed}"][
                "choice_accuracy"
            ],
        }

    result = {
        "schema": "r013-results/1",
        "study": manifest["study"],
        "status": "completed_exploratory",
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"],
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "reporting_amendment": {
            "predecessor_manifest_sha256": AMENDMENT["predecessor_manifest_sha256"],
            "predecessor_protocol_sha256": AMENDMENT["predecessor_protocol_sha256"],
            "scope": "post-scoring report validation only",
            "scored_outputs_unchanged": True,
        },
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_reports": {
            f"{condition}/{model}": {
                "path": payloads[(condition, model)]["path"],
                "sha256": sha256(ROOT / payloads[(condition, model)]["path"]),
            }
            for condition in CONDITIONS
            for model in MODELS
        },
        "training_updates": {
            f"{arm}/s{seed}": {
                "attempts": training_runs[(seed, arm)]["telemetry"][
                    "optimizer_attempts"
                ],
                "successful": training_runs[(seed, arm)]["telemetry"][
                    "successful_updates"
                ],
                "skipped": training_runs[(seed, arm)]["telemetry"]["skipped_updates"],
            }
            for seed in SEEDS
            for arm in ("seen-only", "all-label")
        },
        "primary": {
            "comparison": EVALUATION["primary_comparison"],
            "support_rule": EVALUATION["support_rule"],
            "per_seed": primary["per_seed"],
            "supported": primary["supported"],
        },
        "all_label_control": {
            "interpretation": "descriptive matched-budget contrast; training composition differs",
            "per_seed": all_label_control["per_seed"],
        },
        "metrics": metrics,
        "generated_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "One selected set of 12 MASSIVE intent names and one 20-option construction; no inference to a population of unseen intents.",
            "The source training partition was used by earlier project checkpoints; the foundation model's MASSIVE pretraining exposure is unknown.",
            "The label options are short English identifiers, some of which may be ambiguous or opaque.",
            "Exact normalized duplicates were grouped and reduced to one representative row; paraphrase leakage was not measured.",
            "Two fixed training seeds do not characterize the seed population.",
        ],
    }
    write_json(RESULTS_JSON, result)
    RESULTS_MD.write_text(_render_markdown(result), encoding="utf-8")
    print(f"R013 summarized: primary support={result['primary']['supported']}")
    return result


def main():
    summarize()


if __name__ == "__main__":
    main()
