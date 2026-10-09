"""Summarize locked R007 frozen reports; no training, calibration or predictions.

Usage: uv run python scripts/summarize_r007.py
The historical row-bootstrap summarizer is intentionally not used for R007.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from laya import evals
from r007_identity import case_identity_tag

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r007-unseen-transfer.json"
OUTPUT = ROOT / "research/R007-unseen-transfer/results.json"
METRICS = (
    "choice_accuracy",
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_jsonl(target, split):
    files = target["export_sha256"][split]
    for relative, digest in files.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Frozen export fingerprint mismatch: {relative}")
    data = next(relative for relative in files if relative.endswith(".jsonl"))
    ids = next(relative for relative in files if relative.endswith("-ids.json"))
    rows = [json.loads(line) for line in (ROOT / data).read_text().splitlines()]
    provenance = json.loads((ROOT / ids).read_text())
    if len(rows) != len(provenance) or len(rows) != target["partitions"][split]["rows"]:
        raise ValueError("Export/provenance length mismatch")
    if len({row["source_id"] for row in provenance}) != len(rows):
        raise ValueError("Duplicate source IDs")
    return data, rows, provenance


def paired_group_bootstrap(correct, groups, samples=10000, seed=13):
    """Uniform group draws, retaining every row; row-weighted ratio per resample.

    One shared draw stream for all model columns. Large groups contribute more
    rows, not one equally weighted group mean. Chunking only bounds memory.
    """
    matrix = np.asarray(correct, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(groups) or not len(groups):
        raise ValueError("Expected nonempty rows x models and one group per row")
    unique = sorted(set(groups))
    lookup = {group: i for i, group in enumerate(unique)}
    index = np.array([lookup[group] for group in groups])
    sizes = np.bincount(index, minlength=len(unique))
    totals = np.zeros((len(unique), matrix.shape[1]))
    np.add.at(totals, index, matrix)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for start in range(0, samples, 256):
        draws = rng.integers(0, len(unique), (min(256, samples - start), len(unique)))
        bootstrap.append(totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None])
    return np.concatenate(bootstrap)


def recall_and_f1(golds, predictions, labels):
    totals, guessed, hits = Counter(golds), Counter(predictions), Counter()
    for gold, prediction in zip(golds, predictions):
        hits[gold] += gold == prediction
    recall = {
        label: hits[label] / totals[label] if totals[label] else None
        for label in labels
    }
    f1 = [
        2 * hits[label] / (totals[label] + guessed[label])
        if totals[label] + guessed[label]
        else 0.0
        for label in labels
    ]
    return recall, sum(f1) / len(labels)


def audit_baseline(audit, test):
    counts = Counter(row["expected"]["label"] for row in audit)
    label = min(counts, key=lambda key: (-counts[key], key))
    return {
        "fit_partition": "audit",
        "selected_label": label,
        "audit_label_counts": dict(sorted(counts.items())),
        "test_accuracy": sum(row["expected"]["label"] == label for row in test)
        / len(test),
        "uniform_random_expected_accuracy": sum(
            1 / len(row["questions"]["label"]["criteria"]) for row in test
        )
        / len(test),
    }


def check_report(
    report, model, checkpoint, target, dataset, rows, provenance, manifest_hash
):
    config = report["config"]
    required = {
        "model": model,
        "dataset_sha256": sha256(ROOT / dataset),
        "frozen_manifest_sha256": manifest_hash,
        "weights_sha256": checkpoint["sha256"]["model.safetensors"],
        "budget_override": {key: target[key] for key in ("max_len", "head_max_len")},
        "measurement_protocol": "isolated-warm-v1",
        "coverage_metric_definition": evals.COVERAGE_METRIC_DEFINITION,
        "questions_sha256": evals.questions_fingerprint(
            evals.Dataset.from_jsonl(str(ROOT / dataset))
        ),
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f"Frozen report identity mismatch: {model}")
    if Path(config["dataset"]).resolve() != (ROOT / dataset).resolve():
        raise ValueError(f"Frozen report dataset mismatch: {model}")
    cases = report["cases"]
    if len(cases) != len(rows):
        raise ValueError(f"Report row count mismatch: {model}")
    for case, row, metadata in zip(cases, rows, provenance):
        answer = case["answer"]
        probabilities = answer["probabilities"]
        gold = row["expected"]["label"]
        identity_tag = case_identity_tag(row, metadata)
        if (
            case["qid"] != "label"
            or case["model"] != model
            or case["expected"] != gold
            or case.get("tags") != [identity_tag]
            or answer["type"] != "choice"
            or set(probabilities) != set(row["questions"]["label"]["criteria"])
            or answer["choice"] not in probabilities
            or case["correct"] != (answer["choice"] == gold)
            or case["confidence"] != answer["answer_confidence"]
            or not math.isfinite(case["confidence"])
            or not 0 <= case["confidence"] <= 1
            or any(
                not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
            )
            or abs(math.fsum(probabilities.values()) - 1.0)
            > len(probabilities) * 0.00005 + 1e-12
            or max(probabilities.values()) - probabilities[answer["choice"]]
            > 0.0001 + 1e-12
            or abs(case["confidence"] - max(probabilities.values())) > 0.0001 + 1e-12
        ):
            raise ValueError(f"Report case alignment/answer mismatch: {model}")
    confidence = [case["confidence"] for case in cases]
    correct = [case["correct"] for case in cases]
    computed = {
        "choice_accuracy": sum(correct) / len(correct),
        "ece": evals.ece(confidence, correct, bins=15),
        "brier": evals.brier(confidence, correct),
        "aurc": evals.aurc(confidence, correct),
        "selective_accuracy@50": evals.selective_accuracy(confidence, correct, 0.5),
        "selective_accuracy@80": evals.selective_accuracy(confidence, correct, 0.8),
    }
    if any(
        value is None or not math.isclose(report["overall"][key], value, abs_tol=1e-10)
        for key, value in computed.items()
    ):
        raise ValueError(f"Report metric mismatch: {model}")
    return cases


def summarize_target(name, target, checkpoints, manifest_hash):
    dataset, rows, ids = verified_jsonl(target, "test")
    _, audit, audit_ids = verified_jsonl(target, "audit")
    groups = [row["group_id"] for row in ids]
    if set(groups) & {row["group_id"] for row in audit_ids}:
        raise ValueError("Audit/test group intersection")
    if set(target["audit_source_ids"]) & {row["source_id"] for row in ids}:
        raise ValueError("Persisted audit source in test")
    models = list(checkpoints)
    golds = [row["expected"]["label"] for row in rows]
    labels = list("01234" if name == "asag" else "ABCD")
    entries, columns = {}, []
    for model, checkpoint in checkpoints.items():
        path = ROOT / f"artifacts/runs/r007-{name}-frozen/{model}.json"
        report = json.loads(path.read_text())
        cases = check_report(
            report, model, checkpoint, target, dataset, rows, ids, manifest_hash
        )
        predictions = [case["answer"]["choice"] for case in cases]
        columns.append([int(case["correct"]) for case in cases])
        recall, macro_f1 = recall_and_f1(golds, predictions, labels)
        entry = {key: report["overall"][key] for key in METRICS}
        entry["recall_by_grade" if name == "asag" else "recall_by_answer_position"] = (
            recall
        )
        if name == "asag":
            entry["macro_f1"] = macro_f1
        confidences = np.array([case["confidence"] for case in cases])
        entry["actual_selective_coverage"] = {
            str(coverage): float(
                np.mean(
                    confidences
                    >= np.sort(confidences)[-math.ceil(coverage * len(cases))]
                )
            )
            for coverage in (0.5, 0.8)
        }
        entry["report"] = str(path.relative_to(ROOT))
        entry["report_sha256"] = sha256(path)
        entry["run_config"] = report["config"]
        entries[model] = entry
    matrix = np.asarray(columns, dtype=float).T
    primary = name == "mkqa-entity-choice"
    bootstrap = paired_group_bootstrap(matrix, groups) if primary else None
    differences = {}
    pairs = [("full-e2-8bit", "mt-s13"), ("full-e2-8bit", "mt-s14")]
    pairs += [("control-base-maxlen512", model) for model in models[1:]]
    for a, b in pairs:
        i, j = models.index(a), models.index(b)
        entry = {"difference_pp": float(100 * (matrix[:, j] - matrix[:, i]).mean())}
        if bootstrap is not None:
            entry["ci95_pp"] = (
                100 * np.quantile(bootstrap[:, j] - bootstrap[:, i], [0.025, 0.975])
            ).tolist()
        differences[f"{b} - {a}"] = entry
    return {
        "scope": "exploratory primary"
        if primary
        else "descriptive diagnostic; crossed students, no inferential CIs",
        "test_rows": len(rows),
        "test_groups": len(set(groups)),
        "dataset_sha256": sha256(ROOT / dataset),
        "baseline": audit_baseline(audit, rows),
        "models": entries,
        "accuracy_differences": differences,
        "h1_positive_pattern": bool(
            all(
                np.quantile(
                    bootstrap[:, models.index(model)]
                    - bootstrap[:, models.index("full-e2-8bit")],
                    0.025,
                )
                > 0
                for model in ("mt-s13", "mt-s14")
            )
        )
        if bootstrap is not None
        else None,
    }


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "locked_exploratory":
        raise ValueError("R007 analysis requires a locked manifest")
    manifest_hash = sha256(MANIFEST)
    results = {
        "manifest_sha256": manifest_hash,
        "analysis": {
            "method": "paired uniform-group bootstrap; row-weighted accuracy ratio",
            "samples": 10000,
            "seed": 13,
            "interval": "percentile 2.5/97.5",
            "ece_bins": 15,
            "brier": "top-answer confidence versus correctness",
            "asag_macro_f1_labels": list("01234"),
            "baseline_tie_break": "lexicographically first audit maximum-frequency label",
        },
        "targets": {
            name: summarize_target(
                name, target, manifest["checkpoint_fingerprints"], manifest_hash
            )
            for name, target in manifest["targets"].items()
        },
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "accuracy": {
                        model: entry["choice_accuracy"]
                        for model, entry in target["models"].items()
                    },
                    "differences": target["accuracy_differences"],
                    "h1_positive_pattern": target["h1_positive_pattern"],
                }
                for name, target in results["targets"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
