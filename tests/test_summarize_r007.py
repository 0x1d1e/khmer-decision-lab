"""R007 analysis invariants, using synthetic rows and recorded-answer fixtures."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from laya import evals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r007_identity")
SPEC = importlib.util.spec_from_file_location(
    "summarize_r007", ROOT / "scripts/summarize_r007.py"
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_bootstrap_samples_groups_not_rows_with_shared_model_draws():
    # Unequal group sizes distinguish a row-weighted ratio from group means.
    matrix = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    groups = ["large", "large", "small"]
    actual = summary.paired_group_bootstrap(matrix, groups, samples=1000, seed=13)
    draws = np.random.default_rng(13).integers(0, 2, (1000, 2))
    expected = []
    for draw in draws:
        selected = [row for group in draw for row in ([0, 1] if group == 0 else [2])]
        expected.append(matrix[selected].mean(axis=0))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert set(np.round(actual[:, 0], 6)) == {0.0, 0.666667, 1.0}
    assert any(actual[:, 0] == 2 / 3)
    np.testing.assert_array_equal(
        actual, summary.paired_group_bootstrap(matrix, groups, samples=1000)
    )
    np.testing.assert_array_equal(
        actual, summary.paired_group_bootstrap(matrix[::-1], groups[::-1], samples=1000)
    )


def test_audit_fitted_baseline_does_not_choose_test_majority():
    audit = [{"expected": {"label": label}} for label in ("B", "A", "B", "A")]
    test = [
        {
            "expected": {"label": label},
            "questions": {"label": {"criteria": dict.fromkeys("ABCD", "name")}},
        }
        for label in ("D", "D", "A")
    ]
    baseline = summary.audit_baseline(audit, test)
    assert baseline["selected_label"] == "A"
    assert baseline["test_accuracy"] == 1 / 3
    assert baseline["uniform_random_expected_accuracy"] == 0.25


def test_grade_metrics_include_all_five_ordinal_labels():
    recall, f1 = summary.recall_and_f1(["0", "0", "1"], ["0", "1", "1"], list("01234"))
    assert recall == {"0": 0.5, "1": 1.0, "2": None, "3": None, "4": None}
    assert f1 == pytest.approx((2 / 3 + 2 / 3) / 5)


@pytest.fixture
def aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    rows = [
        {
            "state": {"question": f"synthetic {i}"},
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"A": "Alpha", "B": "Beta"},
                }
            },
            "expected": {"label": gold},
        }
        for i, gold in enumerate(("A", "B", "A"))
    ]
    dataset = "synthetic.jsonl"
    (tmp_path / dataset).write_text("".join(json.dumps(row) + "\n" for row in rows))

    class RecordedAnswers:
        def predict(self, state, questions, model=None):
            assert state and questions and model == "synthetic"
            return {
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": "A",
                        "probabilities": {"A": 0.8, "B": 0.2},
                        "answer_confidence": 0.8,
                    }
                }
            }

    provenance = [
        {"source_id": f"source:{i}", "group_id": f"group:{i}"} for i in range(len(rows))
    ]
    data = evals.Dataset.from_jsonl(str(tmp_path / dataset))
    identity.attach_case_identity_tags(data, rows, provenance)
    for example in data.examples:
        example.model = "synthetic"
    target = {"max_len": 512, "head_max_len": 256}
    checkpoint = {"sha256": {"model.safetensors": "synthetic-weights-hash"}}
    report = evals.evaluate(
        RecordedAnswers(),
        data,
        config={
            "dataset": str(tmp_path / dataset),
            "dataset_sha256": summary.sha256(tmp_path / dataset),
            "model": "synthetic",
            "weights_sha256": "synthetic-weights-hash",
            "frozen_manifest_sha256": "frozen-hash",
            "budget_override": target,
            "measurement_protocol": "isolated-warm-v1",
        },
    ).to_json()
    return report, checkpoint, target, dataset, rows, provenance


def check(aligned):
    report, checkpoint, target, dataset, rows, provenance = aligned
    return summary.check_report(
        report,
        "synthetic",
        checkpoint,
        target,
        dataset,
        rows,
        provenance,
        "frozen-hash",
    )


def refresh_calibration_metrics(report):
    cases = report["cases"]
    confidence = [case["confidence"] for case in cases]
    correct = [case["correct"] for case in cases]
    report["overall"]["ece"] = evals.ece(confidence, correct, bins=15)
    report["overall"]["brier"] = evals.brier(confidence, correct)
    report["overall"]["aurc"] = evals.aurc(confidence, correct)
    for coverage in (0.5, 0.8):
        key = f"selective_accuracy@{round(coverage * 100)}"
        report["overall"][key] = evals.selective_accuracy(confidence, correct, coverage)


def test_aligned_report_passes_with_tied_confidence(aligned):
    cases = check(aligned)
    assert len(cases) == 3
    assert aligned[0]["overall"]["selective_accuracy@50"] == 2 / 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("frozen_manifest_sha256", "changed"),
        ("weights_sha256", "changed"),
        ("dataset_sha256", "changed"),
        ("budget_override", {"max_len": 2048, "head_max_len": 256}),
        ("coverage_metric_definition", -1),
    ],
)
def test_report_identity_changes_rejected(aligned, key, value):
    aligned[0]["config"][key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        check(aligned)


def test_same_gold_different_group_case_reordering_rejected(aligned):
    # Rows 0 and 2 share a gold label and option schema but come from different groups.
    aligned[0]["cases"][0], aligned[0]["cases"][2] = (
        aligned[0]["cases"][2],
        aligned[0]["cases"][0],
    )
    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        check(aligned)


def test_non_normalized_probability_payload_rejected(aligned):
    aligned[0]["cases"][0]["answer"]["probabilities"] = {"A": 0.8, "B": 0.8}
    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        check(aligned)


def test_non_argmax_selected_choice_rejected(aligned):
    case = aligned[0]["cases"][1]
    case["answer"]["choice"] = "B"
    case["answer"]["probabilities"] = {"A": 0.8, "B": 0.2}
    case["answer"]["answer_confidence"] = 0.8
    case["confidence"] = 0.8
    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        check(aligned)


def test_confidence_must_match_probability_maximum(aligned):
    case = aligned[0]["cases"][0]
    case["answer"]["answer_confidence"] = 0.5
    case["confidence"] = 0.5
    refresh_calibration_metrics(aligned[0])
    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        check(aligned)


def test_rounded_probability_sum_and_maximum_tie_are_valid(aligned):
    case = aligned[0]["cases"][0]
    case["answer"]["probabilities"] = {"A": 0.5001, "B": 0.5}
    case["answer"]["answer_confidence"] = 0.5001
    case["confidence"] = 0.5001
    refresh_calibration_metrics(aligned[0])
    assert len(check(aligned)) == 3
    case["answer"]["probabilities"] = {"A": 0.5, "B": 0.5}
    case["answer"]["answer_confidence"] = 0.5
    case["confidence"] = 0.5
    refresh_calibration_metrics(aligned[0])
    assert len(check(aligned)) == 3


def test_tampered_metrics_rejected(aligned):
    aligned[0]["overall"]["choice_accuracy"] = 1.0
    with pytest.raises(ValueError, match="metric mismatch"):
        check(aligned)
