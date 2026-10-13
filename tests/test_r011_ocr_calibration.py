import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r011_identity")
evaluator = importlib.import_module("evaluate_r011")
summarizer = importlib.import_module("summarize_r011")
inference = importlib.import_module("r011_inference")


def _row(text, label):
    return {"state": {"utterance": text}, "expected": {"intent": label}}


def test_calibration_selection_excludes_overlap_conflicts_and_duplicate_groups():
    rows = [
        _row("eligible", "a"),
        _row(" eligible ", "a"),
        _row("ambiguous", "a"),
        _row("ambiguous", "b"),
        _row("train copy", "a"),
        _row("test copy", "b"),
    ]
    selected, audit = identity.calibration_selection(
        rows,
        {identity.normalize_input("train copy")},
        {identity.normalize_input("test copy")},
    )
    assert [index for index, _row_value in selected] == [0]
    assert audit == {
        "validation_rows": 6,
        "validation_normalized_groups": 4,
        "validation_conflicting_label_groups": 1,
        "validation_conflicting_label_rows": 2,
        "validation_train_overlap_groups": 1,
        "validation_train_overlap_rows": 1,
        "validation_test_overlap_groups": 1,
        "validation_test_overlap_rows": 1,
        "validation_excluded_group_union": 3,
        "eligible_calibration_groups": 1,
        "eligible_calibration_rows": 1,
    }


def test_temperature_metrics_preserve_accuracy_and_match_laya_confidence_metrics():
    rows = [
        {
            "logits": [4.0, 0.0, -1.0],
            "target_index": 0,
            "expected": "a",
            "baseline_prediction": "a",
        },
        {
            "logits": [3.0, 2.0, 0.0],
            "target_index": 1,
            "expected": "b",
            "baseline_prediction": "a",
        },
    ]
    baseline = summarizer._metrics(rows, 1.0)
    softer = summarizer._metrics(rows, 2.0)
    assert baseline["choice_accuracy"] == softer["choice_accuracy"] == 0.5
    assert baseline["mean_choice_nll"] != softer["mean_choice_nll"]
    assert baseline["mean_answer_confidence"] > softer["mean_answer_confidence"]
    assert baseline["top_answer_brier"] != softer["top_answer_brier"]


def test_metrics_reject_inconsistent_saved_predictions():
    rows = [
        {
            "logits": [2.0, 1.0],
            "target_index": 0,
            "expected": "a",
            "baseline_prediction": "b",
        }
    ]
    with pytest.raises(ValueError, match="stored correctness"):
        summarizer._metrics(rows, 1.0)


def test_group_bootstrap_is_row_weighted_paired_and_deterministic(monkeypatch):
    monkeypatch.setattr(summarizer, "DRAWS", 200)
    differences = [0.2, 0.2, -0.1]
    groups = ["duplicate", "duplicate", "singleton"]
    first = summarizer._paired_group_bootstrap(differences, groups)
    second = summarizer._paired_group_bootstrap(differences, groups)
    assert first == second
    assert first["mean_nll_difference_ocr_fit_minus_existing"] == pytest.approx(0.1)
    assert first["groups"] == 2
    assert first["draws"] == 200
    assert first["seed"] == 13


def test_group_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="different lengths or are empty"):
        summarizer._paired_group_bootstrap([0.1], [])


def test_logit_capture_matches_laya_question_schema_batches():
    class Model(torch.nn.Module):
        def forward(self, values):
            logits = torch.zeros((len(values), 20))
            logits[:, 0] = values
            logits[:, 1] = -values
            return logits, torch.zeros((len(values), 1))

    class Agent:
        def __init__(self):
            self.model = Model()
            self.calls = []

        def _predict(self, states, questions):
            values = torch.tensor([state["score"] for state in states])
            logits, _ = self.model(values)
            labels = list(questions["intent"]["criteria"])
            return [
                {
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": labels[int(vector.argmax())],
                        }
                    }
                }
                for vector in logits
            ]

        def predict(self, state, questions):
            self.calls.append(("predict", 1))
            return self._predict([state], questions)[0]

        def predict_batch(
            self, states, questions, batch_size=None, sort_by_length=False
        ):
            self.calls.append(("batch", len(states), batch_size, sort_by_length))
            return self._predict(states, questions)

    labels = [f"intent-{index}" for index in range(20)]
    reversed_labels = list(reversed(labels))
    base_question = {"intent": {"type": "choice", "criteria": dict.fromkeys(labels)}}
    other_question = {
        "intent": {"type": "choice", "criteria": dict.fromkeys(reversed_labels)}
    }
    rows = [
        {
            "state": {"score": score},
            "questions": base_question,
            "expected": {"intent": labels[0]},
        }
        for score in range(1, 11)
    ]
    rows.append(
        {
            "state": {"score": -1},
            "questions": other_question,
            "expected": {"intent": reversed_labels[1]},
        }
    )
    agent = Agent()

    logits, cases = inference.capture_laya_logits(agent, rows, "mt-s13")

    expected_logits = [[0.0] * 20 for _ in rows]
    for index, score in enumerate(range(1, 11)):
        expected_logits[index][0] = float(score)
        expected_logits[index][1] = float(-score)
    expected_logits[-1][0] = -1.0
    expected_logits[-1][1] = 1.0
    assert logits.tolist() == expected_logits
    assert agent.calls == [
        ("batch", 8, 8, False),
        ("batch", 2, 8, False),
        ("predict", 1),
    ]
    assert [case["answer"]["choice"] for case in cases] == [
        *([labels[0]] * 10),
        reversed_labels[1],
    ]


def test_baseline_case_alignment_uses_each_rows_option_order():
    rows = []
    cases = []
    for index, prefix in enumerate(("first", "second")):
        labels = [f"{prefix}{choice}" for choice in range(20)]
        expected = labels[1]
        prediction = labels[1] if index == 0 else labels[0]
        tags = [f"r010-row:row-{index}", f"r010-group:group-{index}"]
        rows.append(
            {
                "questions": {"intent": {"criteria": labels}},
                "expected": {"intent": expected},
                "tags": tags,
            }
        )
        cases.append(
            {
                "expected": expected,
                "answer": {"choice": prediction},
                "correct": prediction == expected,
                "model": "mt-s13",
                "qid": "intent",
                "tags": tags,
            }
        )

    evaluator._validate_baseline_cases(rows, cases, "mt-s13")


def test_evaluation_validation_authenticates_target_index(monkeypatch, tmp_path):
    labels = [f"c{index}" for index in range(20)]
    test_row = {
        "questions": {"intent": {"criteria": labels}},
        "expected": {"intent": "c1"},
        "tags": [
            "r010-row:row-0",
            "r010-group:group-0",
            "r010-train-overlap:0",
        ],
    }
    logits = [4.0, 3.0] + [0.0] * 18
    reports = {}
    records: dict[str, dict[str, Any]] = {}
    for model_name in summarizer.MODELS:
        report_record = {
            "report": f"{model_name}.json",
            "sha256": f"{model_name}-hash",
            "choice_accuracy": 0.0,
        }
        reports[model_name] = (
            {
                "cases": [
                    {
                        "expected": "c1",
                        "answer": {"choice": "c0"},
                        "model": model_name,
                        "qid": "intent",
                    }
                ]
            },
            report_record,
        )
        records[model_name] = {
            "baseline_report": report_record["report"],
            "baseline_report_sha256": report_record["sha256"],
            "baseline_accuracy": report_record["choice_accuracy"],
            "existing_temperature": [1.0, 1.0, 1.0],
            "ocr_temperature": [2.0, 1.0, 1.0],
            "calibration_sha256": f"{model_name}-calibration-hash",
            "rows": [
                {
                    "row_index": 0,
                    "row_id": "row-0",
                    "group_id": "group-0",
                    "train_overlap": False,
                    "expected": "c1",
                    "baseline_prediction": "c0",
                    "target_index": 1,
                    "logits": logits,
                }
            ],
        }

    manifest = {
        "study": "R011-ocr-domain-calibration",
        "source": {"test_rows": 1},
        "recipe": {"evaluation_batch_size": 8},
        "evaluation": {
            "inference_grouping": "Laya evals.evaluate: consecutive identical question schemas, singleton predict, batch_size=8"
        },
    }
    ready = {
        model_name: {
            "metadata": {
                "existing_temperature": [1.0, 1.0, 1.0],
                "calibration_sha256": f"{model_name}-calibration-hash",
            },
            "calibration": {"temperature": [2.0, 1.0, 1.0]},
        }
        for model_name in summarizer.MODELS
    }
    preparation = {
        "outputs_sha256": {identity.CALIBRATION_DATA: "calibration-data-hash"}
    }
    evaluation = {
        "study": manifest["study"],
        "manifest_sha256": "manifest-hash",
        "r010_manifest_sha256": "r010-manifest-hash",
        "preparation_sha256": "preparation-hash",
        "r010_preparation_sha256": "preparation-hash",
        "calibration_data_sha256": "calibration-data-hash",
        "inference_batch_size": 8,
        "inference_grouping": manifest["evaluation"]["inference_grouping"],
        "models": records,
    }
    monkeypatch.setattr(summarizer, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)
    monkeypatch.setattr(summarizer, "read_json", lambda _path: evaluation)
    monkeypatch.setattr(summarizer, "sha256", lambda _path: "preparation-hash")
    monkeypatch.setattr(summarizer, "read_jsonl", lambda _path: [test_row])
    monkeypatch.setattr(
        summarizer,
        "verify_r010_prepared",
        lambda *_args: {"outputs": {evaluator.CONDITION: {"file": "test.jsonl"}}},
    )
    monkeypatch.setattr(
        summarizer, "_baseline_report", lambda *_args: reports[_args[-1]]
    )

    arguments = (
        manifest,
        "manifest-hash",
        {},
        "r010-manifest-hash",
        ready,
        preparation,
    )
    summarizer._validate_evaluation(*arguments)

    records["mt-s13"]["rows"][0]["target_index"] = 2
    with pytest.raises(ValueError, match="scored row integrity mismatch"):
        summarizer._validate_evaluation(*arguments)
