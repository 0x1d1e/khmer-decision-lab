import importlib
import json
import sys
from pathlib import Path

import pytest
from laya import evals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
summarizer = importlib.import_module("summarize_r008")
evaluator = importlib.import_module("evaluate_r008")
identity = importlib.import_module("r008_identity")


def _rows():
    return [
        {
            "state": {"Question": f"item {index}"},
            "questions": {"topic": {"criteria": {"A": "a", "B": "b", "C": "c"}}},
            "expected": {"topic": "A"},
            "tags": [f"source-{index}"],
            "language": "km",
        }
        for index in range(3)
    ]


def _report(rows, model="multitask-s13"):
    cases = []
    confidence = []
    correct = []
    for index, row in enumerate(rows):
        gold = row["expected"]["topic"]
        choice = "A" if index != 1 else "B"
        probabilities = {"A": 0.7, "B": 0.2, "C": 0.1}
        if choice == "B":
            probabilities = {"A": 0.2, "B": 0.7, "C": 0.1}
        is_correct = choice == gold
        confidence.append(0.7)
        correct.append(is_correct)
        cases.append(
            {
                "qid": "topic",
                "model": model,
                "expected": gold,
                "tags": [*row["tags"], identity.row_identity_tag(row)],
                "answer": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "answer_confidence": 0.7,
                },
                "confidence": 0.7,
                "correct": is_correct,
            }
        )
    return {
        "cases": cases,
        "overall": summarizer.expected_metrics(confidence, correct),
    }


def test_report_cases_accepts_aligned_rows_and_recomputes_metrics():
    rows = _rows()
    report = _report(rows)

    confidence, correct, golds, predictions = summarizer.check_report_cases(
        report, "multitask-s13", rows
    )

    assert confidence == [0.7, 0.7, 0.7]
    assert correct == [True, False, True]
    assert golds == ["A", "A", "A"]
    assert predictions == ["A", "B", "A"]


def test_report_cases_rejects_same_gold_rows_reordered():
    rows = _rows()
    report = _report(rows)
    report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]

    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        summarizer.check_report_cases(report, "multitask-s13", rows)


def test_report_cases_rejects_contradictory_probabilities():
    rows = _rows()
    report = _report(rows)
    report["cases"][0]["answer"]["probabilities"]["C"] = 0.5

    with pytest.raises(ValueError, match="alignment/answer mismatch"):
        summarizer.check_report_cases(report, "multitask-s13", rows)


class _FakeAgent:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    @staticmethod
    def _response(questions):
        return {
            "answers": {
                qid: {
                    "type": "choice",
                    "choice": "A",
                    "probabilities": {"A": 0.7, "B": 0.2, "C": 0.1},
                    "answer_confidence": 0.7,
                }
                for qid in questions
            }
        }

    def predict(self, _state, questions, **_kwargs):
        self.single_calls += 1
        return self._response(questions)

    def predict_batch(self, states, questions, **_kwargs):
        self.batch_calls += 1
        return [self._response(questions) for _ in states]


def test_laya_evaluator_adapter_handles_qid_model_and_singleton_tail(tmp_path):
    rows = _rows()
    path = tmp_path / "r008-eval.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset = evals.Dataset.from_jsonl(str(path))
    evaluator.bind_dataset_rows(dataset, rows, "multitask-s13")
    agent = _FakeAgent()

    report = evaluator.evaluate_agent(
        agent, dataset, "multitask-s13", {}, batch_size=2
    ).to_json()

    assert agent.batch_calls == 1
    assert agent.single_calls == 1
    assert [case["model"] for case in report["cases"]] == ["multitask-s13"] * len(rows)
    assert [case["qid"] for case in report["cases"]] == ["topic"] * len(rows)
    _confidence, correct, golds, predictions = summarizer.check_report_cases(
        report, "multitask-s13", rows
    )
    assert correct == [True, True, True]
    assert golds == predictions == ["A"] * len(rows)


@pytest.mark.parametrize(
    "readiness_failure", ["checkpoint", "calibration", "invalid_calibration"]
)
def test_score_requires_study_wide_readiness_before_opening_test(
    tmp_path, monkeypatch, readiness_failure
):
    seeds = [13, 14]
    manifest = {
        "recipe": {"seeds": seeds},
        "training_split": {"arms": {arm: {} for arm in identity.ARM_TASKS}},
        "validation_calibration": {
            task: {
                "test_file": f"data/processed/{task}-test.jsonl",
                "test_sha256": "test-hash",
                "test_rows": 1,
            }
            for task in evaluator.TASKS
        },
    }
    monkeypatch.setattr(
        evaluator,
        "verify_locked_manifest",
        lambda _path: (manifest, "manifest-hash"),
    )
    checkpoint_calls = []
    calibration_calls = []
    missing_checkpoint = "R008-sib-s14"
    missing_task = "sib200-km"

    def fake_checkpoint_provenance(checkpoint, _task, _manifest, _manifest_hash):
        checkpoint_name = Path(checkpoint).name
        checkpoint_calls.append(checkpoint_name)
        if readiness_failure == "checkpoint" and checkpoint_name == missing_checkpoint:
            raise FileNotFoundError("missing sibling checkpoint")
        for arm, prefix in identity.ARM_OUTPUT_PREFIXES.items():
            if checkpoint_name.startswith(prefix):
                return {
                    "training_arm": arm,
                    "seed": int(checkpoint_name.rsplit("-s", 1)[1]),
                }
        raise AssertionError(f"unexpected checkpoint path: {checkpoint}")

    def fake_calibration(checkpoint, task, _provenance, _manifest, _manifest_hash):
        artifact = (Path(checkpoint).name, task)
        calibration_calls.append(artifact)
        if readiness_failure in ("calibration", "invalid_calibration") and artifact == (
            missing_checkpoint,
            missing_task,
        ):
            if readiness_failure == "calibration":
                raise ValueError("missing sibling calibration")
            raise ValueError("invalid sibling Laya calibration payload")
        return Path(checkpoint) / f"calibration-{task}.json", f"{task}-hash"

    monkeypatch.setattr(
        identity,
        "verify_checkpoint_training_provenance",
        fake_checkpoint_provenance,
    )
    monkeypatch.setattr(identity, "verify_calibration_artifact", fake_calibration)
    test_loads = []
    monkeypatch.setattr(
        evals.Dataset,
        "from_jsonl",
        lambda path: test_loads.append(path),
    )

    expected_error = (
        FileNotFoundError if readiness_failure == "checkpoint" else ValueError
    )
    expected_message = (
        "missing sibling"
        if readiness_failure != "invalid_calibration"
        else "invalid sibling"
    )
    with pytest.raises(expected_error, match=expected_message):
        evaluator.score(
            evaluator.ROOT / "artifacts/checkpoints/R008-mt-s13",
            "khmer-news",
            tmp_path / "manifest.json",
        )

    assert not test_loads
    assert len(checkpoint_calls) == 6
    assert len(calibration_calls) == (7 if readiness_failure == "checkpoint" else 8)


def test_report_verification_rejects_wrong_inference_dtype(monkeypatch):
    rows = _rows()
    report = _report(rows)
    report["overall"].update(
        {
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
            "cost_per_decision_p50_ms": 1.0,
            "cost_per_decision_p95_ms": 2.0,
        }
    )
    task_data = {
        "test_file": "data/processed/khmer-news-test.jsonl",
        "test_sha256": "test-hash",
        "test_rows": len(rows),
    }
    provenance = {
        "weights_sha256": "weights-hash",
        "training_file_sha256": "train-hash",
    }
    manifest = {
        "recipe": {
            "base_revision": "base-revision",
            "max_len": 512,
            "head_max_len": 256,
        },
        "inference": {"device": "cuda", "autocast_dtype": "bfloat16"},
        "environment": {"laya": "0.4.1"},
    }
    report["config"] = {
        "study": "R008-specialist-controls",
        "task": "khmer-news",
        "model": "multitask-s13",
        "training_arm": "multitask",
        "seed": 13,
        "dataset": task_data["test_file"],
        "dataset_sha256": task_data["test_sha256"],
        "locked_manifest_sha256": "manifest-hash",
        "weights_sha256": provenance["weights_sha256"],
        "calibration_sha256": "calibration-hash",
        "training_provenance_sha256": "provenance-hash",
        "training_file_sha256": provenance["training_file_sha256"],
        "base_revision": "base-revision",
        "max_len": 512,
        "head_max_len": 256,
        "device": "cuda",
        "inference_dtype": "float16",
        "laya_version": "0.4.1",
    }
    monkeypatch.setattr(summarizer, "sha256", lambda _path: "provenance-hash")

    with pytest.raises(ValueError, match="frozen report identity mismatch"):
        summarizer.verify_report(
            report,
            "multitask-s13",
            "khmer-news",
            "multitask",
            13,
            task_data,
            provenance,
            "calibration-hash",
            "manifest-hash",
            manifest,
            rows,
        )


def test_paired_group_bootstrap_is_reproducible_and_paired():
    groups = ["duplicate", "duplicate", "singleton-a", "singleton-b"]
    multitask = [True, True, True, False]
    specialist = [False, False, True, True]

    first = summarizer.paired_group_bootstrap(
        multitask, specialist, groups, samples=500, seed=13
    )
    second = summarizer.paired_group_bootstrap(
        multitask, specialist, groups, samples=500, seed=13
    )

    assert first.shape == (500,)
    assert (first == second).all()
    assert first.min() < 0 < first.max()
    assert first.mean() > 0
