import importlib
import json
import sys
from pathlib import Path

import pytest
from laya import evals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
evaluator = importlib.import_module("evaluate_r009")
calibrator = importlib.import_module("calibrate_r009")
identity = importlib.import_module("r009_identity")


def test_macro_ovr_auc_handles_perfect_ranking():
    auc = evaluator.macro_ovr_auc(
        ["a", "b", "a", "b"],
        [
            {"a": 0.9, "b": 0.1},
            {"a": 0.2, "b": 0.8},
            {"a": 0.7, "b": 0.3},
            {"a": 0.4, "b": 0.6},
        ],
        ["a", "b"],
    )
    assert auc == 1.0


def test_macro_ovr_auc_scores_ties_as_half():
    auc = evaluator.macro_ovr_auc(
        ["a", "b"], [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}], ["a", "b"]
    )
    assert auc == 0.5


def test_classification_metrics_compute_macro_f1_and_auc():
    confidence = [0.8, 0.7]
    correct = [True, False]
    report = {
        "overall": {
            "choice_accuracy": sum(correct) / len(correct),
            "mean_confidence": sum(confidence) / len(confidence),
            "ece": evals.ece(confidence, correct, bins=15),
            "brier": evals.brier(confidence, correct),
            "aurc": evals.aurc(confidence, correct),
            "selective_accuracy@50": evals.selective_accuracy(confidence, correct, 0.5),
            "selective_accuracy@80": evals.selective_accuracy(confidence, correct, 0.8),
        }
    }
    metrics = evaluator.classification_metrics(
        report,
        confidence,
        correct,
        ["a", "b"],
        ["a", "a"],
        [{"a": 0.8, "b": 0.2}, {"a": 0.7, "b": 0.3}],
        ["a", "b"],
    )
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["macro_ovr_auc"] == 1.0


def calibration_case(tmp_path, seed=13):
    from laya.calibrate import CALIBRATION_VERSION, calibration_payload

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    config = {
        "max_len": 512,
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
    }
    (checkpoint / "rl_agent_config.json").write_text(json.dumps(config))
    (checkpoint / "training_provenance.json").write_text("{}\n")
    provenance = {"weights_sha256": "weights-sha256"}
    manifest = {
        "source": {
            "validation_file": "data/processed/validation.jsonl",
            "validation_sha256": "validation-sha256",
            "validation_rows": 99,
        },
        "inference": {"device": "cuda", "autocast_dtype": "bfloat16"},
    }
    r009_hash = "r009-manifest-sha256"
    r008_hash = "r008-manifest-sha256"
    payload = calibration_payload(
        [1.0, 1.0, 1.0],
        {},
        model_id_or_path=str(checkpoint.resolve()),
        subfolder=None,
        config=config,
    )
    assert payload["version"] == CALIBRATION_VERSION
    payload["r009"] = {
        "study": "R009-sib-specialist-continuation",
        "seed": seed,
        "checkpoint": f"artifacts/checkpoints/R009-sib-s{seed}-cont",
        "checkpoint_weights_sha256": provenance["weights_sha256"],
        "training_provenance_sha256": identity.sha256(
            checkpoint / "training_provenance.json"
        ),
        "validation_file": manifest["source"]["validation_file"],
        "validation_sha256": manifest["source"]["validation_sha256"],
        "validation_rows": 99,
        "usable_validation_items": 99,
        "inference_device": "cuda",
        "inference_dtype": "bfloat16",
        "r009_manifest_sha256": r009_hash,
        "r008_manifest_sha256": r008_hash,
    }
    return payload, checkpoint, provenance, manifest, r009_hash, r008_hash


def test_calibrator_records_laya_tokenizer_normalization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    original_path = checkpoint / "checkpoint_latest/tokenizer/tokenizer_config.json"
    original_path.parent.mkdir(parents=True)
    original = {
        "tokenizer_class": "TokenizersBackend",
        "backend": "tokenizers",
        "is_local": True,
        "extra_special_tokens": ["<start>", "<end>"],
    }
    original_path.write_text(json.dumps(original))
    normalized = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {"extra_0": "<start>", "extra_1": "<end>"},
    }
    tokenizer_path = checkpoint / "tokenizer/tokenizer_config.json"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_text(json.dumps(normalized))
    before = {
        "tokenizer/tokenizer_config.json": identity.sha256(original_path),
        "model.safetensors": "unchanged-model",
    }
    after = {
        "tokenizer/tokenizer_config.json": identity.sha256(tokenizer_path),
        "model.safetensors": "unchanged-model",
    }
    monkeypatch.setattr(
        calibrator,
        "checkpoint_inference_inventory",
        lambda _checkpoint: after,
    )
    provenance = {"inference_files_sha256": before}

    result = calibrator.record_laya_tokenizer_normalization(checkpoint, provenance)

    assert result["inference_files_sha256"] == after
    assert (
        result["post_training_inference_normalization"]["before_sha256"]
        == before["tokenizer/tokenizer_config.json"]
    )
    assert (
        result["post_training_inference_normalization"]["after_sha256"]
        == after["tokenizer/tokenizer_config.json"]
    )
    assert json.loads((checkpoint / "training_provenance.json").read_text()) == result
    assert calibrator.record_laya_tokenizer_normalization(checkpoint, result) is result


def test_calibrator_rejects_unrelated_inference_file_change(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    original_path = checkpoint / "checkpoint_latest/tokenizer/tokenizer_config.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
    normalized_path = checkpoint / "tokenizer/tokenizer_config.json"
    normalized_path.parent.mkdir(parents=True)
    normalized_path.write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    before = {
        "tokenizer/tokenizer_config.json": identity.sha256(original_path),
        "model.safetensors": "original-model",
    }
    after = {
        "tokenizer/tokenizer_config.json": identity.sha256(normalized_path),
        "model.safetensors": "changed-model",
    }
    monkeypatch.setattr(
        calibrator,
        "checkpoint_inference_inventory",
        lambda _checkpoint: after,
    )

    with pytest.raises(ValueError, match="changed beyond tokenizer normalization"):
        calibrator.record_laya_tokenizer_normalization(
            checkpoint, {"inference_files_sha256": before}
        )


def test_r009_calibration_payload_is_bound_and_temperature_only(tmp_path):
    payload, checkpoint, provenance, manifest, r009_hash, r008_hash = calibration_case(
        tmp_path
    )
    identity.validate_r009_calibration_payload(
        payload, 13, checkpoint, provenance, manifest, r009_hash, r008_hash
    )

    payload["binning_map"] = {"choice_2": [0.1, 0.9]}
    with pytest.raises(ValueError, match="unexpected Laya payload shape"):
        identity.validate_r009_calibration_payload(
            payload, 13, checkpoint, provenance, manifest, r009_hash, r008_hash
        )


def test_r009_calibration_rejects_missing_or_invalid_temperatures(tmp_path):
    payload, checkpoint, provenance, manifest, r009_hash, r008_hash = calibration_case(
        tmp_path
    )
    del payload["temperature"]
    with pytest.raises(ValueError, match="unexpected Laya payload shape"):
        identity.validate_r009_calibration_payload(
            payload, 13, checkpoint, provenance, manifest, r009_hash, r008_hash
        )

    payload, checkpoint, provenance, manifest, r009_hash, r008_hash = calibration_case(
        tmp_path / "again"
    )
    payload["temperature"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid temperature"):
        identity.validate_r009_calibration_payload(
            payload, 13, checkpoint, provenance, manifest, r009_hash, r008_hash
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("version", "unsupported Laya version"),
        ("temperature", "invalid temperature"),
        ("config", "identity differs"),
        ("validation_count", "not bound"),
    ],
)
def test_r009_calibration_rejects_wrong_version_or_provenance(
    tmp_path, mutation, message
):
    payload, checkpoint, provenance, manifest, r009_hash, r008_hash = calibration_case(
        tmp_path
    )
    if mutation == "version":
        payload["version"] += 1
    elif mutation == "temperature":
        payload["temperature"][0] = 1e100
    elif mutation == "config":
        payload["config"]["max_len"] = 256
    else:
        payload["r009"]["usable_validation_items"] = 98

    with pytest.raises(ValueError, match=message):
        identity.validate_r009_calibration_payload(
            payload, 13, checkpoint, provenance, manifest, r009_hash, r008_hash
        )


def test_invalid_seed14_calibration_blocks_baselines_and_test_access(
    tmp_path, monkeypatch
):
    case13 = calibration_case(tmp_path / "seed13", seed=13)
    case14 = calibration_case(tmp_path / "seed14", seed=14)
    payload13, checkpoint13, provenance13, manifest, r009_hash, r008_hash = case13
    payload14, checkpoint14, provenance14, _, _, _ = case14
    payload14["binning_map"] = {"choice_2": [0.1, 0.9]}
    for checkpoint, payload in (
        (checkpoint13, payload13),
        (checkpoint14, payload14),
    ):
        (checkpoint / "calibration-sib200-km.json").write_text(json.dumps(payload))

    checkpoints = {13: checkpoint13, 14: checkpoint14}
    provenances = {13: provenance13, 14: provenance14}
    monkeypatch.setattr(
        identity,
        "verify_r008_anchor",
        lambda seed, arm, *_args: ({"checkpoint": f"anchor-{arm}-{seed}"}, {}, None),
    )
    monkeypatch.setattr(
        identity,
        "verify_r009_training",
        lambda seed, *_args: (checkpoints[seed], provenances[seed], {}),
    )
    monkeypatch.setattr(
        identity,
        "verify_anchor_report",
        lambda *_args: pytest.fail(
            "baseline report accessed before all calibrations passed"
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "verify_test_file",
        lambda *_args: pytest.fail("test file accessed before all calibrations passed"),
    )
    monkeypatch.setattr(
        evaluator,
        "read_jsonl",
        lambda *_args: pytest.fail("test JSONL read before all calibrations passed"),
    )

    with pytest.raises(ValueError, match="unexpected Laya payload shape"):
        evaluator.read_test_rows_after_readiness(manifest, r009_hash, {}, r008_hash)


def test_case_alignment_requires_row_identity_tag():
    row = {
        "state": {"text": "sample"},
        "questions": {"topic": {"criteria": ["a", "b"]}},
        "expected": {"topic": "a"},
    }
    model = "r009-test"
    report = {
        "overall": {"choice_accuracy": 1.0},
        "cases": [
            {
                "qid": "topic",
                "model": model,
                "tags": [evaluator.row_tag(row)],
                "expected": "a",
                "answer": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "answer_confidence": 0.7,
                },
                "confidence": 0.7,
                "correct": True,
            }
        ],
    }
    result = evaluator.check_cases(report, [row], model, evaluator.row_tag)
    assert result[1] == [True]

    report["cases"][0]["tags"] = ["r009:wrong-row"]
    with pytest.raises(ValueError, match="identity or answer mismatch"):
        evaluator.check_cases(report, [row], model, evaluator.row_tag)
