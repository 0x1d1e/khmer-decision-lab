import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from laya import evals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r012_identity")
r010_identity = importlib.import_module("r010_identity")
trainer = importlib.import_module("train_r012_continuation")
evaluator = importlib.import_module("evaluate_r012")
summarizer = importlib.import_module("summarize_r012")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(text, label, group="group"):
    labels = [f"intent-{index}" for index in range(20)]
    return {
        "state": {"utterance": text},
        "questions": {"intent": {"type": "choice", "criteria": labels}},
        "expected": {"intent": label},
        "tags": [f"r010-group:{group}"],
        "language": "km",
    }


def _valid_training_run(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    artifact_dir = tmp_path / "artifacts/runs/R012"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(identity, "ARTIFACT_DIR", artifact_dir)
    identity.write_json(artifact_dir / "preparation.json", {})

    anchor_rel = "artifacts/checkpoints/R006-mt-s13"
    output_rel = identity.training_output_path(13, "clean")
    anchor = tmp_path / anchor_rel
    output = tmp_path / output_rel
    latest = output / "checkpoint_latest"
    for checkpoint in (anchor, output):
        (checkpoint / "encoder").mkdir(parents=True)
        (checkpoint / "tokenizer").mkdir()
        (checkpoint / "encoder/config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
        (checkpoint / "tokenizer/tokenizer_config.json").write_text(
            "{}", encoding="utf-8"
        )
    latest.mkdir()
    (latest / "tokenizer").mkdir()
    (latest / "tokenizer/tokenizer_config.json").write_text("{}", encoding="utf-8")

    source_config = {
        "temperature": [1.25, 1.0, 1.0],
        "temperature_by_options": {},
        "max_len": 512,
        "head_max_len": 256,
        "amp_dtype": "bf16",
    }
    source_questions = {
        "intent": {"criteria": ["intent-a"]},
        "topic": {"criteria": ["topic-a"]},
    }
    (anchor / "model.safetensors").write_bytes(b"frozen weights")
    (anchor / "rl_agent_config.json").write_text(
        json.dumps(source_config), encoding="utf-8"
    )
    (anchor / "questions.json").write_text(
        json.dumps(source_questions), encoding="utf-8"
    )
    for path in (
        output / "rl_agent_config.json",
        latest / "rl_agent_config.json",
    ):
        path.write_text(json.dumps(source_config), encoding="utf-8")
    for path in (output / "questions.json", latest / "questions.json"):
        path.write_text(json.dumps(source_questions), encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"continued weights")

    anchor_inventory = identity.checkpoint_inference_inventory(anchor)
    output_inventory = identity.checkpoint_inference_inventory(output)
    recipe = {
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 32,
        "encoder_lr": 0.000025,
        "head_lr": 0.0001,
        "min_lr": 0.000001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "loss": "soft-ce",
        "shuffle_options": ["choice"],
        "expected_optimizer_attempts": 360,
        "minimum_successful_updates": 342,
        "maximum_paired_update_difference": 7,
        "optimizer": "bitsandbytes.optim.adamw.PagedAdamW8bit",
        "base_revision": "base-revision",
    }
    training_config = {
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 32,
        "encoder_lr": 0.000025,
        "head_lr": 0.0001,
        "min_lr": 0.000001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "loss": "soft-ce",
        "freeze_encoder": False,
        "shuffle_options": ["choice"],
        "seed": 13,
        "max_len": 512,
        "head_max_len": 256,
        "amp": True,
        "gradient_checkpointing": True,
        "calib_max": 0,
        "calib_frac": 0.0,
        "eval_data": None,
    }
    model = {
        "checkpoint": anchor_rel,
        "inference_files_sha256": anchor_inventory,
        "max_len": 512,
        "head_max_len": 256,
    }
    manifest = {
        "study": "R012-ocr-continuation",
        "models": {"mt-s13": model},
        "source": {
            "training_rows": 11514,
            "train_file": "data/train.jsonl",
            "train_sha256": "train-sha256",
            "r006_manifest_sha256": "r006-sha256",
            "r010_manifest_sha256": "r010-sha256",
        },
        "recipe": recipe,
        "implementation_sha256": {
            "scripts/train_r012_continuation.py": "trainer-sha256",
            "scripts/r012_identity.py": "identity-sha256",
        },
    }
    telemetry = {
        "optimizer": recipe["optimizer"],
        "optimizer_attempts": 360,
        "successful_updates": 360,
        "skipped_updates": 0,
        "scheduler_steps": 360,
        "failure": None,
        "telemetry_cleanup_failure": None,
        "training_seconds": 1.0,
        "peak_allocated_bytes": 1024,
    }
    report = {
        "train_items": 11514,
        "calibration_items": 0,
        "skipped": {},
        "temperature": [1.0, 1.0, 1.0],
    }
    telemetry_path = output / "training_telemetry.json"
    report_path = output / "r012_training_report.json"
    identity.write_json(telemetry_path, telemetry)
    identity.write_json(report_path, report)
    restoration = {
        "laya_training_report_temperature": report["temperature"],
        "restored_temperature": source_config["temperature"],
        "changed_model_weights": False,
        "files": ["rl_agent_config.json", "checkpoint_latest/rl_agent_config.json"],
    }
    provenance = {
        "schema": "r012-training-provenance/1",
        "study": manifest["study"],
        "arm": "clean",
        "seed": 13,
        "checkpoint_directory": output_rel,
        "source_checkpoint": anchor_rel,
        "source_weights_sha256": anchor_inventory["model.safetensors"],
        "source_inference_files_sha256": anchor_inventory,
        "training_file": manifest["source"]["train_file"],
        "training_file_sha256": manifest["source"]["train_sha256"],
        "training_rows": manifest["source"]["training_rows"],
        "r006_manifest_sha256": manifest["source"]["r006_manifest_sha256"],
        "r010_manifest_sha256": manifest["source"]["r010_manifest_sha256"],
        "r012_manifest_sha256": "manifest-sha256",
        "preparation_sha256": identity.sha256(artifact_dir / "preparation.json"),
        "validation_used": False,
        "checkpoint_selection": False,
        "weights_sha256": identity.sha256(output / "model.safetensors"),
        "inference_files_sha256": output_inventory,
        "training_telemetry_sha256": identity.sha256(telemetry_path),
        "training_report_sha256": identity.sha256(report_path),
        "trainer_sha256": manifest["implementation_sha256"][
            "scripts/train_r012_continuation.py"
        ],
        "identity_sha256": manifest["implementation_sha256"][
            "scripts/r012_identity.py"
        ],
        "base_revision": recipe["base_revision"],
        "optimizer_reset": True,
        "optimizer_attempts": 360,
        "successful_updates": 360,
        "skipped_updates": 0,
        "training_config": training_config,
        "temperature_restoration": restoration,
        "inference_metadata_restoration": {
            "source_checkpoint": anchor_rel,
            "changed_model_weights": False,
            "files": {
                relative: {
                    "sha256": anchor_inventory[relative],
                    "paths": [relative, f"checkpoint_latest/{relative}"],
                }
                for relative in (
                    "questions.json",
                    "tokenizer/tokenizer_config.json",
                )
            },
        },
    }
    identity.write_json(output / "training_provenance.json", provenance)
    return manifest, output, provenance


def test_verify_preparation_recomputes_identity_and_ocr_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    artifact_dir = tmp_path / "artifacts/runs/R012"
    monkeypatch.setattr(identity, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(identity, "PREPARED_TRAIN", "data/ocr-train.jsonl")
    monkeypatch.setattr(
        identity, "TRAIN_TRANSCRIPTS", "artifacts/runs/R012/transcripts.jsonl"
    )
    source_rows = [
        {"state": {"utterance": "ខ្មែរ"}, "expected": {"intent": "a"}},
        {"state": {"utterance": "សួស្តី"}, "expected": {"intent": "b"}},
    ]
    source_path = tmp_path / "data/train.jsonl"
    _write_jsonl(source_path, source_rows)
    source_hash = identity.sha256(source_path)
    prepared_rows = []
    transcript_rows = []
    transcript_cers = []
    for index, original in enumerate(source_rows):
        text = original["state"]["utterance"] if index == 0 else "សួស្តី OCR"
        recognized = {"text": text, "image_sha256": "a" * 64}
        counts = r010_identity.cer_counts(original["state"]["utterance"], text)
        transcript_cers.append(counts)
        prepared = json.loads(json.dumps(original, ensure_ascii=False))
        prepared["state"]["utterance"] = text
        prepared_rows.append(prepared)
        transcript_rows.append(
            {
                "row_index": index,
                "row_id": identity.train_row_id(source_hash, index, original),
                "reference": original["state"]["utterance"],
                **recognized,
                "cer": counts,
            }
        )
    prepared_path = tmp_path / "data/ocr-train.jsonl"
    transcript_path = artifact_dir / "transcripts.jsonl"
    _write_jsonl(prepared_path, prepared_rows)
    _write_jsonl(transcript_path, transcript_rows)
    codepoint_edits = sum(counts["codepoint_edits"] for counts in transcript_cers)
    reference_codepoints = sum(
        counts["reference_codepoints"] for counts in transcript_cers
    )
    grapheme_edits = sum(counts["grapheme_edits"] for counts in transcript_cers)
    reference_graphemes = sum(
        counts["reference_graphemes"] for counts in transcript_cers
    )
    metrics = {
        "rows": 2,
        "unique_raw_utterances": 2,
        "codepoint_edits": codepoint_edits,
        "reference_codepoints": reference_codepoints,
        "codepoint_cer": codepoint_edits / reference_codepoints,
        "grapheme_edits": grapheme_edits,
        "reference_graphemes": reference_graphemes,
        "grapheme_cer": grapheme_edits / reference_graphemes,
        "exact_rows": sum(counts["exact"] for counts in transcript_cers),
        "blank_rows": 0,
    }
    identity.write_json(
        artifact_dir / "preparation.json",
        {
            "study": "R012-ocr-continuation",
            "manifest_sha256": "manifest",
            "preparer_sha256": "preparer",
            "source_train_sha256": source_hash,
            "training_rows": 2,
            "prepared_train_file": "data/ocr-train.jsonl",
            "transcript_artifact": "artifacts/runs/R012/transcripts.jsonl",
            "outputs_sha256": {
                "data/ocr-train.jsonl": identity.sha256(prepared_path),
                "artifacts/runs/R012/transcripts.jsonl": identity.sha256(
                    transcript_path
                ),
            },
            "ocr_metrics": metrics,
        },
    )
    manifest = {
        "study": "R012-ocr-continuation",
        "source": {
            "train_file": "data/train.jsonl",
            "train_sha256": source_hash,
            "training_rows": 2,
        },
        "implementation_sha256": {"scripts/prepare_r012.py": "preparer"},
    }

    assert identity.verify_preparation(manifest, "manifest")["ocr_metrics"] == metrics

    transcript_rows[1]["text"] = "corrupted"
    _write_jsonl(transcript_path, transcript_rows)
    with pytest.raises(ValueError, match="file inventory"):
        identity.verify_preparation(manifest, "manifest")


def test_validate_report_cases_checks_row_labels_and_probability_distribution():
    row = _row("query", "intent-3")
    labels = row["questions"]["intent"]["criteria"]
    probabilities = {label: 0.0 for label in labels}
    probabilities["intent-3"] = 1.0
    case = {
        "model": "mt-s13",
        "qid": "intent",
        "language": "km",
        "expected": "intent-3",
        "tags": row["tags"],
        "correct": True,
        "answer": {
            "type": "choice",
            "choice": "intent-3",
            "probabilities": probabilities,
        },
    }
    identity.validate_report_cases([case], [row], "mt-s13")

    case["answer"]["probabilities"]["intent-4"] = 0.5
    with pytest.raises(ValueError, match="probabilities are invalid"):
        identity.validate_report_cases([case], [row], "mt-s13")


def test_laya_eval_preserves_r010_schema_grouping_and_singleton_path():
    first_labels = [f"intent-{index}" for index in range(20)]
    second_labels = list(reversed(first_labels))
    rows = [_row(f"query-{index}", first_labels[0], "first") for index in range(10)]
    rows.append(_row("last-query", second_labels[0], "last"))
    rows[-1]["questions"]["intent"]["criteria"] = second_labels

    class StubAgent:
        def __init__(self):
            self.calls = []

        def _predict(self, states, questions):
            labels = questions["intent"]["criteria"]
            prediction = labels[0]
            probabilities = {label: 0.0 for label in labels}
            probabilities[prediction] = 1.0
            return [
                {
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": prediction,
                            "probabilities": probabilities,
                        }
                    }
                }
                for _ in states
            ]

        def predict(self, state, questions, **_kwargs):
            self.calls.append(("predict", 1))
            return self._predict([state], questions)[0]

        def predict_batch(
            self, states, questions, batch_size=None, sort_by_length=False, **_kwargs
        ):
            self.calls.append(("batch", len(states), batch_size, sort_by_length))
            return self._predict(states, questions)

    agent = StubAgent()
    dataset = evaluator._dataset(rows, "mt-s13")
    report = evals.evaluate(
        evaluator.AgentRunner(agent, "mt-s13"),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=8,
    )

    assert agent.calls == [
        ("batch", 8, 8, False),
        ("batch", 2, 8, False),
        ("predict", 1),
    ]
    assert len(report.cases) == len(rows)
    assert all(case["correct"] for case in report.cases)


def test_overlap_audit_excludes_whole_groups_but_retains_conflicting_labels(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)
    train_path = tmp_path / "data/train.jsonl"
    _write_jsonl(train_path, [{"state": {"utterance": "Clean copy"}}])
    transcript_path = tmp_path / "artifacts/runs/R012/transcripts.jsonl"
    _write_jsonl(transcript_path, [{"text": "OCR copy"}])
    overlap_id = "overlap-group"
    eligible_id = "eligible-conflict-group"
    clean_rows = [
        _row("clean copy", "intent-0", overlap_id),
        _row("clean copy", "intent-1", overlap_id),
        _row("eligible", "intent-2", eligible_id),
        _row("eligible", "intent-3", eligible_id),
    ]
    ocr_rows = [
        _row("ocr copy", "intent-0", overlap_id),
        _row("ocr copy", "intent-1", overlap_id),
        _row("ocr eligible", "intent-2", eligible_id),
        _row("ocr eligible", "intent-3", eligible_id),
    ]
    manifest = {"source": {"train_file": "data/train.jsonl"}}
    preparation = {"transcript_artifact": "artifacts/runs/R012/transcripts.jsonl"}

    audit = summarizer._test_eligibility(
        manifest, preparation, {"clean": clean_rows, "ocr_12px": ocr_rows}
    )

    for condition in ("clean", "ocr_12px"):
        assert audit[condition]["excluded_train_overlap_groups"] == 1
        assert audit[condition]["excluded_train_overlap_rows"] == 2
        assert audit[condition]["eligible_rows"] == 2
        assert audit[condition]["eligible_conflicting_label_groups"] == 1
        assert audit[condition]["selected_indices"] == [2, 3]


def test_group_bootstrap_is_deterministic_row_weighted_and_paired(monkeypatch):
    monkeypatch.setattr(summarizer, "DRAWS", 200)
    reference = [False, False, True]
    comparison = [True, False, True]
    groups = ["duplicate", "duplicate", "singleton"]

    first = summarizer._group_bootstrap(reference, comparison, groups)
    second = summarizer._group_bootstrap(reference, comparison, groups)

    assert first == second
    assert first["difference_percentage_points"] == pytest.approx(100 / 3)
    assert first["groups"] == 2
    assert first["draws"] == 200
    assert first["seed"] == 13

    by_group = {"duplicate": [0, 1], "singleton": [2]}
    group_order = sorted(by_group)
    rng = np.random.default_rng(13)
    expanded_samples = []
    for _ in range(200):
        sampled_groups = rng.integers(0, len(group_order), size=len(group_order))
        sampled_differences = [
            comparison[index] - reference[index]
            for sampled_group in sampled_groups
            for index in by_group[group_order[sampled_group]]
        ]
        expanded_samples.append(np.mean(sampled_differences) * 100)
    expected_interval = np.quantile(expanded_samples, [0.025, 0.975], method="linear")
    assert first[
        "paired_group_bootstrap_95_percent_interval_percentage_points"
    ] == pytest.approx(expected_interval)


def test_temperature_restoration_updates_both_configs_without_touching_weights(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    latest = checkpoint / "checkpoint_latest"
    latest.mkdir(parents=True)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"fixed test weights")
    original_weights = weights.read_bytes()
    for path in (checkpoint / "rl_agent_config.json", latest / "rl_agent_config.json"):
        path.write_text(json.dumps({"temperature": [1.0, 1.0, 1.0]}), encoding="utf-8")
    source = {"temperature": [1.25, 1.0, 1.0], "temperature_by_options": {}}

    restoration = trainer.restore_temperature(checkpoint, source)

    assert restoration["restored_temperature"] == source["temperature"]
    assert restoration["changed_model_weights"] is False
    assert restoration["files"] == [
        "rl_agent_config.json",
        "checkpoint_latest/rl_agent_config.json",
    ]
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["temperature"]
        == source["temperature"]
        for path in (
            checkpoint / "rl_agent_config.json",
            latest / "rl_agent_config.json",
        )
    )
    assert weights.read_bytes() == original_weights


def test_inference_metadata_restoration_repairs_laya_serialization_changes(tmp_path):
    anchor = tmp_path / "R006"
    output = tmp_path / "R012"
    latest = output / "checkpoint_latest"
    anchor_tokenizer = anchor / "tokenizer"
    output_tokenizer = output / "tokenizer"
    latest_tokenizer = latest / "tokenizer"
    for directory in (anchor_tokenizer, output_tokenizer, latest_tokenizer):
        directory.mkdir(parents=True)
    latest.mkdir(exist_ok=True)
    anchor_questions = {
        "intent": {"criteria": ["intent-a"]},
        "topic": {"criteria": ["topic-a"]},
    }
    laya_questions = {"intent": {"criteria": ["intent-a"]}}
    anchor_question_path = anchor / "questions.json"
    anchor_question_path.write_text(json.dumps(anchor_questions), encoding="utf-8")
    anchor_tokenizer_path = anchor_tokenizer / "tokenizer_config.json"
    anchor_tokenizer_path.write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
        encoding="utf-8",
    )
    for path in (output / "questions.json", latest / "questions.json"):
        path.write_text(json.dumps(laya_questions), encoding="utf-8")
    for path in (
        output_tokenizer / "tokenizer_config.json",
        latest_tokenizer / "tokenizer_config.json",
    ):
        path.write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend"}),
            encoding="utf-8",
        )

    anchor_inventory = {
        "questions.json": identity.sha256(anchor_question_path),
        "tokenizer/tokenizer_config.json": identity.sha256(anchor_tokenizer_path),
    }
    provenance = {
        "inference_metadata_restoration": {
            "source_checkpoint": "R006-anchor",
            "changed_model_weights": False,
            "files": {
                relative: {
                    "sha256": digest,
                    "paths": [relative, f"checkpoint_latest/{relative}"],
                }
                for relative, digest in anchor_inventory.items()
            },
        }
    }
    with pytest.raises(ValueError, match="differs from its anchor"):
        identity.verify_inference_metadata_restoration(
            output, anchor_inventory, provenance, "R006-anchor"
        )

    restored = trainer.restore_inference_metadata(output, anchor, "R006-anchor")

    assert restored == provenance["inference_metadata_restoration"]
    assert (
        identity.verify_inference_metadata_restoration(
            output, anchor_inventory, provenance, "R006-anchor"
        )
        == restored
    )
    for relative in anchor_inventory:
        assert (output / relative).read_bytes() == (anchor / relative).read_bytes()
        assert (latest / relative).read_bytes() == (anchor / relative).read_bytes()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("latest_schema", "inference metadata differs from its anchor"),
        ("latest_tokenizer", "inference metadata differs from its anchor"),
        ("provenance", "inference-metadata restoration provenance mismatch"),
    ],
)
def test_training_readiness_authenticates_anchor_metadata_and_provenance(
    tmp_path, monkeypatch, corruption, message
):
    manifest, output, provenance = _valid_training_run(tmp_path, monkeypatch)
    if corruption == "latest_schema":
        (output / "checkpoint_latest/questions.json").write_text(
            json.dumps({"intent": {"criteria": ["intent-a"]}}), encoding="utf-8"
        )
    elif corruption == "latest_tokenizer":
        (output / "checkpoint_latest/tokenizer/tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend"}), encoding="utf-8"
        )
    else:
        provenance["inference_metadata_restoration"]["changed_model_weights"] = True
        identity.write_json(output / "training_provenance.json", provenance)

    with pytest.raises(ValueError, match=message):
        identity.verify_training_run(
            13, "clean", manifest, "manifest-sha256", {"outputs_sha256": {}}
        )


def test_evaluator_authenticates_baselines_with_r010_preparation(monkeypatch):
    events = []
    manifest = {"source": {"r010_preparation_artifact": "R010/preparation.json"}}
    r010 = {"study": "R010-massive-ocr-cascade"}
    r012_preparation = {"outputs_sha256": {"train.jsonl": "train-hash"}}
    r010_preparation = {"outputs": {"clean": {}, "ocr_12px": {}}}

    def verify_locked_manifest(*, test_access):
        events.append(("lock", test_access))
        return manifest, "r012-hash", r010, "r010-hash"

    def verify_training_readiness(*_args):
        events.append("readiness")
        return {"all-four-runs": True}, r012_preparation

    def read_json(path):
        assert str(path).endswith("R010/preparation.json")
        events.append("read-r010-preparation")
        return r010_preparation

    class BaselinesAuthenticated(Exception):
        pass

    def verify_r010_baselines(_manifest, _r010, _r010_hash, preparation):
        assert preparation is r010_preparation
        events.append("baselines")
        raise BaselinesAuthenticated

    monkeypatch.setattr(evaluator, "verify_locked_manifest", verify_locked_manifest)
    monkeypatch.setattr(
        evaluator, "verify_training_readiness", verify_training_readiness
    )
    monkeypatch.setattr(evaluator, "read_json", read_json)
    monkeypatch.setattr(evaluator, "verify_r010_baselines", verify_r010_baselines)

    with pytest.raises(BaselinesAuthenticated):
        evaluator.evaluate()

    assert events == [
        ("lock", False),
        "readiness",
        ("lock", True),
        "readiness",
        "read-r010-preparation",
        "baselines",
    ]


@pytest.mark.parametrize(
    "module,entrypoint", [(evaluator, "evaluate"), (summarizer, "summarize")]
)
def test_test_access_is_not_opened_when_training_readiness_fails(
    module, entrypoint, monkeypatch
):
    accesses = []

    def verify_locked_manifest(*, test_access):
        accesses.append(test_access)
        return {}, "manifest", {}, "r010-manifest"

    def fail_readiness(*_args):
        raise ValueError("training not ready")

    monkeypatch.setattr(module, "verify_locked_manifest", verify_locked_manifest)
    monkeypatch.setattr(module, "verify_training_readiness", fail_readiness)

    with pytest.raises(ValueError, match="training not ready"):
        getattr(module, entrypoint)()

    assert accesses == [False]


def test_training_readiness_rejects_unmatched_optimizer_success_counts(
    monkeypatch,
):
    manifest = {"recipe": {"maximum_paired_update_difference": 7}}
    monkeypatch.setattr(identity, "verify_preparation", lambda *_args: {"ok": True})
    counts = {
        (13, "clean"): 360,
        (13, "ocr12"): 352,
        (14, "clean"): 360,
        (14, "ocr12"): 360,
    }
    monkeypatch.setattr(
        identity,
        "verify_training_run",
        lambda seed, arm, *_args: {"successful_updates": counts[(seed, arm)]},
    )

    with pytest.raises(ValueError, match="not matched for seed 13"):
        identity.verify_training_readiness(manifest, "manifest")
