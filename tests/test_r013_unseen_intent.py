import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r013_identity")
preparer = importlib.import_module("prepare_r013")
evaluator = importlib.import_module("evaluate_r013")
trainer = importlib.import_module("train_r013")
summarizer = importlib.import_module("summarize_r013")
inference = importlib.import_module("r013_inference")


def _row(text, label, criteria, group="group"):
    return {
        "state": {"utterance": text},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is the user's intent in this utterance?",
                "criteria": criteria,
            }
        },
        "expected": {"intent": label},
        "tags": [f"r013-group:{group}", f"r013-row:{group}-row"],
        "language": "km",
    }


def _preparation_fixture(tmp_path, monkeypatch):
    source_labels = sorted(
        (*identity.HELDOUT_INTENTS, *(f"seen_{i:02}" for i in range(48)))
    )
    heldout = set(identity.HELDOUT_INTENTS)
    seen_labels = sorted(set(source_labels) - heldout)
    train_files = {
        "seen-only": "prepared/seen-only.jsonl",
        "all-label": "prepared/all-label.jsonl",
    }
    split_files = {
        "validation-unseen": "splits/validation-unseen.jsonl",
        "validation-seen": "splits/validation-seen.jsonl",
        "test-unseen": "splits/test-unseen.jsonl",
        "test-seen": "splits/test-seen.jsonl",
    }
    expected_split = {
        "validation_unseen_groups": 1,
        "validation_seen_groups": 1,
        "heldout_test_groups": 1,
        "known_test_groups": 1,
    }
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    monkeypatch.setattr(identity, "SOURCE_FILE", "source.jsonl")
    monkeypatch.setattr(identity, "PREPARATION_FILE", "artifacts/r013/preparation.json")
    monkeypatch.setattr(identity, "TRAIN_FILES", train_files)
    monkeypatch.setattr(identity, "SPLIT_FILES", split_files)
    monkeypatch.setattr(identity, "EXPECTED_SPLIT", expected_split)
    monkeypatch.setattr(
        identity,
        "RECIPE",
        {**identity.RECIPE, "training_rows_per_arm": len(source_labels)},
    )

    source_path = tmp_path / identity.SOURCE_FILE
    identity.write_jsonl(
        source_path,
        [{"expected": {"intent": label}} for label in source_labels],
    )
    source_hash = identity.sha256(source_path)
    output_hashes = {}
    for arm, relative in train_files.items():
        pool = seen_labels if arm == "seen-only" else source_labels
        labels = seen_labels + seen_labels[:12] if arm == "seen-only" else source_labels
        rows = []
        for index, gold in enumerate(labels):
            candidates = [gold, *(label for label in pool if label != gold)][:20]
            criteria = {label: label.replace("_", " ") for label in candidates}
            rows.append(
                _row(f"training-{arm}-{index}", gold, criteria, f"{arm}-{index}")
            )
        path = tmp_path / relative
        identity.write_jsonl(path, rows)
        output_hashes[relative] = identity.sha256(path)
    for relative in split_files.values():
        path = tmp_path / relative
        identity.write_jsonl(path, [])
        output_hashes[relative] = identity.sha256(path)

    implementation = {
        relative: f"implementation:{relative}"
        for relative in identity.IMPLEMENTATION_FILES
    }
    manifest = {
        "study": "R013-unseen-intent-transfer",
        "source": {"sha256": source_hash},
        "implementation_sha256": implementation,
    }
    manifest_sha256 = "manifest-hash"
    preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_sha256,
        "preparer_sha256": implementation["scripts/prepare_r013.py"],
        "source_sha256": source_hash,
        "heldout_intents": list(identity.HELDOUT_INTENTS),
        "seen_intents": seen_labels,
        "heldout_test_distractors": seen_labels[:8],
        "split_counts": expected_split,
        "training_rows": {arm: len(source_labels) for arm in identity.ARMS},
        "validation_rows": {"unseen": 1, "seen": 1},
        "test_rows": {"unseen": 1, "seen": 1},
        "outputs_sha256": output_hashes,
    }
    preparation_path = tmp_path / identity.PREPARATION_FILE
    identity.write_json(preparation_path, preparation)
    return manifest, manifest_sha256, preparation_path, train_files


def test_prepare_splits_groups_and_builds_leakage_free_candidate_pools(
    tmp_path, monkeypatch
):
    labels = [f"intent_{index:02}" for index in range(60)]
    heldout = preparer._selected_intents(
        {label: [f"{label}-{index}" for index in range(40)] for label in labels}
    )
    expected = {
        "source_rows": 2400,
        "distinct_normalized_groups": 2400,
        "same_label_duplicate_groups": 0,
        "conflicting_label_groups": 0,
        "conflicting_label_rows": 0,
        "eligible_groups": 2400,
        "eligible_labels": 60,
        "train_groups": 1680,
        "validation_groups": 240,
        "test_groups": 480,
        "heldout_test_groups": 96,
        "known_test_groups": 384,
        "seen_only_train_groups": 1344,
        "validation_unseen_groups": 48,
        "validation_seen_groups": 192,
    }
    train_files = {
        "seen-only": "prepared/seen.jsonl",
        "all-label": "prepared/all.jsonl",
    }
    split_files = {
        "validation-unseen": "splits/validation-unseen.jsonl",
        "validation-seen": "splits/validation-seen.jsonl",
        "test-unseen": "splits/test-unseen.jsonl",
        "test-seen": "splits/test-seen.jsonl",
    }
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    monkeypatch.setattr(preparer, "SOURCE_FILE", "source.jsonl")
    monkeypatch.setattr(preparer, "PREPARATION_FILE", "artifacts/r013/preparation.json")
    monkeypatch.setattr(preparer, "TRAIN_FILES", train_files)
    monkeypatch.setattr(preparer, "SPLIT_FILES", split_files)
    monkeypatch.setattr(preparer, "EXPECTED_SPLIT", expected)
    monkeypatch.setattr(preparer, "HELDOUT_INTENTS", heldout)

    source_path = tmp_path / "source.jsonl"
    identity.write_jsonl(
        source_path,
        [
            {
                "state": {"utterance": f"request {label} {index}"},
                "expected": {"intent": label},
            }
            for label in labels
            for index in range(40)
        ],
    )
    source_hash = identity.sha256(source_path)
    monkeypatch.setattr(
        preparer,
        "verify_locked_manifest",
        lambda _path: (
            {
                "study": "R013-unseen-intent-transfer",
                "source": {"sha256": source_hash},
            },
            "lock-hash",
        ),
    )

    preparation = preparer.prepare()

    seen_rows = identity.read_jsonl(tmp_path / train_files["seen-only"])
    all_rows = identity.read_jsonl(tmp_path / train_files["all-label"])
    seen_labels = set(labels) - set(heldout)
    assert len(seen_rows) == len(all_rows) == expected["seen_only_train_groups"]
    assert all(
        row["expected"]["intent"] in seen_labels
        and set(row["questions"]["intent"]["criteria"]) <= seen_labels
        and len(row["questions"]["intent"]["criteria"]) == 20
        for row in seen_rows
    )
    assert {row["expected"]["intent"] for row in all_rows} == set(labels)
    assert all(
        row["expected"]["intent"] in row["questions"]["intent"]["criteria"]
        for row in (*seen_rows, *all_rows)
    )
    assert set(preparation["all_label_sample_per_intent"]) == set(labels)
    assert sum(preparation["all_label_sample_per_intent"].values()) == len(all_rows)


def test_seen_only_builder_uses_only_seen_label_candidates():
    assert "scripts/r013_inference.py" in identity.IMPLEMENTATION_FILES
    heldout = set(identity.HELDOUT_INTENTS)
    seen = sorted(f"seen_{index:02}" for index in range(48))
    source = {"state": {"utterance": "request"}, "expected": {"intent": seen[0]}}

    first = preparer._choice_row(source, "row-id", "group-id", seen, "train")
    second = preparer._choice_row(source, "row-id", "group-id", seen, "train")
    criteria = first["questions"]["intent"]["criteria"]

    assert first == second
    assert len(criteria) == 20
    assert first["expected"]["intent"] in criteria
    assert set(criteria) <= set(seen)
    assert not set(criteria) & heldout


def test_base_inventory_accepts_pinned_snapshot_without_questions_file(tmp_path):
    checkpoint = (
        tmp_path
        / "hub"
        / "models--convaiinnovations--laya"
        / "snapshots"
        / identity.BASE_REVISION
        / identity.BASE_SUBDIR
    )
    (checkpoint / "encoder").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    files = {
        "model.safetensors": b"weights",
        "rl_agent_config.json": b"{}",
        "encoder/config.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
    }
    for relative, contents in files.items():
        path = checkpoint / relative
        path.write_bytes(contents)

    inventory = identity.base_checkpoint_inventory(checkpoint)

    assert set(inventory) == set(files)
    assert "questions.json" not in inventory


def test_base_normalization_rejects_escaping_symlink_before_reading(tmp_path):
    checkpoint = (
        tmp_path
        / "hub"
        / "models--convaiinnovations--laya"
        / "snapshots"
        / identity.BASE_REVISION
        / identity.BASE_SUBDIR
    )
    (checkpoint / "encoder").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    for relative, contents in {
        "model.safetensors": b"weights",
        "rl_agent_config.json": b"{}",
        "encoder/config.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
    }.items():
        (checkpoint / relative).write_bytes(contents)
    outside = tmp_path / "outside-tokenizer-config.json"
    outside.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "backend": {}}),
        encoding="utf-8",
    )
    tokenizer_config = checkpoint / "tokenizer/tokenizer_config.json"
    tokenizer_config.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its Hugging Face cache"):
        identity.normalized_base_checkpoint_inventory(checkpoint)

    assert tokenizer_config.is_symlink()


def _write_synthetic_inference_checkpoint(checkpoint, tokenizer_config):
    checkpoint = Path(checkpoint)
    (checkpoint / "encoder").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    for relative, contents in {
        "model.safetensors": b"weights",
        "rl_agent_config.json": b"{}",
        "questions.json": b"{}",
        "encoder/config.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": tokenizer_config,
    }.items():
        (checkpoint / relative).write_bytes(contents)


def test_restore_known_laya_tokenizer_rewrite_only_changes_authenticated_metadata(
    tmp_path, monkeypatch
):
    output = tmp_path / "checkpoint"
    latest = output / "checkpoint_latest"
    original = b'{"tokenizer_class":"PreTrainedTokenizerFast"}'
    normalized = b'{"backend":"tokenizers","tokenizer_class":"TokenizersBackend"}'
    _write_synthetic_inference_checkpoint(output, normalized)
    _write_synthetic_inference_checkpoint(latest, original)
    monkeypatch.setattr(
        identity,
        "RUNTIME_TOKENIZER_CONFIG_SHA256",
        hashlib.sha256(normalized).hexdigest(),
    )
    expected_inventory = identity.checkpoint_inference_inventory(latest)

    first = identity.restore_known_laya_tokenizer_rewrite(output, expected_inventory)
    second = identity.restore_known_laya_tokenizer_rewrite(output, expected_inventory)

    assert first["restored"] is True
    assert first["before_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert first["after_sha256"] == expected_inventory[identity.TOKENIZER_CONFIG]
    assert second["restored"] is False
    assert identity.checkpoint_inference_inventory(output) == expected_inventory

    (output / identity.TOKENIZER_CONFIG).write_bytes(b"unexpected metadata")
    with pytest.raises(ValueError, match="known locked states"):
        identity.restore_known_laya_tokenizer_rewrite(output, expected_inventory)


def test_training_metadata_recovery_does_not_open_prepared_splits(
    tmp_path, monkeypatch
):
    original = b'{"tokenizer_class":"PreTrainedTokenizerFast"}'
    normalized = b'{"backend":"tokenizers","tokenizer_class":"TokenizersBackend"}'
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    monkeypatch.setattr(
        identity,
        "OUTPUTS",
        {
            (seed, arm): f"checkpoints/{arm}-s{seed}"
            for seed in identity.SEEDS
            for arm in identity.ARMS
        },
    )
    monkeypatch.setattr(identity, "IMPLEMENTATION_FILES", ("scripts/test.py",))
    monkeypatch.setattr(
        identity,
        "RUNTIME_TOKENIZER_CONFIG_SHA256",
        hashlib.sha256(normalized).hexdigest(),
    )
    implementation = {"scripts/test.py": "implementation-hash"}
    manifest = {
        "study": "R013-unseen-intent-transfer",
        "protocol_sha256": "protocol-hash",
        "implementation_sha256": implementation,
    }
    manifest_sha256 = "manifest-hash"
    monkeypatch.setattr(
        identity,
        "read_jsonl",
        lambda _path: pytest.fail("metadata recovery must not open split data"),
    )
    expected_by_run = {}
    for seed in identity.SEEDS:
        for arm in identity.ARMS:
            output = identity.output_directory(seed, arm)
            latest = output / "checkpoint_latest"
            _write_synthetic_inference_checkpoint(output, normalized)
            _write_synthetic_inference_checkpoint(latest, original)
            expected = identity.checkpoint_inference_inventory(latest)
            expected_by_run[(seed, arm)] = expected
            identity.write_json(
                output / "training_provenance.json",
                {
                    "study": manifest["study"],
                    "manifest_sha256": manifest_sha256,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "implementation_sha256": implementation,
                    "seed": seed,
                    "arm": arm,
                    "output_inventory": expected,
                },
            )

    events = identity.restore_training_tokenizer_metadata(manifest, manifest_sha256)

    assert len(events) == 4
    assert all(event["restored"] for event in events)
    for key, expected in expected_by_run.items():
        assert (
            identity.checkpoint_inference_inventory(identity.output_directory(*key))
            == expected
        )


@pytest.mark.parametrize("leak", ["gold", "candidate"])
def test_preparation_readiness_rejects_heldout_gold_or_candidate_leakage(
    leak, tmp_path, monkeypatch
):
    manifest, manifest_hash, preparation_path, train_files = _preparation_fixture(
        tmp_path, monkeypatch
    )
    identity.verify_preparation(manifest, manifest_hash, test_access=False)

    relative = train_files["seen-only"]
    train_path = tmp_path / relative
    rows = identity.read_jsonl(train_path)
    gold = rows[0]["expected"]["intent"]
    criteria = rows[0]["questions"]["intent"]["criteria"]
    replacement = identity.HELDOUT_INTENTS[0]
    if leak == "gold":
        rows[0]["expected"]["intent"] = replacement
    else:
        candidate = next(label for label in criteria if label != gold)
        del criteria[candidate]
        criteria[replacement] = replacement.replace("_", " ")
    identity.write_jsonl(train_path, rows)
    preparation = identity.read_json(preparation_path)
    preparation["outputs_sha256"][relative] = identity.sha256(train_path)
    identity.write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="option semantics|held-out label leaked"):
        identity.verify_preparation(manifest, manifest_hash, test_access=False)


def test_preparation_readiness_accepts_predecessor_manifest(tmp_path, monkeypatch):
    manifest, current_hash, preparation_path, _ = _preparation_fixture(
        tmp_path, monkeypatch
    )
    manifest["amendment"] = identity.AMENDMENT
    predecessor_implementation = identity.AMENDMENT["predecessor_implementation_sha256"]
    monkeypatch.setattr(
        identity,
        "_verified_manifest_lineage",
        lambda _manifest, _sha: {
            current_hash: {
                "protocol_sha256": "protocol-hash",
                "implementation_sha256": manifest["implementation_sha256"],
            },
            identity.AMENDMENT["predecessor_manifest_sha256"]: {
                "protocol_sha256": identity.AMENDMENT["predecessor_protocol_sha256"],
                "implementation_sha256": predecessor_implementation,
            },
        },
    )
    preparation = identity.read_json(preparation_path)
    preparation["manifest_sha256"] = identity.AMENDMENT["predecessor_manifest_sha256"]
    preparation["preparer_sha256"] = identity.AMENDMENT[
        "predecessor_implementation_sha256"
    ]["scripts/prepare_r013.py"]
    identity.write_json(preparation_path, preparation)

    verified = identity.verify_preparation(manifest, current_hash, test_access=False)

    assert (
        verified["manifest_sha256"] == identity.AMENDMENT["predecessor_manifest_sha256"]
    )


def test_trainer_base_provenance_passes_readiness_with_synthetic_checkpoints(
    tmp_path, monkeypatch
):
    seed, arm = 13, "seen-only"
    output_relative = "artifacts/checkpoints/r013-synthetic"
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    monkeypatch.setattr(identity, "OUTPUTS", {(seed, arm): output_relative})
    output = tmp_path / output_relative
    latest = output / "checkpoint_latest"
    config = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    questions = {"intent": {"type": "choice", "criteria": ["intent-a"]}}
    for checkpoint in (output, latest):
        (checkpoint / "encoder").mkdir(parents=True)
        (checkpoint / "tokenizer").mkdir()
        (checkpoint / "encoder/config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"changed weights")
        (checkpoint / "rl_agent_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (checkpoint / "questions.json").write_text(
            json.dumps(questions), encoding="utf-8"
        )
    preparation_path = tmp_path / identity.PREPARATION_FILE
    preparation_path.parent.mkdir(parents=True)
    preparation_path.write_text("{}", encoding="utf-8")
    training_file = identity.TRAIN_FILES[arm]
    preparation = {"outputs_sha256": {training_file: "training-data-hash"}}
    identity.write_json(
        preparation_path,
        {"outputs_sha256": preparation["outputs_sha256"]},
    )

    report_training = {
        key: trainer.RECIPE[key]
        for key in (
            "epochs",
            "micro_batch",
            "grad_accum",
            "encoder_lr",
            "head_lr",
            "min_lr",
            "weight_decay",
            "grad_clip",
            "loss",
            "freeze_encoder",
            "shuffle_options",
            "max_len",
            "head_max_len",
            "amp",
            "gradient_checkpointing",
        )
    }
    report_training.update(
        {"seed": seed, "calib_max": 0, "calib_frac": 0.0, "eval_data": None}
    )
    report = {
        "eval_source": None,
        "eval_items": 0,
        "eval_mode": None,
        "calibration": {
            qtype: {"items": 0, "temperature": 1.0, "issues": []}
            for qtype in ("choice", "score", "noul")
        },
        "training": report_training,
    }
    identity.write_json(output / "train_report.json", report)
    telemetry = {
        "optimizer": trainer.RECIPE["optimizer"],
        "optimizer_attempts": trainer.RECIPE["expected_optimizer_attempts"],
        "successful_updates": trainer.RECIPE["minimum_successful_updates"],
        "skipped_updates": trainer.RECIPE["expected_optimizer_attempts"]
        - trainer.RECIPE["minimum_successful_updates"],
        "scheduler_steps": trainer.RECIPE["minimum_successful_updates"],
        "failure": None,
        "telemetry_cleanup_failure": None,
        "device": "cuda",
        "gpu": "synthetic GPU",
        "torch_version": "synthetic torch",
        "cuda_version": "synthetic CUDA",
        "peak_allocated_bytes": 1024,
    }
    identity.write_json(output / "training_telemetry.json", telemetry)
    current_implementation = {
        relative: f"current:{relative}" for relative in identity.IMPLEMENTATION_FILES
    }
    manifest = {
        "study": "R013-unseen-intent-transfer",
        "protocol_sha256": "protocol-hash",
        "implementation_sha256": current_implementation,
        "amendment": identity.AMENDMENT,
        "environment": {
            "gpu": "synthetic GPU",
            "torch_build": "synthetic torch",
            "torch_cuda": "synthetic CUDA",
        },
        "base": {
            "repo": trainer.BASE_REPO,
            "revision": trainer.BASE_REVISION,
            "subdir": trainer.BASE_SUBDIR,
            "inference_files_sha256": {"model.safetensors": "a" * 64},
        },
    }
    manifest_hash = "manifest-hash"
    predecessor_implementation = identity.AMENDMENT["predecessor_implementation_sha256"]
    monkeypatch.setattr(
        identity,
        "_verified_manifest_lineage",
        lambda _manifest, _sha: {
            manifest_hash: {
                "protocol_sha256": manifest["protocol_sha256"],
                "implementation_sha256": current_implementation,
            },
            identity.AMENDMENT["predecessor_manifest_sha256"]: {
                "protocol_sha256": identity.AMENDMENT["predecessor_protocol_sha256"],
                "implementation_sha256": predecessor_implementation,
            },
        },
    )
    expected_base = trainer._base_provenance(manifest["base"]["inference_files_sha256"])
    provenance = {
        "study": manifest["study"],
        "manifest_sha256": identity.AMENDMENT["predecessor_manifest_sha256"],
        "preparation_sha256": identity.sha256(preparation_path),
        "protocol_sha256": identity.AMENDMENT["predecessor_protocol_sha256"],
        "implementation_sha256": identity.AMENDMENT[
            "predecessor_implementation_sha256"
        ],
        "seed": seed,
        "arm": arm,
        "recipe": {**trainer.RECIPE, "seed": seed, "arm": arm},
        "training_file": training_file,
        "training_file_sha256": preparation["outputs_sha256"][training_file],
        "training_rows": trainer.RECIPE["training_rows_per_arm"],
        "output_inventory": identity.checkpoint_inference_inventory(output),
        "training_telemetry_sha256": identity.sha256(
            output / "training_telemetry.json"
        ),
        "train_report_sha256": identity.sha256(output / "train_report.json"),
        "base": expected_base,
    }
    identity.write_json(output / "training_provenance.json", provenance)

    ready = identity.verify_training_run(
        manifest, manifest_hash, preparation, seed, arm
    )

    assert (
        ready["telemetry"]["successful_updates"]
        == trainer.RECIPE["minimum_successful_updates"]
    )
    assert provenance["base"]["subdir"] == trainer.BASE_SUBDIR


def test_known_intent_rows_accept_sampled_seen_options_and_reject_invalid_options():
    seen = sorted(f"seen_{index:02}" for index in range(48))
    gold = seen[0]
    options = [gold, *seen[1:20]]
    preparation = {"seen_intents": seen, "heldout_test_distractors": seen[:8]}

    valid = _row(
        "known request",
        gold,
        {label: label.replace("_", " ") for label in options},
        "valid-known",
    )
    assert evaluator._validate_test_rows([valid], "test-seen", preparation) == 1

    missing_gold = _row(
        "known request without gold",
        gold,
        {label: label.replace("_", " ") for label in [*options[1:], seen[20]]},
        "missing-gold",
    )
    with pytest.raises(ValueError, match="test row schema mismatch"):
        evaluator._validate_test_rows([missing_gold], "test-seen", preparation)

    heldout_candidate = _row(
        "known request with heldout candidate",
        gold,
        {
            label: label.replace("_", " ")
            for label in [*options[:-1], identity.HELDOUT_INTENTS[0]]
        },
        "heldout-candidate",
    )
    with pytest.raises(ValueError, match="test row schema mismatch"):
        evaluator._validate_test_rows([heldout_candidate], "test-seen", preparation)


def test_test_rows_require_the_locked_option_set():
    seen = sorted(f"seen_{index:02}" for index in range(48))
    distractors = seen[:8]
    options = [*identity.HELDOUT_INTENTS, *distractors]
    row = _row(
        "unseen request",
        identity.HELDOUT_INTENTS[0],
        {label: label.replace("_", " ") for label in options},
        "test-group",
    )
    preparation = {"seen_intents": seen, "heldout_test_distractors": distractors}

    assert evaluator._validate_test_rows([row], "test-unseen", preparation) == 1

    row["questions"]["intent"]["criteria"].pop(distractors[-1])
    row["questions"]["intent"]["criteria"][seen[-1]] = seen[-1].replace("_", " ")
    with pytest.raises(ValueError, match="test row schema mismatch"):
        evaluator._validate_test_rows([row], "test-unseen", preparation)


def test_test_files_are_opened_only_after_all_training_readiness_checks(
    tmp_path, monkeypatch
):
    events = []
    split_files = {
        "validation-unseen": "validation-unseen.jsonl",
        "validation-seen": "validation-seen.jsonl",
        "test-unseen": "test-unseen.jsonl",
        "test-seen": "test-seen.jsonl",
    }
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    monkeypatch.setattr(identity, "SPLIT_FILES", split_files)

    def verify_preparation(_manifest, _manifest_hash, *, test_access=False):
        events.append(f"preparation:{test_access}")
        return {"ready": True}

    def verify_training(_manifest, _manifest_hash, _preparation):
        events.append("training-readiness")
        return {"all-runs-ready": True}

    def read_jsonl(path):
        events.append(f"read:{Path(path).name}")
        return []

    monkeypatch.setattr(identity, "verify_preparation", verify_preparation)
    monkeypatch.setattr(identity, "verify_all_training_runs", verify_training)
    monkeypatch.setattr(identity, "read_jsonl", read_jsonl)

    identity.load_test_rows_after_readiness({}, "manifest-hash")

    assert events[:3] == ["preparation:False", "training-readiness", "preparation:True"]
    assert events[3:] == [f"read:{Path(path).name}" for path in split_files.values()]


def test_laya_logit_capture_uses_locked_singleton_scoring_and_preserves_alignment():
    class Model(torch.nn.Module):
        def forward(self, scores):
            logits = torch.zeros((len(scores), 20))
            logits[:, 0] = scores
            logits[:, 1] = -scores
            return logits, torch.zeros((len(scores), 1))

    class Agent:
        def __init__(self):
            self.model = Model()
            self.calls = []

        def _predict(self, states, questions):
            labels = list(questions["intent"]["criteria"])
            scores = torch.tensor([state["score"] for state in states])
            logits, _ = self.model(scores)
            probabilities = torch.softmax(logits, dim=1).tolist()
            results = []
            for row_probabilities in probabilities:
                prediction = labels[int(np.argmax(row_probabilities))]
                results.append(
                    {
                        "answers": {
                            "intent": {
                                "type": "choice",
                                "choice": prediction,
                                "probabilities": dict(
                                    zip(labels, row_probabilities, strict=True)
                                ),
                            }
                        }
                    }
                )
            return results

        def predict(self, state, questions, **kwargs):
            self.calls.append(
                (
                    "predict",
                    kwargs.get("max_len"),
                    kwargs.get("head_max_len"),
                )
            )
            return self._predict([state], questions)[0]

        def predict_batch(
            self, states, questions, batch_size=None, sort_by_length=False, **kwargs
        ):
            self.calls.append(
                (
                    "batch",
                    len(states),
                    batch_size,
                    sort_by_length,
                    kwargs.get("max_len"),
                    kwargs.get("head_max_len"),
                )
            )
            return self._predict(states, questions)

    labels = [f"intent_{index:02}" for index in range(20)]
    reversed_labels = list(reversed(labels))
    rows = [
        _row(
            f"query-{index}",
            labels[0],
            dict.fromkeys(labels, "label"),
            f"group-{index}",
        )
        for index in range(10)
    ]
    rows.append(
        _row(
            "last-query",
            reversed_labels[1],
            dict.fromkeys(reversed_labels, "label"),
            "group-last",
        )
    )
    for index, row in enumerate(rows):
        row["state"]["score"] = -1.0 if index == len(rows) - 1 else 1.0

    agent = Agent()
    logits, payload = inference.capture_laya_logits(agent, rows, "stub-model")

    assert logits.shape == (len(rows), 20)
    assert logits[0, :2].tolist() == [1.0, -1.0]
    assert logits[-1, :2].tolist() == [-1.0, 1.0]
    assert agent.calls == [("predict", 512, 256)] * len(rows)
    assert len(payload["cases"]) == len(rows)
    assert payload["cases"][-1]["answer"]["choice"] == reversed_labels[1]
    assert payload["config"]["timing"]["batch_size"] == 1
    assert payload["config"]["timing"]["rows_alone"] == len(rows)
    assert payload["config"]["timing"]["max_chunk"] == 1


def test_verify_cases_requires_exact_laya_probability_rounding():
    labels = [f"intent_{index:02}" for index in range(20)]
    row = _row(
        "probability-rounding boundary",
        labels[0],
        {label: label for label in labels},
        "rounding-boundary",
    )
    raw_probabilities = np.array(
        [0.05015] * 4 + [0.04995] * 14 + [0.05005] * 2, dtype=np.float32
    )
    logits = np.log(raw_probabilities).astype(np.float32).reshape(1, 20)
    row_logits = logits[0] / 1.0
    expected_probabilities = np.exp(row_logits - row_logits.max())
    expected_probabilities /= expected_probabilities.sum()
    probabilities = {
        label: round(float(probability), 4)
        for label, probability in zip(labels, expected_probabilities, strict=True)
    }
    case = {
        "model": "rounding-test",
        "qid": "intent",
        "expected": labels[0],
        "tags": row["tags"],
        "correct": True,
        "answer": {
            "type": "choice",
            "choice": labels[0],
            "probabilities": probabilities,
            "answer_confidence": round(float(expected_probabilities.max()), 4),
        },
    }

    evaluator._verify_cases([case], [row], "rounding-test", logits)

    one_quantum_high = {
        **case,
        "answer": {
            **case["answer"],
            "probabilities": {
                **probabilities,
                labels[0]: round(probabilities[labels[0]] + 0.0001, 4),
            },
        },
    }
    with pytest.raises(ValueError, match="report probabilities are invalid"):
        evaluator._verify_cases([one_quantum_high], [row], "rounding-test", logits)

    one_quantum_confidence_high = {
        **case,
        "answer": {
            **case["answer"],
            "answer_confidence": round(case["answer"]["answer_confidence"] + 0.0001, 4),
        },
    }
    with pytest.raises(ValueError, match="report probabilities are invalid"):
        evaluator._verify_cases(
            [one_quantum_confidence_high], [row], "rounding-test", logits
        )


def test_metrics_and_group_bootstrap_match_fixed_estimands():
    options = [
        "heldout_a",
        "heldout_b",
        *(f"distractor_{index}" for index in range(18)),
    ]
    criteria = {label: label for label in options}
    rows = [
        _row("query-a", "heldout_a", criteria, "duplicate-group"),
        _row("query-b", "heldout_b", criteria, "duplicate-group"),
        _row("query-c", "heldout_b", criteria, "single-group"),
    ]
    logits = np.zeros((3, 20), dtype=np.float64)
    logits[0, 0] = 5.0
    logits[1, 2] = 5.0
    logits[2, 1] = 5.0
    predictions = [options[0], options[2], options[1]]
    cases = [
        {
            "expected": row["expected"]["intent"],
            "answer": {"choice": prediction},
            "correct": prediction == row["expected"]["intent"],
            "tags": row["tags"],
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]

    metrics = summarizer.compute_metrics(rows, cases, logits)
    assert metrics["choice_accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_intent_recall"]["heldout_a"] == 1.0
    assert metrics["per_intent_recall"]["heldout_b"] == 0.5
    assert metrics["groups"] == 2
    assert metrics["macro_ovr_auc"] == pytest.approx(0.875)

    reference = [
        dict(case, correct=correct)
        for case, correct in zip(cases, [False, False, True], strict=True)
    ]
    candidate = [
        dict(case, correct=correct)
        for case, correct in zip(cases, [True, False, True], strict=True)
    ]
    first = summarizer.paired_group_bootstrap(reference, candidate, draws=300, seed=13)
    second = summarizer.paired_group_bootstrap(reference, candidate, draws=300, seed=13)
    assert first == second
    assert first["difference_pp"] == pytest.approx(100 / 3)
    assert first["groups"] == 2

    grouped = {"duplicate-group": [1.0, 0.0], "single-group": [0.0]}
    group_ids = sorted(grouped)
    rng = np.random.default_rng(13)
    samples = []
    for _ in range(300):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True)
        draws = [value for group in selected for value in grouped[str(group)]]
        samples.append(100 * np.mean(draws))
    expected = np.quantile(samples, [0.025, 0.975], method="linear")
    assert first["ci95_pp"] == pytest.approx(expected.tolist())


def test_summary_requires_effective_singleton_inference_configuration():
    config = {
        "device": "cuda",
        "autocast_dtype": "bfloat16",
        "max_len": 512,
        "head_max_len": 256,
        "laya_cuda_amp": "bf16",
        "batch_size": 1,
        "timing": {"batch_size": 1, "max_chunk": 1},
    }

    summarizer._verify_report_inference_config(config, ("test-unseen", "base"))

    for field, value in (
        ("batch_size", 8),
        ("timing", {"batch_size": 8, "max_chunk": 8}),
        ("timing", {"batch_size": 1, "max_chunk": 8}),
    ):
        invalid = {**config, field: value}
        with pytest.raises(ValueError, match="report inference configuration mismatch"):
            summarizer._verify_report_inference_config(invalid, ("test-unseen", "base"))
