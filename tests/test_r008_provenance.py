import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r008_identity")


def _locked_manifest(root, training_file, training_arm):
    relative_training = str(training_file.relative_to(root))
    return {
        "recipe": {
            "base_revision": "base-revision",
            "seeds": [13, 14],
            "expected_optimizer_updates_per_seed": {
                "multitask": 5,
                "khmer-news-specialist": 4,
                "sib200-specialist": 3,
            },
        },
        "implementation_sha256": {
            "scripts/train_full_8bit.py": "trainer-digest",
            "scripts/laya_amp.py": "guard-digest",
        },
        "training_split": {
            "arms": {
                training_arm: {
                    "file": relative_training,
                    "sha256": identity.sha256(training_file),
                    "rows": 1,
                }
            }
        },
    }


def _write_inference_files(checkpoint):
    (checkpoint / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "questions.json").write_text("{}", encoding="utf-8")
    (checkpoint / "encoder").mkdir()
    (checkpoint / "encoder" / "config.json").write_text("{}", encoding="utf-8")
    tokenizer = checkpoint / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_conversion_audit(checkpoint, training_file, manifest, arm, updates):
    preflight = {
        "rows_read": 1,
        "valid_items": 1,
        "train_items": 1,
        "calibration_items": 0,
        "optimizer_updates": updates,
        "skipped": {},
    }
    audit = {
        "schema": "r008-training-conversion-audit/1",
        "training_arm": arm,
        "seed": 13,
        "data_file": str(training_file.relative_to(identity.ROOT)),
        "training_file_sha256": identity.sha256(training_file),
        "locked_manifest_sha256": "locked-manifest-digest",
        "expected_rows_and_items": 1,
        "expected_optimizer_updates": updates,
        "preflight": preflight,
        "observed": {
            "train_items": 1,
            "calibration_items": 0,
            "skipped": {},
        },
    }
    path = checkpoint / identity.CONVERSION_AUDIT_FILENAME
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def _make_checkpoint(root, monkeypatch, arm="khmer-news-specialist", updates=4):
    monkeypatch.setattr(identity, "ROOT", root)
    training_file = root / "data" / "training.jsonl"
    training_file.parent.mkdir(parents=True, exist_ok=True)
    training_file.write_text('{"row":1}\n', encoding="utf-8")
    prefix = identity.ARM_OUTPUT_PREFIXES[arm]
    checkpoint = root / "artifacts" / "checkpoints" / f"{prefix}-s13"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"synthetic weights")
    _write_inference_files(checkpoint)
    telemetry = {
        "optimizer_attempts": updates,
        "successful_updates": updates,
        "skipped_updates": 0,
        "scheduler_steps": updates,
        "failure": None,
        "telemetry_cleanup_failure": None,
    }
    (checkpoint / "training_telemetry.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )
    manifest = _locked_manifest(root, training_file, arm)
    conversion_audit = _write_conversion_audit(
        checkpoint, training_file, manifest, arm, updates
    )
    provenance = {
        "schema": "r008-training-provenance/1",
        "study": "R008-specialist-controls",
        "training_arm": arm,
        "calibration_tasks": list(identity.ARM_TASKS[arm]),
        "seed": 13,
        "checkpoint_directory": str(checkpoint.resolve()),
        "weights_sha256": identity.sha256(checkpoint / "model.safetensors"),
        "inference_files_sha256": identity.checkpoint_inference_inventory(checkpoint),
        "training_conversion_audit_sha256": identity.sha256(conversion_audit),
        "training_file": str(training_file.relative_to(root)),
        "training_file_sha256": identity.sha256(training_file),
        "locked_manifest_sha256": "locked-manifest-digest",
        "base_revision": "base-revision",
        "trainer_sha256": "trainer-digest",
        "amp_guard_sha256": "guard-digest",
        "training_telemetry_sha256": identity.sha256(
            checkpoint / "training_telemetry.json"
        ),
        "optimizer_updates": {
            "attempts": updates,
            "successful": updates,
            "skipped": 0,
            "scheduler_steps": updates,
        },
    }
    (checkpoint / identity.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return checkpoint, manifest, training_file


def test_trainer_writes_provenance_that_binds_weights_arm_and_data(
    tmp_path, monkeypatch
):
    root = tmp_path
    monkeypatch.setattr(identity, "ROOT", root)
    training_file = root / "data" / "news.jsonl"
    training_file.parent.mkdir(parents=True)
    training_file.write_text('{"row":1}\n', encoding="utf-8")
    checkpoint = root / "artifacts" / "checkpoints" / "R008-news-s13"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    _write_inference_files(checkpoint)
    telemetry = {
        "optimizer_attempts": 4,
        "successful_updates": 3,
        "skipped_updates": 1,
        "scheduler_steps": 3,
        "failure": None,
        "telemetry_cleanup_failure": None,
    }
    (checkpoint / "training_telemetry.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )
    manifest = _locked_manifest(root, training_file, "khmer-news-specialist")
    manifest["recipe"]["expected_optimizer_updates_per_seed"][
        "khmer-news-specialist"
    ] = 4
    conversion_audit = _write_conversion_audit(
        checkpoint, training_file, manifest, "khmer-news-specialist", 4
    )
    monkeypatch.setattr(
        identity,
        "verify_locked_manifest",
        lambda _path: (manifest, "locked-manifest-digest"),
    )

    provenance_path = identity.write_training_provenance(
        checkpoint,
        "khmer-news-specialist",
        13,
        training_file,
        root / "manifest.json",
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["training_arm"] == "khmer-news-specialist"
    assert provenance["calibration_tasks"] == ["khmer-news"]
    assert provenance["weights_sha256"] == identity.sha256(
        checkpoint / "model.safetensors"
    )
    assert provenance[
        "inference_files_sha256"
    ] == identity.checkpoint_inference_inventory(checkpoint)
    assert provenance["training_conversion_audit_sha256"] == identity.sha256(
        conversion_audit
    )
    assert (
        identity.verify_checkpoint_training_provenance(
            checkpoint, "khmer-news", manifest, "locked-manifest-digest"
        )
        == provenance
    )
    try:
        identity.verify_checkpoint_training_provenance(
            checkpoint, "sib200-km", manifest, "locked-manifest-digest"
        )
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("News specialist was accepted for SIB-200 calibration")


def test_multitask_checkpoint_allows_both_target_calibration_tasks(
    tmp_path, monkeypatch
):
    checkpoint, manifest, _ = _make_checkpoint(
        tmp_path, monkeypatch, arm="multitask", updates=5
    )
    for task in ("khmer-news", "sib200-km"):
        assert (
            identity.verify_checkpoint_training_provenance(
                checkpoint, task, manifest, "locked-manifest-digest"
            )["training_arm"]
            == "multitask"
        )


def test_checkpoint_provenance_rejects_changed_weights_or_manifest(
    tmp_path, monkeypatch
):
    checkpoint, manifest, _ = _make_checkpoint(tmp_path, monkeypatch)
    (checkpoint / "model.safetensors").write_bytes(b"changed weights")
    try:
        identity.verify_checkpoint_training_provenance(
            checkpoint, "khmer-news", manifest, "locked-manifest-digest"
        )
    except ValueError as exc:
        assert "weights" in str(exc)
    else:
        raise AssertionError("changed weights were accepted")

    checkpoint, manifest, _ = _make_checkpoint(tmp_path / "second", monkeypatch)
    try:
        identity.verify_checkpoint_training_provenance(
            checkpoint, "khmer-news", manifest, "different-manifest-digest"
        )
    except ValueError as exc:
        assert "different R008 manifest" in str(exc)
    else:
        raise AssertionError("different manifest was accepted")


def test_checkpoint_provenance_rejects_changed_inference_files(tmp_path, monkeypatch):
    checkpoint, manifest, _ = _make_checkpoint(tmp_path, monkeypatch)
    config_path = checkpoint / "encoder" / "config.json"
    config_path.write_text('{"local_attention": 256}', encoding="utf-8")

    with pytest.raises(ValueError, match="inference-defining checkpoint files"):
        identity.verify_checkpoint_training_provenance(
            checkpoint, "khmer-news", manifest, "locked-manifest-digest"
        )


@pytest.mark.parametrize("override", ["fp16", "float16", "other"])
def test_inference_precision_rejects_conflicting_laya_override(override):
    with pytest.raises(ValueError, match="conflicting LAYA_CUDA_AMP"):
        identity.validate_laya_cuda_amp_override(override)


@pytest.mark.parametrize("override", [None, "bf16", "bfloat16", "BF16"])
def test_inference_precision_accepts_locked_laya_override(override):
    identity.validate_laya_cuda_amp_override(override)


def test_inference_policy_rejects_nonlocked_manifest_dtype():
    with pytest.raises(ValueError, match="precision policy"):
        identity.validate_inference_policy(
            {"inference": {"device": "cuda", "autocast_dtype": "float16"}}
        )


def test_load_r008_agent_applies_and_checks_locked_dtype(monkeypatch):
    from types import SimpleNamespace

    import laya
    import torch

    calls = []

    class FakeModel:
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            assert with_kwargs
            self.policy_hook = hook

    def fake_load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs, os.environ.get("LAYA_CUDA_AMP")))
        return SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            dtype=torch.bfloat16,
            amp_enabled=True,
            model=FakeModel(),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(laya, "load", fake_load)
    monkeypatch.setenv("LAYA_CUDA_AMP", "bfloat16")
    manifest = {"inference": dict(identity.INFERENCE_POLICY)}

    agent, dtype = identity.load_r008_agent(
        "checkpoint", manifest, calibration=Path("calibration.json")
    )

    assert agent.amp_enabled
    assert dtype == "bfloat16"
    assert hasattr(agent.model, "policy_hook")
    agent.device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="refusing a precision or device fallback"):
        agent.model.policy_hook(agent.model, (), {})
    assert calls == [
        (
            "checkpoint",
            {"device": "cuda", "calibration": "calibration.json"},
            "bf16",
        )
    ]
    assert os.environ["LAYA_CUDA_AMP"] == "bfloat16"


def test_training_conversion_rejects_skips_and_missing_items(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    training_file = tmp_path / "data" / "training.jsonl"
    training_file.parent.mkdir(parents=True)
    training_file.write_text('{"row":1}\n', encoding="utf-8")
    manifest = _locked_manifest(tmp_path, training_file, "khmer-news-specialist")
    summary = {
        "rows_read": 1,
        "valid_items": 1,
        "train_items": 1,
        "calibration_items": 0,
        "optimizer_updates": 4,
        "skipped": {},
    }

    assert identity.validate_training_conversion(
        summary, "khmer-news-specialist", manifest
    ) == (1, 4)
    for changed in (
        {**summary, "skipped": {"empty": 1}},
        {**summary, "train_items": 0},
    ):
        with pytest.raises(ValueError, match="conversion differs"):
            identity.validate_training_conversion(
                changed, "khmer-news-specialist", manifest
            )


def _calibration_fixture(tmp_path, monkeypatch):
    from laya.calibrate import CALIBRATION_VERSION

    checkpoint, manifest, _ = _make_checkpoint(tmp_path, monkeypatch)
    validation_path = (
        tmp_path / "data" / "processed" / "r008-khmer-news-calibration.jsonl"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps({"row": 1}) + chr(10), encoding="utf-8")
    manifest["validation_calibration"] = {
        "khmer-news": {
            "calibration_file": "data/processed/r008-khmer-news-calibration.jsonl",
            "calibration_sha256": identity.sha256(validation_path),
            "calibration_rows": 1,
            "test_file": "data/processed/khmer-news-test.jsonl",
        }
    }
    provenance_path = checkpoint / identity.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    calibration = {
        "version": CALIBRATION_VERSION,
        "temperature": [1.1, 1.2, 1.3],
        "temperature_by_options": {"choice:3-5": 1.4},
        "model_id_or_path": str(checkpoint.resolve()),
        "subfolder": None,
        "config": {},
        "r008": {
            "task": "khmer-news",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_weights_sha256": provenance["weights_sha256"],
            "training_arm": provenance["training_arm"],
            "training_provenance_sha256": identity.sha256(provenance_path),
            "validation_file": "data/processed/r008-khmer-news-calibration.jsonl",
            "validation_sha256": manifest["validation_calibration"]["khmer-news"][
                "calibration_sha256"
            ],
            "validation_rows": 1,
            "usable_validation_items": 1,
            "inference_device": "cuda",
            "inference_dtype": "bfloat16",
            "locked_manifest_sha256": "locked-manifest-digest",
        },
    }
    calibration_path = checkpoint / "calibration-khmer-news.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    return checkpoint, manifest, provenance, calibration_path, calibration


def test_calibration_artifact_is_bound_to_task_validation_and_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint, manifest, provenance, calibration_path, calibration = (
        _calibration_fixture(tmp_path, monkeypatch)
    )

    verified_path, digest = identity.verify_calibration_artifact(
        checkpoint, "khmer-news", provenance, manifest, "locked-manifest-digest"
    )

    assert verified_path == calibration_path
    assert digest == identity.sha256(calibration_path)
    calibration["r008"]["validation_file"] = (
        "data/processed/r008-sib200-km-calibration.jsonl"
    )
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        identity.verify_calibration_artifact(
            checkpoint, "khmer-news", provenance, manifest, "locked-manifest-digest"
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_temperature",
        "wrong_temperature_count",
        "nonfinite_temperature",
        "wrong_checkpoint_identity",
        "wrong_config_identity",
        "unregistered_binning",
        "missing_r008_metadata",
    ],
)
def test_calibration_artifact_rejects_invalid_laya_payload(
    tmp_path, monkeypatch, corruption
):
    checkpoint, manifest, provenance, calibration_path, calibration = (
        _calibration_fixture(tmp_path, monkeypatch)
    )
    if corruption == "missing_temperature":
        calibration.pop("temperature")
    elif corruption == "wrong_temperature_count":
        calibration["temperature"] = [1.0, 1.0]
    elif corruption == "nonfinite_temperature":
        calibration["temperature"] = [float("nan"), 1.0, 1.0]
    elif corruption == "wrong_checkpoint_identity":
        calibration["model_id_or_path"] = "/unrelated/checkpoint"
    elif corruption == "wrong_config_identity":
        calibration["config"] = {"max_len": 7}
    elif corruption == "unregistered_binning":
        calibration["binning_map"] = {}
    elif corruption == "missing_r008_metadata":
        calibration.pop("r008")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(
        (TypeError, ValueError),
        match="R008 calibration|calibration map|calibration identity",
    ):
        identity.verify_calibration_artifact(
            checkpoint, "khmer-news", provenance, manifest, "locked-manifest-digest"
        )


def test_laya_oom_cpu_fallback_is_rejected_before_cpu_forward(monkeypatch):
    from contextlib import nullcontext

    import laya.agent as laya_agent
    import torch
    from laya.agent import Agent

    class BatchTensor:
        shape = (1, 1)

        def to(self, _device):
            return self

    class Model:
        def __init__(self):
            self.devices = []
            self.moves = []
            self.policy_hook = None

        def register_forward_pre_hook(self, hook, *, with_kwargs):
            assert with_kwargs
            self.policy_hook = hook

        def __call__(self, *_args, **_kwargs):
            if self.policy_hook is None:
                raise AssertionError("R008 forward guard was not installed")
            self.policy_hook(self, _args, _kwargs)
            self.devices.append(agent.device.type)
            if agent.device.type == "cuda":
                raise RuntimeError("CUDA out of memory (synthetic E2E reproduction)")
            raise AssertionError("CPU model forward should never run")

        def to(self, device):
            self.moves.append(device.type)
            return self

    agent = object.__new__(Agent)
    agent.device = torch.device("cuda")
    agent.dtype = torch.bfloat16
    agent.amp_enabled = True
    agent._backend = None
    agent._fast = None
    agent._compiled = False
    agent._reduce_overhead = False
    agent.model = Model()
    agent.cpu_fallback_count = 0
    agent.last_fallback_reason = None
    identity.install_r008_forward_guard(agent)

    monkeypatch.setattr(
        laya_agent, "_amp_context", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _device: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda _device: torch.bfloat16)
    batch = {
        key: BatchTensor()
        for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    }

    with (
        pytest.warns(RuntimeWarning, match="Retrying this request on CPU"),
        pytest.raises(
            RuntimeError, match="refusing a precision or device fallback"
        ) as exc,
    ):
        agent._infer(batch)

    assert "CUDA out of memory" in str(exc.value)
    assert agent.model.devices == ["cuda"]
    assert agent.device.type == "cuda"
    assert isinstance(agent.last_fallback_reason, str)
    assert agent.last_fallback_reason.startswith("CUDA out of memory")
