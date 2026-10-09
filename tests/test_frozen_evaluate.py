"""Frozen evaluation CLI preflight with synthetic files, never inference."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "control-base-maxlen512": "control-base-maxlen512",
    "full-e2-8bit": "R003-full-e2-8bit",
    "mt-s13": "R006-mt-s13",
    "mt-s14": "R006-mt-s14",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    dataset = tmp_path / "r007-synthetic-test.jsonl"
    dataset.write_text(
        "Synthetic export, deliberately not a valid inference dataset.\n"
    )
    provenance = tmp_path / "r007-synthetic-test-ids.json"
    provenance.write_text("[]\n")
    checkpoints = {}
    for model, folder in MODELS.items():
        directory = tmp_path / "artifacts/checkpoints" / folder
        hashes = {}
        for relative in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Invalid weight bytes ensure a successful check cannot be inference.
            path.write_text("synthetic")
            hashes[relative] = digest(path)
        checkpoints[model] = {"directory": str(directory), "sha256": hashes}
    manifest = {
        "status": "locked_exploratory",
        "preparer_sha256": digest(ROOT / "scripts/prepare_r007.py"),
        "checkpoint_fingerprints": checkpoints,
        "targets": {
            "synthetic": {
                "max_len": 512,
                "head_max_len": 256,
                "export_sha256": {
                    "test": {
                        str(dataset): digest(dataset),
                        str(provenance): digest(provenance),
                    }
                },
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return dataset, path, manifest


def run_preflight(frozen, **overrides):
    dataset, manifest, _ = frozen
    env = os.environ.copy()
    env.update(MAX_LEN="512", HEAD_MAX_LEN="256", HF_HUB_OFFLINE="1")
    env.update(overrides)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate.py"),
            str(dataset),
            "unused-output",
            ",".join(MODELS),
            "--manifest",
            str(manifest),
            "--verify-only",
        ],
        cwd=dataset.parent,
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def test_valid_frozen_preflight_never_loads_weights_or_predicts(frozen):
    result = run_preflight(frozen)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no predictions generated" in result.stdout
    assert not (frozen[0].parent / "unused-output").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "encoder/config.json",
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ],
)
def test_changed_checkpoint_refuses_before_inference(frozen, relative):
    # Changing encoder behavior or temperature with unchanged weights is rejected.
    directory = Path(frozen[2]["checkpoint_fingerprints"]["mt-s13"]["directory"])
    (directory / relative).write_text("changed configuration or artifact")
    result = run_preflight(frozen)
    assert result.returncode != 0
    assert f"Frozen checkpoint fingerprint mismatch: mt-s13/{relative}" in result.stderr


def test_changed_export_refuses_before_inference(frozen):
    frozen[0].write_text("changed dataset")
    result = run_preflight(frozen)
    assert result.returncode != 0
    assert "Frozen export fingerprint mismatch" in result.stderr


def test_changed_budget_refuses_before_inference(frozen):
    result = run_preflight(frozen, MAX_LEN="2048")
    assert result.returncode != 0
    assert "Frozen token budget mismatch" in result.stderr


def test_unlocked_manifest_refuses_before_inference(frozen):
    manifest = frozen[2]
    manifest["status"] = "prepared_exploratory_pending_protocol_lock"
    frozen[1].write_text(json.dumps(manifest))
    result = run_preflight(frozen)
    assert result.returncode != 0
    assert "requires a locked R007 manifest" in result.stderr


def test_r007_cannot_accidentally_skip_manifest(frozen):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate.py"),
            str(frozen[0]),
            "unused-output",
            ",".join(MODELS),
        ],
        cwd=frozen[0].parent,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "R007 exports require --manifest" in result.stderr
