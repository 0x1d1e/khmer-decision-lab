"""CLI regression: model order must not change inference memory or predictions.

Uses existing local checkpoints and validation data; skips when unavailable.
No training, Hub downloads or new benchmark test-set scoring.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mt-s13", "mt-s14")


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for inference-memory regression")
    source = ROOT / "data/processed/sib200-km-validation.jsonl"
    checkpoints = ROOT / "artifacts/checkpoints"
    if not source.exists() or not all(
        (checkpoints / f"R006-mt-s{seed}/model.safetensors").exists()
        for seed in (13, 14)
    ):
        pytest.skip("Existing R006 checkpoints and SIB validation data required")

    work = tmp_path_factory.mktemp("evaluate")
    dataset = work / "validation-16.jsonl"
    dataset.write_text(
        "".join(source.read_text(encoding="utf-8").splitlines(keepends=True)[:16]),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    # Overrides inherited from a caller must not change this regression's budget.
    for key in ("MAX_LEN", "HEAD_MAX_LEN"):
        env.pop(key, None)
    collected = {}
    for label, models in (
        ("single13", MODELS[:1]),
        ("single14", MODELS[1:]),
        ("multi", MODELS),
    ):
        output = work / label
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                str(dataset),
                str(output),
                ",".join(models),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        collected[label] = {
            model: json.loads((output / f"{model}.json").read_text(encoding="utf-8"))
            for model in models
        }
    return collected


def test_model_order_does_not_inflate_peak_memory(reports):
    for label, model in zip(("single13", "single14"), MODELS):
        alone = reports[label][model]["config"]["peak_vram_mib"]
        sequenced = reports["multi"][model]["config"]["peak_vram_mib"]
        # Allow allocator/kernel variability, but not an extra resident checkpoint.
        assert sequenced <= alone + 64


def test_model_order_preserves_predictions(reports):
    for label, model in zip(("single13", "single14"), MODELS):
        alone = reports[label][model]
        sequenced = reports["multi"][model]
        assert len(sequenced["cases"]) == 16
        assert [c["answer"]["choice"] for c in sequenced["cases"]] == [
            c["answer"]["choice"] for c in alone["cases"]
        ]
        assert (
            sequenced["overall"]["choice_accuracy"]
            == alone["overall"]["choice_accuracy"]
        )


def test_measurement_protocol_is_explicit(reports):
    for report in reports["multi"].values():
        config = report["config"]
        assert config["measurement_protocol"] == "isolated-warm-v1"
        assert config["warmup_examples"] == 8
        assert config["model_load_ms"] > 0
        assert config["warmup_ms"] > 0
        assert (
            config["peak_vram_scope"] == "single resident model, scored inference only"
        )
        assert "cost_per_decision_p50_ms" in report["overall"]
        assert (
            report["overall"]["latency_p50_ms"]
            >= report["overall"]["cost_per_decision_p50_ms"]
        )
