"""Tests for the R017 efficiency measurement study.

Validates the measurement record schema, batch-config validation, rejection of
missing checkpoint paths, and the summarize aggregation math on synthetic
fixtures. No GPU is required.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "measure_r017_efficiency", ROOT / "scripts/measure_r017_efficiency.py"
)
assert SPEC is not None and SPEC.loader is not None
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)

SUMMARIZE_SPEC = importlib.util.spec_from_file_location(
    "summarize_r017", ROOT / "scripts/summarize_r017.py"
)
assert SUMMARIZE_SPEC is not None and SUMMARIZE_SPEC.loader is not None
summarize = importlib.util.module_from_spec(SUMMARIZE_SPEC)
SUMMARIZE_SPEC.loader.exec_module(summarize)


def base_record():
    return {
        "study": "R017-efficiency",
        "measurement_protocol": "isolated-warm-v1",
        "checkpoint": "full-e2-8bit",
        "origin": "R003",
        "seed": 13,
        "checkpoint_path": "artifacts/checkpoints/R003-full-e2-8bit",
        "weights_sha256": "a" * 64,
        "dataset": "data/processed/massive-km-test-full.jsonl",
        "dataset_sha256": "b" * 64,
        "dataset_rows": 2974,
        "batch_size": 4,
        "warmup_examples": 8,
        "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "torch_version": "2.14.1+cu130",
        "cuda_version": "13.0",
        "status": "ok",
        "oom_reason": None,
        "model_load_ms": 1234.5,
        "warmup_ms": 67.5,
        "latency_p50_ms": 12.5,
        "latency_p95_ms": 25.0,
        "latency_mean_ms": 14.2,
        "cost_per_decision_p50_ms": 12.5,
        "cost_per_decision_p95_ms": 25.0,
        "peak_vram_allocated_mib": 1024.0,
        "peak_vram_reserved_mib": 2048.0,
        "timing": {"batch_size": 4},
    }


def test_measurement_record_schema_accepts_valid_record():
    measure.validate_measurement_record(base_record())


def test_measurement_record_schema_rejects_missing_key():
    record = base_record()
    del record["peak_vram_allocated_mib"]
    with pytest.raises(ValueError, match="missing key"):
        measure.validate_measurement_record(record)


def test_measurement_record_schema_rejects_wrong_type():
    record = base_record()
    record["batch_size"] = "4"
    with pytest.raises(TypeError, match="invalid type"):
        measure.validate_measurement_record(record)


def test_measurement_record_schema_rejects_extra_key():
    record = base_record()
    record["extra"] = 1
    with pytest.raises(ValueError, match="unexpected keys"):
        measure.validate_measurement_record(record)


def test_measurement_record_schema_rejects_bool_for_int():
    record = base_record()
    record["seed"] = True
    with pytest.raises(TypeError, match="invalid type"):
        measure.validate_measurement_record(record)


def test_measurement_record_schema_accepts_dropped_record():
    record = base_record()
    record.update(
        status="dropped",
        oom_reason="CUDA out of memory at batch_size 16",
        model_load_ms=None,
        warmup_ms=None,
        latency_p50_ms=None,
        latency_p95_ms=None,
        latency_mean_ms=None,
        cost_per_decision_p50_ms=None,
        cost_per_decision_p95_ms=None,
        peak_vram_allocated_mib=None,
        peak_vram_reserved_mib=None,
    )
    measure.validate_measurement_record(record)


def test_batch_config_validation_defaults_and_range():
    assert measure.validate_batch_sizes((1, 4, 8, 16)) == [1, 4, 8, 16]
    # unsorted input is normalized, duplicates removed.
    assert measure.validate_batch_sizes([16, 8, 4, 4, 1]) == [1, 4, 8, 16]
    assert measure.validate_batch_sizes([32]) == [32]
    for bad in ([0], [-1], [65], [None], [True], ["x"]):
        with pytest.raises(ValueError):
            measure.validate_batch_sizes(bad)


def test_load_plan_uses_all_eight_checkpoints():
    plan, batches = measure.load_plan()
    assert [entry["checkpoint"] for entry in plan] == list(measure.CHECKPOINTS)
    assert batches == [1, 4, 8, 16]
    for entry in plan:
        assert Path(entry["checkpoint_path"]).is_dir()
        assert Path(entry["weights_path"]).is_file()
        assert len(entry["weights_sha256"]) == 64


def test_load_plan_rejects_unknown_checkpoint():
    with pytest.raises(ValueError, match="Unknown checkpoints"):
        measure.load_plan(models=["full-e2-8bit", "no-such-checkpoint"])


def test_load_plan_rejects_missing_checkpoint_directory():
    tampered = {"ghost": ("artifacts/checkpoints/does-not-exist", "X", 9)}
    with pytest.raises(ValueError, match="Missing checkpoint directory"):
        measure.load_plan(models=["ghost"], checkpoints=tampered)


def test_load_plan_rejects_missing_weights():
    # scripts/ is a real directory without model.safetensors.
    tampered = {"ghost": ("scripts", "X", 9)}
    with pytest.raises(ValueError, match="Missing model weights"):
        measure.load_plan(models=["ghost"], checkpoints=tampered)


def _fixture_record(checkpoint, batch_size, status="ok", p50=10.0, peak=1000.0):
    record = base_record()
    origin = "R006" if checkpoint.startswith(("mt", "R008")) else "R003"
    seed = 14 if checkpoint.endswith("-s14") else 13
    record.update(
        origin=origin,
        seed=seed,
        checkpoint=checkpoint,
        batch_size=batch_size,
        status=status,
        latency_p50_ms=p50,
        latency_p95_ms=p50 * 2,
        latency_mean_ms=p50,
        cost_per_decision_p50_ms=p50,
        cost_per_decision_p95_ms=p50 * 2,
        peak_vram_allocated_mib=peak,
        peak_vram_reserved_mib=peak,
        oom_reason=None if status == "ok" else "CUDA out of memory at batch_size 16",
    )
    return record


def test_summarize_aggregation_math_on_synthetic_fixtures():
    records = [
        _fixture_record("mt-s13", 1, p50=20.0, peak=900.0),
        _fixture_record("mt-s13", 4, p50=8.0, peak=1100.0),
        _fixture_record("mt-s13", 16, status="dropped", p50=0.0, peak=0.0),
        _fixture_record("full-e2-8bit", 1, p50=15.0, peak=800.0),
        _fixture_record("full-e2-8bit", 8, p50=6.0, peak=1300.0),
    ]
    summary = summarize.build_summary(records)

    assert summary["study"] == "R017-efficiency"
    assert summary["configs_total"] == 5
    assert summary["configs_ok"] == 4
    assert summary["configs_dropped"] == 1
    assert summary["dropped"] == [
        {
            "checkpoint": "mt-s13",
            "batch_size": 16,
            "reason": "CUDA out of memory at batch_size 16",
        }
    ]

    by_name = {entry["checkpoint"]: entry for entry in summary["checkpoints"]}
    assert set(by_name) == {"full-e2-8bit", "mt-s13"}

    mt = by_name["mt-s13"]
    assert mt["origin"] == "R006" and mt["seed"] == 13
    assert [c["batch_size"] for c in mt["batch_configs"]] == [1, 4, 16]
    assert mt["fastest_batch_size"] == 4
    assert mt["fastest_latency_p50_ms"] == 8.0
    assert mt["peak_vram_allocated_mib_min"] == 900.0
    assert mt["peak_vram_allocated_mib_max"] == 1100.0
    assert mt["batch_configs"][2]["status"] == "dropped"

    e2 = by_name["full-e2-8bit"]
    assert e2["fastest_batch_size"] == 8
    assert e2["fastest_latency_p50_ms"] == 6.0


def test_summarize_writes_summary_json(tmp_path, monkeypatch):
    records = [_fixture_record("mt-s13", 4, p50=8.0, peak=1100.0)]
    out = tmp_path / "summary.json"
    summarize.write_json(out, summarize.build_summary(records))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["configs_ok"] == 1
    assert data["checkpoints"][0]["batch_configs"][0]["latency_p50_ms"] == 8.0


def test_summarize_rejects_incomplete_records():
    record = base_record()
    del record["latency_p50_ms"]
    with pytest.raises(ValueError, match="missing key"):
        summarize.build_summary([record])


def test_render_markdown_includes_limitations_and_dropped():
    records = [
        _fixture_record("mt-s13", 16, status="dropped"),
        _fixture_record("mt-s13", 1),
    ]
    summary = summarize.build_summary(records)
    table = summarize.render_markdown(summary)
    assert "dropped" in table
    assert "not comparable to historical reports" in table
    assert "RTX 3050" in table
