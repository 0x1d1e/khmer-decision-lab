"""Evaluate the base, R002 and R003 checkpoints on one prepared dataset.

Usage: uv run python scripts/evaluate.py DATA_JSONL OUT_DIR [MODEL,...]

Writes OUT_DIR/{model}.json (laya-evals report) for each model. No training.
The default models are base, R002 and R003; controls must be named explicitly.
Optional env MAX_LEN / HEAD_MAX_LEN override the checkpoints' token budgets
for every call (R005 long-head control); they are recorded in the report.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import laya
import torch
from laya import evals
from laya.evals_cli import RouterRunner

DATASET, OUT_DIR = sys.argv[1], Path(sys.argv[2])

BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
CHECKPOINTS = {
    "head-e2": "artifacts/checkpoints/R002-head",
    "full-e2-8bit": "artifacts/checkpoints/R003-full-e2-8bit",
    # R004 control: base weights with max_len 512 (see research/R004-cross-task).
    "control-base-maxlen512": "artifacts/checkpoints/control-base-maxlen512",
}
MODELS = ("multilingual", "head-e2", "full-e2-8bit")
if len(sys.argv) > 3:
    MODELS = tuple(sys.argv[3].split(","))
BUDGET = {
    key: int(os.environ[env])
    for key, env in (("max_len", "MAX_LEN"), ("head_max_len", "HEAD_MAX_LEN"))
    if env in os.environ
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


models = {
    name: str(Path(path).resolve())
    for name, path in CHECKPOINTS.items()
    if name in MODELS
}
for name, path in models.items():
    assert Path(path).is_dir(), f"Missing checkpoint: {name}"
weights = {
    name: sha256(Path(path) / "model.safetensors") for name, path in models.items()
}


def set_budget(ctx):
    for key, value in BUDGET.items():
        setattr(ctx, key, value)


dataset = evals.Dataset.from_jsonl(DATASET)
router = laya.Router(
    models=models,
    revisions={"multilingual": BASE_REVISION},
    device="cuda",
    max_loaded=1,
    preload=False,
    on_predict_start=set_budget if BUDGET else None,
)

results = {}
for name in MODELS:
    print(f"\n=== {name} ===", flush=True)
    for example in dataset.examples:
        example.model = name

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report = evals.evaluate(
        RouterRunner(router),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=8,
        config={
            "dataset": DATASET,
            "dataset_sha256": evals.file_fingerprint(DATASET),
            "model": name,
            "base_revision": BASE_REVISION,
            "weights_sha256": weights.get(name),
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "budget_override": BUDGET or None,
        },
    )
    payload = report.to_json()
    payload["config"]["peak_vram_mib"] = round(
        torch.cuda.max_memory_allocated() / 2**20, 1
    )

    output = OUT_DIR / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    results[name] = report.overall
    print(f"peak_vram_mib {payload['config']['peak_vram_mib']}")
    for key, value in sorted(report.overall.items()):
        print(f"{key:<30} {value:.4f}")

print("\n=== Comparison ===")
print(f"{'Metric':<28}" + "".join(f"{name:>14}" for name in MODELS))
for metric in ("choice_accuracy", "ece", "brier", "latency_p50_ms", "latency_p95_ms"):
    values = [results[name].get(metric, float("nan")) for name in MODELS]
    print(f"{metric:<28}" + "".join(f"{value:>14.4f}" for value in values))
