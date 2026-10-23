"""Score R016 budget-matched SIB-200 specialists against R008 multitask baselines.

Usage: uv run python scripts/evaluate_r016_budget_specialist.py [OUT_DIR]

Scores the R016 budget-matched specialists and the R008 multitask checkpoints
on the locked 204-row SIB-200 test split using evaluate.py's isolated-warm-v1
protocol: models are unloaded between runs, and loading plus one 8-example
warm-up pass are excluded from scored timing and peak-memory measurement.

Writes OUT_DIR/{model}.json (laya-evals report) for each model. No training,
no calibration -- each checkpoint is scored as shipped, so the specialist and
multitask arms are compared under one identical protocol.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import laya
import torch
from laya import evals
from laya.evals_cli import RouterRunner

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/sib200-km-test.jsonl"
DATASET_SHA256 = "6d539da9c2767143237bfc9f8cdf9a7395e9680400c3a761c9000a53b4f057cc"
DEFAULT_OUT_DIR = ROOT / "artifacts/runs/R016"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
MAX_LEN = 512
HEAD_MAX_LEN = 256
CHECKPOINTS = {
    "R016-budget-sib-s13": "artifacts/checkpoints/R016-budget-sib-s13",
    "R016-budget-sib-s14": "artifacts/checkpoints/R016-budget-sib-s14",
    "R008-mt-s13": "artifacts/checkpoints/R008-mt-s13",
    "R008-mt-s14": "artifacts/checkpoints/R008-mt-s14",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("models", nargs="?", default=",".join(CHECKPOINTS))
    return parser.parse_args()


def set_budget(ctx):
    ctx.max_len = MAX_LEN
    ctx.head_max_len = HEAD_MAX_LEN


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    models = tuple(name for name in args.models.split(",") if name)
    unknown = [name for name in models if name not in CHECKPOINTS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if not models:
        raise ValueError("no models selected")

    if sha256(DATASET) != DATASET_SHA256:
        raise ValueError("SIB-200 test file fingerprint mismatch")
    dataset = evals.Dataset.from_jsonl(str(DATASET))
    if len(dataset.examples) != 204:
        raise ValueError(f"SIB-200 test has {len(dataset.examples)} rows, expected 204")

    checkpoints = {
        name: str((ROOT / path).resolve())
        for name, path in CHECKPOINTS.items()
        if name in models
    }
    for name, path in checkpoints.items():
        assert Path(path).is_dir(), f"Missing checkpoint: {name}"
    weights = {
        name: sha256(Path(path) / "model.safetensors")
        for name, path in checkpoints.items()
    }

    router = laya.Router(
        models=checkpoints,
        revisions=None,
        device="cuda",
        max_loaded=1,
        preload=False,
        on_predict_start=set_budget,
    )

    results = {}
    for name in models:
        print(f"\n=== {name} ===", flush=True)
        for example in dataset.examples:
            example.model = name

        router.unload()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        started = time.perf_counter()
        router.load(name)
        torch.cuda.synchronize()
        model_load_ms = (time.perf_counter() - started) * 1000

        runner = RouterRunner(router)
        warmup = evals.Dataset(examples=dataset.examples[:8])
        started = time.perf_counter()
        evals.evaluate(runner, warmup, evaluators=[], batch_size=8)
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - started) * 1000
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        report = evals.evaluate(
            runner,
            dataset,
            evaluators=evals.default_evaluators(),
            batch_size=8,
            config={
                "study": "R016-budget-matched-specialist",
                "task": "sib200-km",
                "model": name,
                "checkpoint": checkpoints[name],
                "dataset": str(DATASET),
                "dataset_sha256": DATASET_SHA256,
                "weights_sha256": weights.get(name),
                "base_revision": BASE_REVISION,
                "max_len": MAX_LEN,
                "head_max_len": HEAD_MAX_LEN,
                "device": "cuda",
                "gpu": torch.cuda.get_device_name(0),
                "measurement_protocol": "isolated-warm-v1",
                "model_load_ms": round(model_load_ms, 4),
                "warmup_ms": round(warmup_ms, 4),
                "warmup_examples": len(warmup.examples),
                "peak_vram_scope": "single resident model, scored inference only",
                "brier_definition": "mean((P(reported answer is correct) - correct)**2)",
            },
        )
        payload = report.to_json()
        payload["config"]["peak_vram_mib"] = round(
            torch.cuda.max_memory_allocated() / 2**20, 1
        )

        output = out_dir / f"{name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results[name] = report.overall
        print(f"peak_vram_mib {payload['config']['peak_vram_mib']}")
        for key, value in sorted(report.overall.items()):
            print(f"{key:<30} {value:.4f}")

    print("\n=== Comparison ===")
    print(f"{'Model':<24}" + "".join(f"{name:>16}" for name in models))
    for metric in ("choice_accuracy", "ece", "brier", "aurc"):
        values = [results[name].get(metric, float("nan")) for name in models]
        print(f"{metric:<24}" + "".join(f"{value:>16.4f}" for value in values))


if __name__ == "__main__":
    main()
