"""Evaluate the base, R002 and R003 checkpoints on one prepared dataset.

Usage: uv run python scripts/evaluate.py DATA_JSONL OUT_DIR [MODEL,...]
       [--manifest FROZEN_MANIFEST] [--verify-only]

R007 canonical exports require a locked manifest. --verify-only checks frozen
inputs and budgets without loading weights or making any predictions.

Writes OUT_DIR/{model}.json (laya-evals report) for each model. No training.
The default models are base, R002 and R003; controls must be named explicitly.
Optional env MAX_LEN / HEAD_MAX_LEN override the checkpoints' token budgets
for every call (R005 long-head control); they are recorded in the report.
Models are unloaded between runs. Loading and one warm-up pass over the first
8 examples are outside scored timing and inference peak-memory measurement.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import laya
import torch
from laya import evals
from laya.evals_cli import RouterRunner
from r007_identity import attach_case_identity_tags

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset")
parser.add_argument("out_dir", type=Path)
parser.add_argument("models", nargs="?", default="multilingual,head-e2,full-e2-8bit")
parser.add_argument("--manifest", type=Path)
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()
DATASET, OUT_DIR = args.dataset, args.out_dir

BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
CHECKPOINTS = {
    "head-e2": "artifacts/checkpoints/R002-head",
    "full-e2-8bit": "artifacts/checkpoints/R003-full-e2-8bit",
    # R004 control: base weights with max_len 512 (see research/R004-cross-task).
    "control-base-maxlen512": "artifacts/checkpoints/control-base-maxlen512",
    "mt-s13": "artifacts/checkpoints/R006-mt-s13",
    "mt-s14": "artifacts/checkpoints/R006-mt-s14",
}
MODELS = tuple(args.models.split(","))
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


def verify_frozen_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "locked_exploratory":
        raise ValueError("Frozen scoring requires a locked R007 manifest")
    if (
        sha256(Path(__file__).with_name("prepare_r007.py"))
        != manifest["preparer_sha256"]
    ):
        raise ValueError("Frozen preparer fingerprint mismatch")
    if tuple(MODELS) != tuple(manifest["checkpoint_fingerprints"]):
        raise ValueError("Frozen checkpoint selection/order mismatch")
    selected = None
    selected_ids = None
    for target in manifest["targets"].values():
        files = target["export_sha256"]["test"]
        for relative in files:
            if Path(relative).resolve() == Path(
                DATASET
            ).resolve() and relative.endswith(".jsonl"):
                selected = target
                selected_ids = next(
                    (name for name in files if name.endswith("-ids.json")), None
                )
    if selected is None or selected_ids is None:
        raise ValueError("Dataset is not a frozen target test export")
    if BUDGET != {key: selected[key] for key in ("max_len", "head_max_len")}:
        raise ValueError("Frozen token budget mismatch")
    for target in manifest["targets"].values():
        for files in target["export_sha256"].values():
            for relative, digest in files.items():
                if sha256(relative) != digest:
                    raise ValueError(f"Frozen export fingerprint mismatch: {relative}")
    for model, checkpoint in manifest["checkpoint_fingerprints"].items():
        directory = Path(CHECKPOINTS[model])
        if directory.resolve() != Path(checkpoint["directory"]).resolve():
            raise ValueError(f"Frozen checkpoint directory mismatch: {model}")
        required = {
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        }
        if not required <= checkpoint["sha256"].keys():
            raise ValueError(f"Incomplete frozen checkpoint fingerprints: {model}")
        for relative, digest in checkpoint["sha256"].items():
            if sha256(directory / relative) != digest:
                raise ValueError(
                    f"Frozen checkpoint fingerprint mismatch: {model}/{relative}"
                )
    return sha256(path), selected_ids


if Path(DATASET).name.startswith("r007-") and args.manifest is None:
    parser.error("R007 exports require --manifest")
if args.verify_only and args.manifest is None:
    parser.error("--verify-only requires --manifest")
manifest_sha256, identity_provenance_path = (
    verify_frozen_manifest(args.manifest) if args.manifest else (None, None)
)
if args.verify_only:
    print("Frozen inputs and budgets verified; no predictions generated.")
    sys.exit(0)

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
if not dataset.examples:
    raise ValueError("Evaluation dataset is empty")
if identity_provenance_path is not None:
    raw_rows = [json.loads(line) for line in Path(DATASET).read_text().splitlines()]
    provenance = json.loads(Path(identity_provenance_path).read_text())
    attach_case_identity_tags(dataset, raw_rows, provenance)
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

    # max_loaded=1 evicts only after building the next model, so explicitly
    # unload first to avoid counting two checkpoints in the next model's peak.
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
            "dataset": DATASET,
            "frozen_manifest_sha256": manifest_sha256,
            "dataset_sha256": evals.file_fingerprint(DATASET),
            "model": name,
            "base_revision": BASE_REVISION,
            "weights_sha256": weights.get(name),
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "budget_override": BUDGET or None,
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
for metric in (
    "choice_accuracy",
    "ece",
    "brier",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_decision_p50_ms",
):
    values = [results[name].get(metric, float("nan")) for name in MODELS]
    print(f"{metric:<28}" + "".join(f"{value:>14.4f}" for value in values))
