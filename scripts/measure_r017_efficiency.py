"""Measure latency, cost, and peak VRAM of existing checkpoints (study R017).

Usage: uv run --frozen python scripts/measure_r017_efficiency.py [--dry-run]
       [--batch-sizes 1,4,8,16] [--models a,b,c] [--out-dir DIR]

Protocol (isolated-warm-v1), one checkpoint at a time, GPU-serialized under an
exclusive flock on /tmp/khmer-gpu.lock:

  1. load the model fresh,
  2. warm up on the first 8 examples (excluded from scored timing/memory),
  3. run data/processed/massive-km-test-full.jsonl (2,974 rows) at each fixed
     batch size,
  4. record latency p50/p95/mean, per-decision cost (ms/row), peak VRAM
     (allocated and reserved), load time, warm-up time,
  5. unload and repeat for the next checkpoint.

Batch sizes are run ascending; a config that OOMs is dropped with the reason
recorded and no larger batch size is attempted for that checkpoint. Writes one
JSON per (checkpoint, batch size) under --out-dir/measurements/ plus
preparation.json. --dry-run lists the plan without touching the GPU.

Every number is specific to this RTX 3050 (6 GB), this one dataset, and this
one batch configuration; results are not comparable to historical reports that
include cold loading or count a preceding model's peak VRAM.
"""

import argparse
import datetime
import fcntl
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

ROOT = Path(__file__).resolve().parents[1]

LOCK_FILE = "/tmp/khmer-gpu.lock"
DATASET = ROOT / "data/processed/massive-km-test-full.jsonl"
WARMUP_EXAMPLES = 8
WARMUP_BATCH_SIZE = 8
DEFAULT_BATCH_SIZES = (1, 4, 8, 16)
MAX_BATCH_SIZE = 64

# name -> (relative checkpoint path, study origin, seed)
CHECKPOINTS = {
    "full-e2-8bit": ("artifacts/checkpoints/R003-full-e2-8bit", "R003", 13),
    "full-e2-8bit-s14": ("artifacts/checkpoints/R003-full-e2-8bit-s14", "R003", 14),
    "mt-s13": ("artifacts/checkpoints/R006-mt-s13", "R006", 13),
    "mt-s14": ("artifacts/checkpoints/R006-mt-s14", "R006", 14),
    "R008-sib-s13": ("artifacts/checkpoints/R008-sib-s13", "R008", 13),
    "R008-mt-s13": ("artifacts/checkpoints/R008-mt-s13", "R008", 13),
    "R013-seen-only-s13": ("artifacts/checkpoints/R013-seen-only-s13", "R013", 13),
    "R013-all-label-s13": ("artifacts/checkpoints/R013-all-label-s13", "R013", 13),
}

MEASUREMENT_SCHEMA = {
    "study": str,
    "measurement_protocol": str,
    "checkpoint": str,
    "origin": str,
    "seed": int,
    "checkpoint_path": str,
    "weights_sha256": str,
    "dataset": str,
    "dataset_sha256": str,
    "dataset_rows": int,
    "batch_size": int,
    "warmup_examples": int,
    "gpu": str,
    "torch_version": str,
    "cuda_version": str,
    "status": str,
    "oom_reason": (str, type(None)),
    "model_load_ms": (float, int, type(None)),
    "warmup_ms": (float, int, type(None)),
    "latency_p50_ms": (float, int, type(None)),
    "latency_p95_ms": (float, int, type(None)),
    "latency_mean_ms": (float, int, type(None)),
    "cost_per_decision_p50_ms": (float, int, type(None)),
    "cost_per_decision_p95_ms": (float, int, type(None)),
    "peak_vram_allocated_mib": (float, int, type(None)),
    "peak_vram_reserved_mib": (float, int, type(None)),
    "timing": dict,
}

LIMITATIONS = (
    "Numbers are specific to one RTX 3050 (6 GB), one dataset "
    "(data/processed/massive-km-test-full.jsonl, 2,974 rows), and one batch "
    "configuration per run. They are not comparable to historical reports that "
    "include cold loading or count a preceding model's peak VRAM, and they do "
    "not generalize beyond this hardware, dataset, and configuration."
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_batch_sizes(batch_sizes):
    """Return a sorted, deduplicated list of batch sizes in [1, MAX_BATCH_SIZE]."""
    out = []
    for raw in batch_sizes:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid batch size: {raw!r} (must be an integer)")
        if isinstance(raw, bool) or value < 1 or value > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size out of range [1, {MAX_BATCH_SIZE}]: {raw!r}")
        if value not in out:
            out.append(value)
    out.sort()
    return out


def validate_checkpoint(name, relative):
    directory = (ROOT / relative).resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing checkpoint directory for {name}: {directory}")
    weights = directory / "model.safetensors"
    if not weights.is_file():
        raise ValueError(f"Missing model weights for {name}: {weights}")
    return directory


def load_plan(models=None, batch_sizes=DEFAULT_BATCH_SIZES, checkpoints=CHECKPOINTS):
    """Validate and build the measurement plan. Rejects unknown/missing checkpoints."""
    if models is None:
        selected = list(checkpoints.items())
    else:
        unknown = [m for m in models if m not in checkpoints]
        if unknown:
            raise ValueError(f"Unknown checkpoints: {', '.join(sorted(unknown))}")
        selected = [(name, checkpoints[name]) for name in models]

    batches = validate_batch_sizes(batch_sizes)
    if not batches:
        raise ValueError("At least one batch size is required")

    plan = []
    for name, (relative, origin, seed) in selected:
        directory = validate_checkpoint(name, relative)
        plan.append(
            {
                "checkpoint": name,
                "origin": origin,
                "seed": seed,
                "checkpoint_path": str(directory),
                "weights_path": str(directory / "model.safetensors"),
                "weights_sha256": sha256(directory / "model.safetensors"),
            }
        )
    return plan, batches


def _valid_type(value, expected):
    if isinstance(expected, tuple):
        return any(_valid_type(value, t) for t in expected)
    if expected is type(None):
        return value is None
    if isinstance(value, bool):
        return expected is bool
    return isinstance(value, expected)


def validate_measurement_record(record):
    if not isinstance(record, dict):
        raise TypeError("Measurement record must be a JSON object")
    for key, expected in MEASUREMENT_SCHEMA.items():
        if key not in record:
            raise ValueError(f"Measurement record missing key: {key}")
        if not _valid_type(record[key], expected):
            raise TypeError(
                f"Measurement record key {key!r} has invalid type "
                f"{type(record[key]).__name__}"
            )
    extra = set(record) - set(MEASUREMENT_SCHEMA)
    if extra:
        raise ValueError(f"Measurement record has unexpected keys: {sorted(extra)}")


def base_record(plan_entry, dataset, model_load_ms, warmup_ms, batch_size):
    return {
        "study": "R017-efficiency",
        "measurement_protocol": "isolated-warm-v1",
        "checkpoint": plan_entry["checkpoint"],
        "origin": plan_entry["origin"],
        "seed": plan_entry["seed"],
        "checkpoint_path": plan_entry["checkpoint_path"],
        "weights_sha256": plan_entry["weights_sha256"],
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "dataset_rows": len(dataset.examples),
        "batch_size": batch_size,
        "warmup_examples": WARMUP_EXAMPLES,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_load_ms": round(float(model_load_ms), 4),
        "warmup_ms": round(float(warmup_ms), 4),
        "status": "ok",
        "oom_reason": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_mean_ms": None,
        "cost_per_decision_p50_ms": None,
        "cost_per_decision_p95_ms": None,
        "peak_vram_allocated_mib": None,
        "peak_vram_reserved_mib": None,
        "timing": None,
    }


def _is_cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run_checkpoint(plan_entry, dataset, batch_sizes, measurements_dir):
    """Load once, warm up once, score each batch size, then unload."""
    name = plan_entry["checkpoint"]
    router = laya.Router(
        models={name: plan_entry["checkpoint_path"]},
        device="cuda",
        max_loaded=1,
        preload=False,
    )
    router.unload()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    started = time.perf_counter()
    router.load(name)
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - started) * 1000

    warmup = evals.Dataset(examples=dataset.examples[:WARMUP_EXAMPLES])
    runner = RouterRunner(router)
    started = time.perf_counter()
    evals.evaluate(runner, warmup, evaluators=[], batch_size=WARMUP_BATCH_SIZE)
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - started) * 1000

    results = []
    for batch_size in batch_sizes:
        record = base_record(plan_entry, dataset, model_load_ms, warmup_ms, batch_size)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            report = evals.evaluate(
                runner,
                dataset,
                evaluators=evals.default_evaluators(),
                batch_size=batch_size,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            overall = report.overall
            record.update(
                {
                    "status": "ok",
                    "n_rows": len(dataset.examples),
                    "latency_p50_ms": round(float(overall["latency_p50_ms"]), 4),
                    "latency_p95_ms": round(float(overall["latency_p95_ms"]), 4),
                    "latency_mean_ms": round(elapsed_ms / len(dataset.examples), 4),
                    "cost_per_decision_p50_ms": round(
                        float(overall["cost_per_decision_p50_ms"]), 4
                    ),
                    "cost_per_decision_p95_ms": round(
                        float(overall["cost_per_decision_p95_ms"]), 4
                    ),
                    "peak_vram_allocated_mib": round(
                        torch.cuda.max_memory_allocated() / 2**20, 1
                    ),
                    "peak_vram_reserved_mib": round(
                        torch.cuda.max_memory_reserved() / 2**20, 1
                    ),
                    "timing": report.config["timing"],
                }
            )
            print(
                f"{name:>20} bs={batch_size:<3} "
                f"p50={record['latency_p50_ms']:.2f}ms "
                f"p95={record['latency_p95_ms']:.2f}ms "
                f"mean={record['latency_mean_ms']:.2f}ms/row "
                f"peak={record['peak_vram_allocated_mib']:.1f}MiB",
                flush=True,
            )
        except Exception as exc:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if not _is_cuda_oom(exc):
                router.unload()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                raise
            record.update(
                {
                    "status": "dropped",
                    "oom_reason": (
                        f"CUDA out of memory at batch_size {batch_size}: {exc}"
                    ),
                }
            )
            print(
                f"{name:>20} bs={batch_size:<3} DROPPED OOM: {exc}",
                flush=True,
            )
            # Larger batches allocate at least as much, so stop here for this
            # checkpoint and let the next checkpoint load fresh.
            results.append(record)
            write_json(measurements_dir / f"{name}_bs{batch_size}.json", record)
            break

        write_json(measurements_dir / f"{name}_bs{batch_size}.json", record)
        results.append(record)

    router.unload()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return results


def preparation_payload(plan, batch_sizes, status, measured=None):
    payload = {
        "study": "R017-efficiency",
        "status": status,
        "measurement_protocol": "isolated-warm-v1",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "dataset_rows": None,  # filled in after the dataset is loaded
        "batch_configs": list(batch_sizes),
        "limitations": LIMITATIONS,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoints": [
            {
                "checkpoint": entry["checkpoint"],
                "origin": entry["origin"],
                "seed": entry["seed"],
                "checkpoint_path": entry["checkpoint_path"],
                "weights_sha256": entry["weights_sha256"],
            }
            for entry in plan
        ],
        "implementation_sha256": {
            "scripts/measure_r017_efficiency.py": sha256(
                ROOT / "scripts/measure_r017_efficiency.py"
            ),
            "scripts/summarize_r017.py": sha256(ROOT / "scripts/summarize_r017.py"),
        },
        "measurements": "artifacts/runs/R017/measurements/",
        "summary": "artifacts/runs/R017/summary.json",
    }
    if measured is not None:
        payload["configs_measured"] = measured["ok"]
        payload["configs_dropped"] = measured["dropped"]
        payload["configs_total"] = measured["total"]
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES))
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/runs/R017")
    args = parser.parse_args()

    batch_sizes = [part for part in args.batch_sizes.split(",") if part.strip()]
    models = (
        [part for part in args.models.split(",") if part.strip()]
        if args.models
        else None
    )

    try:
        plan, batch_sizes = load_plan(models, batch_sizes)
    except ValueError as exc:
        parser.error(str(exc))

    dataset = evals.Dataset.from_jsonl(str(DATASET))
    if not dataset.examples:
        raise ValueError("Evaluation dataset is empty")

    if args.dry_run:
        print("R017 efficiency measurement plan (dry run, no weights loaded):")
        print(f"  dataset: {DATASET} ({len(dataset.examples)} rows)")
        print(f"  batch sizes: {batch_sizes}")
        print(
            f"  warm-up: {WARMUP_EXAMPLES} examples at batch_size {WARMUP_BATCH_SIZE}"
        )
        print("  checkpoints:")
        for entry in plan:
            print(
                f"    - {entry['checkpoint']:<22} "
                f"{entry['checkpoint_path']} "
                f"(origin={entry['origin']}, seed={entry['seed']}, "
                f"weights_sha256={entry['weights_sha256'][:16]}...)"
            )
        return 0

    if not torch.cuda.is_available():
        raise ValueError("R017 measurement requires CUDA")

    out_dir = args.out_dir.resolve()
    measurements_dir = out_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)

    preparation = preparation_payload(plan, batch_sizes, "in_progress")
    preparation["dataset_rows"] = len(dataset.examples)
    write_json(out_dir / "preparation.json", preparation)

    lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        all_records = []
        for entry in plan:
            print(f"\n=== {entry['checkpoint']} ===", flush=True)
            all_records.extend(
                run_checkpoint(entry, dataset, batch_sizes, measurements_dir)
            )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    measured = {
        "ok": sum(1 for r in all_records if r["status"] == "ok"),
        "dropped": sum(1 for r in all_records if r["status"] == "dropped"),
        "total": len(all_records),
    }
    preparation = preparation_payload(plan, batch_sizes, "completed", measured)
    preparation["dataset_rows"] = len(dataset.examples)
    preparation["gpu"] = torch.cuda.get_device_name(0)
    write_json(out_dir / "preparation.json", preparation)

    print(
        f"\n=== R017 complete: {measured['ok']} configs measured, "
        f"{measured['dropped']} dropped, {measured['total']} total ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
