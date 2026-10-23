"""Aggregate R017 efficiency measurements into summary.json and a table.

Usage: uv run --frozen python scripts/summarize_r017.py
       [--measurements artifacts/runs/R017/measurements]
       [--out artifacts/runs/R017/summary.json]

Reads one JSON per (checkpoint, batch size) written by
scripts/measure_r017_efficiency.py and writes summary.json plus a markdown
table on stdout. No GPU is needed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_MEASUREMENTS = ROOT / "artifacts/runs/R017/measurements"
DEFAULT_OUT = ROOT / "artifacts/runs/R017/summary.json"

LIMITATIONS = (
    "Numbers are specific to one RTX 3050 (6 GB), one dataset "
    "(data/processed/massive-km-test-full.jsonl, 2,974 rows), and one batch "
    "configuration per run. They are not comparable to historical reports that "
    "include cold loading or count a preceding model's peak VRAM, and they do "
    "not generalize beyond this hardware, dataset, and configuration."
)


def load_measurements(directory=DEFAULT_MEASUREMENTS):
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def validate_record(record):
    required = (
        "checkpoint",
        "origin",
        "seed",
        "checkpoint_path",
        "weights_sha256",
        "dataset",
        "dataset_sha256",
        "dataset_rows",
        "batch_size",
        "warmup_examples",
        "gpu",
        "torch_version",
        "cuda_version",
        "status",
        "model_load_ms",
        "warmup_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "cost_per_decision_p50_ms",
        "cost_per_decision_p95_ms",
        "peak_vram_allocated_mib",
        "peak_vram_reserved_mib",
    )
    if not isinstance(record, dict):
        raise TypeError("Measurement record must be a JSON object")
    for key in required:
        if key not in record:
            raise ValueError(f"Measurement record missing key: {key}")


def group_by_checkpoint(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["checkpoint"], []).append(record)
    return grouped


def _config_row(record, file_name):
    if record["status"] == "ok":
        return {
            "batch_size": record["batch_size"],
            "status": "ok",
            "n_rows": record["dataset_rows"],
            "latency_p50_ms": record["latency_p50_ms"],
            "latency_p95_ms": record["latency_p95_ms"],
            "latency_mean_ms": record["latency_mean_ms"],
            "cost_per_decision_p50_ms": record["cost_per_decision_p50_ms"],
            "cost_per_decision_p95_ms": record["cost_per_decision_p95_ms"],
            "peak_vram_allocated_mib": record["peak_vram_allocated_mib"],
            "peak_vram_reserved_mib": record["peak_vram_reserved_mib"],
            "file": file_name,
        }
    return {
        "batch_size": record["batch_size"],
        "status": "dropped",
        "reason": record.get("oom_reason"),
        "file": None,
    }


def build_summary(records):
    for record in records:
        validate_record(record)
    if not records:
        raise ValueError("No measurement records found")

    base = records[0]
    grouped = group_by_checkpoint(records)
    checkpoints = []
    dropped = []
    ok_count = 0

    for name in sorted(grouped):
        recs = sorted(grouped[name], key=lambda r: r["batch_size"])
        configs = []
        for record in recs:
            file_name = (
                f"measurements/{record['checkpoint']}_bs{record['batch_size']}.json"
            )
            row = _config_row(record, file_name)
            configs.append(row)
            if record["status"] == "ok":
                ok_count += 1
            else:
                dropped.append(
                    {
                        "checkpoint": name,
                        "batch_size": record["batch_size"],
                        "reason": record.get("oom_reason"),
                    }
                )
        ok_configs = [c for c in configs if c["status"] == "ok"]
        fastest = (
            min(ok_configs, key=lambda c: c["latency_p50_ms"]) if ok_configs else None
        )
        peaks = [c["peak_vram_allocated_mib"] for c in ok_configs]
        checkpoints.append(
            {
                "checkpoint": name,
                "origin": recs[0]["origin"],
                "seed": recs[0]["seed"],
                "checkpoint_path": recs[0]["checkpoint_path"],
                "weights_sha256": recs[0]["weights_sha256"],
                "model_load_ms": recs[0]["model_load_ms"],
                "warmup_ms": recs[0]["warmup_ms"],
                "warmup_examples": recs[0]["warmup_examples"],
                "batch_configs": configs,
                "fastest_batch_size": fastest["batch_size"] if fastest else None,
                "fastest_latency_p50_ms": fastest["latency_p50_ms"]
                if fastest
                else None,
                "peak_vram_allocated_mib_min": min(peaks) if peaks else None,
                "peak_vram_allocated_mib_max": max(peaks) if peaks else None,
            }
        )

    return {
        "study": "R017-efficiency",
        "measurement_protocol": "isolated-warm-v1",
        "dataset": base["dataset"],
        "dataset_sha256": base["dataset_sha256"],
        "dataset_rows": base["dataset_rows"],
        "gpu": base["gpu"],
        "torch_version": base["torch_version"],
        "cuda_version": base["cuda_version"],
        "configs_total": len(records),
        "configs_ok": ok_count,
        "configs_dropped": len(dropped),
        "dropped": dropped,
        "checkpoints": checkpoints,
        "limitations": LIMITATIONS,
    }


def render_markdown(summary):
    lines = [
        "# R017 efficiency results",
        "",
        (
            f"Protocol: {summary['measurement_protocol']}  |  "
            f"Dataset: {summary['dataset']} ({summary['dataset_rows']} rows)  |  "
            f"GPU: {summary['gpu']}"
        ),
        "",
        (
            "Checkpoint | Origin | Seed | BS | p50 ms | p95 ms | mean ms/row | "
            "cost p50 ms/row | peak alloc MiB | peak reserv MiB | load ms | warmup ms | Status"
        ),
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
    ]
    for entry in summary["checkpoints"]:
        load = (
            f"{entry['model_load_ms']:.0f}"
            if entry["model_load_ms"] is not None
            else "-"
        )
        warmup = f"{entry['warmup_ms']:.0f}" if entry["warmup_ms"] is not None else "-"
        for config in entry["batch_configs"]:
            if config["status"] == "ok":
                lines.append(
                    f"{entry['checkpoint']} | {entry['origin']} | {entry['seed']} | "
                    f"{config['batch_size']} | {config['latency_p50_ms']:.2f} | "
                    f"{config['latency_p95_ms']:.2f} | {config['latency_mean_ms']:.2f} | "
                    f"{config['cost_per_decision_p50_ms']:.2f} | "
                    f"{config['peak_vram_allocated_mib']:.1f} | "
                    f"{config['peak_vram_reserved_mib']:.1f} | "
                    f"{load} | {warmup} | ok"
                )
            else:
                lines.append(
                    f"{entry['checkpoint']} | {entry['origin']} | {entry['seed']} | "
                    f"{config['batch_size']} | - | - | - | - | - | - | "
                    f"{load} | {warmup} | dropped ({config.get('reason')})"
                )
    lines.append("")
    lines.append("## Dropped configurations")
    if summary["dropped"]:
        for item in summary["dropped"]:
            lines.append(
                f"- {item['checkpoint']} batch_size={item['batch_size']}: {item['reason']}"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_measurements(args.measurements)
    summary = build_summary(records)
    write_json(args.out, summary)
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
