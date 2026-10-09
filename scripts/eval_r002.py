import json
from pathlib import Path

import laya
from laya import evals
from laya.evals_cli import RouterRunner

DATASET = "data/processed/massive-km-test-full.jsonl"

CHECKPOINTS = {
    "head-e2": "artifacts/checkpoints/R002-head",
    "full-e2-8bit": "artifacts/checkpoints/R003-full-e2-8bit",
}

dataset = evals.Dataset.from_jsonl(DATASET)

models = {
    name: str(Path(path).resolve())
    for name, path in CHECKPOINTS.items()
}

for name, path in models.items():
    assert Path(path).is_dir(), f"Missing checkpoint: {name}"

router = laya.Router(
    models=models,
    device="cuda",
    max_loaded=1,
    preload=False,
)

results = {}

for name in ("multilingual", "head-e2", "full-e2-8bit"):
    print(f"\n=== {name} ===", flush=True)

    for example in dataset.examples:
        example.model = name

    report = evals.evaluate(
        RouterRunner(router),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=8,
        config={
            "dataset": DATASET,
            "dataset_sha256": evals.file_fingerprint(DATASET),
            "model": name,
            "device": "cuda",
        },
    )

    output = Path(f"artifacts/runs/{name}-test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results[name] = report.overall

    for key, value in sorted(report.overall.items()):
        print(f"{key:<30} {value:.4f}")

print("\n=== Comparison ===")
print(f"{'Metric':<28} {'Base':>10} {'Head E2':>10} {'Full E2':>10}")

for metric in (
    "choice_accuracy",
    "ece",
    "brier",
    "latency_p50_ms",
    "latency_p95_ms",
):
    values = [
        results[name].get(metric, float("nan"))
        for name in ("multilingual", "head-e2", "full-e2-8bit")
    ]
    print(
        f"{metric:<28}"
        + "".join(f"{value:>10.4f}" for value in values)
    )
