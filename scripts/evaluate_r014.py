"""Score the R014 SEA-NLI Khmer NLI exports with four fixed models.

R014-local evaluator, modelled on `scripts/evaluate.py` (which is untouched).
Four models are scored on the two Khmer exports and the mixed non-Khmer export:
the frozen multilingual base, the two R006 multitask seeds, and the R003
single-task checkpoint. No training, tuning, or calibration.

Usage: uv run --frozen python scripts/evaluate_r014.py

`r014-group:<group_id>` tags are attached to every example by reading the
prepared sidecar id maps, so each group identity survives into `cases[].tags`.
Models unload before the next load, and loading plus a one-pass warm-up are
outside scored timing and peak-memory measurement (`isolated-warm-v1`). The
warm-up runs the model that is actually being scored: every example carries its
target model name before the checkpoint loads. Token limits are pinned to
`max_len=512` and `head_max_len=256` and applied to every call, so the frozen
base (whose config may allow 1024) and the adapted checkpoints are scored under
one shared context window. The three adapted checkpoints are locked to captured
`model.safetensors` digests and the lock is asserted before loading and again
before scoring.

Writes artifacts/runs/R014/<condition>/<model>.json.
"""

import hashlib
import json
import time
from pathlib import Path

import laya
import torch
from huggingface_hub import snapshot_download
from laya import evals
from laya.evals_cli import RouterRunner

ROOT = Path(__file__).resolve().parents[1]

BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
CHECKPOINTS = {
    "mt-s13": "artifacts/checkpoints/R006-mt-s13",
    "mt-s14": "artifacts/checkpoints/R006-mt-s14",
    "full-e2-8bit": "artifacts/checkpoints/R003-full-e2-8bit",
}
MODELS = ("multilingual", "mt-s13", "mt-s14", "full-e2-8bit")

DATASETS = {
    "km-normal": "data/processed/sea-nli-km-normal-test.jsonl",
    "km-hard": "data/processed/sea-nli-km-hard-test.jsonl",
    "en-normal": "data/processed/sea-nli-en-normal-test.jsonl",
    "en-hard": "data/processed/sea-nli-en-hard-test.jsonl",
    "other": "data/processed/sea-nli-other-test.jsonl",
}
IDS = {
    "km-normal": "data/processed/sea-nli-km-normal-test-ids.json",
    "km-hard": "data/processed/sea-nli-km-hard-test-ids.json",
    "en-normal": "data/processed/sea-nli-en-normal-test-ids.json",
    "en-hard": "data/processed/sea-nli-en-hard-test-ids.json",
    "other": "data/processed/sea-nli-other-test-ids.json",
}
CONDITIONS = ("km-normal", "km-hard", "en-normal", "en-hard", "other")

BATCH_SIZE = 8
TAG_PREFIX = "r014-group:"
OUT = ROOT / "artifacts/runs/R014"

# Pinned inference context window (R013 pinned the same pair for the same
# reason). The frozen base config may permit max_len 1024 while the adapted
# checkpoints use 512; every call gets these limits so the arms are comparable.
MAX_LEN = 512
HEAD_MAX_LEN = 256
BUDGET = {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}

# sha256 of each adapted checkpoint's model.safetensors, captured at staging
# time from the frozen local R003/R006 artifacts. A mismatch is a hard failure:
# scoring a swapped or corrupted checkpoint under an R003/R006 name is worse
# than not scoring at all.
EXPECTED_WEIGHTS = {
    "full-e2-8bit": (
        "b5c682c025001fdd2d7f5dcbfeef7a7022edc1bfe7e87fbafc08a8907b587f93"
    ),
    "mt-s13": "ae4042ab1eb5748d5618f3996e222fd696c617d293a430717c79a1bbdb492da9",
    "mt-s14": "e2ff8c7d7149af6692e653e35ed625781919789c35ac6a6017ee01a5062e5d92",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def assert_pinned_checkpoint(name, path):
    """Re-hash one pinned checkpoint and require the captured digest."""
    expected = EXPECTED_WEIGHTS[name]
    actual = sha256(Path(path) / "model.safetensors")
    if actual != expected:
        raise ValueError(
            f"R014 checkpoint weights do not match the pinned digest: {name}"
        )
    return actual


def load_condition(condition):
    """Load one prepared export and attach its sidecar group tags."""
    dataset_path = ROOT / DATASETS[condition]
    ids_path = ROOT / IDS[condition]
    if not dataset_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            f"R014 prepared export is missing for {condition}: "
            f"{DATASETS[condition]} / {IDS[condition]}. Run prepare_r014.py first."
        )
    dataset = evals.Dataset.from_jsonl(str(dataset_path))
    sidecar = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, list) or len(sidecar) != len(dataset.examples):
        raise ValueError(f"R014 sidecar does not align with {DATASETS[condition]}")
    for example, entry in zip(dataset.examples, sidecar, strict=True):
        if not isinstance(entry, dict):
            raise TypeError(f"R014 sidecar entry is invalid: {DATASETS[condition]}")
        group = entry.get("group_id")
        if not isinstance(group, str) or not group:
            raise ValueError(f"R014 sidecar entry is invalid: {DATASETS[condition]}")
        if any(tag.startswith(TAG_PREFIX) for tag in example.tags):
            raise ValueError("R014 group tag already present")
        example.tags = (*example.tags, f"{TAG_PREFIX}{group}")
    return dataset, sha256(ids_path)


def resolve_models():
    exempt = set(MODELS) - set(EXPECTED_WEIGHTS)
    if exempt != {"multilingual"}:
        raise ValueError(
            f"R014 weight lock exempts unexpected models: {sorted(exempt)}; "
            "only the frozen multilingual base may be unpinned"
        )
    checkpoints = {
        name: str((ROOT / relative).resolve()) for name, relative in CHECKPOINTS.items()
    }
    for name, path in checkpoints.items():
        assert Path(path).is_dir(), f"Missing checkpoint: {name}"
    snapshot = Path(
        snapshot_download(
            BASE_REPO, revision=BASE_REVISION, allow_patterns="multilingual/*"
        )
    )
    checkpoints["multilingual"] = str((snapshot / "multilingual").resolve())
    weights = {}
    for name, path in checkpoints.items():
        digest = sha256(Path(path) / "model.safetensors")
        expected = EXPECTED_WEIGHTS.get(name)
        if expected is not None and digest != expected:
            raise ValueError(
                f"R014 checkpoint weights do not match the pinned digest: {name}"
            )
        weights[name] = digest
    return checkpoints, weights


def evaluate():
    conditions = {condition: load_condition(condition) for condition in CONDITIONS}
    checkpoints, weights = resolve_models()
    models = {name: checkpoints[name] for name in CHECKPOINTS}

    def set_budget(ctx):
        for key, value in BUDGET.items():
            setattr(ctx, key, value)

    router = laya.Router(
        models=models,
        revisions={"multilingual": BASE_REVISION},
        device="cuda",
        max_loaded=1,
        preload=False,
        on_predict_start=set_budget,
    )
    reports = []
    for name in MODELS:
        # Route every example to the model being scored before it loads, so the
        # warm-up exercises the target checkpoint instead of auto-routing or a
        # stale previous name (which would evict and re-load it while timed).
        for condition in CONDITIONS:
            for example in conditions[condition][0].examples:
                example.model = name
        if name in EXPECTED_WEIGHTS:
            assert_pinned_checkpoint(name, checkpoints[name])
        router.unload()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        started = time.perf_counter()
        router.load(name)
        torch.cuda.synchronize()
        model_load_ms = (time.perf_counter() - started) * 1000

        runner = RouterRunner(router)
        warmup = evals.Dataset(examples=conditions[CONDITIONS[0]][0].examples[:8])
        started = time.perf_counter()
        evals.evaluate(runner, warmup, evaluators=[], batch_size=BATCH_SIZE)
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - started) * 1000

        for condition in CONDITIONS:
            dataset, sidecar_sha256 = conditions[condition]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report = evals.evaluate(
                runner,
                dataset,
                evaluators=evals.default_evaluators(),
                batch_size=BATCH_SIZE,
                config={
                    "study": "R014-sea-nli-khmer-transfer",
                    "condition": condition,
                    "dataset": DATASETS[condition],
                    "dataset_sha256": evals.file_fingerprint(
                        str(ROOT / DATASETS[condition])
                    ),
                    "sidecar": IDS[condition],
                    "sidecar_sha256": sidecar_sha256,
                    "group_tag_prefix": TAG_PREFIX,
                    "groups": len(
                        {
                            tag
                            for ex in dataset.examples
                            for tag in ex.tags
                            if tag.startswith(TAG_PREFIX)
                        }
                    ),
                    "model": name,
                    "model_list": list(MODELS),
                    "checkpoint": checkpoints[name],
                    "base_revision": BASE_REVISION,
                    "weights_sha256": weights[name],
                    "device": "cuda",
                    "gpu": torch.cuda.get_device_name(0),
                    "max_len": MAX_LEN,
                    "head_max_len": HEAD_MAX_LEN,
                    "batch_size": BATCH_SIZE,
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
            output = OUT / condition / f"{name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            reports.append(str(output.relative_to(ROOT)))
            print(
                f"{condition:>10} {name:>14} n={len(payload['cases'])} "
                f"accuracy={payload['overall']['choice_accuracy']:.4f}",
                flush=True,
            )
    router.unload()
    return reports


def main():
    reports = evaluate()
    print(f"R014 scoring wrote {len(reports)} reports under artifacts/runs/R014")


if __name__ == "__main__":
    main()
