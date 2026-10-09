"""Score one locked R008 checkpoint on its matching task's test split.

Usage: uv run --frozen python scripts/evaluate_r008.py CHECKPOINT TASK
TASK is khmer-news or sib200-km. No training or calibration is performed.
"""

import argparse
import json
from pathlib import Path

import laya
import torch
from laya import evals
from r008_identity import (
    ARM_TASKS,
    INFERENCE_POLICY,
    ROOT,
    load_r008_agent,
    row_identity_tag,
    sha256,
    verify_locked_manifest,
    verify_study_readiness,
)

MANIFEST = Path("datasets/manifests/r008-specialist-controls.json")
TASKS = ("khmer-news", "sib200-km")
MAX_LEN = 512
HEAD_MAX_LEN = 256


class AgentRunner:
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name

    def _check_model(self, model):
        if model is not None and model != self.model_name:
            raise ValueError(f"runner serves {self.model_name!r}, not {model!r}")

    def predict(self, state, questions, model=None, min_confidence=None):
        self._check_model(model)
        kwargs = {} if min_confidence is None else {"min_confidence": min_confidence}
        return self.agent.predict(state, questions, **kwargs)

    def predict_batch(
        self,
        states,
        questions,
        model=None,
        batch_size=None,
        sort_by_length=False,
        min_confidence=None,
    ):
        self._check_model(model)
        kwargs = {"batch_size": batch_size, "sort_by_length": sort_by_length}
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        return self.agent.predict_batch(states, questions, **kwargs)


def bind_dataset_rows(dataset, rows, model_name):
    """Attach frozen row identities and checkpoint names before evaluation."""
    if len(dataset.examples) != len(rows):
        raise ValueError("R008 dataset and raw row counts differ")
    for example, row in zip(dataset.examples, rows):
        if example.model not in (None, model_name):
            raise ValueError("R008 dataset row names a different model")
        if any(tag.startswith("r008:") for tag in example.tags):
            raise ValueError("R008 row already has an identity tag")
        example.model = model_name
        example.tags = (*example.tags, row_identity_tag(row))


def evaluate_agent(agent, dataset, model_name, config, batch_size=8):
    """Run Laya's evaluator with explicit per-case checkpoint identity."""
    for example in dataset.examples:
        if example.model != model_name:
            raise ValueError("R008 dataset rows must be bound to this model first")
    return evals.evaluate(
        AgentRunner(agent, model_name),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=batch_size,
        config=config,
    )


def score(checkpoint, task, manifest_path=MANIFEST, output=None):
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    manifest, manifest_hash = verify_locked_manifest(manifest_path)
    readiness = verify_study_readiness(manifest, manifest_hash)
    checkpoint = Path(checkpoint).resolve()
    selected = next(
        (entry for entry in readiness.values() if entry["checkpoint"] == checkpoint),
        None,
    )
    if selected is None:
        raise ValueError("checkpoint is not one of the preregistered R008 runs")
    provenance = selected["provenance"]
    if task not in ARM_TASKS[provenance["training_arm"]]:
        raise ValueError("checkpoint training arm is incompatible with the task")
    task_data = manifest["validation_calibration"][task]
    calibration_path, calibration_hash = selected["calibrations"][task]

    agent, inference_dtype = load_r008_agent(
        checkpoint, manifest, calibration=calibration_path
    )
    if (
        agent.cfg.get("max_len") != MAX_LEN
        or agent.cfg.get("head_max_len") != HEAD_MAX_LEN
    ):
        raise ValueError("checkpoint token budgets differ from the locked R008 recipe")

    dataset_path = ROOT / task_data["test_file"]
    if sha256(dataset_path) != task_data["test_sha256"]:
        raise ValueError("R008 test file fingerprint mismatch")
    dataset = evals.Dataset.from_jsonl(str(dataset_path))
    if len(dataset.examples) != task_data["test_rows"]:
        raise ValueError("R008 test row count differs from the locked manifest")
    raw_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_rows) != len(dataset.examples):
        raise ValueError("R008 parsed test row count mismatch")
    arm = provenance["training_arm"]
    seed = provenance["seed"]
    model_name = f"{arm}-s{seed}"
    bind_dataset_rows(dataset, raw_rows, model_name)

    report = evaluate_agent(
        agent,
        dataset,
        model_name,
        {
            "study": "R008-specialist-controls",
            "task": task,
            "model": model_name,
            "training_arm": arm,
            "seed": seed,
            "dataset": task_data["test_file"],
            "dataset_sha256": task_data["test_sha256"],
            "locked_manifest_sha256": manifest_hash,
            "weights_sha256": provenance["weights_sha256"],
            "calibration_sha256": calibration_hash,
            "training_provenance_sha256": sha256(
                checkpoint / "training_provenance.json"
            ),
            "training_file_sha256": provenance["training_file_sha256"],
            "base_revision": provenance["base_revision"],
            "max_len": MAX_LEN,
            "head_max_len": HEAD_MAX_LEN,
            "device": agent.device.type,
            "inference_dtype": inference_dtype,
            "gpu": torch.cuda.get_device_name(0),
            "laya_version": laya.__version__,
        },
    )
    if (
        agent.device.type != INFERENCE_POLICY["device"]
        or str(agent.dtype).removeprefix("torch.") != inference_dtype
        or not agent.amp_enabled
    ):
        raise ValueError("R008 inference runtime changed precision during scoring")
    output = (
        Path(output)
        if output
        else Path(
            manifest["evaluation"]["test_reports"].format(task=task, arm=arm, seed=seed)
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, report.overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output, overall = score(args.checkpoint, args.task, args.manifest, args.output)
    print(json.dumps({"report": str(output), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
