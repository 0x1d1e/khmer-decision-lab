"""Score matched R012 continuations on authenticated R010 test inputs."""

import gc

import torch
from laya import evals
from r008_identity import (
    INFERENCE_POLICY,
    checkpoint_inference_inventory,
    load_r008_agent,
)
from r012_identity import (
    ARMS,
    ARTIFACT_DIR,
    CONDITIONS,
    ROOT,
    SEEDS,
    intent_labels,
    read_json,
    read_jsonl,
    sha256,
    training_output_path,
    validate_report_cases,
    verify_locked_manifest,
    verify_r010_baselines,
    verify_training_readiness,
    write_json,
)


class AgentRunner:
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name

    def _check_model(self, model):
        if model is not None and model != self.model_name:
            raise ValueError(f"R012 runner serves {self.model_name!r}, not {model!r}")

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


def _dataset(rows, model_name):
    dataset = evals.Dataset(examples=[evals.Example.from_dict(row) for row in rows])
    if len(dataset.examples) != len(rows):
        raise ValueError("R012 Laya dataset conversion changed row count")
    for example in dataset.examples:
        if example.model not in (None, model_name):
            raise ValueError("R012 test row names a different model")
        example.model = model_name
    return dataset


def evaluate():
    # This first pass authenticates code, training data, and all four continuations only.
    manifest, manifest_hash, _r010, _r010_hash = verify_locked_manifest(
        test_access=False
    )
    ready, _r012_preparation = verify_training_readiness(manifest, manifest_hash)

    # Test source, prepared inputs, and baseline reports are opened only after readiness.
    manifest, manifest_hash, r010, r010_hash = verify_locked_manifest(test_access=True)
    ready, _r012_preparation = verify_training_readiness(manifest, manifest_hash)
    r010_preparation = read_json(ROOT / manifest["source"]["r010_preparation_artifact"])
    baselines = verify_r010_baselines(manifest, r010, r010_hash, r010_preparation)
    rows_by_condition = {
        condition: read_jsonl(ROOT / r010_preparation["outputs"][condition]["file"])
        for condition in CONDITIONS
    }
    for condition, rows in rows_by_condition.items():
        if len(rows) != manifest["source"]["test_rows"]:
            raise ValueError(f"R012 test row count mismatch: {condition}")
        for index, row in enumerate(rows):
            if (
                row.get("questions", {}).keys() != {"intent"}
                or row.get("expected", {}).keys() != {"intent"}
                or len(intent_labels(row)) != 20
                or row["expected"]["intent"] not in intent_labels(row)
                or row.get("state", {}).get("utterance") is None
            ):
                raise ValueError(f"R012 malformed {condition} test row {index}")

    for condition in CONDITIONS:
        for seed in SEEDS:
            model_name = f"mt-s{seed}"
            validate_report_cases(
                baselines[f"{condition}/{model_name}"]["cases"],
                rows_by_condition[condition],
                model_name,
            )

    output_records = []
    expected_reports = set()
    for seed in SEEDS:
        model_name = f"mt-s{seed}"
        model = manifest["models"][model_name]
        for arm in ARMS:
            checkpoint_rel = training_output_path(seed, arm)
            checkpoint = ROOT / checkpoint_rel
            inventory = ready[f"{arm}-s{seed}"]["inference_files_sha256"]
            if checkpoint_inference_inventory(checkpoint) != inventory:
                raise ValueError(
                    f"R012 continuation changed before evaluation: {arm}/s{seed}"
                )
            agent, inference_dtype = load_r008_agent(
                checkpoint, {"inference": INFERENCE_POLICY}
            )
            if (
                agent.device.type != "cuda"
                or agent.dtype != torch.bfloat16
                or not agent.amp_enabled
                or inference_dtype != "bfloat16"
                or agent.cfg.get("max_len") != model["max_len"]
                or agent.cfg.get("head_max_len") != model["head_max_len"]
                or agent.temperature != model["choice_temperature"]
                or agent.temperature_by_options
                or agent.lang_temperatures
                or getattr(agent, "binning_map", None) is not None
            ):
                raise ValueError(f"R012 inference settings mismatch: {arm}/s{seed}")

            for condition in CONDITIONS:
                dataset = _dataset(rows_by_condition[condition], model_name)
                config = {
                    "study": manifest["study"],
                    "arm": arm,
                    "seed": seed,
                    "condition": condition,
                    "manifest_sha256": manifest_hash,
                    "r010_manifest_sha256": r010_hash,
                    "source_test_sha256": manifest["source"]["test_sha256"],
                    "dataset": r010_preparation["outputs"][condition]["file"],
                    "dataset_sha256": r010_preparation["outputs"][condition]["sha256"],
                    "r012_training_provenance_sha256": ready[f"{arm}-s{seed}"][
                        "provenance_sha256"
                    ],
                    "checkpoint": checkpoint_rel,
                    "weights_sha256": inventory["model.safetensors"],
                    "inference_files_sha256": inventory,
                    "max_len": model["max_len"],
                    "head_max_len": model["head_max_len"],
                    "choice_temperature": model["choice_temperature"],
                    "device": INFERENCE_POLICY["device"],
                    "autocast_dtype": INFERENCE_POLICY["autocast_dtype"],
                    "laya_cuda_amp": INFERENCE_POLICY["laya_cuda_amp"],
                    "inference_grouping": manifest["recipe"]["inference_grouping"],
                    "evaluation_batch_size": manifest["recipe"][
                        "evaluation_batch_size"
                    ],
                    "gpu": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
                runner = AgentRunner(agent, model_name)
                report = evals.evaluate(
                    runner,
                    dataset,
                    evaluators=evals.default_evaluators(),
                    batch_size=manifest["recipe"]["evaluation_batch_size"],
                    config=config,
                )
                payload = report.to_json()
                validate_report_cases(
                    payload.get("cases", []), rows_by_condition[condition], model_name
                )
                path = ARTIFACT_DIR / arm / model_name / f"{condition}.json"
                if path.exists():
                    raise FileExistsError(
                        f"R012 evaluation report already exists: {path}"
                    )
                write_json(path, payload)
                relative = str(path.relative_to(ROOT))
                expected_reports.add((arm, model_name, condition))
                output_records.append(
                    {
                        "arm": arm,
                        "model": model_name,
                        "condition": condition,
                        "report": relative,
                        "report_sha256": sha256(path),
                        "rows": len(payload["cases"]),
                        "choice_accuracy": payload["overall"]["choice_accuracy"],
                    }
                )
                print(
                    f"{arm:>5} {model_name:>6} {condition:>8} "
                    f"n={len(payload['cases'])} "
                    f"accuracy={payload['overall']['choice_accuracy']:.4f}",
                    flush=True,
                )
                del runner, dataset
                if checkpoint_inference_inventory(checkpoint) != inventory:
                    raise ValueError(
                        f"R012 checkpoint changed during inference: {arm}/s{seed}"
                    )

            del agent
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if len(output_records) != len(expected_reports):
        raise ValueError("R012 evaluation did not produce all eight condition reports")
    evaluation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "r010_manifest_sha256": r010_hash,
        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
        "training_ready": {
            key: value["provenance_sha256"] for key, value in ready.items()
        },
        "reports": output_records,
    }
    write_json(ARTIFACT_DIR / "evaluation.json", evaluation)
    return evaluation


def main():
    evaluate()


if __name__ == "__main__":
    main()
