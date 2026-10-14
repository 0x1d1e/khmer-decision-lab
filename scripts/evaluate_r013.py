"""Score R013 only after all four training outputs pass readiness."""

import math
import os
import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

import laya
import numpy as np
import torch
from huggingface_hub import snapshot_download
from r008_identity import INFERENCE_POLICY, install_r008_forward_guard
from r013_identity import (
    ARTIFACT_DIR,
    BASE_REPO,
    BASE_REVISION,
    BASE_SUBDIR,
    HELDOUT_INTENTS,
    INFERENCE_BATCH_SIZE,
    INFERENCE_LIMITS,
    OUTPUTS,
    ROOT,
    SEEDS,
    SPLIT_FILES,
    TOKENIZER_RECOVERY_FILE,
    load_test_rows_after_readiness,
    normalized_base_checkpoint_inventory,
    restore_known_laya_tokenizer_rewrite,
    restore_training_tokenizer_metadata,
    sha256,
    verify_cuda_environment,
    verify_locked_manifest,
    write_json,
)
from r013_inference import capture_laya_logits

CONDITIONS = ("test-unseen", "test-seen")
MODELS = (
    "base",
    "seen-only-s13",
    "all-label-s13",
    "seen-only-s14",
    "all-label-s14",
)


def _identity_temperature(value):
    values = value if isinstance(value, list) else [value]
    return bool(values) and all(
        not isinstance(item, bool)
        and isinstance(item, (int, float))
        and np.isfinite(item)
        and abs(float(item) - 1.0) <= 1e-8
        for item in values
    )


def _tags(row, prefix):
    matches = [
        tag[len(prefix) :] for tag in row.get("tags", []) if tag.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"R013 row must have one {prefix} tag")
    return matches[0]


def _validate_test_rows(rows, condition, preparation):
    if not rows:
        raise ValueError(f"R013 {condition} dataset is empty")
    groups = set()
    seen_labels = set(preparation["seen_intents"])
    distractors = preparation["heldout_test_distractors"]
    if (
        len(seen_labels) != 48
        or len(distractors) != 8
        or not set(distractors) <= seen_labels
    ):
        raise ValueError("R013 prepared candidate-label pools are invalid")
    expected_labels = (
        set(HELDOUT_INTENTS) if condition == "test-unseen" else seen_labels
    )
    allowed = (
        set(HELDOUT_INTENTS) | set(distractors)
        if condition == "test-unseen"
        else seen_labels
    )
    source_ids = set()
    for index, row in enumerate(rows):
        gold = row.get("expected", {}).get("intent")
        question = row.get("questions", {}).get("intent", {})
        criteria = question.get("criteria")
        if (
            not isinstance(gold, str)
            or gold not in expected_labels
            or not isinstance(criteria, dict)
            or len(criteria) != 20
            or (
                set(criteria) != allowed
                if condition == "test-unseen"
                else not set(criteria) <= allowed
            )
            or gold not in criteria
            or any(criteria[label] != label.replace("_", " ") for label in criteria)
            or question.get("instructions")
            != "What is the user's intent in this utterance?"
        ):
            raise ValueError(f"R013 test row schema mismatch: {condition}/{index}")
        group = _tags(row, "r013-group:")
        source_id = _tags(row, "r013-row:")
        if group in groups or source_id in source_ids:
            raise ValueError(f"R013 test repeats an input group: {condition}/{group}")
        groups.add(group)
        source_ids.add(source_id)
    return len(groups)


def _verify_cases(cases, rows, model, logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(rows), 20) or not np.isfinite(logits).all():
        raise ValueError(f"R013 logits are invalid for {model}")
    if len(cases) != len(rows):
        raise ValueError(f"R013 scored row count mismatch for {model}")
    for index, (case, row) in enumerate(zip(cases, rows, strict=True)):
        labels = list(row["questions"]["intent"]["criteria"])
        answer = case.get("answer")
        probabilities = (
            answer.get("probabilities") if isinstance(answer, dict) else None
        )
        if (
            case.get("model") != model
            or case.get("qid") != "intent"
            or case.get("expected") != row["expected"]["intent"]
            or case.get("tags") != row.get("tags", [])
            or not isinstance(answer, dict)
            or answer.get("type") != "choice"
            or answer.get("choice") not in labels
            or type(case.get("correct")) is not bool
            or case["correct"] != (answer.get("choice") == row["expected"]["intent"])
            or not isinstance(probabilities, dict)
            or set(probabilities) != set(labels)
            or not isinstance(answer.get("answer_confidence"), (int, float))
            or isinstance(answer.get("answer_confidence"), bool)
        ):
            raise ValueError(f"R013 report row alignment mismatch: {model}/{index}")
        values = list(probabilities.values())
        row_logits = logits[index] / temperature
        expected_probabilities = np.exp(row_logits - row_logits.max())
        expected_probabilities /= expected_probabilities.sum()
        rounding_bound = 0.00005 * len(labels)
        normalization_error = abs(
            math.fsum(float(value) for value in expected_probabilities) - 1.0
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
                for value in values
            )
            or abs(math.fsum(cast(list[float], values)) - 1.0)
            > rounding_bound + normalization_error + 1e-12
            or labels[int(expected_probabilities.argmax())] != answer["choice"]
            or any(
                probabilities[label] != round(float(expected), 4)
                for label, expected in zip(labels, expected_probabilities, strict=True)
            )
            or answer["answer_confidence"]
            != round(float(expected_probabilities.max()), 4)
        ):
            raise ValueError(f"R013 report probabilities are invalid: {model}/{index}")


def evaluate():
    manifest, manifest_sha256 = verify_locked_manifest()
    verify_cuda_environment(manifest)
    expected_reports = [
        ARTIFACT_DIR / "evaluation" / condition / f"{model}.json"
        for condition in CONDITIONS
        for model in MODELS
    ]
    if (ARTIFACT_DIR / "evaluation.json").exists() or any(
        path.exists() for path in expected_reports
    ):
        raise FileExistsError("R013 evaluation outputs already exist")
    recovery_path = ROOT / TOKENIZER_RECOVERY_FILE
    if recovery_path.exists() or recovery_path.is_symlink():
        raise FileExistsError("R013 tokenizer recovery record already exists")
    recovery_record = {
        "study": manifest["study"],
        "manifest_sha256": manifest_sha256,
        "pre_readiness": restore_training_tokenizer_metadata(manifest, manifest_sha256),
        "post_load": [],
    }
    write_json(recovery_path, recovery_record)
    preparation, runs, prepared_rows = load_test_rows_after_readiness(
        manifest, manifest_sha256
    )
    for condition in CONDITIONS:
        if _validate_test_rows(prepared_rows[condition], condition, preparation) == 0:
            raise ValueError(f"R013 test condition has no groups: {condition}")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R013 scoring requires CUDA BF16")
    base_snapshot = Path(
        snapshot_download(
            BASE_REPO,
            revision=BASE_REVISION,
            allow_patterns=f"{BASE_SUBDIR}/*",
        )
    )
    base = base_snapshot / BASE_SUBDIR
    base_inventory = normalized_base_checkpoint_inventory(base)
    if base_inventory != manifest["base"]["inference_files_sha256"]:
        raise ValueError("R013 multilingual base checkpoint fingerprint mismatch")

    checkpoints = {"base": str(base.resolve())}
    for seed in SEEDS:
        for arm in ("seen-only", "all-label"):
            model_name = f"{arm}-s{seed}"
            checkpoints[model_name] = str((ROOT / OUTPUTS[(seed, arm)]).resolve())
    router = None
    reports = []
    agent = None
    try:
        with patch.dict(
            os.environ, {"LAYA_CUDA_AMP": INFERENCE_POLICY["laya_cuda_amp"]}
        ):
            router = laya.Router(
                models=checkpoints,
                device=INFERENCE_POLICY["device"],
                max_loaded=1,
                preload=False,
            )
            for condition in CONDITIONS:
                relative = SPLIT_FILES[condition]
                rows = prepared_rows[condition]
                for model in MODELS:
                    if agent is not None:
                        del agent
                        agent = None
                    router.unload()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    agent = router.load(model)
                    torch.cuda.synchronize()
                    model_load_ms = (time.perf_counter() - started) * 1000
                    if model != "base":
                        arm, seed_text = model.rsplit("-s", 1)
                        recovery_event = restore_known_laya_tokenizer_rewrite(
                            ROOT / OUTPUTS[(int(seed_text), arm)],
                            runs[(int(seed_text), arm)]["inventory"],
                        )
                        recovery_event.update(
                            {"seed": int(seed_text), "arm": arm, "phase": "post-load"}
                        )
                        recovery_record["post_load"].append(recovery_event)
                        write_json(recovery_path, recovery_record)
                    if (
                        agent.device.type != INFERENCE_POLICY["device"]
                        or agent.dtype != torch.bfloat16
                        or not agent.amp_enabled
                        or not _identity_temperature(agent.temperature)
                        or agent.temperature_by_options
                        or agent.lang_temperatures
                    ):
                        raise ValueError(
                            f"R013 model failed locked inference settings: {model}"
                        )
                    install_r008_forward_guard(agent)

                    started = time.perf_counter()
                    warmup_rows = rows[:8]
                    capture_laya_logits(agent, warmup_rows, model)
                    torch.cuda.synchronize()
                    warmup_ms = (time.perf_counter() - started) * 1000
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    report_config = {
                        "study": manifest["study"],
                        "condition": condition,
                        "manifest_sha256": manifest_sha256,
                        "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
                        "prepared_file": relative,
                        "prepared_file_sha256": preparation["outputs_sha256"][relative],
                        "model": model,
                        "checkpoint": checkpoints[model],
                        "inference_files_sha256": (
                            base_inventory
                            if model == "base"
                            else runs[
                                (int(model.rsplit("s", 1)[1]), model.rsplit("-s", 1)[0])
                            ]["inventory"]
                        ),
                        "training_provenance_sha256": (
                            None
                            if model == "base"
                            else sha256(
                                ROOT
                                / OUTPUTS[
                                    (
                                        int(model.rsplit("s", 1)[1]),
                                        model.rsplit("-s", 1)[0],
                                    )
                                ]
                                / "training_provenance.json"
                            )
                        ),
                        "device": INFERENCE_POLICY["device"],
                        "autocast_dtype": INFERENCE_POLICY["autocast_dtype"],
                        "max_len": INFERENCE_LIMITS["max_len"],
                        "head_max_len": INFERENCE_LIMITS["head_max_len"],
                        "laya_cuda_amp": INFERENCE_POLICY["laya_cuda_amp"],
                        "batch_size": INFERENCE_BATCH_SIZE,
                        "gpu": torch.cuda.get_device_name(0),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "model_load_ms": round(model_load_ms, 4),
                        "warmup_ms": round(warmup_ms, 4),
                        "warmup_examples": len(warmup_rows),
                        "peak_vram_scope": "single resident model, scored inference only",
                        "top_answer_brier_definition": "mean((P(reported answer is correct) - correct)**2)",
                    }
                    started = time.perf_counter()
                    logits, payload = capture_laya_logits(
                        agent, rows, model, config=report_config
                    )
                    torch.cuda.synchronize()
                    payload["r013_logits"] = logits.tolist()
                    payload["config"]["scored_inference_ms"] = round(
                        (time.perf_counter() - started) * 1000, 4
                    )
                    payload["config"]["peak_vram_mib"] = round(
                        torch.cuda.max_memory_allocated() / 2**20, 1
                    )
                    _verify_cases(
                        payload["cases"],
                        rows,
                        model,
                        logits,
                        temperature=agent.temperature[0],
                    )
                    output = ARTIFACT_DIR / "evaluation" / condition / f"{model}.json"
                    write_json(output, payload)
                    reports.append(
                        {
                            "condition": condition,
                            "model": model,
                            "report": str(output.relative_to(ROOT)),
                            "report_sha256": sha256(output),
                            "rows": len(payload["cases"]),
                            "choice_accuracy": payload["overall"]["choice_accuracy"],
                        }
                    )
                    print(
                        f"{condition:>12} {model:>16} n={len(payload['cases'])} "
                        f"accuracy={payload['overall']['choice_accuracy']:.4f}",
                        flush=True,
                    )
    finally:
        if agent is not None:
            del agent
        if router is not None:
            router.unload()
        recovery_record["post_evaluation"] = restore_training_tokenizer_metadata(
            manifest, manifest_sha256
        )
        write_json(recovery_path, recovery_record)
    write_json(
        ARTIFACT_DIR / "evaluation.json",
        {
            "study": manifest["study"],
            "manifest_sha256": manifest_sha256,
            "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
            "tokenizer_config_recovery": str(recovery_path.relative_to(ROOT)),
            "tokenizer_config_recovery_sha256": sha256(recovery_path),
            "reports": reports,
        },
    )
    return reports


def main():
    evaluate()


if __name__ == "__main__":
    main()
