"""Score the locked R010 clean-text and synthetic OCR test conditions."""

import json
import time
from pathlib import Path

import laya
import torch
from laya import evals
from laya.evals_cli import RouterRunner
from r008_identity import (
    INFERENCE_POLICY,
    checkpoint_inference_inventory,
    install_r008_forward_guard,
)
from r010_identity import (
    ARTIFACT_DIR,
    CONDITIONS,
    MODEL_PATHS,
    ROOT,
    sha256,
    verify_locked_manifest,
    verify_prepared,
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    manifest, manifest_sha256 = verify_locked_manifest()
    preparation = verify_prepared(manifest, manifest_sha256)
    if not torch.cuda.is_available():
        raise ValueError("R010 scoring requires the locked CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("R010 scoring requires CUDA BF16 support")

    checkpoints = {
        model: str((ROOT / relative).resolve())
        for model, relative in MODEL_PATHS.items()
    }
    router = laya.Router(
        models=checkpoints,
        device="cuda",
        max_loaded=1,
        preload=False,
    )
    reports = []
    agent = None
    for condition in CONDITIONS:
        dataset_path = ROOT / preparation["outputs"][condition]["file"]
        dataset = evals.Dataset.from_jsonl(str(dataset_path))
        for model, checkpoint in checkpoints.items():
            expected_files = manifest["models"][model]["inference_files_sha256"]
            if checkpoint_inference_inventory(checkpoint) != expected_files:
                raise ValueError(
                    f"R010 model checkpoint changed during scoring: {model}"
                )
            for example in dataset.examples:
                example.model = model

            # The forward guard closes over `agent`; clear it before Router's collection.
            if agent is not None:
                del agent
            router.unload()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            started = time.perf_counter()
            agent = router.load(model)
            torch.cuda.synchronize()
            model_load_ms = (time.perf_counter() - started) * 1000
            if (
                agent.device.type != INFERENCE_POLICY["device"]
                or agent.dtype != torch.bfloat16
                or not agent.amp_enabled
            ):
                raise ValueError(f"R010 model failed the locked BF16 policy: {model}")
            install_r008_forward_guard(agent)

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
                    "study": manifest["study"],
                    "condition": condition,
                    "manifest_sha256": manifest_sha256,
                    "source_test_file": manifest["source"]["test_file"],
                    "source_test_sha256": manifest["source"]["test_sha256"],
                    "dataset": preparation["outputs"][condition]["file"],
                    "dataset_sha256": preparation["outputs"][condition]["sha256"],
                    "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
                    "preparer_sha256": manifest["implementation_sha256"][
                        "scripts/prepare_r010.py"
                    ],
                    "model": model,
                    "checkpoint": manifest["models"][model]["checkpoint"],
                    "weights_sha256": expected_files["model.safetensors"],
                    "inference_files_sha256": expected_files,
                    "max_len": manifest["models"][model]["max_len"],
                    "head_max_len": manifest["models"][model]["head_max_len"],
                    "device": INFERENCE_POLICY["device"],
                    "autocast_dtype": INFERENCE_POLICY["autocast_dtype"],
                    "laya_cuda_amp": INFERENCE_POLICY["laya_cuda_amp"],
                    "gpu": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
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
            output = ARTIFACT_DIR / "massive-km" / condition / f"{model}.json"
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
                f"{condition:>8} {model:>12} "
                f"n={len(payload['cases'])} "
                f"accuracy={payload['overall']['choice_accuracy']:.4f}"
            )

    write_json(
        ARTIFACT_DIR / "evaluation.json",
        {
            "study": manifest["study"],
            "manifest_sha256": manifest_sha256,
            "preparation_sha256": sha256(ARTIFACT_DIR / "preparation.json"),
            "reports": reports,
        },
    )


if __name__ == "__main__":
    main()
