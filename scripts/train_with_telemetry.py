"""Run scripts/train_full_8bit.py unchanged and record training telemetry.

Usage: uv run python scripts/train_with_telemetry.py [DATA OUTPUT SEED]

The trainer is executed through runpy with the arguments passed here, so the
recipe is exactly the one train_full_8bit.py defines. Nothing about the
optimizer, the scheduler or the loss is modified: this wrapper only observes.

What it records in OUTPUT/training_telemetry.json:

- optimizer class actually constructed by the trainer
- scheduler attempts (Laya calls scheduler.step() once per accumulation window)
- successful optimizer updates (post-step hook; GradScaler skips do not fire it)
- skipped updates (attempts minus successful updates)
- training-only wall clock and peak allocated VRAM (reset before train_model)
- whole-run peak allocated VRAM and wall clock
- GPU, torch and CUDA versions
- the failure, if the run raises

The scheduler is deliberately not gated on optimizer updates, so a skipped
update still advances the cosine schedule exactly as it did for R003 seed 13.
"""

import json
import os
import runpy
import sys
import time
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TRAINER = SCRIPTS / "train_full_8bit.py"


def main():
    if len(sys.argv) not in (1, 2, 3, 4):
        raise SystemExit(__doc__)
    argv = list(sys.argv[1:])
    output = Path(
        argv[1] if len(argv) > 1 else "artifacts/checkpoints/R003-full-e2-8bit"
    )
    telemetry_path = output / "training_telemetry.json"

    state = {
        "optimizer": None,
        "optimizer_attempts": 0,
        "successful_updates": 0,
        "skipped_updates": 0,
        "scheduler_steps": 0,
        "expected_updates": None,
        "training_seconds": None,
        "peak_allocated_bytes": None,
        "failure": None,
    }

    import bitsandbytes.optim as bnb_optim
    from laya import train as laya_train

    base_optimizer = bnb_optim.PagedAdamW8bit

    class TelemetryPagedAdamW8bit(base_optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state["optimizer"] = (
                f"{type(self).__mro__[1].__module__}.{type(self).__mro__[1].__qualname__}"
            )
            self.register_step_post_hook(_record_update)

    def _record_update(_optimizer, _args, _kwargs):
        state["successful_updates"] += 1

    original_scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR.step
    # PyTorch calls step() once while the scheduler is being constructed; that
    # call is initialization, not an update attempt, so it is not counted.
    initialized = weakref.WeakKeyDictionary()

    def counted_scheduler_step(self, *args, **kwargs):
        if self not in initialized:
            initialized[self] = True
        else:
            state["scheduler_steps"] += 1
        return original_scheduler_step(self, *args, **kwargs)

    original_train_model = laya_train.train_model

    def measured_train_model(*args, **kwargs):
        items = kwargs["items"] if "items" in kwargs else args[2]
        config = kwargs["config"] if "config" in kwargs else args[3]
        state["expected_updates"] = laya_train.optimizer_updates(
            len(items), config.micro_batch, config.grad_accum, config.epochs
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            return original_train_model(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            state["training_seconds"] = time.perf_counter() - started
            state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            state["optimizer_attempts"] = state["scheduler_steps"]
            state["skipped_updates"] = (
                state["scheduler_steps"] - state["successful_updates"]
            )

    def write():
        payload = dict(state)
        payload.update(
            {
                "device": "cuda",
                "gpu": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "bootstrap": {
                    "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
                    "trainer_sha256": _sha256(TRAINER),
                },
                "instrumentation": (
                    "passive: optimizer post-step hook plus pass-through "
                    "CosineAnnealingLR.step; scheduler not gated"
                ),
            }
        )
        if payload["peak_allocated_bytes"] is not None:
            payload["peak_vram_mib"] = round(payload["peak_allocated_bytes"] / 2**20, 1)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = telemetry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, telemetry_path)

    run_started = time.time()
    torch.cuda.reset_peak_memory_stats()
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    sys.argv = [str(TRAINER)] + argv
    try:
        with (
            patch.object(bnb_optim, "PagedAdamW8bit", TelemetryPagedAdamW8bit),
            patch.object(
                torch.optim.lr_scheduler.CosineAnnealingLR,
                "step",
                counted_scheduler_step,
            ),
            patch.object(laya_train, "train_model", measured_train_model),
        ):
            runpy.run_path(str(TRAINER), run_name="__main__")
    except BaseException as exc:
        state["failure"] = f"{type(exc).__name__}: {exc}"[:1000]
        raise
    finally:
        state["run_wall_clock_seconds"] = round(time.time() - run_started, 1)
        state["run_peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        write()


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
