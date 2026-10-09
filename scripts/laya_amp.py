"""Guard Laya's cosine schedule against GradScaler-skipped optimizer updates."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter
from typing import Any
from unittest.mock import patch
from weakref import WeakKeyDictionary

import torch

_GUARD_LOCK = Lock()


@dataclass
class TrainingTelemetry:
    optimizer: str | None = None
    optimizer_attempts: int = 0
    successful_updates: int = 0
    skipped_updates: int = 0
    scheduler_steps: int = 0
    training_seconds: float | None = None
    peak_allocated_bytes: int | None = None
    failure: str | None = None
    telemetry_cleanup_failure: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@contextmanager
def laya_training_guard() -> Iterator[TrainingTelemetry]:
    """Preserve Laya's scheduler initialization and gate later steps on optimizer updates.

    Laya 0.4.1 unconditionally calls ``scheduler.step()`` after GradScaler may skip
    ``optimizer.step()``. The scoped scheduler subclass delegates initialization unchanged
    and gates only later calls using a PyTorch optimizer post-step hook.

    This process-global patch is intended for one standalone, serial Laya training call.
    """
    if not _GUARD_LOCK.acquire(blocking=False):
        raise RuntimeError("Laya AMP scheduler guard is already active")

    handles = []
    telemetry = TrainingTelemetry()
    try:
        if not hasattr(torch.optim.Optimizer, "register_step_post_hook"):
            raise RuntimeError(
                "PyTorch optimizer post-step hooks are required for the Laya scheduler guard"
            )

        from laya import train as laya_train

        original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        original_train_model = laya_train.train_model
        scheduler_states: WeakKeyDictionary[Any, dict[str, bool]] = WeakKeyDictionary()
        scheduler_count = 0
        train_call_count = 0

        class GuardedCosineAnnealingLR(original_scheduler):
            def step(self, *step_args, **step_kwargs):
                state = scheduler_states.get(self)
                if state is None:
                    # PyTorch calls step once during scheduler initialization. Leave that
                    # call exactly as the original class implements it.
                    return super().step(*step_args, **step_kwargs)

                telemetry.optimizer_attempts += 1
                if not state["optimizer_stepped"]:
                    telemetry.skipped_updates += 1
                    return None

                state["optimizer_stepped"] = False
                result = super().step(*step_args, **step_kwargs)
                telemetry.scheduler_steps += 1
                return result

        def guarded_scheduler_factory(optimizer, *args, **kwargs):
            nonlocal scheduler_count
            if scheduler_count:
                raise RuntimeError(
                    "Laya created more than one cosine scheduler in a training run"
                )

            # The subclass's step delegates to the original implementation until after
            # __init__ completes, so PyTorch's initial scheduler state is unchanged.
            scheduler = GuardedCosineAnnealingLR(optimizer, *args, **kwargs)
            scheduler_states[scheduler] = {"optimizer_stepped": False}
            scheduler_count += 1

            optimizer_type = type(optimizer)
            telemetry.optimizer = (
                f"{optimizer_type.__module__}.{optimizer_type.__qualname__}"
            )

            def record_optimizer_step(_optimizer, _args, _kwargs):
                state = scheduler_states[scheduler]
                if state["optimizer_stepped"]:
                    raise RuntimeError(
                        "multiple optimizer updates occurred before one scheduler attempt"
                    )
                state["optimizer_stepped"] = True
                telemetry.successful_updates += 1

            handles.append(optimizer.register_step_post_hook(record_optimizer_step))
            return scheduler

        def measured_train_model(*args, **kwargs):
            nonlocal train_call_count
            if train_call_count:
                raise RuntimeError(
                    "Laya AMP scheduler guard supports one training call"
                )
            train_call_count += 1

            def argument(name, index):
                if name in kwargs:
                    return kwargs[name]
                return args[index]

            started = perf_counter()
            device = None
            primary_failure = None
            try:
                device = torch.device(argument("device", 4))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)

                result = original_train_model(*args, **kwargs)
                items = argument("items", 2)
                config = argument("config", 3)
                expected_updates = laya_train.optimizer_updates(
                    len(items), config.micro_batch, config.grad_accum, config.epochs
                )
                if scheduler_count != 1:
                    raise RuntimeError(
                        f"expected one Laya cosine scheduler, found {scheduler_count}"
                    )
                if telemetry.optimizer_attempts != expected_updates:
                    raise RuntimeError(
                        "Laya scheduler attempts do not match the configured optimizer budget: "
                        f"{telemetry.optimizer_attempts} != {expected_updates}"
                    )
                if (
                    telemetry.successful_updates + telemetry.skipped_updates
                    != expected_updates
                ):
                    raise RuntimeError(
                        "optimizer update telemetry does not match the update budget"
                    )
                if telemetry.scheduler_steps != telemetry.successful_updates:
                    raise RuntimeError(
                        "scheduler and successful optimizer update counts differ"
                    )
                return result
            except BaseException as exc:
                primary_failure = exc
                telemetry.failure = f"{type(exc).__name__}: {exc}"[:1000]
                raise
            finally:
                telemetry.training_seconds = perf_counter() - started
                if device is not None and device.type == "cuda":
                    try:
                        torch.cuda.synchronize(device)
                        telemetry.peak_allocated_bytes = (
                            torch.cuda.max_memory_allocated(device)
                        )
                    except BaseException as cleanup_exc:
                        telemetry.telemetry_cleanup_failure = (
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"[:1000]
                        )
                        if primary_failure is None:
                            telemetry.failure = telemetry.telemetry_cleanup_failure
                            raise

        with (
            patch.object(
                torch.optim.lr_scheduler,
                "CosineAnnealingLR",
                guarded_scheduler_factory,
            ),
            patch.object(laya_train, "train_model", measured_train_model),
        ):
            try:
                yield telemetry
            except BaseException as exc:
                if telemetry.failure is None:
                    telemetry.failure = f"{type(exc).__name__}: {exc}"[:1000]
                raise
    finally:
        for handle in handles:
            handle.remove()
        _GUARD_LOCK.release()
