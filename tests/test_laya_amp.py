import importlib.util
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from bitsandbytes.optim import PagedAdamW8bit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("laya_amp", ROOT / "scripts/laya_amp.py")
assert SPEC is not None and SPEC.loader is not None
laya_amp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = laya_amp
SPEC.loader.exec_module(laya_amp)
laya_training_guard = laya_amp.laya_training_guard


def test_scheduler_only_advances_after_optimizer_step():
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    reference_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    reference_optimizer = torch.optim.AdamW([reference_parameter], lr=0.1)
    reference_scheduler = scheduler_class(reference_optimizer, T_max=2, eta_min=0.0)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    with laya_training_guard() as telemetry:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=2, eta_min=0.0
        )
        assert isinstance(scheduler, scheduler_class)
        assert scheduler.state_dict() == reference_scheduler.state_dict()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        after_update = scheduler.last_epoch

        scheduler.step()

    assert after_update == 1
    assert scheduler.last_epoch == after_update
    assert torch.optim.lr_scheduler.CosineAnnealingLR is scheduler_class
    assert telemetry.optimizer == (
        f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    )
    assert telemetry.optimizer_attempts == 2
    assert telemetry.successful_updates == 1
    assert telemetry.skipped_updates == 1
    assert telemetry.scheduler_steps == 1


def test_guard_rejects_reentrant_scope():
    with laya_training_guard(), pytest.raises(RuntimeError, match="already active"):
        laya_training_guard().__enter__()


def test_guard_restores_global_patches_after_failure():
    from laya import train as laya_train

    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    train_model = laya_train.train_model
    with pytest.raises(RuntimeError, match="forced failure"), laya_training_guard():
        raise RuntimeError("forced failure")

    assert torch.optim.lr_scheduler.CosineAnnealingLR is scheduler_class
    assert laya_train.train_model is train_model
    assert laya_amp._GUARD_LOCK.acquire(blocking=False)
    laya_amp._GUARD_LOCK.release()


def test_cuda_cleanup_does_not_mask_training_failure():
    from laya import train as laya_train

    original_train_model = laya_train.train_model
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    sync_calls = 0

    def fail_training(*args, **kwargs):
        raise ValueError("original training failure")

    def synchronize(_device):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise RuntimeError("cleanup CUDA synchronization failure")

    with (
        patch.object(laya_train, "train_model", fail_training),
        patch.object(torch.cuda, "synchronize", synchronize),
        patch.object(torch.cuda, "reset_peak_memory_stats"),
        patch.object(torch.cuda, "max_memory_allocated", return_value=0),
    ):
        guarded_train_model = laya_train.train_model
        with laya_training_guard() as telemetry:
            with pytest.raises(ValueError, match="original training failure"):
                laya_train.train_model(
                    object(),
                    object(),
                    [{}],
                    laya_train.TrainConfig(epochs=1),
                    torch.device("cuda"),
                    8,
                    4,
                )

            assert laya_train.train_model is not guarded_train_model
            assert telemetry.failure == "ValueError: original training failure"
            assert telemetry.telemetry_cleanup_failure == (
                "RuntimeError: cleanup CUDA synchronization failure"
            )
            assert telemetry.training_seconds is not None

        assert laya_train.train_model is guarded_train_model
        assert torch.optim.lr_scheduler.CosineAnnealingLR is original_scheduler

    assert laya_train.train_model is original_train_model


def test_cuda_cleanup_failure_is_reported_after_successful_training():
    from laya import train as laya_train

    original_train_model = laya_train.train_model
    sync_calls = 0

    def fake_training(_model, _tok, _items, _config, _device, _max_len, _head_max_len):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        return [0.0]

    def synchronize(_device):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise RuntimeError("cleanup CUDA synchronization failure")

    config = laya_train.TrainConfig(
        micro_batch=1,
        grad_accum=1,
        epochs=1,
    )
    with (
        patch.object(laya_train, "train_model", fake_training),
        patch.object(torch.cuda, "synchronize", synchronize),
        patch.object(torch.cuda, "reset_peak_memory_stats"),
        patch.object(torch.cuda, "max_memory_allocated", return_value=0),
    ):
        with laya_training_guard() as telemetry:
            with pytest.raises(
                RuntimeError, match="cleanup CUDA synchronization failure"
            ):
                laya_train.train_model(
                    object(),
                    object(),
                    [{}],
                    config,
                    torch.device("cuda"),
                    8,
                    4,
                )

            assert telemetry.failure == (
                "RuntimeError: cleanup CUDA synchronization failure"
            )
            assert telemetry.telemetry_cleanup_failure == telemetry.failure
            assert telemetry.training_seconds is not None

        assert laya_train.train_model is fake_training

    assert laya_train.train_model is original_train_model


def test_guard_measures_and_guards_laya_training_loop():
    from laya import train as laya_train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.head = torch.nn.Parameter(torch.tensor([0.2, -0.2]))

    def fake_encode(*args, **kwargs):
        return {}

    def fake_collate(*args, **kwargs):
        return {
            "marker_mask": torch.tensor([[True, True]]),
            "target": torch.tensor([[1.0, 0.0]]),
        }

    def fake_forward(model, batch, device, amp, detach_encoder):
        return model.head.unsqueeze(0)

    items = [{"q": {"t": "choice"}, "k": 2} for _ in range(2)]
    config = laya_train.TrainConfig(
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        loss="soft-ce",
        freeze_encoder=True,
        amp=False,
        gradient_checkpointing=False,
        log_every=0,
    )
    with (
        patch.object(laya_train, "encode_item", side_effect=fake_encode),
        patch.object(laya_train, "collate_items", side_effect=fake_collate),
        patch.object(laya_train, "_forward", side_effect=fake_forward),
        laya_training_guard() as telemetry,
    ):
        history = laya_train.train_model(
            TinyModel(),
            SimpleNamespace(pad_token_id=0),
            items,
            config,
            torch.device("cpu"),
            8,
            4,
        )

    assert len(history) == 1
    assert telemetry.optimizer == "torch.optim.adamw.AdamW"
    assert telemetry.optimizer_attempts == 2
    assert telemetry.successful_updates == 2
    assert telemetry.skipped_updates == 0
    assert telemetry.scheduler_steps == 2
    assert telemetry.training_seconds is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_laya_paged_8bit_training_recovers_after_skipped_update():
    from laya import train as laya_train

    head_size = 100_001
    config = laya_train.TrainConfig(
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        loss="soft-ce",
        freeze_encoder=True,
        amp=True,
        gradient_checkpointing=False,
        log_every=0,
    )
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1)
    model.head = torch.nn.Parameter(torch.zeros(head_size))

    reference_parameter = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        reference_optimizer, T_max=3, eta_min=config.min_lr
    )
    optimizer_holder = []
    scheduler_holder = []
    observations = []
    inject_overflow = False
    forward_calls = 0

    def advance_reference_schedule(_optimizer, _args, _kwargs):
        reference_parameter.grad = torch.ones_like(reference_parameter)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        reference_scheduler.step()

    class RecordingPagedAdamW8bit(PagedAdamW8bit):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optimizer_holder.append(self)
            self.register_step_post_hook(advance_reference_schedule)

    def record_scheduler_factory(*args, **kwargs):
        scheduler = guarded_scheduler_factory(*args, **kwargs)
        scheduler_holder.append(scheduler)
        return scheduler

    def overflow_on_second_update(gradient):
        if inject_overflow:
            return torch.full_like(gradient, float("inf"))
        return gradient

    model.head.register_hook(overflow_on_second_update)

    def fake_encode(*args, **kwargs):
        return {}

    def fake_collate(*args, **kwargs):
        return {
            "marker_mask": torch.tensor([[True, True]]),
            "target": torch.tensor([[1.0, 0.0]]),
        }

    def fake_forward(model, batch, device, amp, detach_encoder):
        nonlocal forward_calls, inject_overflow
        forward_calls += 1
        inject_overflow = forward_calls == 2
        optimizer = optimizer_holder[0]
        scheduler = scheduler_holder[0]
        state = optimizer.state.get(model.head, {})
        state1 = state.get("state1")
        state2 = state.get("state2")
        observations.append(
            {
                "parameter": model.head.detach().clone(),
                "state1": state1.clone() if state1 is not None else None,
                "state2": state2.clone() if state2 is not None else None,
                "optimizer_step": state.get("step"),
                "scheduler_epoch": scheduler.last_epoch,
                "reference_epoch": reference_scheduler.last_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "reference_learning_rate": reference_optimizer.param_groups[0]["lr"],
            }
        )
        return model.head[:2].unsqueeze(0)

    items = [{"q": {"t": "choice"}, "k": 2} for _ in range(3)]
    with (
        patch.object(torch.optim, "AdamW", RecordingPagedAdamW8bit),
        patch.object(laya_train, "encode_item", side_effect=fake_encode),
        patch.object(laya_train, "collate_items", side_effect=fake_collate),
        patch.object(laya_train, "_forward", side_effect=fake_forward),
        warnings.catch_warnings(record=True) as recorded,
        laya_training_guard() as telemetry,
    ):
        warnings.simplefilter("always")
        guarded_scheduler_factory = torch.optim.lr_scheduler.CosineAnnealingLR
        with patch.object(
            torch.optim.lr_scheduler,
            "CosineAnnealingLR",
            record_scheduler_factory,
        ):
            history = laya_train.train_model(
                model,
                SimpleNamespace(pad_token_id=0),
                items,
                config,
                torch.device("cuda"),
                8,
                4,
            )

    optimizer = optimizer_holder[0]
    scheduler = scheduler_holder[0]
    state = optimizer.state[model.head]
    assert len(history) == 1
    assert forward_calls == 3
    assert state["state1"].dtype is torch.uint8
    assert state["state2"].dtype is torch.uint8
    assert state["state1"].is_paged
    assert state["state2"].is_paged

    assert observations[0]["optimizer_step"] is None
    assert observations[1]["optimizer_step"] == 1
    assert observations[2]["optimizer_step"] == 1
    assert torch.equal(observations[1]["parameter"], observations[2]["parameter"])
    assert torch.equal(observations[1]["state1"], observations[2]["state1"])
    assert torch.equal(observations[1]["state2"], observations[2]["state2"])
    assert observations[1]["scheduler_epoch"] == 1
    assert observations[2]["scheduler_epoch"] == 1
    assert observations[1]["reference_epoch"] == 1
    assert observations[2]["reference_epoch"] == 1
    assert observations[1]["learning_rate"] == pytest.approx(
        observations[2]["learning_rate"]
    )
    assert observations[1]["reference_learning_rate"] == pytest.approx(
        observations[2]["reference_learning_rate"]
    )

    assert not torch.equal(observations[0]["parameter"], observations[1]["parameter"])
    assert not torch.equal(observations[2]["parameter"], model.head.detach())
    assert state["step"] == 2
    assert scheduler.last_epoch == reference_scheduler.last_epoch == 2
    assert scheduler.get_last_lr() == pytest.approx(reference_scheduler.get_last_lr())
    assert telemetry.optimizer == (
        f"{RecordingPagedAdamW8bit.__module__}.{RecordingPagedAdamW8bit.__qualname__}"
    )
    assert telemetry.optimizer_attempts == 3
    assert telemetry.successful_updates == 2
    assert telemetry.skipped_updates == 1
    assert telemetry.scheduler_steps == 2
    assert telemetry.training_seconds is not None
    assert telemetry.peak_allocated_bytes is not None
    assert not any(
        "lr_scheduler.step() before optimizer.step()" in str(w.message)
        for w in recorded
    )
