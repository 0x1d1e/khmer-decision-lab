import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
calibrator = importlib.import_module("calibrate_r008")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="R008 calibration requires CUDA BF16",
)
def test_calibration_logits_use_cuda_bf16_autocast(monkeypatch):
    observed = {}

    def fake_calibration_records(
        _model, _tokenizer, items, device, max_len, head_max_len, parallel
    ):
        observed.update(
            autocast=torch.is_autocast_enabled("cuda"),
            dtype=torch.get_autocast_dtype("cuda"),
            device=device.type,
            max_len=max_len,
            head_max_len=head_max_len,
            parallel=parallel,
            items=items,
        )
        return ["record"]

    monkeypatch.setattr(calibrator, "calibration_records", fake_calibration_records)
    agent = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        amp_enabled=True,
        model=object(),
        tok=object(),
        parallel_options=False,
    )
    items = [object()]

    records = calibrator.calibration_records_bf16(agent, items)

    assert records == ["record"]
    assert observed == {
        "autocast": True,
        "dtype": torch.bfloat16,
        "device": "cuda",
        "max_len": 512,
        "head_max_len": 256,
        "parallel": False,
        "items": items,
    }


def test_calibration_records_reject_non_bf16_agent():
    agent = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.float16,
        amp_enabled=True,
    )

    with pytest.raises(ValueError, match="requires CUDA BF16 inference"):
        calibrator.calibration_records_bf16(agent, [])
