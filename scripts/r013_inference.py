"""Capture full-precision choice logits through Laya's locked evaluation path."""

import numpy as np
import torch
from laya import evals
from r013_identity import INFERENCE_BATCH_SIZE, INFERENCE_LIMITS

BATCH_SIZE = INFERENCE_BATCH_SIZE


class _LogitRunner:
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name
        self._active_capture = None
        self._logit_batches = []
        self._hook = agent.model.register_forward_hook(self._capture_forward)

    def _capture_forward(self, _module, _args, output):
        if self._active_capture is None:
            raise RuntimeError("R013 observed a model forward outside a Laya eval call")
        if not isinstance(output, (tuple, list)) or not output:
            raise ValueError("R013 model forward did not return logits")
        logits = output[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("R013 model forward logits are not a tensor")
        self._active_capture.append(logits.detach().float().cpu().numpy())

    def _run(self, method, expected_rows, *args, **kwargs):
        if self._active_capture is not None:
            raise RuntimeError("R013 prediction calls cannot be nested")
        captured = []
        self._active_capture = captured
        try:
            result = method(*args, **kwargs)
        finally:
            self._active_capture = None
        if len(captured) != 1:
            raise ValueError(
                f"R013 expected one model forward per Laya eval chunk, saw {len(captured)}"
            )
        logits = np.asarray(captured[0])
        if (
            logits.ndim != 2
            or logits.shape[0] != expected_rows
            or logits.shape[1] != 20
            or not np.isfinite(logits).all()
        ):
            raise ValueError(
                f"R013 logit chunk shape or values are invalid: {logits.shape}"
            )
        self._logit_batches.append(logits)
        return result

    def predict(self, state, questions, model=None, min_confidence=None):
        if model != self.model_name:
            raise ValueError(f"R013 unexpected model in singleton eval: {model!r}")
        kwargs = {
            "max_len": INFERENCE_LIMITS["max_len"],
            "head_max_len": INFERENCE_LIMITS["head_max_len"],
        }
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        return self._run(self.agent.predict, 1, state, questions, **kwargs)

    def predict_batch(
        self,
        states,
        questions,
        model=None,
        batch_size=None,
        sort_by_length=False,
        min_confidence=None,
    ):
        if model != self.model_name:
            raise ValueError(f"R013 unexpected model in batch eval: {model!r}")
        if sort_by_length or batch_size != BATCH_SIZE:
            raise ValueError(
                "R013 inference must preserve unsorted batch_size=1 evaluation"
            )
        kwargs = {
            "batch_size": batch_size,
            "sort_by_length": False,
            "max_len": INFERENCE_LIMITS["max_len"],
            "head_max_len": INFERENCE_LIMITS["head_max_len"],
        }
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        return self._run(
            self.agent.predict_batch,
            len(states),
            states,
            questions,
            **kwargs,
        )

    def close(self):
        self._hook.remove()


def capture_laya_logits(agent, rows, model_name, config=None):
    """Run ordered 20-option rows and retain logits before Laya's rounded report output."""
    if not rows:
        raise ValueError("R013 cannot capture logits for an empty dataset")
    examples = [
        evals.Example.from_dict(row, where=f"R013 row {index}")
        for index, row in enumerate(rows)
    ]
    if any(set(example.questions) != {"intent"} for example in examples):
        raise ValueError("R013 inputs must contain exactly one intent question per row")
    if any(
        len(example.questions["intent"].get("criteria", [])) != 20
        for example in examples
    ):
        raise ValueError("R013 inputs must use 20-option intent questions")
    for example in examples:
        example.model = model_name

    dataset = evals.Dataset(examples)
    runner = _LogitRunner(agent, model_name)
    try:
        report = evals.evaluate(
            runner,
            dataset,
            evaluators=evals.default_evaluators(),
            batch_size=BATCH_SIZE,
            config=config,
        )
    finally:
        runner.close()

    logits = np.concatenate(runner._logit_batches, axis=0)
    if logits.shape != (len(rows), 20) or len(report.cases) != len(rows):
        raise ValueError("R013 captured logits do not align with Laya report rows")
    return logits, report.to_json()
