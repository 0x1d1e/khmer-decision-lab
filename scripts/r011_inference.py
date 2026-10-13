"""Capture Laya logits with R010's exact evaluation batching semantics."""

import numpy as np
import torch
from laya import evals

BATCH_SIZE = 8
GROUPING = "Laya evals.evaluate: consecutive identical question schemas, singleton predict, batch_size=8"


class _LogitRunner:
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name
        self._active_capture = None
        self._logit_batches = []
        self._hook = agent.model.register_forward_hook(self._capture_forward)

    def _capture_forward(self, _module, _args, output):
        if self._active_capture is None:
            raise RuntimeError("R011 observed a model forward outside a Laya eval call")
        if not isinstance(output, (tuple, list)) or not output:
            raise ValueError("R011 model forward did not return logits")
        logits = output[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("R011 model forward logits are not a tensor")
        self._active_capture.append(logits.detach().float().cpu().numpy())

    def _run(self, method, expected_rows, *args, **kwargs):
        if self._active_capture is not None:
            raise RuntimeError("R011 prediction calls cannot be nested")
        captured = []
        self._active_capture = captured
        try:
            result = method(*args, **kwargs)
        finally:
            self._active_capture = None
        if len(captured) != 1:
            raise ValueError(
                f"R011 expected one forward per Laya eval chunk, saw {len(captured)}"
            )
        logits = np.asarray(captured[0])
        if (
            logits.ndim != 2
            or logits.shape[0] != expected_rows
            or logits.shape[1] != 20
        ):
            raise ValueError(
                f"R011 logit chunk shape mismatch: {logits.shape}, expected ({expected_rows}, 20)"
            )
        self._logit_batches.append(logits)
        return result

    def predict(self, state, questions, model=None, min_confidence=None):
        if model != self.model_name:
            raise ValueError(f"R011 unexpected model in singleton eval: {model!r}")
        kwargs = {} if min_confidence is None else {"min_confidence": min_confidence}
        return self._run(
            self.agent.predict,
            1,
            state,
            questions,
            **kwargs,
        )

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
            raise ValueError(f"R011 unexpected model in batch eval: {model!r}")
        if sort_by_length:
            raise ValueError("R011 must preserve R010's unsorted batch inference")
        kwargs = {"batch_size": batch_size, "sort_by_length": False}
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


def capture_laya_logits(agent, rows, model_name):
    """Score ordered one-question rows using Laya's locked R010 eval batching."""
    if not rows:
        raise ValueError("R011 cannot capture logits for an empty dataset")
    examples = [
        evals.Example.from_dict(row, where=f"R011 row {index}")
        for index, row in enumerate(rows)
    ]
    if any(set(example.questions) != {"intent"} for example in examples):
        raise ValueError("R011 inputs must contain exactly one intent question per row")
    if any(
        len(example.questions["intent"].get("criteria", [])) != 20
        for example in examples
    ):
        raise ValueError("R011 inputs must use 20-option intent questions")
    for example in examples:
        example.model = model_name

    dataset = evals.Dataset(examples)
    runner = _LogitRunner(agent, model_name)
    try:
        report = evals.evaluate(
            runner,
            dataset,
            evaluators=[],
            batch_size=BATCH_SIZE,
        )
    finally:
        runner.close()

    logits = np.concatenate(runner._logit_batches, axis=0)
    if logits.shape != (len(rows), 20) or len(report.cases) != len(rows):
        raise ValueError(
            "R011 captured rows do not align with the Laya evaluation report"
        )
    return logits, report.cases
