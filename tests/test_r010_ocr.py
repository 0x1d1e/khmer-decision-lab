import gc
import hashlib
import importlib
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from laya import evals
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
identity = importlib.import_module("r010_identity")
preparer = importlib.import_module("prepare_r010")
summarizer = importlib.import_module("summarize_r010")
evaluator = importlib.import_module("evaluate_r010")


def test_cer_counts_distinguish_codepoints_from_khmer_graphemes():
    counts = identity.cer_counts("ក្ខ", "ក")
    assert counts == {
        "codepoint_edits": 2,
        "reference_codepoints": 3,
        "grapheme_edits": 1,
        "reference_graphemes": 1,
        "exact": False,
    }


def test_cer_normalizes_unicode_and_collapses_whitespace():
    assert identity.cer_counts("  ក្ខ  ", "ក្ខ\n") == {
        "codepoint_edits": 0,
        "reference_codepoints": 3,
        "grapheme_edits": 0,
        "reference_graphemes": 1,
        "exact": True,
    }


def test_input_group_normalization_collapses_whitespace():
    assert identity.normalize_input("  ក  ខ \t") == identity.normalize_input("ក ខ")


def test_render_png_is_deterministic_and_shaped():
    font = ImageFont.truetype(str(identity.FONT_PATH), 24)
    first = preparer.render_png("សូមស្ងាត់", font)
    second = preparer.render_png("សូមស្ងាត់", font)
    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")


def test_macro_f1_includes_each_benchmark_label():
    result = summarizer._classification_metrics(
        [
            {"expected": "a", "answer": {"choice": "a"}},
            {"expected": "b", "answer": {"choice": "a"}},
        ],
        ["a", "b"],
    )
    assert result["choice_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_label"]["b"]["recall"] == 0.0


def test_macro_f1_rejects_choices_outside_the_locked_label_set():
    with pytest.raises(ValueError, match="outside the locked choice set"):
        summarizer._classification_metrics(
            [{"expected": "a", "answer": {"choice": "unknown"}}], ["a"]
        )


def test_group_bootstrap_resamples_duplicate_utterances_as_one_cluster():
    result = summarizer._paired_group_bootstrap(
        [False, False, False],
        [True, True, False],
        ["duplicate", "duplicate", "other"],
    )
    assert result["groups"] == 2
    assert result["difference_percentage_points"] == pytest.approx(200 / 3)


def test_group_bootstrap_is_paired_and_deterministic():
    first = summarizer._paired_group_bootstrap(
        [False, False], [True, False], ["group-a", "group-b"]
    )
    second = summarizer._paired_group_bootstrap(
        [False, False], [True, False], ["group-a", "group-b"]
    )
    assert first == second
    assert first["difference_percentage_points"] == 50.0
    assert first["paired_group_bootstrap_95_percent_interval_percentage_points"] == [
        0.0,
        100.0,
    ]
    assert first["groups"] == 2


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepared_fixture(tmp_path, monkeypatch, *, corrupt_overlap=False):
    monkeypatch.setattr(identity, "ROOT", tmp_path)
    artifact_dir = tmp_path / "artifacts/runs/R010"
    monkeypatch.setattr(identity, "ARTIFACT_DIR", artifact_dir)
    source_rows: list[dict[str, Any]] = [
        {
            "state": {"utterance": text},
            "questions": {"intent": {"criteria": {"a": "A", "b": "B"}}},
            "expected": {"intent": label},
            "tags": [],
            "language": "km",
        }
        for text, label in (("same", "a"), ("different", "b"))
    ]
    test_path = tmp_path / "data/test.jsonl"
    train_path = tmp_path / "data/train.jsonl"
    _write_jsonl(test_path, source_rows)
    _write_jsonl(train_path, [{"state": {"utterance": "same"}}])
    test_sha256 = identity.sha256(test_path)
    train_inputs = {"same"}
    metadata: list[dict[str, Any]] = []
    group_sizes: dict[str, int] = {}
    for index, row in enumerate(source_rows):
        normalized = identity.normalize_input(row["state"]["utterance"])
        group_id = hashlib.sha256(normalized.encode()).hexdigest()
        row_id = hashlib.sha256(
            f"{test_sha256}\0{index}\0{identity.canonical_json(row)}".encode()
        ).hexdigest()
        metadata.append(
            {
                "row_index": index,
                "row_id": row_id,
                "group_id": group_id,
                "train_overlap": normalized in train_inputs,
                "reference": row["state"]["utterance"],
            }
        )
        group_sizes[normalized] = 1
    audit = {
        "test_rows": 2,
        "normalized_test_groups": 2,
        "conflicting_label_groups": 0,
        "conflicting_label_rows": 0,
        "exact_train_overlap_groups": 1,
        "exact_train_overlap_rows": 1,
    }
    manifest = {
        "study": "R010-massive-ocr-cascade",
        "source": {
            "test_file": "data/test.jsonl",
            "test_sha256": test_sha256,
            "test_rows": 2,
            "train_file": "data/train.jsonl",
            "audit": audit,
        },
        "implementation_sha256": {"scripts/prepare_r010.py": "preparer"},
    }
    prepared_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in identity.CONDITIONS
    }
    ocr_records: list[dict[str, Any]] = []
    for index, (source_row, row_metadata) in enumerate(zip(source_rows, metadata)):
        overlap = row_metadata["train_overlap"]
        recorded_overlap = not overlap if corrupt_overlap and index == 0 else overlap
        tags = [
            f"r010-row:{row_metadata['row_id']}",
            f"r010-group:{row_metadata['group_id']}",
            f"r010-train-overlap:{int(overlap)}",
        ]
        record = {**row_metadata, "train_overlap": recorded_overlap, "ocr": {}}
        clean = {**source_row, "tags": tags}
        prepared_rows["clean"].append(clean)
        for condition in identity.CONDITIONS[1:]:
            text = f"ocr {condition} {index}"
            result = {
                "text": text,
                "image_sha256": "a" * 64,
                **identity.cer_counts(row_metadata["reference"], text),
            }
            record["ocr"][condition] = result
            row = json.loads(json.dumps(source_row))
            row["state"]["utterance"] = text
            row["tags"] = tags
            prepared_rows[condition].append(row)
        ocr_records.append(record)

    outputs = {}
    for condition, rows in prepared_rows.items():
        relative = f"data/processed/r010-{condition}.jsonl"
        path = tmp_path / relative
        _write_jsonl(path, rows)
        outputs[condition] = {
            "file": relative,
            "sha256": identity.sha256(path),
            "rows": len(rows),
        }
    transcript_path = artifact_dir / "ocr-transcripts.jsonl"
    _write_jsonl(transcript_path, ocr_records)
    outputs["ocr_transcripts"] = {
        "file": str(transcript_path.relative_to(tmp_path)),
        "sha256": identity.sha256(transcript_path),
        "rows": len(ocr_records),
    }
    preparation = {
        "study": manifest["study"],
        "manifest_sha256": "locked",
        "source_test_sha256": test_sha256,
        "source_audit": audit,
        "preparer_sha256": "preparer",
        "outputs": outputs,
        "files_sha256": {value["file"]: value["sha256"] for value in outputs.values()},
    }
    identity.write_json(artifact_dir / "preparation.json", preparation)
    return manifest, preparation, artifact_dir


def test_verify_prepared_rejects_transcript_overlap_disagreeing_with_source(
    tmp_path, monkeypatch
):
    manifest, _preparation, _artifact_dir = _prepared_fixture(
        tmp_path, monkeypatch, corrupt_overlap=True
    )
    with pytest.raises(ValueError, match="row identity mismatch"):
        identity.verify_prepared(manifest, "locked")


def test_report_cases_accepts_expected_label_from_laya_evaluator(tmp_path, monkeypatch):
    manifest, preparation, artifact_dir = _prepared_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)
    monkeypatch.setattr(summarizer, "ARTIFACT_DIR", artifact_dir)
    summarizer._prepared_rows.cache_clear()
    dataset_path = tmp_path / preparation["outputs"]["clean"]["file"]
    dataset = evals.Dataset.from_jsonl(str(dataset_path))
    for example in dataset.examples:
        example.model = "mt-s13"

    class StubRunner:
        def predict(self, state, questions, model=None):
            if model not in (None, "mt-s13"):
                raise AssertionError(f"unexpected model: {model}")
            choice = "a" if state["utterance"] == "same" else "b"
            return {
                "answers": {
                    qid: {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": {"a": 0.9, "b": 0.1}
                        if choice == "a"
                        else {"a": 0.1, "b": 0.9},
                        "answer_confidence": 0.9,
                    }
                    for qid in questions
                }
            }

    checkpoint = {
        "checkpoint": "artifacts/checkpoints/R006-mt-s13",
        "inference_files_sha256": {"model.safetensors": "weights"},
        "max_len": 512,
        "head_max_len": 256,
    }
    model = "mt-s13"
    manifest["models"] = {model: checkpoint}
    manifest["environment"] = {
        "gpu": "test-gpu",
        "torch_version": "test-torch",
        "torch_cuda": "test-cuda",
    }
    inference = __import__("r008_identity").INFERENCE_POLICY
    config = {
        "study": manifest["study"],
        "condition": "clean",
        "manifest_sha256": "locked",
        "source_test_file": manifest["source"]["test_file"],
        "source_test_sha256": manifest["source"]["test_sha256"],
        "dataset": preparation["outputs"]["clean"]["file"],
        "dataset_sha256": preparation["outputs"]["clean"]["sha256"],
        "preparation_sha256": identity.sha256(artifact_dir / "preparation.json"),
        "preparer_sha256": "preparer",
        "model": model,
        "checkpoint": checkpoint["checkpoint"],
        "weights_sha256": "weights",
        "inference_files_sha256": checkpoint["inference_files_sha256"],
        "max_len": 512,
        "head_max_len": 256,
        "device": inference["device"],
        "autocast_dtype": inference["autocast_dtype"],
        "laya_cuda_amp": inference["laya_cuda_amp"],
        "gpu": "test-gpu",
        "torch_version": "test-torch",
        "cuda_version": "test-cuda",
    }
    report = evals.evaluate(
        StubRunner(),
        dataset,
        evaluators=evals.default_evaluators(),
        batch_size=1,
        config=config,
    ).to_json()
    report_path = tmp_path / "artifacts/runs/R010/massive-km/clean/mt-s13.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    evaluation = {
        "manifest_sha256": "locked",
        "reports": [
            {
                "condition": "clean",
                "model": model,
                "report": "artifacts/runs/R010/massive-km/clean/mt-s13.json",
                "report_sha256": identity.sha256(report_path),
                "rows": 2,
                "choice_accuracy": report["overall"]["choice_accuracy"],
            }
        ],
    }

    loaded_report, cases = summarizer._report_cases(
        manifest, "clean", model, preparation, evaluation
    )
    assert loaded_report["cases"][0]["expected"] == "a"
    assert [case["expected"] for case in cases] == ["a", "b"]


def test_evaluator_releases_previous_agent_before_loading_replacement(
    tmp_path, monkeypatch
):
    router_class = evaluator.laya.Router
    load_observations = []
    expected_files = {
        "first": {"model.safetensors": "first-weights"},
        "second": {"model.safetensors": "second-weights"},
    }

    class StubAgent:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.device = SimpleNamespace(type="cuda")
            self.dtype = torch.bfloat16
            self.amp_enabled = True
            self.last_fallback_reason = None

    def make_router(**kwargs):
        router = router_class(**kwargs)
        agent_refs = []

        def build(_name):
            if agent_refs:
                load_observations.append(agent_refs[-1]() is None)
            agent = StubAgent()
            agent_refs.append(weakref.ref(agent))
            return agent

        router._build = build
        return router

    class StubDataset:
        def __init__(self, examples):
            self.examples = examples

        @classmethod
        def from_jsonl(cls, _path):
            return cls([SimpleNamespace(model=None)])

    class StubReport:
        def __init__(self, config):
            self.config = config or {}

        def to_json(self):
            return {
                "config": self.config,
                "cases": [{}],
                "overall": {"choice_accuracy": 1.0},
            }

    monkeypatch.setattr(evaluator.laya, "Router", make_router)
    monkeypatch.setattr(evaluator.evals, "Dataset", StubDataset)
    monkeypatch.setattr(
        evaluator.evals,
        "evaluate",
        lambda _runner, _dataset, _evaluators=None, _batch_size=8, **kwargs: StubReport(
            kwargs.get("config")
        ),
    )
    monkeypatch.setattr(evaluator, "RouterRunner", lambda _router: object())
    monkeypatch.setattr(evaluator, "verify_locked_manifest", lambda: (manifest, "lock"))
    monkeypatch.setattr(evaluator, "verify_prepared", lambda *_args: preparation)
    monkeypatch.setattr(
        evaluator,
        "checkpoint_inference_inventory",
        lambda checkpoint: expected_files[
            Path(checkpoint).name.removeprefix("checkpoint-")
        ],
    )
    monkeypatch.setattr(evaluator, "sha256", lambda _path: "artifact-hash")
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "ARTIFACT_DIR", tmp_path / "artifacts/runs/R010")
    monkeypatch.setattr(evaluator, "CONDITIONS", ("clean",))
    monkeypatch.setattr(
        evaluator,
        "MODEL_PATHS",
        {"first": "checkpoint-first", "second": "checkpoint-second"},
    )
    monkeypatch.setattr(
        evaluator,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                is_bf16_supported=lambda: True,
                empty_cache=lambda: None,
                synchronize=lambda: None,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0,
                get_device_name=lambda *_args: "test-gpu",
            ),
            bfloat16=torch.bfloat16,
            __version__="test-torch",
            version=SimpleNamespace(cuda="test-cuda"),
        ),
    )

    manifest = {
        "study": "R010-massive-ocr-cascade",
        "implementation_sha256": {"scripts/prepare_r010.py": "preparer"},
        "models": {
            name: {
                "checkpoint": f"checkpoint-{name}",
                "inference_files_sha256": expected_files[name],
                "max_len": 512,
                "head_max_len": 256,
            }
            for name in expected_files
        },
        "environment": {
            "gpu": "test-gpu",
            "torch_version": "test-torch",
            "torch_cuda": "test-cuda",
        },
        "source": {"test_file": "data/test.jsonl", "test_sha256": "source-hash"},
    }
    preparation = {
        "outputs": {"clean": {"file": "data/clean.jsonl", "sha256": "data-hash"}}
    }

    evaluator.main()
    gc.collect()
    assert load_observations == [True]
