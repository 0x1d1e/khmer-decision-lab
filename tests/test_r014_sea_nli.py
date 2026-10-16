"""R014 SEA-NLI preparation invariants. Fixtures only: no GPU, network, or gated data."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
preparer = importlib.import_module("prepare_r014")
evaluator = importlib.import_module("evaluate_r014")
summarizer = importlib.import_module("summarize_r014")

LANGUAGE_FIELD = "culture"
KHMER_CULTURE = "cambodian"


def _khmer(text):
    """Fixtures must carry Khmer script in native text like the source."""
    if any(0x1780 <= ord(char) <= 0x17FF for char in text):
        return text
    return f"ក{text}"


def _row(
    premise,
    hypothesis,
    label,
    *,
    culture=KHMER_CULTURE,
    english=None,
    entry_id=None,
    **extra,
):
    hypothesis = _khmer(hypothesis)
    if english is None:
        english = (f"EN {premise}", f"EN {hypothesis}")
    if entry_id is None:
        entry_id = f"id-{premise}-{hypothesis}"
    return {
        "premise_native": premise,
        "hypothesis_native": hypothesis,
        "premise_english": english[0],
        "hypothesis_english": english[1],
        "culture": culture,
        "true_label": label,
        "entry_id": entry_id,
        **extra,
    }


def _prepare(tmp_path, monkeypatch, rows_by_config, **kwargs):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    out_dir = tmp_path / "data/processed"
    manifest_path = tmp_path / "datasets/manifests/sea-nli-km.json"
    manifest = preparer.prepare(
        rows_by_config,
        source_files=[
            {"config": config, "path": f"{config}/test.parquet", "sha256": "0" * 64}
            for config in rows_by_config
        ],
        out_dir=out_dir,
        manifest_path=manifest_path,
        **kwargs,
    )
    return manifest, out_dir, manifest_path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_culture_selection_is_casefolded_and_not_script_based(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row("ក ខគ", "h1", "entailment", culture="Cambodian"),
            _row("ក ខគ", "h2", "neutral", culture="thai"),
            _row("hello", "h3", "neutral", culture="singaporean"),
        ],
        "hard": [_row("ខគ ក", "h4", "contradiction", culture="cambodian")],
    }
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, rows)
    selection = manifest["selection"]
    assert selection["field"] == "culture"
    assert selection["value"] == "cambodian"
    assert selection["counts"]["selected"] == {"normal": 1, "hard": 1}
    assert selection["counts"]["non_khmer"] == {"normal": 2, "hard": 0}
    assert manifest["validation"]["checked_rows"] == 2
    assert manifest["validation"]["passed"] is True
    khmer_rows = [
        json.loads(line)
        for line in (out_dir / "sea-nli-km-normal-test.jsonl").read_text().splitlines()
    ]
    # The `thai` row carries Khmer script but is not cambodian culture.
    assert len(khmer_rows) == 1
    assert khmer_rows[0]["state"]["premise"] == "ក ខគ"
    assert khmer_rows[0]["language"] == "km"


def test_validation_raises_on_non_khmer_native_row(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row(
                "latin premise",
                "h1",
                "entailment",
                culture="cambodian",
                entry_id="bad-1",
            )
        ],
        "hard": [_row("ក ខគ", "h2", "neutral")],
    }
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="bad-1"):
        preparer.prepare(
            rows,
            out_dir=tmp_path / "data/processed",
            manifest_path=tmp_path / "datasets/manifests/sea-nli-km.json",
        )


def test_selection_fails_loudly_when_no_khmer_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    rows = {"normal": [_row("hello", "h1", "neutral", culture="singaporean")]}
    with pytest.raises(ValueError, match="no Khmer rows"):
        preparer.prepare(
            rows,
            out_dir=tmp_path / "data/processed",
            manifest_path=tmp_path / "datasets/manifests/sea-nli-km.json",
        )


def test_zwsp_is_stripped_from_grouping_but_not_model_text(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row("ក ខ\u200bគ", "h1", "entailment"),
            _row("ក ខគ", "h2", "neutral"),
        ],
        "hard": [_row("ក ខគ", "h3", "contradiction")],
    }
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, rows)
    examples = [
        json.loads(line)
        for line in (out_dir / "sea-nli-km-normal-test.jsonl").read_text().splitlines()
    ]
    sidecar = json.loads((out_dir / "sea-nli-km-normal-test-ids.json").read_text())
    assert len(examples) == 2
    assert examples[0]["state"]["premise"] == "ក ខ\u200bគ"
    assert "\u200b" in examples[0]["state"]["premise"]
    assert sidecar[0]["group_id"] == sidecar[1]["group_id"]
    assert (
        manifest["exports"]["data/processed/sea-nli-km-normal-test.jsonl"][
            "rows_sharing_premise"
        ]
        == 2
    )


def test_duplicate_removal_keeps_lowest_source_index(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row("ក ខគ", "h", "entailment"),
            _row("other-premise", "h2", "neutral", culture="singaporean"),
            _row("ក ខគ", "h", "entailment"),  # exact duplicate
            _row("ក ខគ", "h", "contradiction"),  # different label
        ],
        "hard": [_row("ក ខគ", "h9", "neutral")],
    }
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, rows)
    sidecar = json.loads((out_dir / "sea-nli-km-normal-test-ids.json").read_text())
    entry = manifest["exports"]["data/processed/sea-nli-km-normal-test.jsonl"]
    assert entry["rows"] == 2
    assert entry["duplicates_removed"] == 1
    assert sidecar[0]["source_index"] == 0
    assert {item["source_index"] for item in sidecar} == {0, 3}


def test_field_allowlist_drops_metadata_columns(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row(
                "ក ខគ",
                "h1",
                "entailment",
                is_misclassified=True,
                step_misclassified=0,
                improved_v1=1,
                entailment_type="lexical",
                row_id=42,
            )
        ],
        "hard": [_row("ក ខគ", "h9", "neutral")],
    }
    _, out_dir, _ = _prepare(tmp_path, monkeypatch, rows)
    path = out_dir / "sea-nli-km-normal-test.jsonl"
    text = path.read_text()
    assert "is_misclassified" not in text
    assert "entailment_type" not in text
    assert "row_id" not in text
    example = json.loads(text.splitlines()[0])
    assert set(example) == {"state", "language", "questions", "expected"}
    assert set(example["state"]) == {"premise", "hypothesis"}


def test_rendered_options_are_bare_labels():
    from laya.common import render_options

    question = {"t": "choice", "crit": preparer.CRITERIA}
    assert render_options(question) == ["entailment", "neutral", "contradiction"]
    assert all(value == "" for value in preparer.CRITERIA.values())


def test_resolve_gold_encodings_and_khmer_raise(tmp_path, monkeypatch):
    names = ["entailment", "neutral", "contradiction"]
    assert preparer.resolve_gold(0, names) == "entailment"
    assert preparer.resolve_gold(2, names) == "contradiction"
    assert preparer.resolve_gold("Entailment", names) == "entailment"
    with pytest.raises(ValueError):
        preparer.resolve_gold(3, names)
    with pytest.raises(ValueError):
        preparer.resolve_gold(0, None)
    with pytest.raises(ValueError):
        preparer.resolve_gold(None, names)
    with pytest.raises(ValueError):
        preparer.resolve_gold("maybe", names)
    with pytest.raises(TypeError):
        preparer.resolve_gold(True, names)

    monkeypatch.setattr(preparer, "ROOT", tmp_path)
    rows = {
        "normal": [_row("ក ខគ", "h1", "maybe")],
        "hard": [_row("ក ខគ", "h2", "entailment")],
    }
    with pytest.raises(ValueError, match="maybe"):
        preparer.prepare(
            rows,
            out_dir=tmp_path / "data/processed",
            manifest_path=tmp_path / "datasets/manifests/sea-nli-km.json",
        )


def test_non_khmer_unmapped_labels_are_counted_not_silently_dropped(
    tmp_path, monkeypatch
):
    rows = {
        "normal": [
            _row("ក ខគ", "h1", "entailment"),
            _row("hello", "h2", "maybe", culture="singaporean"),
        ],
        "hard": [_row("ក ខគ", "h3", "neutral")],
    }
    manifest, _, _ = _prepare(tmp_path, monkeypatch, rows)
    other = manifest["exports"]["data/processed/sea-nli-other-test.jsonl"]
    assert other["rows"] == 0
    assert other["exclusions"] == {"unmapped_true_label": 1}
    assert manifest["exclusion_counts_by_reason"] == {"unmapped_true_label": 1}


def test_manifest_is_self_consistent(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row("ក ខគ", "h1", "entailment"),
            _row("hello", "h2", "neutral", culture="singaporean"),
        ],
        "hard": [_row("ក ខគ", "h3", "contradiction")],
    }
    manifest, _, manifest_path = _prepare(tmp_path, monkeypatch, rows)
    assert manifest["dataset"] == "aisingapore/SEA-NLI"
    assert manifest["revision"] == "f3a73ccb4213ff995ef2bf8bcb903eaf53195713"
    assert manifest["task"]["labels"] == ["entailment", "neutral", "contradiction"]
    assert manifest["preparer_sha256"] == _sha256(ROOT / "scripts/prepare_r014.py")
    assert manifest_path.is_file()
    for relative, entry in manifest["exports"].items():
        path = tmp_path / relative
        assert entry["sha256"] == _sha256(path)
        rows_out = path.read_text().splitlines()
        assert len(rows_out) == entry["rows"]
        ids_path = tmp_path / entry["ids"]
        assert entry["ids_sha256"] == _sha256(ids_path)
        sidecar = json.loads(ids_path.read_text())
        assert len(sidecar) == entry["rows"]
        assert entry["groups"] == len({item["group_id"] for item in sidecar})
        assert sum(entry["label_distribution"].values()) == entry["rows"]
    assert set(manifest["exports"]) == {
        "data/processed/sea-nli-km-normal-test.jsonl",
        "data/processed/sea-nli-km-hard-test.jsonl",
        "data/processed/sea-nli-en-normal-test.jsonl",
        "data/processed/sea-nli-en-hard-test.jsonl",
        "data/processed/sea-nli-other-test.jsonl",
    }
    assert manifest["descriptors"]["by_config"]["normal"]["selected_rows"] == 1
    assert manifest["arms"]["english"]["state_source"] == preparer.ENGLISH_FIELDS


def test_other_export_never_dedups_across_configs(tmp_path, monkeypatch):
    rows = {
        "normal": [
            _row("hello world", "h1", "neutral", culture="singaporean"),
            _row("ក ខគ", "h2", "entailment"),
        ],
        "hard": [
            _row("hello world", "h1", "neutral", culture="singaporean"),
            _row("ក ខគ", "h2", "entailment"),
        ],
    }
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, rows)
    entry = manifest["exports"]["data/processed/sea-nli-other-test.jsonl"]
    assert entry["rows"] == 2
    assert entry["duplicates_removed"] == 0
    assert entry["source_configs"] == ["normal", "hard"]
    assert len((out_dir / "sea-nli-other-test.jsonl").read_text().splitlines()) == 2


def test_english_arm_shares_identity_and_gold_and_differs_only_in_state(
    tmp_path, monkeypatch
):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    for native, english in (("km-normal", "en-normal"), ("km-hard", "en-hard")):
        native_examples = _read_examples(out_dir, native)
        english_examples = _read_examples(out_dir, english)
        assert native_examples and len(native_examples) == len(english_examples)
        assert [example["expected"] for example in native_examples] == [
            example["expected"] for example in english_examples
        ]
        assert _read_sidecar(out_dir, native) == _read_sidecar(out_dir, english)
        for native_example, english_example in zip(
            native_examples, english_examples, strict=True
        ):
            assert native_example["state"] != english_example["state"]
            assert english_example["state"]["premise"].startswith("EN ")
            # The Khmer arms carry `km`; the English-text control arms carry `en`.
            assert native_example["language"] == "km"
            assert english_example["language"] == "en"
    en_entry = manifest["exports"]["data/processed/sea-nli-en-normal-test.jsonl"]
    assert en_entry["arm"] == "english"
    assert en_entry["state_fields"] == preparer.ENGLISH_FIELDS


def test_language_field_pins_arm_text(tmp_path, monkeypatch):
    """`language` tracks the model-visible text, not the source culture."""
    _, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    expected = {
        "sea-nli-km-normal-test": "km",
        "sea-nli-km-hard-test": "km",
        "sea-nli-en-normal-test": "en",
        "sea-nli-en-hard-test": "en",
        "sea-nli-other-test": None,
    }
    for stem, language in expected.items():
        examples = [
            json.loads(line)
            for line in (out_dir / f"{stem}.jsonl").read_text().splitlines()
        ]
        assert examples, stem
        assert {example["language"] for example in examples} == {language}


def test_bootstrap_parameters_and_quantile_method():
    assert summarizer.EVALUATION["bootstrap_draws"] == 10000
    assert summarizer.EVALUATION["bootstrap_seed"] == 13
    assert "method='linear'" in summarizer.EVALUATION["bootstrap_interval"]

    def case(group, expected, correct, choice, confidence=0.9):
        return {
            "tags": [f"r014-group:{group}"],
            "expected": expected,
            "correct": correct,
            "confidence": confidence,
            "answer": {"choice": choice},
        }

    reference = [
        case("a", "entailment", True, "entailment"),
        case("a", "neutral", True, "neutral"),
        case("b", "contradiction", True, "contradiction"),
    ]
    candidate = [
        case("a", "entailment", False, "neutral"),
        case("a", "neutral", False, "neutral"),
        case("b", "contradiction", False, "neutral"),
    ]
    result = summarizer.paired_group_bootstrap(reference, candidate, draws=200, seed=13)
    assert result["draws"] == 200
    assert result["seed"] == 13
    assert result["quantile_method"] == "numpy.quantile(method='linear')"
    assert result["groups"] == 2
    assert result["rows"] == 3
    assert result["difference_pp"] == pytest.approx(-100.0)
    assert result["ci95_pp"] == pytest.approx([-100.0, -100.0])
    again = summarizer.paired_group_bootstrap(reference, candidate, draws=200, seed=13)
    assert again == result


def _unclustered_ci(reference, candidate, draws, seed):
    """Row-resampling reference used to catch an unclustered implementation."""
    differences = np.asarray(
        [
            float(bool(candidate_case.get("correct")))
            - float(bool(reference_case.get("correct")))
            for reference_case, candidate_case in zip(reference, candidate, strict=True)
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.choice(len(differences), size=len(differences), replace=True)
        estimates[draw] = 100.0 * float(differences[sample].mean())
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def test_paired_group_bootstrap_resamples_clusters_not_rows():
    def case(group, correct):
        return {
            "tags": [f"r014-group:{group}"],
            "expected": "entailment",
            "correct": correct,
            "confidence": 0.9,
            "answer": {"choice": "entailment" if correct else "neutral"},
        }

    reference, candidate = [], []
    for group, difference in zip("abcd", (1, 1, 1, -1), strict=True):
        for _ in range(10):
            reference.append(case(group, difference != 1))
            candidate.append(case(group, difference == 1))

    result = summarizer.paired_group_bootstrap(
        reference, candidate, draws=2000, seed=13
    )
    assert result["groups"] == 4
    assert result["rows"] == 40
    assert result["difference_pp"] == pytest.approx(50.0)
    unclustered_lower, unclustered_upper = _unclustered_ci(
        reference, candidate, draws=2000, seed=13
    )
    # Heterogeneous clusters: clustered resampling widens the interval well past
    # what row-level resampling gives, so an unclustered implementation fails.
    assert result["ci95_pp"][0] < unclustered_lower
    assert result["ci95_pp"][1] > unclustered_upper


def test_compute_metrics_matches_report_shape():
    def case(expected, choice, correct, confidence):
        return {
            "tags": ["r014-group:g"],
            "expected": expected,
            "correct": correct,
            "confidence": confidence,
            "answer": {"choice": choice},
        }

    cases = [
        case("entailment", "entailment", True, 0.8),
        case("neutral", "entailment", False, 0.6),
        case("contradiction", "contradiction", True, 0.7),
    ]
    metrics = summarizer.compute_metrics(cases)
    assert metrics["rows"] == 3
    assert metrics["choice_accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_confidence"] == pytest.approx(0.7)
    assert metrics["per_class_recall"]["entailment"] == pytest.approx(1.0)
    assert metrics["per_class_recall"]["neutral"] == pytest.approx(0.0)
    assert sum(metrics["predicted_label_distribution"].values()) == 3
    assert 0.0 <= metrics["ece_15_equal_width_bins"] <= 1.0


def test_report_isolation_policy_constants():
    assert evaluator.MODELS == ("multilingual", "mt-s13", "mt-s14", "full-e2-8bit")
    assert evaluator.BATCH_SIZE == 8
    assert tuple(evaluator.CONDITIONS) == (
        "km-normal",
        "km-hard",
        "en-normal",
        "en-hard",
        "other",
    )
    assert evaluator.TAG_PREFIX == "r014-group:"
    assert evaluator.MAX_LEN == 512
    assert evaluator.HEAD_MAX_LEN == 256
    assert evaluator.BUDGET == {"max_len": 512, "head_max_len": 256}
    assert set(evaluator.EXPECTED_WEIGHTS) == {"mt-s13", "mt-s14", "full-e2-8bit"}
    assert len(set(evaluator.EXPECTED_WEIGHTS.values())) == 3
    assert set(evaluator.MODELS) - set(evaluator.EXPECTED_WEIGHTS) == {"multilingual"}


CONDITION_STEMS = {
    "km-normal": "sea-nli-km-normal-test",
    "km-hard": "sea-nli-km-hard-test",
    "en-normal": "sea-nli-en-normal-test",
    "en-hard": "sea-nli-en-hard-test",
    "other": "sea-nli-other-test",
}


def _read_examples(out_dir, condition):
    stem = CONDITION_STEMS[condition]
    return [
        json.loads(line)
        for line in (out_dir / f"{stem}.jsonl").read_text().splitlines()
    ]


def _read_sidecar(out_dir, condition):
    stem = CONDITION_STEMS[condition]
    return json.loads((out_dir / f"{stem}-ids.json").read_text())


def _fixture_cases(out_dir, condition, arm, degenerate=False):
    examples = _read_examples(out_dir, condition)
    sidecar = _read_sidecar(out_dir, condition)
    cases = []
    for index, (example, entry) in enumerate(zip(examples, sidecar, strict=True)):
        gold = example["expected"]["relationship"]
        if degenerate and arm == "multilingual":
            correct = gold == "entailment"
            choice = "entailment"
        elif arm == "multilingual":
            correct = index % 2 == 0
            choice = (
                gold
                if correct
                else next(label for label in preparer.LABELS if label != gold)
            )
        else:
            correct = True
            choice = gold
        cases.append(
            {
                "qid": "relationship",
                "tags": [f"{evaluator.TAG_PREFIX}{entry['group_id']}"],
                "expected": gold,
                "model": arm,
                "answer": {"type": "choice", "choice": choice},
                "confidence": 0.9 if correct else 0.4,
                "correct": correct,
            }
        )
    return cases


def _default_weights():
    return {"multilingual": "base-weights", **evaluator.EXPECTED_WEIGHTS}


def _write_fixture_reports(
    tmp_path,
    manifest,
    out_dir,
    *,
    cases_fn=_fixture_cases,
    weights=None,
    base_revision=None,
    max_len=None,
    head_max_len=None,
    dataset_sha_overrides=None,
):
    reports_dir = tmp_path / "artifacts/runs/R014"
    weights = weights or _default_weights()
    dataset_sha_overrides = dataset_sha_overrides or {}
    for condition in evaluator.CONDITIONS:
        relative = evaluator.DATASETS[condition]
        entry = manifest["exports"][relative]
        for model in evaluator.MODELS:
            payload = {
                "config": {
                    "study": "R014-sea-nli-khmer-transfer",
                    "condition": condition,
                    "dataset": relative,
                    "dataset_sha256": dataset_sha_overrides.get(
                        condition, entry["sha256"]
                    ),
                    "sidecar": entry["ids"],
                    "sidecar_sha256": entry["ids_sha256"],
                    "base_revision": base_revision or evaluator.BASE_REVISION,
                    "max_len": max_len or evaluator.MAX_LEN,
                    "head_max_len": head_max_len or evaluator.HEAD_MAX_LEN,
                    "model": model,
                    "measurement_protocol": "isolated-warm-v1",
                    "batch_size": 8,
                    "weights_sha256": weights[model],
                },
                "overall": {},
                "slices": {},
                "cases": cases_fn(out_dir, condition, model),
            }
            path = reports_dir / condition / f"{model}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
    return reports_dir


def _wire_summary(tmp_path, monkeypatch, manifest_path, reports_dir):
    protocol = tmp_path / "research/R014-sea-nli-khmer-transfer/README.md"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("# R014 protocol\n", encoding="utf-8")
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)
    monkeypatch.setattr(summarizer, "MANIFEST", manifest_path)
    monkeypatch.setattr(summarizer, "OUT", reports_dir)
    monkeypatch.setattr(summarizer, "PROTOCOL", protocol)
    monkeypatch.setattr(summarizer, "RESULTS_JSON", protocol.parent / "results.json")
    monkeypatch.setattr(summarizer, "RESULTS_MD", protocol.parent / "results.md")
    return protocol


def _summary_rows():
    return {
        "normal": [
            _row("ក ខគ", "h1", "entailment"),
            _row("ក ខគ", "h2", "neutral"),
            _row("hello", "h3", "neutral", culture="singaporean"),
        ],
        "hard": [
            _row("ក ខគ", "h1", "contradiction"),
            _row("bonjour", "h2", "neutral", culture="french"),
        ],
    }


def test_summarize_end_to_end_with_fixture_reports(tmp_path, monkeypatch):
    manifest, out_dir, manifest_path = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir)
    protocol = _wire_summary(tmp_path, monkeypatch, manifest_path, reports_dir)

    result = summarizer.summarize()
    assert result["primary"]["valid"] is True
    assert result["primary"]["supported"] is True
    assert result["degeneracy"]["degenerate"] is False
    for seed in summarizer.SEEDS:
        row = result["primary"]["per_seed"][str(seed)]
        assert row["lower_bound_above_zero"] is True
        assert row["ci95_pp"][0] > 0
    assert set(result["metrics"]) == set(evaluator.CONDITIONS)
    assert (protocol.parent / "results.json").is_file()
    assert (protocol.parent / "results.md").is_file()
    assert "Primary comparison" in (protocol.parent / "results.md").read_text()


def test_degeneracy_guard_flips_valid_and_supported(tmp_path, monkeypatch):
    manifest, out_dir, manifest_path = _prepare(tmp_path, monkeypatch, _summary_rows())

    def degenerate_cases(out_dir, condition, arm):
        return _fixture_cases(out_dir, condition, arm, degenerate=True)

    reports_dir = _write_fixture_reports(
        tmp_path, manifest, out_dir, cases_fn=degenerate_cases
    )
    protocol = _wire_summary(tmp_path, monkeypatch, manifest_path, reports_dir)

    result = summarizer.summarize()
    assert result["degeneracy"]["degenerate"] is True
    assert "multilingual" in result["degeneracy"]["arms"]
    assert result["primary"]["valid"] is False
    assert result["primary"]["supported"] is False
    markdown = (protocol.parent / "results.md").read_text()
    assert "Primary support rule met: **False**" in markdown
    assert "valid=False" in markdown


def _checked_load(tmp_path, monkeypatch, reports_dir):
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)
    monkeypatch.setattr(summarizer, "OUT", reports_dir)


def test_load_reports_raises_named_error_on_empty_cases(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir)
    path = reports_dir / "km-normal" / "multilingual.json"
    payload = json.loads(path.read_text())
    payload["cases"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="has no cases"):
        summarizer._load_reports(manifest)


def test_load_reports_rejects_drifted_dataset_sha256(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(
        tmp_path, manifest, out_dir, dataset_sha_overrides={"km-normal": "0" * 64}
    )
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="configuration mismatch"):
        summarizer._load_reports(manifest)


def test_load_reports_rejects_wrong_base_revision(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(
        tmp_path, manifest, out_dir, base_revision="0" * 40
    )
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="configuration mismatch"):
        summarizer._load_reports(manifest)


def test_load_reports_enforces_pinned_weights_lock(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    weights = _default_weights()
    weights["mt-s13"] = "f" * 64
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir, weights=weights)
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="pinned digest"):
        summarizer._load_reports(manifest)


def test_load_reports_rejects_non_distinct_weights(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    monkeypatch.setattr(
        summarizer,
        "EXPECTED_WEIGHTS",
        {"mt-s13": "dup", "mt-s14": "dup", "full-e2-8bit": "uniq"},
    )
    weights = {
        "multilingual": "base-weights",
        "mt-s13": "dup",
        "mt-s14": "dup",
        "full-e2-8bit": "uniq",
    }
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir, weights=weights)
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="not distinct"):
        summarizer._load_reports(manifest)


def test_load_reports_rejects_group_tag_sidecar_mismatch(tmp_path, monkeypatch):
    manifest, out_dir, _ = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir)
    path = reports_dir / "km-normal" / "multilingual.json"
    payload = json.loads(path.read_text())
    payload["cases"][0]["tags"] = [f"{evaluator.TAG_PREFIX}deadbeef"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    _checked_load(tmp_path, monkeypatch, reports_dir)
    with pytest.raises(ValueError, match="group tags"):
        summarizer._load_reports(manifest)


def test_verify_manifest_identity_gates(tmp_path, monkeypatch):
    _manifest, _, manifest_path = _prepare(tmp_path, monkeypatch, _summary_rows())
    monkeypatch.setattr(summarizer, "ROOT", tmp_path)

    def load_with(mutate):
        payload = json.loads(manifest_path.read_text())
        mutate(payload)
        path = tmp_path / "mutated-manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(summarizer, "MANIFEST", path)
        return summarizer._verify_manifest()

    with pytest.raises(ValueError, match="dataset id"):
        load_with(lambda m: m.__setitem__("dataset", "other/dataset"))
    with pytest.raises(ValueError, match="revision"):
        load_with(lambda m: m.__setitem__("revision", "0" * 40))
    with pytest.raises(ValueError, match="preparer"):
        load_with(lambda m: m.__setitem__("preparer_sha256", "0" * 64))

    def drift_sidecar(payload):
        entry = payload["exports"]["data/processed/sea-nli-km-normal-test.jsonl"]
        entry["ids_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="sidecar fingerprint"):
        load_with(drift_sidecar)

    def drift_export(payload):
        payload["exports"]["data/processed/sea-nli-km-normal-test.jsonl"]["sha256"] = (
            "0" * 64
        )

    with pytest.raises(ValueError, match="export fingerprint"):
        load_with(drift_export)


def _diag_case(expected, choice):
    return {
        "tags": [f"{evaluator.TAG_PREFIX}g"],
        "expected": expected,
        "correct": expected == choice,
        "confidence": 0.9,
        "answer": {"choice": choice},
    }


def _diagnostic_payloads(choices_by_model, gold):
    payloads = {}
    for model in evaluator.MODELS:
        choices = choices_by_model[model]
        cases = [_diag_case(g, c) for g, c in zip(gold, choices, strict=True)]
        payloads[("km-normal", model)] = {"cases": cases}
        payloads[("km-hard", model)] = {"cases": []}
    return payloads


def test_cohen_kappa_matches_hand_computed_asymmetric_fixture():
    expected = [
        "entailment",
        "entailment",
        "entailment",
        "neutral",
        "neutral",
        "neutral",
        "contradiction",
        "contradiction",
        "contradiction",
        "contradiction",
    ]
    # Model over-predicts entailment and never predicts the third class here.
    # po = 0.6, pe = 0.32, so kappa = 0.28 / 0.68. An implementation that drops
    # the chance term returns 0.6 and fails.
    predictions = [
        "entailment",
        "entailment",
        "neutral",
        "neutral",
        "neutral",
        "entailment",
        "entailment",
        "entailment",
        "contradiction",
        "contradiction",
    ]
    assert summarizer._cohen_kappa(expected, predictions) == pytest.approx(
        0.4117647058823529
    )
    # Anti-correlated predictions give kappa = -1, not 0.
    assert summarizer._cohen_kappa(
        ["entailment", "entailment", "neutral", "neutral"],
        ["neutral", "neutral", "entailment", "entailment"],
    ) == pytest.approx(-1.0)


def test_mcnemar_exact_two_sided_pins_asymmetric_and_boundary_cases():
    result = summarizer._mcnemar_exact_two_sided(
        [True, True, True, True, True, False],
        [False, False, False, False, False, True],
    )
    # One row helped, five hurt: 2 * (C(6,0)+C(6,1)) / 2**6 = 0.21875.
    assert result == {
        "base_wrong_model_right": 1,
        "base_right_model_wrong": 5,
        "discordant_rows": 6,
        "net_flipped_rows": -4,
        "mcnemar_exact_two_sided_p": pytest.approx(0.21875),
    }
    assert result["mcnemar_exact_two_sided_p"] != pytest.approx(0.5)
    symmetric = summarizer._mcnemar_exact_two_sided(
        [True, True, False, False], [False, False, True, True]
    )
    assert symmetric["mcnemar_exact_two_sided_p"] == 1.0
    assert symmetric["net_flipped_rows"] == 0
    identical = summarizer._mcnemar_exact_two_sided(
        [True, False, True], [True, False, True]
    )
    assert identical["discordant_rows"] == 0
    assert identical["mcnemar_exact_two_sided_p"] == 1.0


def test_label_agreement_counts_disagreements_not_just_accuracy():
    reference = [
        {"answer": {"choice": "entailment"}},
        {"answer": {"choice": "neutral"}},
        {"answer": {"choice": "contradiction"}},
        {"answer": {"choice": "entailment"}},
    ]
    candidate = [
        {"answer": {"choice": "entailment"}},
        {"answer": {"choice": "neutral"}},
        {"answer": {"choice": "neutral"}},
        {"answer": {"choice": "contradiction"}},
    ]
    assert summarizer._label_agreement(reference, candidate) == {
        "rows": 4,
        "agreeing_rows": 2,
        "agreement_fraction": pytest.approx(0.5),
    }


def test_constant_majority_reference_including_two_label_tie():
    reference = summarizer._constant_majority_reference(
        [
            "entailment",
            "entailment",
            "neutral",
            "neutral",
            "neutral",
            "contradiction",
        ]
    )
    assert reference["accuracy"] == pytest.approx(3 / 6)
    assert reference["labels"] == ["neutral"]
    assert reference["label_accuracy"]["contradiction"] == pytest.approx(1 / 6)
    tie = summarizer._constant_majority_reference(
        ["entailment", "entailment", "neutral", "neutral", "contradiction"]
    )
    assert tie["accuracy"] == pytest.approx(2 / 5)
    assert tie["labels"] == ["entailment", "neutral"]


def test_prior_collapse_diagnostics_pins_dominant_fraction_and_margins():
    gold = ["entailment", "neutral", "contradiction", "neutral"]
    payloads = _diagnostic_payloads(
        {
            "multilingual": ["entailment", "entailment", "entailment", "neutral"],
            "mt-s13": ["neutral", "neutral", "neutral", "entailment"],
            "mt-s14": ["entailment", "neutral", "entailment", "neutral"],
            "full-e2-8bit": [
                "contradiction",
                "neutral",
                "contradiction",
                "neutral",
            ],
        },
        gold,
    )
    result = summarizer._prior_collapse_diagnostics(payloads)
    assert result["status"] == "post-scoring, post-hoc diagnostic"
    base = result["per_arm"]["multilingual"]
    assert base["dominant_label"] == "entailment"
    assert base["dominant_fraction"] == pytest.approx(0.75)
    assert base["margin_over_constant_majority_pp"] == pytest.approx(0.0)
    s13 = result["per_arm"]["mt-s13"]
    assert s13["dominant_label"] == "neutral"
    assert s13["dominant_fraction"] == pytest.approx(0.75)
    # Tie on 2/2 breaks to the earlier label, not to whichever was inserted last.
    s14 = result["per_arm"]["mt-s14"]
    assert s14["predicted_label_distribution"] == {
        "entailment": 2,
        "neutral": 2,
        "contradiction": 0,
    }
    assert s14["dominant_label"] == "entailment"
    assert s14["dominant_fraction"] == pytest.approx(0.5)
    assert result["constant_majority_reference"]["labels"] == ["neutral"]
    assert result["constant_majority_reference"]["accuracy"] == pytest.approx(0.5)
    assert result["label_agreement"]["multilingual__mt-s13"] == {
        "rows": 4,
        "agreeing_rows": 0,
        "agreement_fraction": pytest.approx(0.0),
    }
    assert result["net_flips_vs_base"]["mt-s13"]["discordant_rows"] == 3
    assert result["net_flips_vs_base"]["mt-s13"]["net_flipped_rows"] == -1
    counterfactual = result["collapse_threshold_counterfactual"]
    assert counterfactual["locked_threshold"] == 0.90
    # s13 collapses at 0.75: past the stricter 2/3 bar but under the locked 0.90.
    assert counterfactual["locked_guard_triggered"] is False
    assert counterfactual["arms_invalidated_at_stricter_threshold"] == ["mt-s13"]


def test_signed_pp_never_renders_exact_zero_as_positive():
    assert summarizer._signed(0.0) == "0.00"
    assert summarizer._signed(-0.0) == "0.00"
    assert summarizer._signed(13.043478260869565) == "+13.04"
    assert summarizer._signed(-4.477611940298507) == "-4.48"


def test_summarize_emits_post_hoc_diagnostics_block(tmp_path, monkeypatch):
    manifest, out_dir, manifest_path = _prepare(tmp_path, monkeypatch, _summary_rows())
    reports_dir = _write_fixture_reports(tmp_path, manifest, out_dir)
    protocol = _wire_summary(tmp_path, monkeypatch, manifest_path, reports_dir)
    result = summarizer.summarize()
    diagnostics = result["prior_collapse_diagnostics"]
    assert diagnostics["status"] == "post-scoring, post-hoc diagnostic"
    assert set(diagnostics["per_arm"]) == set(evaluator.MODELS)
    assert "changes no threshold" in diagnostics["note"]
    markdown = (protocol.parent / "results.md").read_text()
    assert "Post-scoring prior-collapse diagnostics (post-hoc)" in markdown
    assert "Cohen κ vs gold" in markdown
    assert "+0.00" not in markdown
