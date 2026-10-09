"""Benchmark construction invariants. No model predictions or target scoring."""

import importlib.util
import json
from pathlib import Path

import pytest
from laya.evals import Dataset

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_r007", ROOT / "scripts/prepare_r007.py"
)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


@pytest.mark.parametrize(
    "score,expected",
    [
        ("0", "0"),
        ("0.5", "0"),
        ("1", "1"),
        ("1.5", "2"),
        ("2.5", "2"),
        ("3.5", "4"),
        ("4", "4"),
    ],
)
def test_teacher_bins_use_half_even_rounding(score, expected):
    assert prep.grade(score, "4") == expected


@pytest.mark.parametrize(
    "score,maximum", [("5", "4"), ("-1", "4"), ("0", "0"), ("NaN", "4")]
)
def test_invalid_teacher_scores_rejected(score, maximum):
    with pytest.raises(ValueError):
        prep.grade(score, maximum)


def test_entity_choices_reproducible_without_gold_position_shortcut():
    names = {
        "Q1": "Alice",
        "Q2": "ALICIA",
        "Q3": "Bob",
        "Q4": "Cora",
        "Q5": "Dan",
        "Q6": "Eve",
    }
    aliases = {"Q1": {"alice", "alicia"}}
    positions = set()
    for i in range(100):
        result = prep.entity_choices(f"mkqa:{i}", "Q1", names, aliases)
        assert result == prep.entity_choices(f"mkqa:{i}", "Q1", names, aliases)
        criteria, gold = result
        assert len(criteria) == len(set(criteria.values())) == 4
        assert criteria[gold] == "Alice"
        assert "ALICIA" not in criteria.values()
        positions.add(gold)
    assert positions == set("ABCD")


def test_mkqa_missing_answer_text_is_excluded_not_crashed():
    # Reproduces the initial E2E failure on nullable MKQA answer text.
    rows = [
        {
            "example_id": 1,
            "answers": {
                "km": [
                    {"type": "entity", "entity": "Q1", "text": None, "aliases": None},
                ]
            },
            "queries": {"km": "unused"},
            "query": "unused",
        }
    ]
    records, excluded, _ = prep.mkqa_records(rows, object())
    assert records == []
    assert excluded == [{"source_id": "mkqa:1", "reason": "not_single_entity"}]


def test_shared_concepts_and_queries_form_transitive_groups():
    rows = [
        {"source_id": "1", "keys": ["entity:Q1", "query:first"], "exposed": True},
        {"source_id": "2", "keys": ["entity:Q1", "query:second"], "exposed": False},
        {"source_id": "3", "keys": ["entity:Q2", "query:second"], "exposed": False},
        {"source_id": "4", "keys": ["entity:Q3", "query:third"], "exposed": False},
    ]
    grouped = prep.group_records(rows)
    assert len({r["group_id"] for r in grouped[:3]}) == 1
    assert grouped[0]["group_id"] != grouped[3]["group_id"]
    assert grouped[0]["group_id"] in prep.audit_groups(grouped)
    reordered = prep.group_records(list(reversed(rows)))
    assert {r["source_id"]: r["group_id"] for r in reordered} == {
        r["source_id"]: r["group_id"] for r in grouped
    }
    assert prep.audit_groups(grouped) == prep.audit_groups(reordered)


def test_component_amendment_preserves_inspected_audit_sources():
    rows = [
        {"source_id": str(i), "keys": [f"k:{i}"], "exposed": False} for i in range(20)
    ]
    prep.group_records(rows)
    initial = prep.audit_groups(rows)
    preserved = {r["source_id"] for r in rows if r["group_id"] in initial}
    assert "0" in preserved
    rows.append({"source_id": "new", "keys": ["k:0"], "exposed": False})
    prep.group_records(rows)
    # Reproduces migration under the old seed-only rule, despite fixed seed.
    vulnerable = prep.audit_groups(rows)
    assert rows[0]["group_id"] not in vulnerable
    reserved = prep.audit_groups(rows, preserved)
    assert all(r["group_id"] in reserved for r in rows if r["source_id"] in preserved)
    assert rows[-1]["group_id"] in reserved


def test_disclosed_mkqa_test_exposure_ids_are_explicit_reservations():
    exposed = prep.additional_mkqa_test_exposure_ids()
    assert len(exposed) == 87
    audit = json.loads(
        (ROOT / "research/R007-unseen-transfer/dataset-audit.json").read_text()
    )
    assert exposed == set(audit["additional_test_exposure"]["source_ids"])


def test_prior_audit_sources_survive_exclusions_and_block_locked_overwrite(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    path = tmp_path / "datasets/manifests/r007-unseen-transfer.json"
    path.parent.mkdir(parents=True)
    manifest = {
        "status": "prepared_exploratory_pending_protocol_lock",
        "targets": {"asag": {"audit_source_ids": ["asag:1", "asag:excluded"]}},
    }
    path.write_text(json.dumps(manifest))
    assert prep.prior_audit_sources() == {"asag": {"asag:1", "asag:excluded"}}
    manifest["status"] = "locked_exploratory"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Cannot overwrite a locked"):
        prep.prior_audit_sources()


def asag_row(i, score="4"):
    return {
        "Question": f"Question {i}",
        "Reference": f"Reference {i}",
        "Answer": "Student response",
        "QuestionID": f"Q{i}",
        "Subject": "Math",
        "Student Score": score,
        "Max Score": "4",
        "SchoolID": "PRIVATE-SCHOOL",
        "ClassID": "PRIVATE-CLASS",
        "StudentID": f"PRIVATE-STUDENT-{i}",
        "Year": "2024",
    }


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = ROOT / "artifacts/checkpoints/R006-mt-s13/tokenizer"
    if not path.is_dir():
        pytest.skip("Existing local tokenizer required for production-renderer check")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_mkqa_uniform_english_names_and_cross_language_entity_agreement(tokenizer):
    rows = []
    for i, name in enumerate(("Alice", "Bob", "Cora", "Dan", "Eve"), 1):
        rows.append(
            {
                "example_id": i,
                "query": f"Question {i}",
                "queries": {"km": f"សំណួរ {i}"},
                "answers": {
                    "en": [{"type": "entity", "entity": f"Q{i}", "text": name}],
                    "km": [{"type": "entity", "entity": f"Q{i}", "text": "ชื่อภาษาไทย"}],
                },
            }
        )
    rows.append(
        {
            "example_id": 6,
            "query": "Ambiguous",
            "queries": {"km": "សំណួរ"},
            "answers": {
                "en": [
                    {"type": "entity", "entity": "Q1", "text": "Alice"},
                    {"type": "entity", "entity": "Q2", "text": "Bob"},
                ],
                "km": [{"type": "entity", "entity": "Q1", "text": "ឈ្មោះ"}],
            },
        }
    )
    records, excluded, _ = prep.mkqa_records(rows, tokenizer)
    assert len(records) == 5
    assert excluded == [
        {"source_id": "mkqa:6", "reason": "cross_language_entity_mismatch"}
    ]
    for record, name in zip(records, ("Alice", "Bob", "Cora", "Dan", "Eve")):
        assert record["criteria"][record["label"]] == name
        assert set(record["criteria"].values()) <= {
            "Alice",
            "Bob",
            "Cora",
            "Dan",
            "Eve",
        }
        assert set(record["state"]) == {"question"}


def test_asag_export_complete_inputs_disjoint_groups_no_score_or_identity_leak(
    tmp_path, monkeypatch, tokenizer
):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    rows = [asag_row(i) for i in range(12)]
    rows[0]["Subject"], rows[0]["QuestionID"] = "Biology", "Q7"
    rows.extend([asag_row(1, "0"), asag_row(2)])
    oversized = asag_row(20)
    oversized["Answer"] = "word " * 3000
    rows.append(oversized)
    records, excluded = prep.asag_records(rows)
    target = prep.prepare_target(
        "asag", records, excluded, prep.ASAG_INSTRUCTIONS, 512, {"local": tokenizer}, []
    )
    assert target["group_intersection"] == 0
    assert target["forced_exposed_groups"] == 1
    assert target["exclusions"] == {
        "conflicting_teacher_labels": 2,
        "duplicate_input_and_label": 1,
        "incomplete_input_or_options": 1,
    }
    memberships = {}
    for split in ("audit", "test"):
        data = tmp_path / f"data/processed/r007-asag-{split}.jsonl"
        dataset = Dataset.from_jsonl(str(data))
        assert len(dataset) == target["partitions"][split]["rows"]
        assert "PRIVATE" not in data.read_text()
        for example in dataset.examples:
            assert set(example.state) == {"question", "reference", "answer"}
            assert set(example.questions["label"]["criteria"]) == set("01234")
        ids = json.loads(
            (tmp_path / f"data/processed/r007-asag-{split}-ids.json").read_text()
        )
        memberships.update({row["source_id"]: split for row in ids})
    assert memberships["asag:1"] == "audit"
    assert "asag:2" not in memberships  # Both conflicting grades excluded.
    assert "asag:13" not in memberships
    assert "asag:14" not in memberships  # Duplicate input/label excluded.
    assert "asag:15" not in memberships  # Incomplete evidence excluded.
