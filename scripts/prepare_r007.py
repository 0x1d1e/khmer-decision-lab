"""Prepare exploratory ASAG and derived MKQA choices, never train or predict.

Requires existing R003/R006/base checkpoints and verified R006 adaptation files.
Raw data and row-level provenance stay in git-ignored data/. curl fetches MKQA
because Python's direct HTTPS fetch failed in the initial acquisition audit.
"""

import csv
import gzip
import hashlib
import json
import math
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from huggingface_hub import hf_hub_download
from laya.common import build_sequence, encode_text, render_options
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
ASAG_REV = "5fadaef3422c5784458d662394b5d891e7ebb659"
MKQA_REV = "f964fe1bac4d580cee3d4928572df1f676f8e656"
ASAG_SHA = "4e1087eb397ab8ce2c211d32e4ff86dd9be0103658eb3708f7f160ded0b34065"
MKQA_SHA = "f77ac6bf6306e98816c5c047880db71a3c67d2f1491b169896d63e9bb14ef3d3"
BASE_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
# Audit preview of the superseded km-name rendering. Quarantine entire final groups.
MKQA_EXPOSED = {
    "mkqa:-4767536963343233422",
    "mkqa:7818112641371223126",
    "mkqa:7295881644912577718",
    "mkqa:-6912591881241757083",
    "mkqa:-2310612763144978658",
    "mkqa:8663315281958969051",
    "mkqa:-5408188992455330093",
    "mkqa:2788150498848569248",
}
MODELS = {
    "control-base-maxlen512": "control-base-maxlen512",
    "full-e2-8bit": "R003-full-e2-8bit",
    "mt-s13": "R006-mt-s13",
    "mt-s14": "R006-mt-s14",
}
ASAG_INSTRUCTIONS = (
    "Estimate the teacher's credit for the student answer relative to the reference. "
    "Choose the nearest of 0%, 25%, 50%, 75%, or 100% credit; "
    "break exact ties toward an even level."
)
ASAG_CRITERIA = {str(i): f"{25 * i}% credit" for i in range(5)}
MKQA_INSTRUCTIONS = (
    "Choose the entity that answers the Khmer question. Exactly one option is a "
    "benchmark-accepted answer. Use the question alone, without external retrieval."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text)
        if unicodedata.category(c)[0] in "LNM"
    ).casefold()


def identifier(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def row_rng(source_id, purpose):
    return random.Random(int(identifier(f"{SEED}:{source_id}:{purpose}"), 16))


def grade(score, maximum):
    score, maximum = Decimal(score), Decimal(maximum)
    if (
        not score.is_finite()
        or not maximum.is_finite()
        or not 0 <= score <= maximum
        or maximum <= 0
    ):
        raise ValueError("Invalid teacher score or maximum")
    return str(round(4 * score / maximum))


def asag_records(rows):
    records, excluded = [], []
    for i, row in enumerate(rows, 1):
        source_id = f"asag:{i}"
        state = {
            k: row[c].strip()
            for k, c in (
                ("question", "Question"),
                ("reference", "Reference"),
                ("answer", "Answer"),
            )
        }
        try:
            label = grade(row["Student Score"], row["Max Score"])
        except (ValueError, ArithmeticError):
            excluded.append({"source_id": source_id, "reason": "invalid_score"})
            continue
        if not state["question"] or not state["reference"]:
            excluded.append(
                {"source_id": source_id, "reason": "missing_question_or_reference"}
            )
            continue
        keys = ["question:" + compact(state["question"])]
        keys += [
            k + ":" + compact(state[k])
            for k in ("reference", "answer")
            if len(compact(state[k])) >= 60
        ]
        records.append(
            {
                "source_id": source_id,
                "state": state,
                "label": label,
                "keys": keys,
                "exposed": row["Subject"].casefold() == "biology"
                and row["QuestionID"] in {"Q7", "Q8", "Q9", "Q10"},
                "student_hash": identifier(row["SchoolID"] + ":" + row["StudentID"]),
                "criteria": ASAG_CRITERIA,
            }
        )
    return records, excluded


def entity_catalog(rows, tok):
    names, aliases = {}, defaultdict(set)
    for row in rows:
        for language in ("en", "km"):
            for answer in row["answers"].get(language, []):
                entity, text = answer.get("entity"), (answer.get("text") or "").strip()
                if (
                    answer["type"] != "entity"
                    or not entity
                    or not re.fullmatch(r"Q\d+", entity)
                ):
                    continue
                aliases[entity].update(
                    compact(t) for t in [text, *(answer.get("aliases") or [])] if t
                )
                # One uniform name vocabulary, not mixed km/Thai names or language fallbacks.
                # Laya caps each rendered option at 48 tokens regardless of head_max_len.
                if (
                    language == "en"
                    and entity not in names
                    and text
                    and len(
                        encode_text(tok, " A: " + text, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    <= 48
                ):
                    names[entity] = text
    ambiguous = Counter(compact(text) for text in names.values())
    names = {qid: text for qid, text in names.items() if ambiguous[compact(text)] == 1}
    return names, aliases


def entity_choices(source_id, entity, names, aliases):
    forbidden = aliases[entity] | {compact(names[entity])}
    pool = sorted(
        qid
        for qid, text in names.items()
        if qid != entity and compact(text) not in forbidden
    )
    if len(pool) < 3:
        raise ValueError("Fewer than three unambiguous distractors")
    options = [entity, *row_rng(source_id, "distractors").sample(pool, 3)]
    row_rng(source_id, "position").shuffle(options)
    criteria = dict(zip("ABCD", (names[qid] for qid in options)))
    return criteria, "ABCD"[options.index(entity)]


def mkqa_records(rows, tok):
    names, aliases = entity_catalog(rows, tok)
    records, excluded = [], []
    for row in rows:
        source_id = "mkqa:" + str(row["example_id"])
        answers = row["answers"]["km"]
        entities = {a.get("entity") for a in answers}
        if (
            not answers
            or any(
                a["type"] != "entity" or not (a.get("text") or "").strip()
                for a in answers
            )
            or len(entities) != 1
        ):
            excluded.append({"source_id": source_id, "reason": "not_single_entity"})
            continue
        english = row["answers"].get("en", [])
        if (
            not english
            or any(a["type"] != "entity" for a in english)
            or {a.get("entity") for a in english} != entities
        ):
            excluded.append(
                {"source_id": source_id, "reason": "cross_language_entity_mismatch"}
            )
            continue
        entity = next(iter(entities))
        if entity not in names:
            excluded.append(
                {
                    "source_id": source_id,
                    "reason": "no_short_unambiguous_entity_name",
                }
            )
            continue
        criteria, label = entity_choices(source_id, entity, names, aliases)
        query = row["queries"]["km"].strip()
        if not re.search(r"[\u1780-\u17ff]", query):
            excluded.append({"source_id": source_id, "reason": "missing_khmer_query"})
            continue
        records.append(
            {
                "source_id": source_id,
                "state": {"question": query},
                "label": label,
                "keys": [
                    "entity:" + entity,
                    "query:" + compact(query),
                    "english:" + compact(row["query"]),
                ],
                "exposed": source_id in MKQA_EXPOSED,
                "criteria": criteria,
            }
        )
    return records, excluded, len(names)


def group_records(records):
    """Connected components keep shared concepts and normalized duplicates together."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for key in record["keys"]:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, record in enumerate(records):
        members[root(i)].append(record["source_id"])
    groups = {key: identifier("\n".join(sorted(ids))) for key, ids in members.items()}
    for i, record in enumerate(records):
        record["group_id"] = groups[root(i)]
    return records


def audit_groups(records, preserved_sources=()):
    groups = sorted({r["group_id"] for r in records})
    random.Random(SEED).shuffle(groups)
    return set(groups[: math.ceil(0.2 * len(groups))]) | {
        r["group_id"]
        for r in records
        if r["exposed"] or r["source_id"] in preserved_sources
    }


def prior_audit_sources():
    """Carry audit reservations across amendments, including later-excluded rows."""
    path = ROOT / "datasets/manifests/r007-unseen-transfer.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    if manifest["status"] == "locked_exploratory":
        raise ValueError("Cannot overwrite a locked R007 preparation")
    preserved = {}
    for name, target in manifest["targets"].items():
        if "audit_source_ids" in target:
            preserved[name] = set(target["audit_source_ids"])
            continue
        # Migration from the initial manifest, before source IDs were embedded.
        preserved[name] = set()
        for relative, digest in target["export_sha256"]["audit"].items():
            if relative.endswith("-ids.json"):
                source = ROOT / relative
                if sha256(source) != digest:
                    raise ValueError(f"Prior audit fingerprint mismatch: {relative}")
                preserved[name].update(
                    row["source_id"] for row in json.loads(source.read_text())
                )
    return preserved


def additional_mkqa_test_exposure_ids():
    path = ROOT / "research/R007-unseen-transfer/dataset-audit.json"
    audit = json.loads(path.read_text())["additional_test_exposure"]
    source_ids = set(audit["source_ids"])
    if len(source_ids) != audit["rows_emitted"] or not all(
        source_id.startswith("mkqa:") for source_id in source_ids
    ):
        raise ValueError("Invalid MKQA test-exposure source IDs in dataset audit")
    return source_ids


def adaptation_sources():
    manifest = json.loads((ROOT / "datasets/manifests/r006-multitask.json").read_text())
    sources = []
    for name in ("massive-km-train-full", "khmer-news-train", "sib200-km-train"):
        relative = f"data/processed/{name}.jsonl"
        path = ROOT / relative
        digest = sha256(path)
        if digest != manifest["sha256"][relative]:
            raise ValueError(f"Adaptation fingerprint mismatch: {relative}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        texts = {
            compact(" ".join(str(v) for v in row["state"].values())) for row in rows
        }
        texts |= {compact(str(v)) for row in rows for v in row["state"].values()}
        sources.append(
            {
                "file": relative,
                "sha256": digest,
                "rows": len(rows),
                "texts": sorted(t for t in texts if len(t) >= 60),
            }
        )
    return sources


def input_conflicts(records):
    labels = defaultdict(set)
    signatures = {}
    for record in records:
        # Preserve mathematical punctuation when identifying conflicting grades.
        key = tuple(
            " ".join(unicodedata.normalize("NFKC", str(v)).split())
            for v in record["state"].values()
        )
        signatures[record["source_id"]] = key
        labels[key].add(record["label"])
    return {source_id for source_id, key in signatures.items() if len(labels[key]) > 1}


def prepare_target(
    name,
    records,
    excluded,
    instructions,
    max_len,
    tokenizers,
    sources,
    preserved_sources=(),
):
    records = group_records(records)
    reserved = audit_groups(records, preserved_sources)
    overlap = {}
    contaminated = set()
    for source in sources:
        matched = set()
        for record in records:
            fields = [
                compact(v) for v in record["state"].values() if len(compact(v)) >= 60
            ]
            if any(a in b or b in a for a in fields for b in source["texts"]):
                matched.add(record["group_id"])
        contaminated |= matched
        overlap[source["file"]] = {
            "matching_groups": len(matched),
            "matching_source_rows": sum(r["group_id"] in matched for r in records),
        }
    conflicts = input_conflicts(records) if name == "asag" else set()
    exported = {"audit": [], "test": []}
    provenance = {"audit": [], "test": []}
    seen = set()
    token_failures = Counter()
    for record in records:
        reason = None
        if record["group_id"] in contaminated:
            reason = "adaptation_overlap_group"
        elif record["source_id"] in conflicts:
            reason = "conflicting_teacher_labels"
        question = {"t": "choice", "ins": instructions, "crit": record["criteria"]}
        previous_sequence = None
        for tokenizer in tokenizers.values():
            sequence, markers, stats, truncation = build_sequence(
                tokenizer,
                record["state"],
                question,
                max_len=max_len,
                head_max_len=256,
                return_stats=True,
                return_truncation_stats=True,
            )
            instruction_ids = encode_text(
                tokenizer,
                "choice question: " + instructions.replace(tokenizer.mask_token, " "),
                add_special_tokens=False,
            )["input_ids"]
            instructions_capped = (
                not markers or sequence[1 : markers[0] - 1] != instruction_ids
            )
            if previous_sequence is not None and sequence != previous_sequence:
                raise ValueError(
                    f"Checkpoint tokenizers disagree on {record['source_id']}"
                )
            previous_sequence = sequence
            options_capped = any(
                len(
                    encode_text(tokenizer, " " + text, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                > 48
                for text in render_options(question)
            )
            if (
                truncation["truncated"]
                or instructions_capped
                or options_capped
                or len(markers) != len(record["criteria"])
                or stats["options_distinct"] != len(markers)
            ):
                token_failures[record["source_id"]] += 1
                reason = reason or "incomplete_input_or_options"
        row = {
            "state": record["state"],
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": record["criteria"],
                }
            },
            "expected": {"label": record["label"]},
            "language": "km",
        }
        signature = json.dumps(row, ensure_ascii=False)
        if not reason and signature in seen:
            reason = "duplicate_input_and_label"
        seen.add(signature)
        if reason:
            excluded.append(
                {
                    "source_id": record["source_id"],
                    "group_id": record["group_id"],
                    "reason": reason,
                }
            )
            continue
        split = "audit" if record["group_id"] in reserved else "test"
        exported[split].append(row)
        provenance[split].append(
            {
                k: record[k]
                for k in ("source_id", "group_id", "student_hash")
                if k in record
            }
        )
    paths, counts = {}, {}
    for split in ("audit", "test"):
        if not exported[split]:
            raise ValueError(f"{name}: empty {split} partition")
        prefix = f"data/processed/r007-{name}-{split}"
        paths[split] = {}
        for suffix, content in (
            (
                ".jsonl",
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in exported[split]
                ),
            ),
            ("-ids.json", json.dumps(provenance[split], indent=2) + "\n"),
        ):
            path = ROOT / (prefix + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            paths[split][str(path.relative_to(ROOT))] = sha256(path)
        sizes = Counter(r["group_id"] for r in provenance[split])
        effective_groups = len(exported[split]) ** 2 / sum(
            n * n for n in sizes.values()
        )
        accuracy_bound = math.sqrt(math.log(40) / (2 * effective_groups))
        counts[split] = {
            "precision_assumption": "Independent groups; not established, especially with shared ASAG students. Bounds are design diagnostics, not observed model CIs.",
            "size_weighted_effective_groups": effective_groups,
            "accuracy_95_hoeffding_half_width": accuracy_bound,
            "paired_difference_95_hoeffding_half_width": 2 * accuracy_bound,
            "rows": len(exported[split]),
            "groups": len({r["group_id"] for r in provenance[split]}),
            "label_counts": dict(
                sorted(
                    Counter(row["expected"]["label"] for row in exported[split]).items()
                )
            ),
        }
    exclusion_path = ROOT / f"data/processed/r007-{name}-excluded.json"
    exclusion_path.write_text(json.dumps(excluded, indent=2) + "\n", encoding="utf-8")
    audit_set = {r["group_id"] for r in provenance["audit"]}
    test_set = {r["group_id"] for r in provenance["test"]}
    if audit_set & test_set:
        raise ValueError("Audit/test group intersection")
    student_sets = [
        {r["student_hash"] for r in provenance[s] if "student_hash" in r}
        for s in ("audit", "test")
    ]
    return {
        "max_len": max_len,
        "head_max_len": 256,
        "retained_inputs_and_options_complete": True,
        "checkpoint_token_sequences_equal": True,
        "instructions": instructions,
        "eligible_source_groups": len({r["group_id"] for r in records}),
        "reserved_audit_groups": len(reserved),
        "audit_source_ids": sorted(
            set(preserved_sources)
            | {r["source_id"] for r in records if r["group_id"] in reserved}
        ),
        "forced_exposed_groups": len({r["group_id"] for r in records if r["exposed"]}),
        "partitions": counts,
        "group_intersection": len(audit_set & test_set),
        "shared_student_hashes": len(student_sets[0] & student_sets[1]),
        "exclusions": dict(sorted(Counter(r["reason"] for r in excluded).items())),
        "token_failure_rows_before_other_exclusions": len(token_failures),
        "export_sha256": paths,
        "exclusion_file": str(exclusion_path.relative_to(ROOT)),
        "exclusion_sha256": sha256(exclusion_path),
        "overlap": overlap,
    }


def checkpoint_tokenizers():
    tokenizers, fingerprints = {}, {}
    for model, folder in MODELS.items():
        path = ROOT / "artifacts/checkpoints" / folder
        tokenizers[model] = AutoTokenizer.from_pretrained(
            path / "tokenizer", local_files_only=True
        )
        fingerprints[model] = {
            "directory": str(path.relative_to(ROOT)),
            "sha256": {
                file: sha256(path / file)
                for file in (
                    "model.safetensors",
                    "rl_agent_config.json",
                    "encoder/config.json",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                )
            },
            "choice_temperature": json.loads(
                (path / "rl_agent_config.json").read_text()
            )["temperature"][0],
        }
    return tokenizers, fingerprints


def download_sources():
    asag = Path(
        hf_hub_download(
            "phorknorak/khmer-asag-benchmark",
            "dataset_no_10c_biology.csv",
            repo_type="dataset",
            revision=ASAG_REV,
            token=False,
        )
    )
    mkqa = ROOT / f"data/raw/r007/mkqa-{MKQA_REV}.jsonl.gz"
    if not mkqa.exists():
        mkqa.parent.mkdir(parents=True, exist_ok=True)
        temporary = mkqa.with_suffix(".gz.part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                f"https://raw.githubusercontent.com/apple/ml-mkqa/{MKQA_REV}/dataset/mkqa.jsonl.gz",
                "--output",
                str(temporary),
            ],
            check=True,
        )
        if sha256(temporary) != MKQA_SHA:
            raise ValueError("MKQA source fingerprint mismatch")
        temporary.replace(mkqa)
    for path, expected in ((asag, ASAG_SHA), (mkqa, MKQA_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"Source fingerprint mismatch: {path.name}")
    with asag.open(encoding="utf-8-sig", newline="") as handle:
        asag_rows = list(csv.DictReader(handle))
    with gzip.open(mkqa, "rt", encoding="utf-8") as handle:
        mkqa_rows = [json.loads(line) for line in handle]
    if len({str(r["example_id"]) for r in mkqa_rows}) != len(mkqa_rows):
        raise ValueError("Duplicate MKQA source IDs")
    return asag_rows, mkqa_rows


def main():
    preserved = prior_audit_sources()
    asag_rows, mkqa_rows = download_sources()
    tokenizers, checkpoints = checkpoint_tokenizers()
    sources = adaptation_sources()
    asag, asag_excluded = asag_records(asag_rows)
    mkqa, mkqa_excluded, vocabulary = mkqa_records(mkqa_rows, tokenizers["mt-s13"])
    mkqa_test_exposed = additional_mkqa_test_exposure_ids()
    if not mkqa_test_exposed <= {row["source_id"] for row in mkqa}:
        raise ValueError("An exposed MKQA test source is no longer eligible")
    targets = {
        "asag": prepare_target(
            "asag",
            asag,
            asag_excluded,
            ASAG_INSTRUCTIONS,
            2048,
            tokenizers,
            sources,
            preserved.get("asag", set()),
        ),
        "mkqa-entity-choice": prepare_target(
            "mkqa-entity-choice",
            mkqa,
            mkqa_excluded,
            MKQA_INSTRUCTIONS,
            512,
            tokenizers,
            sources,
            preserved.get("mkqa-entity-choice", set()) | mkqa_test_exposed,
        ),
    }
    manifest = {
        "status": "prepared_exploratory_pending_protocol_lock",
        "primary_target": "mkqa-entity-choice",
        "secondary_target": "asag",
        "target_predictions_generated": False,
        "seed": SEED,
        "preparer_sha256": sha256(Path(__file__)),
        "base_revision": BASE_REV,
        "audit_rule": "Sort group IDs, Random(seed).shuffle, reserve ceil(20%); union exposed groups and all prior audit source-ID components. Persist reservations, including subsequently excluded source IDs, across amendments.",
        "previously_inspected_mkqa_source_ids": sorted(MKQA_EXPOSED),
        "additional_mkqa_test_exposure_source_ids": sorted(mkqa_test_exposed),
        "mkqa_choice_rule": "Single en/km-agreed entity QID; first uncapped name from English annotations in source order; remove duplicate normalized names. Whole-release vocabulary, excluding gold QID and known en/km aliases; three seeded random negatives and independently shuffled display position.",
        "group_rule": "Connected components of compact-normalized questions; ASAG also identical reference/answer >=60 characters; MKQA also target QID and English query.",
        "overlap_rule": "NFKC letters/numbers/marks, casefold; bidirectional full-field substring >=60 characters against individual and joined adaptation state fields, including calibration rows. Exclude matching groups.",
        "checkpoint_fingerprints": checkpoints,
        "adaptation_sources": [
            {k: v for k, v in source.items() if k != "texts"} for source in sources
        ],
        "sources": {
            "asag": {
                "dataset": "phorknorak/khmer-asag-benchmark",
                "revision": ASAG_REV,
                "file": "dataset_no_10c_biology.csv",
                "sha256": ASAG_SHA,
                "rows": len(asag_rows),
                "published_split": "train",
                "license": "CC-BY-NC-4.0",
            },
            "mkqa": {
                "repository": "apple/ml-mkqa",
                "revision": MKQA_REV,
                "file": "dataset/mkqa.jsonl.gz",
                "sha256": MKQA_SHA,
                "rows": len(mkqa_rows),
                "published_split": "single release",
                "license": "CC-BY-SA-3.0",
                "entity_vocabulary": vocabulary,
            },
        },
        "targets": targets,
    }
    path = ROOT / "datasets/manifests/r007-unseen-transfer.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    "partitions": target["partitions"],
                    "exclusions": target["exclusions"],
                    "overlap": target["overlap"],
                }
                for name, target in targets.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
