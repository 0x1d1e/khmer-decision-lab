"""Prepare the R014 SEA-NLI Khmer NLI transfer exports.

Inference-only study. Downloads the gated SEA-NLI test parquet files (the
`normal` and `hard` configs), keeps only the Khmer-culture rows, and writes
Laya choice examples plus a row-index -> group_id sidecar per file. The three
way NLI task is asked as one `choice` question whose three criteria have empty
descriptions, so the rendered option strings are the bare labels.

Two matched arms are exported for the Khmer rows: a `native` arm whose state
text is the dataset's `premise_native` / `hypothesis_native`, and an `english`
control arm whose state text is `premise_english` / `hypothesis_english` (GPT
translation). The two arms share row identity, grouping, and gold; only the
state text differs. The English arm is a descriptive-only secondary endpoint.

Khmer selection is by `culture == "cambodian"` (casefolded). A Khmer-script
check over U+1780-U+17FF is a *validation* on the selected native fields, not
a selection mechanism: a selected row whose native premise or hypothesis
carries no Khmer script raises, naming its `entry_id`.

Every selection, validation, de-duplication, grouping, and schema decision is
recorded in `datasets/manifests/sea-nli-km.json`, which this script writes at
run time like every other study manifest. It holds only ids, hashes, counts,
and rules - no gated text - so it **must be committed together with the
results**. Results reported without a committed manifest are not reportable.
The gated source data itself is not committed.

For the two Khmer exports the label set is closed: a row whose `true_label` is
present but not mappable to {entailment, neutral, contradiction} raises instead
of shrinking the slice silently. Exclusions by reason are recorded per export
and aggregated in the manifest, and printed on every run.

Requires an authenticated Hugging Face account that has accepted the SEA-NLI
terms:

    hf auth login          # or: export HF_TOKEN=hf_...
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TypedDict

from datasets import Dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
PREPARER = Path(__file__).resolve()

DATASET = "aisingapore/SEA-NLI"
REVISION = "f3a73ccb4213ff995ef2bf8bcb903eaf53195713"
CONFIG_ORDER = ("normal", "hard")
CONFIG_FILES = {
    "normal": "normal/test-00000-of-00001.parquet",
    "hard": "hard/test-00000-of-00001.parquet",
}

LABELS = ("entailment", "neutral", "contradiction")
QID = "relationship"
INSTRUCTIONS = (
    "What is the relationship between the premise in `premise` and the "
    "hypothesis in `hypothesis`?"
)
CRITERIA = {label: "" for label in LABELS}

# Model-visible state keys are fixed. The source column each key reads from
# depends on the arm; `native` is the dataset's Khmer text, `english` is the
# GPT translation of the same row.
STATE_FIELDS = ("premise", "hypothesis")
NATIVE_FIELDS = {"premise": "premise_native", "hypothesis": "hypothesis_native"}
ENGLISH_FIELDS = {"premise": "premise_english", "hypothesis": "hypothesis_english"}

QUESTION = {
    "type": "choice",
    "instructions": INSTRUCTIONS,
    "criteria": CRITERIA,
}

# Only these source columns may reach the emitted examples. `culture` is used
# for selection only and never reaches a model input. Everything else,
# including is_misclassified, step_misclassified, improved_* and
# entailment_type, is dropped.
ALLOWLIST = (
    "premise_native",
    "hypothesis_native",
    "premise_english",
    "hypothesis_english",
    "true_label",
)
CULTURE_FIELD = "culture"
KHMER_CULTURE = "cambodian"
ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")
KHMER_START, KHMER_END = 0x1780, 0x17FF
KHMER_SCRIPT_RANGE = "U+1780-U+17FF"

OUT = ROOT / "data/processed"
MANIFEST = ROOT / "datasets/manifests/sea-nli-km.json"


# export key -> stem, configs, arm fields, language, strictness, kind.
class ExportSpec(TypedDict):
    stem: str
    configs: tuple[str, ...]
    fields: dict[str, str]
    arm: str
    language: str | None
    strict: bool
    kind: str


EXPORTS: dict[str, ExportSpec] = {
    "km-normal": {
        "stem": "sea-nli-km-normal-test",
        "configs": ("normal",),
        "fields": NATIVE_FIELDS,
        "arm": "native",
        "language": "km",
        "strict": True,
        "kind": "khmer",
    },
    "km-hard": {
        "stem": "sea-nli-km-hard-test",
        "configs": ("hard",),
        "fields": NATIVE_FIELDS,
        "arm": "native",
        "language": "km",
        "strict": True,
        "kind": "khmer",
    },
    "en-normal": {
        "stem": "sea-nli-en-normal-test",
        "configs": ("normal",),
        "fields": ENGLISH_FIELDS,
        "arm": "english",
        "language": "en",
        "strict": True,
        "kind": "khmer",
    },
    "en-hard": {
        "stem": "sea-nli-en-hard-test",
        "configs": ("hard",),
        "fields": ENGLISH_FIELDS,
        "arm": "english",
        "language": "en",
        "strict": True,
        "kind": "khmer",
    },
    "other": {
        "stem": "sea-nli-other-test",
        "configs": ("normal", "hard"),
        "fields": NATIVE_FIELDS,
        "arm": "native",
        "language": None,
        "strict": False,
        "kind": "non_khmer",
    },
}
# native export key -> matching english export key.
ARM_PAIRS = (("km-normal", "en-normal"), ("km-hard", "en-hard"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_key(text):
    """Grouping and duplicate key only. Never applied to model-visible text."""
    value = unicodedata.normalize("NFKC", text)
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    for char in ZERO_WIDTH:
        value = value.replace(char, "")
    return value.strip()


def group_id(premise):
    return hashlib.sha256(normalize_key(premise).encode("utf-8")).hexdigest()


def dedup_key(premise, hypothesis, gold):
    payload = f"{normalize_key(premise)}\0{normalize_key(hypothesis)}\0{gold}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_khmer_culture(row):
    value = row.get(CULTURE_FIELD)
    return isinstance(value, str) and value.casefold() == KHMER_CULTURE


def has_khmer_script(value):
    return isinstance(value, str) and any(
        KHMER_START <= ord(char) <= KHMER_END for char in value
    )


def validate_khmer_script(selected):
    """Every selected row's native fields must carry Khmer script.

    Raises naming the offending `entry_id`. Returns the recorded outcome.
    """
    checked = 0
    for config in CONFIG_ORDER:
        for index, row in selected[config]:
            checked += 1
            for column in ("premise_native", "hypothesis_native"):
                if not has_khmer_script(row.get(column)):
                    entry_id = row.get("entry_id")
                    raise ValueError(
                        "R014 selected row is not Khmer script in "
                        f"{column}: entry_id={entry_id!r} "
                        f"({config}[{index}])"
                    )
    return {
        "rule": "premise_native and hypothesis_native contain a codepoint in "
        f"{KHMER_SCRIPT_RANGE}",
        "fields": ["premise_native", "hypothesis_native"],
        "range": KHMER_SCRIPT_RANGE,
        "checked_rows": checked,
        "passed": True,
    }


def resolve_gold(value, label_names):
    if isinstance(value, bool):
        raise TypeError("R014 true_label must not be boolean")
    if isinstance(value, str):
        text = value.strip()
        if text in LABELS:
            return text
        if text.casefold() in LABELS:
            return text.casefold()
        raise ValueError(f"R014 true_label is not a known label: {value!r}")
    if isinstance(value, int) and label_names and 0 <= value < len(label_names):
        name = label_names[value]
        if name in LABELS:
            return name
    raise ValueError(f"R014 true_label could not be mapped to a label: {value!r}")


def choice_example(premise, hypothesis, gold, language):
    return {
        "state": {"premise": premise, "hypothesis": hypothesis},
        "language": language,
        "questions": {QID: QUESTION},
        "expected": {QID: gold},
    }


def assert_example_shape(example, language):
    if set(example) != {"state", "language", "questions", "expected"}:
        raise ValueError(f"R014 emitted an unexpected top-level key: {sorted(example)}")
    if set(example["state"]) != set(STATE_FIELDS):
        raise ValueError(
            f"R014 emitted an unexpected state key: {sorted(example['state'])}"
        )
    if example["language"] != language:
        raise ValueError("R014 emitted a mismatched language field")


def _emitted_example(row, language, label_names, strict, fields):
    premise, hypothesis = row.get(fields["premise"]), row.get(fields["hypothesis"])
    if not isinstance(premise, str) or not premise:
        return None, "missing_premise"
    if not isinstance(hypothesis, str) or not hypothesis:
        return None, "missing_hypothesis"
    try:
        gold = resolve_gold(row.get("true_label"), label_names)
    except ValueError as exc:
        if strict:
            raise ValueError(
                f"R014 Khmer true_label is not a known label: {row.get('true_label')!r}"
            ) from exc
        return None, "unmapped_true_label"
    if gold not in CRITERIA:
        return None, "unknown_label"
    example = choice_example(premise, hypothesis, gold, language)
    assert_example_shape(example, language)
    return example, None


def _export_rows(selected, config, language, label_names, strict, fields):
    """Dedup on the native identity, then build examples and groups.

    Row identity, dedup, and grouping always use the native pair, so the native
    and english arms drop the same rows and carry identical sidecars.
    """
    examples, groups, exclusions = [], [], Counter()
    seen_keys = set()
    dropped = 0
    for index, row in selected:
        example, reason = _emitted_example(row, language, label_names, strict, fields)
        if example is None:
            exclusions[reason] += 1
            continue
        key = dedup_key(
            row.get("premise_native"),
            row.get("hypothesis_native"),
            example["expected"][QID],
        )
        if key in seen_keys:
            dropped += 1
            continue
        seen_keys.add(key)
        examples.append(example)
        groups.append(
            {
                "config": config,
                "source_index": index,
                "group_id": group_id(row.get("premise_native")),
            }
        )
    return examples, groups, dropped, exclusions


def _premise_stats(groups):
    counts = Counter(entry["group_id"] for entry in groups)
    shared_groups = sum(1 for value in counts.values() if value > 1)
    shared_rows = sum(value for value in counts.values() if value > 1)
    return shared_rows, shared_groups


def _distribution(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    counted = Counter(present)
    return {
        str(key): count
        for key, count in sorted(counted.items(), key=lambda item: str(item[0]))
    }


def _config_descriptors(selected):
    """Descriptive composition of the selected Khmer rows. No filtering."""
    by_config = {}
    concept_sets = {}
    for config in CONFIG_ORDER:
        rows = [row for _, row in selected[config]]
        concepts = [
            row.get("concept_id") for row in rows if row.get("concept_id") is not None
        ]
        concept_sets[config] = set(concepts)
        disagreements = [
            (row.get("entailment_type"), row.get("true_label"))
            for row in rows
            if row.get("entailment_type") is not None
        ]
        by_config[config] = {
            "selected_rows": len(rows),
            "premise_groups": len(
                {group_id(row.get("premise_native")) for row in rows}
            ),
            "concepts": len(concept_sets[config]),
            "true_label": _distribution([row.get("true_label") for row in rows]),
            "entailment_type_vs_true_label_disagreement": (
                sum(1 for left, right in disagreements if left != right)
                if disagreements
                else None
            ),
            "is_misclassified": _distribution(
                [row.get("is_misclassified") for row in rows]
            ),
            "step_misclassified": _distribution(
                [row.get("step_misclassified") for row in rows]
            ),
            "rows_per_concept": (
                _distribution(list(Counter(concepts).values())) if concepts else None
            ),
        }
    any_concepts = concept_sets["normal"] or concept_sets["hard"]
    overlap = (
        len(concept_sets["normal"] & concept_sets["hard"]) if any_concepts else None
    )
    return {
        "by_config": by_config,
        "concept_id_overlap": overlap,
        "note": (
            "Descriptive composition only. Rows are never filtered on "
            "is_misclassified or step_misclassified: their exact semantics are "
            "unverified and filtering would drop all 23 hard rows and 14 "
            "normal rows. Both fields are dropped before export."
        ),
    }


def _write_export(stem, examples, groups, out_dir):
    rows_path = out_dir / f"{stem}.jsonl"
    ids_path = out_dir / f"{stem}-ids.json"
    write_jsonl(rows_path, examples)
    write_json(ids_path, groups)
    return rows_path, ids_path


def prepare(
    rows_by_config,
    source_files=(),
    *,
    label_names=None,
    out_dir=OUT,
    manifest_path=MANIFEST,
):
    """Select, validate, dedup, write, and describe. Returns the manifest payload."""
    label_names = label_names or {}
    selected_khmer = {
        config: [
            (index, row)
            for index, row in enumerate(rows_by_config.get(config, []))
            if is_khmer_culture(row)
        ]
        for config in CONFIG_ORDER
    }
    selected_other = {
        config: [
            (index, row)
            for index, row in enumerate(rows_by_config.get(config, []))
            if not is_khmer_culture(row)
        ]
        for config in CONFIG_ORDER
    }
    if not any(selected_khmer.values()):
        raise ValueError(
            "R014 selection found no Khmer rows: no row has culture "
            f"{KHMER_CULTURE!r} under the casefolded {CULTURE_FIELD!r} field"
        )
    validation = validate_khmer_script(selected_khmer)

    built = {}
    for key, spec in EXPORTS.items():
        selected = selected_khmer if spec["kind"] == "khmer" else selected_other
        examples, groups, dropped, exclusions = [], [], 0, Counter()
        for config in spec["configs"]:
            part_examples, part_groups, part_dropped, part_exclusions = _export_rows(
                selected[config],
                config,
                spec["language"],
                label_names.get(config),
                spec["strict"],
                spec["fields"],
            )
            examples.extend(part_examples)
            groups.extend(part_groups)
            dropped += part_dropped
            exclusions += part_exclusions
        if spec["kind"] == "khmer" and not examples:
            raise ValueError(f"R014 Khmer export {key} produced no rows")
        built[key] = (spec, examples, groups, dropped, exclusions)

    for native_key, english_key in ARM_PAIRS:
        if built[native_key][2] != built[english_key][2]:
            raise ValueError(
                f"R014 {english_key} does not share row identity and grouping "
                f"with {native_key}"
            )

    exports = {}
    config_label_distribution = {}
    exclusion_counts = Counter()
    for key, (spec, examples, groups, dropped, exclusions) in built.items():
        exclusion_counts += exclusions
        rows_path, ids_path = _write_export(spec["stem"], examples, groups, out_dir)
        shared_rows, shared_groups = _premise_stats(groups)
        relative_rows = rows_path.relative_to(ROOT).as_posix()
        relative_ids = ids_path.relative_to(ROOT).as_posix()
        label_distribution = dict(
            sorted(Counter(example["expected"][QID] for example in examples).items())
        )
        exports[relative_rows] = {
            "export": key,
            "kind": spec["kind"],
            "arm": spec["arm"],
            "source_configs": list(spec["configs"]),
            "state_fields": dict(spec["fields"]),
            "rows": len(examples),
            "groups": len({entry["group_id"] for entry in groups}),
            "duplicates_removed": dropped,
            "rows_sharing_premise": shared_rows,
            "shared_premise_groups": shared_groups,
            "label_distribution": label_distribution,
            "exclusions": dict(sorted(exclusions.items())),
            "sha256": sha256(rows_path),
            "ids": relative_ids,
            "ids_sha256": sha256(ids_path),
        }
        if spec["kind"] == "khmer" and spec["arm"] == "native":
            config_label_distribution[spec["configs"][0]] = label_distribution

    manifest = {
        "study": "R014-sea-nli-khmer-transfer",
        "dataset": DATASET,
        "revision": REVISION,
        "gated": True,
        "access": "manual; test split only; no train or validation split exists",
        "configs": list(CONFIG_ORDER),
        "files": list(source_files),
        "task": {
            "qid": QID,
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "labels": list(LABELS),
            "criterion_descriptions": dict(CRITERIA),
            "option_order": "fixed; entailment, neutral, contradiction for every row",
        },
        "state_fields": list(STATE_FIELDS),
        "field_mapping": {
            "native": dict(NATIVE_FIELDS),
            "english": dict(ENGLISH_FIELDS),
            "note": "state keys are always premise/hypothesis",
        },
        "allowlist": [*ALLOWLIST, f"{CULTURE_FIELD} (selection only, never exported)"],
        "selection": {
            "rule": f"casefolded {CULTURE_FIELD} equals {KHMER_CULTURE!r}",
            "field": CULTURE_FIELD,
            "value": KHMER_CULTURE,
            "counts": {
                "selected": {
                    config: len(selected_khmer[config]) for config in CONFIG_ORDER
                },
                "non_khmer": {
                    config: len(selected_other[config]) for config in CONFIG_ORDER
                },
            },
        },
        "validation": validation,
        "arms": {
            "native": {
                "state_source": dict(NATIVE_FIELDS),
                "exports": ["km-normal", "km-hard"],
            },
            "english": {
                "state_source": dict(ENGLISH_FIELDS),
                "exports": ["en-normal", "en-hard"],
            },
            "note": (
                "native and english arms share row identity (source_index), "
                "grouping (group_id), and gold; only the state text differs"
            ),
        },
        "dedup": {
            "scope": "per export; normal and hard are never deduped together",
            "key": "sha256(normalized native premise, normalized native "
            "hypothesis, true_label)",
            "keep": "lowest source row index",
        },
        "grouping": {
            "key": "sha256(NFKC + casefold + collapse-whitespace + strip "
            "U+200B/U+200C/U+200D/U+FEFF on the native premise)",
            "note": "grouping normalization never touches model-visible text",
        },
        "descriptors": _config_descriptors(selected_khmer),
        "config_label_distribution": config_label_distribution,
        "exports": exports,
        "exclusion_counts_by_reason": dict(sorted(exclusion_counts.items())),
        "preparer": "scripts/prepare_r014.py",
        "preparer_sha256": sha256(PREPARER),
    }
    write_json(manifest_path, manifest)
    return manifest


def download():
    rows_by_config, source_files, label_names = {}, [], {}
    for config in CONFIG_ORDER:
        path = hf_hub_download(
            DATASET,
            CONFIG_FILES[config],
            repo_type="dataset",
            revision=REVISION,
        )
        dataset = Dataset.from_parquet(path)
        rows_by_config[config] = dataset.to_list()
        source_files.append(
            {"config": config, "path": CONFIG_FILES[config], "sha256": sha256(path)}
        )
        true_label = dataset.features.get("true_label")
        label_names[config] = list(getattr(true_label, "names", []) or []) or None
    return rows_by_config, source_files, label_names


def main():
    try:
        rows_by_config, source_files, label_names = download()
    except Exception as exc:
        print(
            f"R014 preparation cannot read the gated dataset {DATASET} at "
            f"{REVISION}. Authenticate and accept the terms first: "
            f"`hf auth login` (or set HF_TOKEN). Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    manifest = prepare(rows_by_config, source_files, label_names=label_names)
    for relative, entry in manifest["exports"].items():
        print(
            f"{relative}: arm={entry['arm']} {entry['rows']} rows, "
            f"{entry['groups']} groups, {entry['duplicates_removed']} duplicates removed"
        )
        print(f"  exclusions: {entry['exclusions']}")
    print(f"exclusions by reason: {manifest['exclusion_counts_by_reason']}")
    print(f"selection: {manifest['selection']}")
    print(f"validation: {manifest['validation']}")
    print("manifest summary:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
