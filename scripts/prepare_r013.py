"""Create the locked R013 group split and matched-budget training files."""

import copy
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from r013_identity import (
    EXPECTED_SPLIT,
    HELDOUT_INTENTS,
    MANIFEST,
    PREPARATION_FILE,
    ROOT,
    SOURCE_FILE,
    SPLIT_FILES,
    TRAIN_FILES,
    group_identity,
    normalize_input,
    read_jsonl,
    row_identity,
    sha256,
    verify_locked_manifest,
    write_json,
    write_jsonl,
)

SELECTION_SEED = 20261014
SPLIT_SEED = 20261014
CHOICE_COUNT = 20
HELDOUT_DISTRACTOR_COUNT = 8
INSTRUCTIONS = "What is the user's intent in this utterance?"


def _selected_intents(groups_by_label):
    eligible = sorted(
        label for label, groups in groups_by_label.items() if len(groups) >= 40
    )
    if len(eligible) < 12:
        raise ValueError("R013 has fewer than 12 eligible intents")
    return tuple(sorted(random.Random(SELECTION_SEED).sample(eligible, 12)))


def _rng(group_id, split, purpose):
    return random.Random(f"{SPLIT_SEED}:{split}:{group_id}:{purpose}")


def _choice_row(source, source_id, group_id, labels, split, fixed_options=None):
    gold = source["expected"]["intent"]
    if fixed_options is None:
        if gold not in labels or len(labels) < CHOICE_COUNT:
            raise ValueError("R013 candidate pool does not contain the gold label")
        negatives = [label for label in labels if label != gold]
        _rng(group_id, split, "candidate-selection").shuffle(negatives)
        options = [gold, *negatives[: CHOICE_COUNT - 1]]
    else:
        options = list(fixed_options)
        if len(options) != CHOICE_COUNT or gold not in options:
            raise ValueError(
                "R013 fixed held-out options do not contain the gold label"
            )
    _rng(group_id, split, "display-order").shuffle(options)
    criteria = {label: label.replace("_", " ") for label in options}
    row = copy.deepcopy(source)
    row["questions"] = {
        "intent": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": criteria,
        }
    }
    row["tags"] = [
        f"r013-group:{group_id}",
        f"r013-row:{source_id}",
    ]
    return row


def _allocate_proportionally(group_ids_by_label, target):
    total = sum(len(groups) for groups in group_ids_by_label.values())
    if target <= 0 or target > total:
        raise ValueError("R013 matched-budget sample is outside the available pool")
    exact = {
        label: len(groups) * target / total
        for label, groups in group_ids_by_label.items()
    }
    allocation = {label: int(value) for label, value in exact.items()}
    remainder = target - sum(allocation.values())
    labels = sorted(group_ids_by_label)
    order = sorted(labels, key=lambda label: (-(exact[label] % 1), label))
    for label in order[:remainder]:
        allocation[label] += 1
    if sum(allocation.values()) != target or any(
        allocation[label] < 1 or allocation[label] > len(group_ids_by_label[label])
        for label in labels
    ):
        raise ValueError("R013 proportional allocation omitted or overfilled a class")
    return allocation


def prepare():
    manifest, manifest_sha256 = verify_locked_manifest(MANIFEST)
    source_path = ROOT / SOURCE_FILE
    source_rows = read_jsonl(source_path)
    if len(source_rows) != EXPECTED_SPLIT["source_rows"]:
        raise ValueError("R013 source row count differs from the lock")

    grouped = defaultdict(list)
    for index, row in enumerate(source_rows):
        text = row.get("state", {}).get("utterance")
        label = row.get("expected", {}).get("intent")
        if not isinstance(text, str) or not text or not isinstance(label, str):
            raise ValueError(f"R013 source row is malformed: {index}")
        normalized = normalize_input(text)
        if not normalized:
            raise ValueError(f"R013 source row normalizes to empty text: {index}")
        grouped[normalized].append((index, row, label))

    conflicting = {
        key
        for key, members in grouped.items()
        if len({member[2] for member in members}) > 1
    }
    same_label_duplicate_groups = sum(
        len(members) > 1 and len({member[2] for member in members}) == 1
        for members in grouped.values()
    )
    conflicting_rows = sum(len(grouped[key]) for key in conflicting)
    representative = {}
    groups_by_label = defaultdict(list)
    for normalized, members in grouped.items():
        if normalized in conflicting:
            continue
        index, row, label = min(members, key=lambda member: member[0])
        gid = group_identity(normalized)
        sid = row_identity(manifest["source"]["sha256"], index, row)
        representative[gid] = (index, row, label, sid)
        groups_by_label[label].append(gid)
    for groups in groups_by_label.values():
        groups.sort()

    actual_split = {
        "source_rows": len(source_rows),
        "distinct_normalized_groups": len(grouped),
        "same_label_duplicate_groups": same_label_duplicate_groups,
        "conflicting_label_groups": len(conflicting),
        "conflicting_label_rows": conflicting_rows,
        "eligible_groups": sum(map(len, groups_by_label.values())),
        "eligible_labels": sum(
            len(groups) >= 40 for groups in groups_by_label.values()
        ),
    }
    if actual_split != {
        key: EXPECTED_SPLIT[key]
        for key in (
            "source_rows",
            "distinct_normalized_groups",
            "same_label_duplicate_groups",
            "conflicting_label_groups",
            "conflicting_label_rows",
            "eligible_groups",
            "eligible_labels",
        )
    }:
        raise ValueError(f"R013 source grouping differs from the lock: {actual_split}")

    heldout = _selected_intents(groups_by_label)
    if heldout != HELDOUT_INTENTS:
        raise ValueError("R013 selected intent labels differ from the locked set")
    heldout_set = set(heldout)
    seen_labels = sorted(set(groups_by_label) - heldout_set)
    if len(seen_labels) != 48:
        raise ValueError("R013 must have exactly 48 seen labels")

    split_groups = {name: defaultdict(list) for name in ("train", "validation", "test")}
    for label, group_ids in groups_by_label.items():
        shuffled = list(group_ids)
        random.Random(f"{SPLIT_SEED}:{label}").shuffle(shuffled)
        n_test = max(1, round(0.20 * len(shuffled)))
        n_validation = max(1, round(0.10 * len(shuffled)))
        if n_test + n_validation >= len(shuffled):
            raise ValueError(
                f"R013 intent has too few groups for 70/10/20 split: {label}"
            )
        split_groups["test"][label] = shuffled[:n_test]
        split_groups["validation"][label] = shuffled[n_test : n_test + n_validation]
        split_groups["train"][label] = shuffled[n_test + n_validation :]

    actual_counts = {
        "train_groups": sum(map(len, split_groups["train"].values())),
        "validation_groups": sum(map(len, split_groups["validation"].values())),
        "test_groups": sum(map(len, split_groups["test"].values())),
        "heldout_test_groups": sum(
            len(split_groups["test"][label]) for label in heldout
        ),
        "known_test_groups": sum(
            len(split_groups["test"][label]) for label in seen_labels
        ),
        "seen_only_train_groups": sum(
            len(split_groups["train"][label]) for label in seen_labels
        ),
        "validation_unseen_groups": sum(
            len(split_groups["validation"][label]) for label in heldout
        ),
        "validation_seen_groups": sum(
            len(split_groups["validation"][label]) for label in seen_labels
        ),
    }
    if any(actual_counts[key] != EXPECTED_SPLIT[key] for key in actual_counts):
        raise ValueError(f"R013 split counts differ from the lock: {actual_counts}")
    train_group_set = set().union(*split_groups["train"].values())
    heldout_group_set = set().union(*split_groups["validation"].values())
    heldout_group_set |= set().union(*split_groups["test"].values())
    if train_group_set & heldout_group_set:
        raise ValueError("R013 train groups overlap validation or test groups")

    existing = [ROOT / PREPARATION_FILE]
    existing.extend(
        ROOT / path for path in (*TRAIN_FILES.values(), *SPLIT_FILES.values())
    )
    if any(path.exists() for path in existing):
        raise FileExistsError("R013 preparation outputs already exist")

    distractors = sorted(
        random.Random(f"{SPLIT_SEED}:heldout-test-distractors").sample(seen_labels, 8)
    )
    heldout_test_options = tuple(sorted((*heldout, *distractors)))
    if len(heldout_test_options) != CHOICE_COUNT:
        raise ValueError("R013 held-out test option set must contain 20 labels")

    source_by_group = {
        gid: (row, label, sid)
        for gid, (_index, row, label, sid) in representative.items()
    }

    train_seen = [gid for label in seen_labels for gid in split_groups["train"][label]]
    if len(train_seen) != EXPECTED_SPLIT["seen_only_train_groups"]:
        raise ValueError("R013 seen-only training row count differs from the lock")
    all_train_by_label = {
        label: sorted(split_groups["train"][label])
        for label in sorted(split_groups["train"])
    }
    allocation = _allocate_proportionally(all_train_by_label, len(train_seen))
    train_all = []
    for label, count in allocation.items():
        train_all.extend(
            random.Random(f"{SPLIT_SEED}:all-label-sample:{label}").sample(
                all_train_by_label[label], count
            )
        )
    if len(train_all) != len(train_seen) or len(set(train_all)) != len(train_all):
        raise ValueError("R013 all-label arm does not match the locked example budget")
    if set(allocation) != set(groups_by_label):
        raise ValueError("R013 all-label sampling must include all 60 intents")

    def make_training_rows(group_ids, label_pool, split_name, require_seen_only=False):
        rows = []
        for gid in sorted(group_ids):
            source, label, sid = source_by_group[gid]
            if require_seen_only and label not in seen_labels:
                raise ValueError("R013 held-out label leaked into seen-only training")
            rows.append(_choice_row(source, sid, gid, label_pool, split_name))
        return rows

    train_rows = {
        "seen-only": make_training_rows(
            train_seen, seen_labels, "train-seen-only", require_seen_only=True
        ),
        "all-label": make_training_rows(
            train_all, sorted(groups_by_label), "train-all-label"
        ),
    }
    if any(
        len(rows) != EXPECTED_SPLIT["seen_only_train_groups"]
        for rows in train_rows.values()
    ):
        raise ValueError("R013 training arm size differs from the locked budget")

    split_rows = {}
    for name in ("validation", "test"):
        unseen_rows, seen_rows = [], []
        for label, gids in split_groups[name].items():
            for gid in gids:
                source, _gold, sid = source_by_group[gid]
                if label in heldout_set:
                    unseen_rows.append(
                        _choice_row(
                            source,
                            sid,
                            gid,
                            heldout_test_options,
                            f"{name}-unseen",
                            fixed_options=heldout_test_options,
                        )
                    )
                else:
                    seen_rows.append(
                        _choice_row(
                            source,
                            sid,
                            gid,
                            seen_labels,
                            f"{name}-seen",
                        )
                    )
        split_rows[f"{name}-unseen"] = sorted(
            unseen_rows, key=lambda row: row["tags"][0]
        )
        split_rows[f"{name}-seen"] = sorted(seen_rows, key=lambda row: row["tags"][0])

    output_rows = {}
    for arm, rows in train_rows.items():
        relative = TRAIN_FILES[arm]
        write_jsonl(ROOT / relative, rows)
        output_rows[relative] = sha256(ROOT / relative)
    for name, rows in split_rows.items():
        relative = SPLIT_FILES[f"{name}"]
        write_jsonl(ROOT / relative, rows)
        output_rows[relative] = sha256(ROOT / relative)

    preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_sha256,
        "preparer_sha256": sha256(Path(__file__)),
        "source_sha256": manifest["source"]["sha256"],
        "heldout_intents": list(heldout),
        "seen_intents": seen_labels,
        "heldout_test_distractors": distractors,
        "split_counts": EXPECTED_SPLIT,
        "training_rows": {arm: len(rows) for arm, rows in train_rows.items()},
        "validation_rows": {
            "unseen": len(split_rows["validation-unseen"]),
            "seen": len(split_rows["validation-seen"]),
        },
        "test_rows": {
            "unseen": len(split_rows["test-unseen"]),
            "seen": len(split_rows["test-seen"]),
        },
        "all_label_sample_per_intent": allocation,
        "outputs_sha256": output_rows,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(ROOT / PREPARATION_FILE, preparation)
    print(
        f"R013 prepared: {len(train_seen)} rows per arm; "
        f"{len(split_rows['test-unseen'])} held-out and "
        f"{len(split_rows['test-seen'])} known-intent test groups"
    )
    return preparation


def main():
    prepare()


if __name__ == "__main__":
    main()
