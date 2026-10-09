"""Derive R005 decision-complexity eval sets from the prepared R000/R004 files.

Run after prepare_massive.py and prepare_r004.py. Validation and test only.
Nothing here looks at model outputs; every variant is fixed by a seed.

- massive-km-k{K}: nested candidate sets. Per row, one seeded random order of
  the other 59 intents picks the negatives: the K-candidate set is the gold
  plus the first K-1 of them, so k=2 is inside k=5 is inside k=7 and so on.
  A second, independent random order of all 60 intents sets the display
  order, so the gold position is uniform within each set. (Reusing the
  selection order for display put the gold last in 98.6% of k=2 rows.)
- {task}-perm{P}: the R001-R004 rows with each row's options reordered by a
  seeded shuffle. The option texts are unchanged.
- {task}-raw: option text is the raw dataset label id.
- {sib200-km,khmer-news}-desc: option text is a short English description,
  written from the label names before any R005 result was seen.
"""

import json
import random
from pathlib import Path

DATA = Path("data/processed")
SEED = 20261009
KS = (2, 5, 7, 20, 60)
PERMUTATIONS = 5
SPLITS = ("validation", "test")
SOURCES = {
    "massive-km": "massive-km-{split}-full.jsonl",
    "sib200-km": "sib200-km-{split}.jsonl",
    "khmer-news": "khmer-news-{split}.jsonl",
}
DESCRIPTIONS = {
    "sib200-km": {
        "entertainment": "entertainment: film, music, television, celebrities",
        "geography": "geography: places, landscapes, climate, natural features",
        "health": "health: medicine, disease, the body, healthcare",
        "politics": "politics: government, elections, laws, international relations",
        "science/technology": "science and technology: research, discoveries, computers, space",
        "sports": "sports: games, athletes, competitions",
        "travel": "travel: tourism, trips, transport, visiting places",
    },
    "khmer-news": {
        "economic": "economy: business, trade, markets, finance",
        "entertainment": "entertainment: celebrities, film, music, television",
        "life": "lifestyle: daily life, health, food, family, culture",
        "politic": "politics: government, elections, policy, diplomacy",
        "sport": "sport: matches, athletes, competitions",
        "technology": "technology: gadgets, internet, software, science",
    },
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(name, split, rows):
    path = DATA / f"{name}-{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} examples")


def with_criteria(row, criteria):
    (qid,) = row["questions"]
    question = {**row["questions"][qid], "criteria": criteria}
    return {**row, "questions": {qid: question}}


def criteria_of(row):
    (qid,) = row["questions"]
    return row["questions"][qid]["criteria"]


def gold_of(row):
    (qid,) = row["expected"]
    return row["expected"][qid]


intents = sorted({gold_of(r) for r in read_jsonl(DATA / "massive-km-train-full.jsonl")})
assert len(intents) == 60

for split in SPLITS:
    sources = {
        task: read_jsonl(DATA / pattern.format(split=split))
        for task, pattern in SOURCES.items()
    }

    rng = random.Random(f"{SEED}-k-{split}")
    by_k = {k: [] for k in KS}
    for row in sources["massive-km"]:
        gold = gold_of(row)
        others = [label for label in intents if label != gold]
        rng.shuffle(others)
        display = intents[:]
        rng.shuffle(display)
        for k in KS:
            chosen = {gold, *others[: k - 1]}
            labels = [label for label in display if label in chosen]
            by_k[k].append(
                with_criteria(row, {lab: lab.replace("_", " ") for lab in labels})
            )
    for k in KS:
        write_jsonl(f"massive-km-k{k}", split, by_k[k])

    for task, rows in sources.items():
        for p in range(PERMUTATIONS):
            rng = random.Random(f"{SEED}-perm{p}-{task}-{split}")
            permuted = []
            for row in rows:
                items = list(criteria_of(row).items())
                rng.shuffle(items)
                permuted.append(with_criteria(row, dict(items)))
            write_jsonl(f"{task}-perm{p}", split, permuted)

        write_jsonl(
            f"{task}-raw",
            split,
            [with_criteria(r, {lab: lab for lab in criteria_of(r)}) for r in rows],
        )
        if task in DESCRIPTIONS:
            descriptions = DESCRIPTIONS[task]
            write_jsonl(
                f"{task}-desc",
                split,
                [
                    with_criteria(r, {lab: descriptions[lab] for lab in criteria_of(r)})
                    for r in rows
                ],
            )
