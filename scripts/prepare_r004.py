"""Export R004 cross-task Khmer eval sets as Laya choice examples.

Every row offers all of the task's labels, in one fixed order, so every model
sees identical candidates. Text is kept exactly as published (no Unicode or
zero-width-space normalization). Train splits are not exported: R004 is
evaluation only.
"""

import csv
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUT = Path("data/processed")
MANIFESTS = Path("datasets/manifests")

SIB = {
    "dataset": "Davlan/sib200",
    "config": "khm_Khmr",
    "revision": "38977a667f6fc264d5c26ec57a01e16db040b358",
    "note": "Original release. mteb/sib200 khm_Khmr was rejected: its "
    "validation split is identical to its test split.",
}
NEWS = {
    "dataset": "CADT-IDRI/Khmer_News_classification_unsegmented",
    "config": "default",
    "revision": "2707fac4b9f1ffb820d4cae1257fece6a0b380c6",
    "note": "Unsegmented variant, closest to real input. License not stated "
    "on the dataset card.",
}

# Option text shown to the model. Keys are the source label names.
SIB_LABELS = {
    "entertainment": "entertainment",
    "geography": "geography",
    "health": "health",
    "politics": "politics",
    "science/technology": "science and technology",
    "sports": "sports",
    "travel": "travel",
}
NEWS_LABELS = {
    "economic": "economy",
    "entertainment": "entertainment",
    "life": "lifestyle",
    "politic": "politics",
    "sport": "sport",
    "technology": "technology",
}


def write(name, split, rows, question, meta):
    path = OUT / f"{name}-{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for state, gold in rows:
            example = {
                "state": state,
                "language": "km",
                "questions": {"topic": question},
                "expected": {"topic": gold},
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} examples")
    meta.setdefault("splits", {})[split] = len(rows)


def sib200():
    question = {
        "type": "choice",
        "instructions": "What is the topic of `text`?",
        "criteria": SIB_LABELS,
    }
    meta = {**SIB, "labels": list(SIB_LABELS)}
    for source, split in (("dev", "validation"), ("test", "test")):
        path = hf_hub_download(
            SIB["dataset"],
            f"data/{SIB['config']}/{source}.tsv",
            repo_type="dataset",
            revision=SIB["revision"],
        )
        with open(path, encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        rows = [({"text": r["text"]}, r["category"]) for r in records]
        assert {gold for _, gold in rows} <= set(SIB_LABELS)
        write("sib200-km", split, rows, question, meta)
    return meta


def khmer_news():
    question = {
        "type": "choice",
        "instructions": "What is the news category of the article in `title` "
        "and `content`?",
        "criteria": NEWS_LABELS,
    }
    meta = {**NEWS, "labels": list(NEWS_LABELS)}
    ds = load_dataset(NEWS["dataset"], revision=NEWS["revision"])
    names = ds["test"].features["label"].names
    assert set(names) == set(NEWS_LABELS), names
    for split in ("validation", "test"):
        rows = [
            ({"title": r["title"], "content": r["content"]}, names[r["label"]])
            for r in ds[split]
        ]
        write("khmer-news", split, rows, question, meta)
    return meta


OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)
for name, prepare in (("sib200-km", sib200), ("khmer-news", khmer_news)):
    meta = prepare()
    path = MANIFESTS / f"{name}.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
