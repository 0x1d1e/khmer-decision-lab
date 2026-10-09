"""Build the R006 multi-task training file.

Run after prepare_massive.py and prepare_r004.py. Exports the SIB-200 and
Khmer News train splits with the same questions and option texts as R004,
drops News train articles that also appear in News validation or test, and
shuffles MASSIVE train + News train + SIB train together at their natural
sizes. Validation and test files are untouched.
"""

import csv
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from prepare_r004 import NEWS, NEWS_QUESTION, SIB, SIB_QUESTION

DATA = Path("data/processed")
OUTPUT = DATA / "r006-multitask-train.jsonl"
MANIFEST = Path("datasets/manifests/r006-multitask.json")
SEED = 13


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def example(state, question, gold):
    return {
        "state": state,
        "language": "km",
        "questions": {"topic": question},
        "expected": {"topic": gold},
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} examples")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


path = hf_hub_download(
    SIB["dataset"],
    f"data/{SIB['config']}/train.tsv",
    repo_type="dataset",
    revision=SIB["revision"],
)
with open(path, encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
sib = [example({"text": r["text"]}, SIB_QUESTION, r["category"]) for r in records]

ds = load_dataset(NEWS["dataset"], revision=NEWS["revision"])
names = ds["train"].features["label"].names
held_out = {
    normalize(r["title"] + r["content"]) for s in ("validation", "test") for r in ds[s]
}
news = [
    example(
        {"title": r["title"], "content": r["content"]}, NEWS_QUESTION, names[r["label"]]
    )
    for r in ds["train"]
    if normalize(r["title"] + r["content"]) not in held_out
]
dropped = len(ds["train"]) - len(news)

write_jsonl(DATA / "sib200-km-train.jsonl", sib)
write_jsonl(DATA / "khmer-news-train.jsonl", news)

massive_path = DATA / "massive-km-train-full.jsonl"
with massive_path.open(encoding="utf-8") as f:
    massive = [json.loads(line) for line in f]
rows = massive + news + sib
random.Random(SEED).shuffle(rows)
write_jsonl(OUTPUT, rows)

MANIFEST.write_text(
    json.dumps(
        {
            "seed": SEED,
            "rows": {
                "massive-km": len(massive),
                "khmer-news": len(news),
                "sib200-km": len(sib),
            },
            "khmer_news_train_dropped_as_in_validation_or_test": dropped,
            "sources": {"sib200-km": SIB, "khmer-news": NEWS},
            "sha256": {
                str(p): sha256(p)
                for p in (
                    massive_path,
                    DATA / "khmer-news-train.jsonl",
                    DATA / "sib200-km-train.jsonl",
                    OUTPUT,
                )
            },
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"wrote {MANIFEST}")
