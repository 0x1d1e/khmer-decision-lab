import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

DATASET = "mteb/amazon_massive_intent"
SEED = 13

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

manifest = Path("datasets/manifests/massive-km.json")
manifest.parent.mkdir(parents=True, exist_ok=True)

# Resolve revision once, then reuse it.
if manifest.exists():
    revision = json.loads(manifest.read_text())["revision"]
else:
    revision = HfApi().dataset_info(DATASET).sha
    manifest.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "config": "km",
                "revision": revision,
                "seed": SEED,
                "choices": 20,
            },
            indent=2,
        )
        + "\n"
    )

ds = load_dataset(DATASET, "km", revision=revision)

labels = sorted(set(ds["train"]["label_text"]))


def export(split, limit=None):
    rows = ds[split]
    rng = random.Random(SEED)

    indices = list(range(len(rows)))
    if limit is not None:
        indices = sorted(rng.sample(indices, min(limit, len(indices))))

    suffix = str(limit) if limit is not None else "full"
    path = out / f"massive-km-{split}-{suffix}.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            row = rows[idx]
            gold = row["label_text"]

            negatives = rng.sample(
                [label for label in labels if label != gold],
                19,
            )
            choices = [gold, *negatives]
            rng.shuffle(choices)

            example = {
                "state": {"utterance": row["text"]},
                "language": "km",
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": ("What is the user asking for in `utterance`?"),
                        "criteria": {
                            label: label.replace("_", " ") for label in choices
                        },
                    }
                },
                "expected": {"intent": gold},
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{path}: {len(indices)} examples")


export("train")
export("validation")
export("test")
