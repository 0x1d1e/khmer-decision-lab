"""Summarize MASSIVE Khmer eval reports into research/results.json.

Reads existing laya-evals reports only; no model is loaded. Adds the metrics
laya-evals does not report: macro-F1 over gold intents, and accuracy split by
whether the utterance also appears verbatim in the training file.
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

RUNS = Path("artifacts/runs")
DATA = Path("data/processed")
OUTPUT = Path("research/results.json")

EXPERIMENTS = {
    "multilingual": "R001",
    "head-e2": "R002",
    "full-e2-8bit": "R003",
}
SPLITS = ("validation", "test")
KEPT = (
    "choice_accuracy",
    "ece",
    "brier",
    "aurc",
    "mean_confidence",
    "selective_accuracy@50",
    "selective_accuracy@80",
    "latency_p50_ms",
    "latency_p95_ms",
)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split()).lower()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def macro_f1(golds, preds):
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, pred in zip(golds, preds):
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    labels = sorted(set(golds))
    scores = []
    for label in labels:
        denom = 2 * tp[label] + fp[label] + fn[label]
        scores.append(2 * tp[label] / denom if denom else 0.0)
    return sum(scores) / len(scores)


def accuracy(cases):
    return sum(c["correct"] for c in cases) / len(cases) if cases else float("nan")


train_texts = {
    normalize(row["state"]["utterance"])
    for row in read_jsonl(DATA / "massive-km-train-full.jsonl")
}

results = {}
for split in SPLITS:
    data_path = DATA / f"massive-km-{split}-full.jsonl"
    rows = read_jsonl(data_path)
    in_train = [normalize(r["state"]["utterance"]) in train_texts for r in rows]

    for model, experiment in EXPERIMENTS.items():
        report = json.loads(
            (RUNS / f"{model}-{split}.json").read_text(encoding="utf-8")
        )
        cases = report["cases"]
        assert report["config"]["dataset"] == str(data_path)
        assert len(cases) == len(rows)
        for case, row in zip(cases, rows):
            assert case["expected"] == row["expected"]["intent"]
            assert set(case["answer"]["probabilities"]) == set(
                row["questions"]["intent"]["criteria"]
            )

        seen = [c for c, flag in zip(cases, in_train) if flag]
        unseen = [c for c, flag in zip(cases, in_train) if not flag]
        entry = {k: round(report["overall"][k], 4) for k in KEPT}
        entry["macro_f1"] = round(
            macro_f1(
                [c["expected"] for c in cases], [c["answer"]["choice"] for c in cases]
            ),
            4,
        )
        entry["n"] = len(cases)
        entry["n_utterance_in_train"] = len(seen)
        entry["accuracy_utterance_in_train"] = round(accuracy(seen), 4)
        entry["accuracy_utterance_not_in_train"] = round(accuracy(unseen), 4)
        entry["dataset_sha256"] = report["config"]["dataset_sha256"]
        entry["report"] = str(RUNS / f"{model}-{split}.json")
        results.setdefault(experiment, {"model": model})[split] = entry

OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

for experiment, entry in results.items():
    for split in SPLITS:
        m = entry[split]
        print(
            f"{experiment} {split:<10} acc {m['choice_accuracy']:.4f}"
            f"  macroF1 {m['macro_f1']:.4f}  ece {m['ece']:.4f}"
            f"  acc seen/unseen {m['accuracy_utterance_in_train']:.4f}"
            f"/{m['accuracy_utterance_not_in_train']:.4f}"
            f" (seen n={m['n_utterance_in_train']})"
        )
print(f"wrote {OUTPUT}")
