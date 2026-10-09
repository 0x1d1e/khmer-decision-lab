"""Summarize laya-evals reports into research/results.json.

Reads existing reports only; no model is loaded. Adds what laya-evals does
not report: macro-F1 over gold labels, chance and majority-class accuracy,
and for MASSIVE, accuracy split by whether the utterance also appears verbatim
in the training file.
"""

import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

RUNS = Path("artifacts/runs")
DATA = Path("data/processed")
OUTPUT = Path("research/results.json")

# task -> (data file pattern, experiments that report on it)
TASKS = {
    "massive-km": ("massive-km-{split}-full.jsonl", ["R001", "R002", "R003"]),
    "sib200-km": ("sib200-km-{split}.jsonl", ["R004"]),
    "khmer-news": ("khmer-news-{split}.jsonl", ["R004"]),
}
SPLITS = ("validation", "test")
MODELS = {
    "multilingual": "base (R001)",
    "head-e2": "R002",
    "full-e2-8bit": "R003",
    "control-base-maxlen512": "base@512",
}
# (reference, compared) pairs for paired accuracy differences
PAIRS = (
    ("multilingual", "head-e2"),
    ("multilingual", "full-e2-8bit"),
    ("control-base-maxlen512", "head-e2"),
    ("control-base-maxlen512", "full-e2-8bit"),
)
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
    "cost_per_decision_p50_ms",
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
    scores = []
    for label in sorted(set(golds)):
        denom = 2 * tp[label] + fp[label] + fn[label]
        scores.append(2 * tp[label] / denom if denom else 0.0)
    return sum(scores) / len(scores)


def accuracy(cases):
    return sum(c["correct"] for c in cases) / len(cases) if cases else float("nan")


def paired_bootstrap(a, b, samples=10_000, seed=13):
    """Accuracy difference b - a with a 95% percentile interval over rows."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return {
        "diff": round((sum(b) - sum(a)) / n, 4),
        "ci95": [
            round(diffs[int(0.025 * samples)], 4),
            round(diffs[int(0.975 * samples)], 4),
        ],
    }


def label_recall(golds, preds):
    hits, totals = Counter(), Counter(golds)
    for gold, pred in zip(golds, preds):
        hits[gold] += gold == pred
    return {label: round(hits[label] / totals[label], 4) for label in sorted(totals)}


massive_train = {
    normalize(row["state"]["utterance"])
    for row in read_jsonl(DATA / "massive-km-train-full.jsonl")
}

results = {}
for task, (pattern, experiments) in TASKS.items():
    for split in SPLITS:
        data_path = DATA / pattern.format(split=split)
        if not (RUNS / f"{task}-{split}").is_dir():
            print(f"skip {task} {split}: no runs")
            continue
        rows = read_jsonl(data_path)
        (qid,) = rows[0]["expected"]
        golds = [r["expected"][qid] for r in rows]
        n_options = {len(r["questions"][qid]["criteria"]) for r in rows}
        majority = Counter(golds).most_common(1)[0][1] / len(golds)
        split_entry = {
            "experiments": experiments,
            "n": len(rows),
            "options_per_row": sorted(n_options),
            "chance_accuracy": round(sum(1 / k for k in n_options) / len(n_options), 4),
            "majority_accuracy": round(majority, 4),
            "dataset_sha256": None,
            "models": {},
        }
        in_train = None
        if task == "massive-km":
            in_train = [
                normalize(r["state"]["utterance"]) in massive_train for r in rows
            ]

        correct = {}
        for model, label in MODELS.items():
            path = RUNS / f"{task}-{split}" / f"{model}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            cases = report["cases"]
            assert report["config"]["dataset"] == str(data_path)
            assert len(cases) == len(rows)
            for case, row in zip(cases, rows):
                assert case["expected"] == row["expected"][qid]
                assert set(case["answer"]["probabilities"]) == set(
                    row["questions"][qid]["criteria"]
                )
            split_entry["dataset_sha256"] = report["config"]["dataset_sha256"]

            entry = {"label": label}
            entry.update({k: round(report["overall"][k], 4) for k in KEPT})
            preds = [c["answer"]["choice"] for c in cases]
            entry["macro_f1"] = round(macro_f1(golds, preds), 4)
            entry["recall_by_label"] = label_recall(golds, preds)
            correct[model] = [int(c["correct"]) for c in cases]
            if "peak_vram_mib" in report["config"]:
                entry["peak_vram_mib"] = report["config"]["peak_vram_mib"]
            if in_train is not None:
                seen = [c for c, flag in zip(cases, in_train) if flag]
                unseen = [c for c, flag in zip(cases, in_train) if not flag]
                entry["n_utterance_in_train"] = len(seen)
                entry["accuracy_utterance_in_train"] = round(accuracy(seen), 4)
                entry["accuracy_utterance_not_in_train"] = round(accuracy(unseen), 4)
            entry["report"] = str(path)
            split_entry["models"][model] = entry
        split_entry["accuracy_diffs"] = {
            f"{b} - {a}": paired_bootstrap(correct[a], correct[b])
            for a, b in PAIRS
            if a in correct and b in correct
        }
        results.setdefault(task, {})[split] = split_entry

OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

for task, splits in results.items():
    for split, s in splits.items():
        print(
            f"\n{task} {split} (n={s['n']}, chance {s['chance_accuracy']:.4f},"
            f" majority {s['majority_accuracy']:.4f})"
        )
        for model, m in s["models"].items():
            print(
                f"  {m['label']:<12} acc {m['choice_accuracy']:.4f}"
                f"  macroF1 {m['macro_f1']:.4f}  ece {m['ece']:.4f}"
                f"  brier {m['brier']:.4f}"
            )
        for pair, d in s["accuracy_diffs"].items():
            print(f"  {pair}: {d['diff']:+.4f} CI95 {d['ci95']}")
print(f"\nwrote {OUTPUT}")
