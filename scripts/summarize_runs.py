"""Summarize laya-evals reports into research/results.json.

Reads existing reports only; no model is loaded (R005 loads the tokenizer to
find options that truncation makes indistinguishable). Adds what laya-evals
does not report: macro-F1 over gold labels, chance and majority-class
accuracy, for MASSIVE accuracy split by whether the utterance also appears
verbatim in the training file, and the R005 cross-variant comparisons.
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

RUNS = Path("artifacts/runs")
DATA = Path("data/processed")
OUTPUT = Path("research/results.json")

# task -> (data file pattern, experiments that report on it)
TASKS = {
    "massive-km": (
        "massive-km-{split}-full.jsonl",
        ["R001", "R002", "R003", "R005", "R006"],
    ),
    "sib200-km": ("sib200-km-{split}.jsonl", ["R004", "R005", "R006"]),
    "khmer-news": ("khmer-news-{split}.jsonl", ["R004", "R005", "R006"]),
}
# R005 variants (scripts/prepare_r005.py); k60-long reruns k60 with a larger
# token budget, so it shares the data file.
KS = (2, 5, 7, 20, 60)
PERMS = tuple(f"perm{p}" for p in range(5))
for k in KS:
    TASKS[f"massive-km-k{k}"] = (f"massive-km-k{k}-{{split}}.jsonl", ["R005"])
TASKS["massive-km-k60-long"] = ("massive-km-k60-{split}.jsonl", ["R005"])
TEXT_VARIANTS = {
    "massive-km": ("raw",),
    "sib200-km": ("raw", "desc"),
    "khmer-news": ("raw", "desc"),
}
for base_task, variants in TEXT_VARIANTS.items():
    for v in PERMS + variants:
        TASKS[f"{base_task}-{v}"] = (f"{base_task}-{v}-{{split}}.jsonl", ["R005"])
R005_MODELS = ("control-base-maxlen512", "head-e2", "full-e2-8bit", "mt-s13")
SPLITS = ("validation", "test")
MODELS = {
    "multilingual": "base (R001)",
    "head-e2": "R002",
    "full-e2-8bit": "R003",
    "control-base-maxlen512": "base@512",
    "mt-s13": "R006 s13",
    "mt-s14": "R006 s14",
}
# (reference, compared) pairs for paired accuracy differences
PAIRS = (
    ("multilingual", "head-e2"),
    ("multilingual", "full-e2-8bit"),
    ("control-base-maxlen512", "head-e2"),
    ("control-base-maxlen512", "full-e2-8bit"),
    ("full-e2-8bit", "mt-s13"),
    ("full-e2-8bit", "mt-s14"),
    ("control-base-maxlen512", "mt-s13"),
    ("control-base-maxlen512", "mt-s14"),
    ("mt-s13", "mt-s14"),
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
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    # Chunked to bound memory at about 1000 x n indices.
    diffs = np.sort(
        np.concatenate(
            [
                d[rng.integers(0, n, (min(1000, samples - i), n))].mean(axis=1)
                for i in range(0, samples, 1000)
            ]
        )
    )
    return {
        "diff": round(float(d.mean()), 4),
        "ci95": [
            round(float(diffs[int(0.025 * samples)]), 4),
            round(float(diffs[int(0.975 * samples)]), 4),
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
outcomes = {}  # (task, split, model) -> (per-row correct, per-row prediction)
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
        if task.startswith("massive-km"):
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
            outcomes[task, split, model] = (correct[model], preds)
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


def colliding_intents(head_max_len):
    """MASSIVE intents whose option span at k=60 equals another intent's."""
    from laya.common import build_head
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "artifacts/checkpoints/R003-full-e2-8bit/tokenizer"
    )
    row = read_jsonl(DATA / "massive-km-k60-test.jsonl")[0]
    question = row["questions"]["intent"]
    labels = list(question["criteria"])
    q = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
    ids, markers, _ = build_head(tok, q, head_max_len)
    ends = markers[1:] + [len(ids) - 1]
    spans = [tuple(ids[a:b]) for a, b in zip(markers, ends)]
    counts = Counter(spans)
    return sorted(lab for lab, span in zip(labels, spans) if counts[span] > 1)


def r005():
    """Cross-variant comparisons, all paired over the same rows and model."""
    out = {"complexity": {}, "order": {}, "text": {}}
    collided = colliding_intents(256)
    out["k60_colliding_intents_at_head_max_len_256"] = collided
    for split in SPLITS:
        golds = [gold for _, gold in _golds("massive-km-k60", split)]
        for model in R005_MODELS:
            entry = {}
            for k in KS:
                if (f"massive-km-k{k}", split, model) in outcomes:
                    entry[f"k{k}"] = round(
                        accuracy_of(f"massive-km-k{k}", split, model), 4
                    )
            pair = [
                ("massive-km-k60", split, model),
                ("massive-km-k60-long", split, model),
            ]
            if all(key in outcomes for key in pair):
                default, long = (outcomes[key][0] for key in pair)
                entry["k60_long - k60"] = paired_bootstrap(default, long)
                for name, (flags, _) in zip(
                    ("k60", "k60_long"), (outcomes[key] for key in pair)
                ):
                    hit = [f for f, g in zip(flags, golds) if g in collided]
                    miss = [f for f, g in zip(flags, golds) if g not in collided]
                    entry[f"{name}_acc_gold_colliding"] = round(sum(hit) / len(hit), 4)
                    entry[f"{name}_acc_gold_distinct"] = round(sum(miss) / len(miss), 4)
                entry["n_gold_colliding"] = len(hit)
            if entry:
                out["complexity"].setdefault(split, {})[model] = entry

        for base_task, variants in TEXT_VARIANTS.items():
            for model in R005_MODELS:
                keys = [(f"{base_task}-{p}", split, model) for p in PERMS]
                if all(key in outcomes for key in keys):
                    accs = [accuracy_of(*key) for key in keys]
                    preds = list(zip(*(outcomes[key][1] for key in keys)))
                    flips = [int(len(set(p)) > 1) for p in preds]
                    out["order"].setdefault(base_task, {}).setdefault(split, {})[
                        model
                    ] = {
                        "accuracy_by_permutation": [round(a, 4) for a in accs],
                        "accuracy_range": round(max(accs) - min(accs), 4),
                        "flip_rate": round(sum(flips) / len(flips), 4),
                        "_flips": flips,
                    }
                ref = (base_task, split, model)
                if ref not in outcomes:
                    continue
                text = {"short": round(accuracy_of(*ref), 4)}
                for v in variants:
                    key = (f"{base_task}-{v}", split, model)
                    if key in outcomes:
                        text[v] = round(accuracy_of(*key), 4)
                        text[f"{v} - short"] = paired_bootstrap(
                            outcomes[ref][0], outcomes[key][0]
                        )
                out["text"].setdefault(base_task, {}).setdefault(split, {})[model] = (
                    text
                )
            order = out["order"].get(base_task, {}).get(split, {})
            base = order.get("control-base-maxlen512")
            for model in R005_MODELS[1:]:
                if base and model in order:
                    order[model]["flip_rate - base"] = paired_bootstrap(
                        base["_flips"], order[model]["_flips"]
                    )
    for splits in out["order"].values():
        for models in splits.values():
            for entry in models.values():
                del entry["_flips"]
    return out


def accuracy_of(task, split, model):
    flags = outcomes[task, split, model][0]
    return sum(flags) / len(flags)


def _golds(task, split):
    pattern = TASKS[task][0]
    rows = read_jsonl(DATA / pattern.format(split=split))
    return [next(iter(r["expected"].items())) for r in rows]


results = {"tasks": results, "R005": r005()}
OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

for task, splits in results["tasks"].items():
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
