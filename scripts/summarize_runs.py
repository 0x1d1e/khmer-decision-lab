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

# ---- New tasks for R007-R014 and R016 (non-flat run layouts) ----

# R007: frozen budget comparison on ASAG and MKQA entity choice
TASKS["asag-r007"] = ("r007-asag-test.jsonl", ["R007"])
TASKS["mkqa-entity-choice-r007"] = ("r007-mkqa-entity-choice-test.jsonl", ["R007"])

# R008: specialist controls on khmer-news and sib200-km
TASKS["khmer-news-r008-mt"] = ("khmer-news-test.jsonl", ["R008"])
TASKS["khmer-news-r008-specialist"] = ("khmer-news-test.jsonl", ["R008"])
TASKS["sib200-km-r008-mt"] = ("sib200-km-test.jsonl", ["R008"])
TASKS["sib200-km-r008-specialist"] = ("sib200-km-test.jsonl", ["R008"])

# R009: sib200-km specialist continuation
TASKS["sib200-km-r009"] = ("sib200-km-test.jsonl", ["R009"])

# R010: OCR robustness cascade on massive-km (4 conditions)
TASKS["massive-km-r010-clean"] = ("r010-massive-km-clean.jsonl", ["R010"])
TASKS["massive-km-r010-ocr_48px"] = ("r010-massive-km-ocr_48px.jsonl", ["R010"])
TASKS["massive-km-r010-ocr_24px"] = ("r010-massive-km-ocr_24px.jsonl", ["R010"])
TASKS["massive-km-r010-ocr_12px"] = ("r010-massive-km-ocr_12px.jsonl", ["R010"])

# R012: OCR continuation (clean / ocr_12px)
TASKS["massive-km-r012-clean"] = ("r010-massive-km-clean.jsonl", ["R012"])
TASKS["massive-km-r012-ocr12"] = ("r010-massive-km-ocr_12px.jsonl", ["R012"])

# R013: unseen intent transfer (test-seen / test-unseen)
TASKS["r013-test-seen"] = ("r013-unseen-intent-test-seen.jsonl", ["R013"])
TASKS["r013-test-unseen"] = ("r013-unseen-intent-test-unseen.jsonl", ["R013"])

# R014: SEA-NLI transfer across languages and difficulty
TASKS["sea-nli-r014-km-hard"] = ("sea-nli-km-hard-test.jsonl", ["R014"])
TASKS["sea-nli-r014-km-normal"] = ("sea-nli-km-normal-test.jsonl", ["R014"])
TASKS["sea-nli-r014-en-normal"] = ("sea-nli-en-normal-test.jsonl", ["R014"])
TASKS["sea-nli-r014-en-hard"] = ("sea-nli-en-hard-test.jsonl", ["R014"])
TASKS["sea-nli-r014-other"] = ("sea-nli-other-test.jsonl", ["R014"])

# R016: budget-matched specialist on sib200-km
TASKS["sib200-km-r016"] = ("sib200-km-test.jsonl", ["R016"])

# Run path templates relative to RUNS, with {split} and {model} placeholders.
# Tasks not listed here use the default: "{task}-{split}/{model}.json".
RUN_PATHS = {
    # R007
    "asag-r007": "r007-asag-frozen/{model}.json",
    "mkqa-entity-choice-r007": "r007-mkqa-entity-choice-frozen/{model}.json",
    # R008
    "khmer-news-r008-mt": "R008/khmer-news/{model}.json",
    "khmer-news-r008-specialist": "R008/khmer-news/{model}.json",
    "sib200-km-r008-mt": "R008/sib200-km/{model}.json",
    "sib200-km-r008-specialist": "R008/sib200-km/{model}.json",
    # R009
    "sib200-km-r009": "R009/sib200-km/{model}.json",
    # R010
    "massive-km-r010-clean": "R010/massive-km/clean/{model}.json",
    "massive-km-r010-ocr_48px": "R010/massive-km/ocr_48px/{model}.json",
    "massive-km-r010-ocr_24px": "R010/massive-km/ocr_24px/{model}.json",
    "massive-km-r010-ocr_12px": "R010/massive-km/ocr_12px/{model}.json",
    # R012
    "massive-km-r012-clean": "R012/clean/{model}/clean.json",
    "massive-km-r012-ocr12": "R012/ocr12/{model}/ocr_12px.json",
    # R013
    "r013-test-seen": "R013/evaluation/test-seen/{model}.json",
    "r013-test-unseen": "R013/evaluation/test-unseen/{model}.json",
    # R014
    "sea-nli-r014-km-hard": "R014/km-hard/{model}.json",
    "sea-nli-r014-km-normal": "R014/km-normal/{model}.json",
    "sea-nli-r014-en-normal": "R014/en-normal/{model}.json",
    "sea-nli-r014-en-hard": "R014/en-hard/{model}.json",
    "sea-nli-r014-other": "R014/other/{model}.json",
    # R016
    "sib200-km-r016": "R016/{model}.json",
}

# Variants to iterate per task (replaces SPLITS for custom-layout tasks).
TASK_VARIANTS = {
    "asag-r007": ("test",),
    "mkqa-entity-choice-r007": ("test",),
    "khmer-news-r008-mt": ("test",),
    "khmer-news-r008-specialist": ("test",),
    "sib200-km-r008-mt": ("test",),
    "sib200-km-r008-specialist": ("test",),
    "sib200-km-r009": ("test",),
    "massive-km-r010-clean": ("test",),
    "massive-km-r010-ocr_48px": ("test",),
    "massive-km-r010-ocr_24px": ("test",),
    "massive-km-r010-ocr_12px": ("test",),
    "massive-km-r012-clean": ("test",),
    "massive-km-r012-ocr12": ("test",),
    "r013-test-seen": ("test-seen",),
    "r013-test-unseen": ("test-unseen",),
    "sea-nli-r014-km-hard": ("test",),
    "sea-nli-r014-km-normal": ("test",),
    "sea-nli-r014-en-normal": ("test",),
    "sea-nli-r014-en-hard": ("test",),
    "sea-nli-r014-other": ("test",),
    "sib200-km-r016": ("test",),
}

# R011 (calibration), R015 (OCR CER), and R017 (latency) produce non-choice-
# accuracy metrics and do not fit this schema, so they are excluded.
R005_MODELS = ("control-base-maxlen512", "head-e2", "full-e2-8bit", "mt-s13")
SPLITS = ("validation", "test")
MODELS = {
    "multilingual": "base (R001)",
    "head-e2": "R002",
    "full-e2-8bit": "R003",
    "control-base-maxlen512": "base@512",
    "mt-s13": "R006 s13",
    "mt-s14": "R006 s14",
    # R008
    "multitask-s13": "R008 multitask s13",
    "multitask-s14": "R008 multitask s14",
    "sib200-specialist-s13": "R008 sib200-specialist s13",
    "sib200-specialist-s14": "R008 sib200-specialist s14",
    "khmer-news-specialist-s13": "R008 khmer-news-specialist s13",
    "khmer-news-specialist-s14": "R008 khmer-news-specialist s14",
    # R009
    "continued-s13": "R009 continued s13",
    "continued-s14": "R009 continued s14",
    # R013
    "base": "R013 base",
    "seen-only-s13": "R013 seen-only s13",
    "all-label-s13": "R013 all-label s13",
    "seen-only-s14": "R013 seen-only s14",
    "all-label-s14": "R013 all-label s14",
    # R016
    "R008-mt-s13": "R016 R008-mt-s13",
    "R008-mt-s14": "R016 R008-mt-s14",
    "R016-budget-sib-s13": "R016 budget-sib-s13",
    "R016-budget-sib-s14": "R016 budget-sib-s14",
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
    # R008
    ("multitask-s13", "multitask-s14"),
    ("sib200-specialist-s13", "sib200-specialist-s14"),
    ("khmer-news-specialist-s13", "khmer-news-specialist-s14"),
    # R009
    ("continued-s13", "continued-s14"),
    # R016
    ("R008-mt-s13", "R008-mt-s14"),
    ("R016-budget-sib-s13", "R016-budget-sib-s14"),
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
outcomes = {}  # (task, variant, model) -> (per-row correct, per-row prediction)
for task, (pattern, experiments) in TASKS.items():
    variants = TASK_VARIANTS.get(task, SPLITS)
    run_path_template = RUN_PATHS.get(task)
    for variant in variants:
        data_path = DATA / pattern.format(split=variant)
        if not data_path.exists():
            print(f"skip {task} {variant}: no data")
            continue
        if run_path_template is None and not (RUNS / f"{task}-{variant}").is_dir():
            print(f"skip {task} {variant}: no runs")
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
            if run_path_template:
                path = RUNS / run_path_template.format(split=variant, model=model)
            else:
                path = RUNS / f"{task}-{variant}" / f"{model}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            cases = report["cases"]
            report_data_path = report["config"].get("dataset", report["config"].get("prepared_file"))
            assert Path(report_data_path).name == data_path.name
            assert len(cases) == len(rows)
            for case, row in zip(cases, rows):
                assert case["expected"] == row["expected"][qid]
                assert set(case["answer"]["probabilities"]) == set(
                    row["questions"][qid]["criteria"]
                )
            split_entry["dataset_sha256"] = report["config"].get("dataset_sha256", report["config"].get("prepared_file_sha256"))

            entry = {"label": label}
            entry.update({k: round(report["overall"][k], 4) for k in KEPT})
            preds = [c["answer"]["choice"] for c in cases]
            entry["macro_f1"] = round(macro_f1(golds, preds), 4)
            entry["recall_by_label"] = label_recall(golds, preds)
            correct[model] = [int(c["correct"]) for c in cases]
            outcomes[task, variant, model] = (correct[model], preds)
            for key in ("peak_vram_mib", "measurement_protocol", "peak_vram_scope"):
                if key in report["config"]:
                    entry[key] = report["config"][key]
            if in_train is not None:
                seen = [c for c, flag in zip(cases, in_train) if flag]
                unseen = [c for c, flag in zip(cases, in_train) if not flag]
                entry["n_utterance_in_train"] = len(seen)
                entry["accuracy_utterance_in_train"] = round(accuracy(seen), 4)
                entry["accuracy_utterance_not_in_train"] = round(accuracy(unseen), 4)
            entry["report"] = str(path)
            split_entry["models"][model] = entry
        if not correct:
            print(f"skip {task} {variant}: no model reports")
            continue
        split_entry["accuracy_diffs"] = {
            f"{b} - {a}": paired_bootstrap(correct[a], correct[b])
            for a, b in PAIRS
            if a in correct and b in correct
        }
        results.setdefault(task, {})[variant] = split_entry


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
