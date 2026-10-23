"""Summarize the R003 seed-14 extension: paired comparisons on MASSIVE.

Usage: uv run --frozen python scripts/summarize_r003_seed14.py

Reads existing laya-evals reports only; no model is loaded and nothing is
scored. Writes research/R003-full-finetune/results-seed14.json.

Grouping follows the established convention (scripts/summarize_r008.py):
groups are sha256 of the NFC-normalized, whitespace-collapsed full state JSON,
and the bootstrap resamples groups while keeping every row, so accuracy stays
row-weighted. 10,000 draws, seed 13.
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/runs"
DATA = ROOT / "data/processed"
OUTPUT = ROOT / "research/R003-full-finetune/results-seed14.json"
SPLITS = ("validation", "test")
MODELS = {
    "full-e2-8bit": "R003 s13",
    "full-e2-8bit-s14": "R003 s14",
    "mt-s13": "R006 s13",
    "mt-s14": "R006 s14",
}
# Seed 13 was originally scored before the isolated-warm-v1 protocol metadata
# existed. Its re-score under the current protocol reproduced the same
# accuracies, so it is used when present and both paths are recorded.
PROTOCOL_MATCHED = {"full-e2-8bit"}
# (reference, candidate): the reported difference is candidate - reference.
PAIRS = (
    ("full-e2-8bit", "full-e2-8bit-s14"),
    ("full-e2-8bit", "mt-s13"),
    ("full-e2-8bit", "mt-s14"),
    ("full-e2-8bit-s14", "mt-s13"),
    ("full-e2-8bit-s14", "mt-s14"),
    ("mt-s13", "mt-s14"),
)
KEPT = (
    "choice_accuracy",
    "mean_confidence",
    "ece",
    "brier",
    "aurc",
    "selective_accuracy@50",
    "selective_accuracy@80",
)
DRAWS = 10_000
SEED = 13


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def normalize_value(value):
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, dict):
        return {key: normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


def state_group_id(row):
    import hashlib

    key = json.dumps(
        normalize_value(row["state"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def macro_f1(golds, predictions):
    true_positive, false_positive, false_negative = Counter(), Counter(), Counter()
    for gold, prediction in zip(golds, predictions, strict=True):
        if gold == prediction:
            true_positive[gold] += 1
        else:
            false_positive[prediction] += 1
            false_negative[gold] += 1
    scores = []
    for label in sorted(set(golds)):
        denominator = (
            2 * true_positive[label] + false_positive[label] + false_negative[label]
        )
        scores.append(2 * true_positive[label] / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def recall_by_label(golds, predictions):
    totals, hits = Counter(golds), Counter()
    for gold, prediction in zip(golds, predictions, strict=True):
        hits[gold] += gold == prediction
    return {label: round(hits[label] / totals[label], 4) for label in sorted(totals)}


def paired_group_bootstrap(reference, candidate, groups, draws=DRAWS, seed=SEED):
    """Row-weighted accuracy difference candidate - reference, groups resampled."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not (len(reference) == len(candidate) == len(groups)):
        raise ValueError("Paired bootstrap requires aligned vectors")
    unique = sorted(set(groups))
    if not unique:
        raise ValueError("Paired bootstrap requires at least one group")
    lookup = {group: index for index, group in enumerate(unique)}
    row_groups = np.asarray([lookup[group] for group in groups])
    sizes = np.bincount(row_groups, minlength=len(unique)).astype(float)
    totals = np.zeros((len(unique), 2), dtype=float)
    np.add.at(totals, row_groups, np.stack([reference, candidate], axis=1))

    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0, draws, 256):
        batch = min(256, draws - start)
        index = rng.integers(0, len(unique), (batch, len(unique)))
        sampled = totals[index].sum(axis=1)
        sampled_rows = sizes[index].sum(axis=1)
        accuracies = sampled / sampled_rows[:, None]
        differences.append(accuracies[:, 1] - accuracies[:, 0])
    estimates = np.concatenate(differences)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    point = (candidate.mean() - reference.mean()) * 100
    return {
        "difference_pp": round(float(point), 4),
        "ci95_pp": [round(float(100 * lower), 4), round(float(100 * upper), 4)],
        "draws": draws,
        "seed": seed,
        "groups": len(unique),
        "rows": len(reference),
        "quantile_method": "numpy.quantile(method='linear')",
    }


def main():
    results = {"comparisons": {}, "splits": {}}
    for split in SPLITS:
        rows = read_jsonl(DATA / f"massive-km-{split}-full.jsonl")
        (question_id,) = rows[0]["expected"]
        golds = [row["expected"][question_id] for row in rows]
        groups = [state_group_id(row) for row in rows]
        entry = {
            "n": len(rows),
            "groups": len(set(groups)),
            "dataset": f"data/processed/massive-km-{split}-full.jsonl",
            "models": {},
        }
        correct = {}
        for model, label in MODELS.items():
            canonical = RUNS / f"massive-km-{split}" / f"{model}.json"
            recheck = RUNS / f"massive-km-{split}-s13-isolated" / f"{model}.json"
            report_path = (
                recheck if model in PROTOCOL_MATCHED and recheck.exists() else canonical
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cases = report["cases"]
            if len(cases) != len(rows):
                raise ValueError(f"{model}/{split} report is not aligned with rows")
            predictions = [case["answer"]["choice"] for case in cases]
            flags = [int(case["correct"]) for case in cases]
            correct[model] = flags
            item = {"label": label}
            item.update(
                {
                    key: round(report["overall"][key], 4)
                    for key in KEPT
                    if key in report["overall"]
                }
            )
            item["macro_f1"] = round(macro_f1(golds, predictions), 4)
            item["recall_by_label"] = recall_by_label(golds, predictions)
            for key in ("measurement_protocol", "peak_vram_mib", "weights_sha256"):
                if key in report["config"]:
                    item[key] = report["config"][key]
            item["report"] = str(
                Path("artifacts/runs") / f"massive-km-{split}" / f"{model}.json"
            )
            if model in PROTOCOL_MATCHED:
                item["protocol_matched_report"] = str(
                    Path("artifacts/runs")
                    / f"massive-km-{split}-s13-isolated"
                    / f"{model}.json"
                )
                if canonical.exists():
                    historical = json.loads(canonical.read_text(encoding="utf-8"))
                    item["historical_report"] = str(
                        Path("artifacts/runs") / f"massive-km-{split}" / f"{model}.json"
                    )
                    item["historical_choice_accuracy"] = round(
                        historical["overall"]["choice_accuracy"], 4
                    )
                    item["historical_report_has_isolated_protocol"] = (
                        "measurement_protocol" in historical["config"]
                    )
            entry["models"][model] = item
        entry["comparisons"] = {
            f"{candidate} - {reference}": paired_group_bootstrap(
                correct[reference], correct[candidate], groups
            )
            for reference, candidate in PAIRS
            if reference in correct and candidate in correct
        }
        results["splits"][split] = entry

    results["comparisons"]["seed_spread_test_s14_minus_s13"] = results["splits"][
        "test"
    ]["comparisons"]["full-e2-8bit-s14 - full-e2-8bit"]

    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    for split, entry in results["splits"].items():
        print(f"\nMASSIVE {split} (n={entry['n']}, groups={entry['groups']})")
        for model, item in entry["models"].items():
            print(
                f"  {item['label']:<10} acc {item['choice_accuracy']:.4f}"
                f"  macroF1 {item['macro_f1']:.4f}  ece {item['ece']:.4f}"
                f"  brier {item['brier']:.4f}"
            )
        for pair, value in entry["comparisons"].items():
            print(f"  {pair}: {value['difference_pp']:+.2f} pp CI95 {value['ci95_pp']}")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
