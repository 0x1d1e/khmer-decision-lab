"""Summarize R016 budget-matched SIB-200 specialist results.

Usage: uv run python scripts/summarize_r016.py

Reads the evaluation reports written by scripts/evaluate_r016_budget_specialist.py
plus the per-seed training telemetry/training reports, recomputes the single
pre-registered endpoint

    D_s = acc(SP@budget, seed s) - acc(MT, seed s)

on the locked 204-row SIB-200 test split, with paired group-bootstrap 95% CIs
(10,000 draws, seed 13; group = exact normalized input state), and writes
artifacts/runs/R016/summary.json.

This study cannot resolve R008's non-inferiority question: at n=204 the
minimum detectable effect is ~5.4 pp and a pass requires a point estimate
above ~+3.39 pp, so all results here are exploratory.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r008_identity import state_group_id
from summarize_r008 import paired_group_bootstrap

RUNS = ROOT / "artifacts/runs/R016"
CHECKPOINTS = ROOT / "artifacts/checkpoints"
SEEDS = (13, 14)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 13
TEST_ROWS = 204
TARGET_UPDATES = 1060


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_report(model):
    return read_json(RUNS / f"{model}.json")


def check_case(report, rows, model):
    cases = report.get("cases", [])
    if len(cases) != len(rows):
        raise ValueError(
            f"{model} report case count mismatch: {len(cases)} != {len(rows)}"
        )
    confidence, correct = [], []
    for case, row in zip(cases, rows):
        questions = row.get("questions", {})
        expected = row.get("expected", {})
        if len(questions) != 1 or set(expected) != set(questions):
            raise ValueError(f"{model} report row must label one question")
        qid = next(iter(questions))
        gold = expected[qid]
        answer = case.get("answer", {})
        choice = answer.get("choice")
        case_confidence = case.get("confidence")
        probabilities = answer.get("probabilities", {})
        options = questions[qid]["criteria"]
        if (
            case.get("qid") != qid
            or case.get("model") != model
            or case.get("expected") != gold
            or answer.get("type") != "choice"
            or choice not in options
            or set(probabilities) != set(options)
            or case.get("correct") != (choice == gold)
            or not isinstance(case_confidence, (int, float))
            or not math.isfinite(case_confidence)
            or not 0 <= case_confidence <= 1
            or abs(case_confidence - max(probabilities.values())) > 0.0001 + 1e-12
        ):
            raise ValueError(f"{model} report case alignment mismatch")
        confidence.append(float(case_confidence))
        correct.append(choice == gold)
    accuracy = sum(correct) / len(correct)
    if abs(accuracy - report["overall"]["choice_accuracy"]) > 1e-12:
        raise ValueError(f"{model} report accuracy does not recompute")
    return confidence, correct


def bootstrap_ci(mt_correct, sp_correct, groups, samples, seed):
    distribution = paired_group_bootstrap(
        mt_correct, sp_correct, groups, samples=samples, seed=seed
    )
    # paired_group_bootstrap returns mt - sp; R016's endpoint is sp - mt.
    diff = 100 * (sum(sp_correct) - sum(mt_correct)) / len(sp_correct)
    low, high = (100 * np.quantile(-distribution, [0.025, 0.975])).tolist()
    return diff, [float(low), float(high)]


def telemetry_for(seed):
    return read_json(
        CHECKPOINTS / f"R016-budget-sib-s{seed}" / "training_telemetry.json"
    )


def training_report_for(seed):
    return read_json(
        CHECKPOINTS / f"R016-budget-sib-s{seed}" / "r016_training_report.json"
    )


def summarize():
    rows_path = ROOT / "data/processed/sib200-km-test.jsonl"
    raw_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_rows) != TEST_ROWS:
        raise ValueError(f"SIB-200 test has {len(raw_rows)} rows, expected {TEST_ROWS}")
    groups = [state_group_id(row) for row in raw_rows]
    if len(set(groups)) != TEST_ROWS:
        raise ValueError("expected 204 unique SIB-200 test state groups")

    per_seed = {}
    interval_half_widths = {}
    for seed in SEEDS:
        specialist_model = f"R016-budget-sib-s{seed}"
        multitask_model = f"R008-mt-s{seed}"
        sp_report = read_report(specialist_model)
        mt_report = read_report(multitask_model)
        _, sp_correct = check_case(sp_report, raw_rows, specialist_model)
        _, mt_correct = check_case(mt_report, raw_rows, multitask_model)

        sp_acc = sp_report["overall"]["choice_accuracy"]
        mt_acc = mt_report["overall"]["choice_accuracy"]
        difference, ci = bootstrap_ci(
            mt_correct, sp_correct, groups, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
        )
        interval_half_widths[str(seed)] = round((ci[1] - ci[0]) / 2, 2)

        telemetry = telemetry_for(seed)
        training_report = training_report_for(seed)
        epoch_losses = training_report.get("epoch_loss", [])
        comparison = training_report.get("comparison", {}) or {}
        validation_before = (comparison.get("before") or {}).get("loss")
        validation_after = (comparison.get("after") or {}).get("loss")

        per_seed[str(seed)] = {
            "specialist_model": specialist_model,
            "multitask_model": multitask_model,
            "specialist_accuracy": sp_acc,
            "multitask_accuracy": mt_acc,
            "difference_percentage_points": difference,
            "paired_group_bootstrap_ci95_percentage_points": ci,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "test_rows": TEST_ROWS,
            "test_state_groups": len(set(groups)),
            "optimizer_updates": {
                "attempts": telemetry.get("optimizer_attempts"),
                "successful": telemetry.get("successful_updates"),
                "skipped": telemetry.get("skipped_updates"),
                "scheduler_steps": telemetry.get("scheduler_steps"),
                "target": TARGET_UPDATES,
                "per_epoch": len(epoch_losses)
                and round(telemetry.get("optimizer_attempts", 0) / len(epoch_losses)),
            },
            "training_seconds": telemetry.get("training_seconds"),
            "peak_allocated_bytes": telemetry.get("peak_allocated_bytes"),
            "epoch_mean_losses": [round(float(loss), 6) for loss in epoch_losses],
            "validation_loss_before": None
            if validation_before is None
            else round(float(validation_before), 6),
            "validation_loss_after": None
            if validation_after is None
            else round(float(validation_after), 6),
            "telemetry": str(
                (
                    CHECKPOINTS / f"R016-budget-sib-s{seed}" / "training_telemetry.json"
                ).relative_to(ROOT)
            ),
            "training_report": str(
                (
                    CHECKPOINTS
                    / f"R016-budget-sib-s{seed}"
                    / "r016_training_report.json"
                ).relative_to(ROOT)
            ),
            "specialist_report": str(
                (RUNS / f"{specialist_model}.json").relative_to(ROOT)
            ),
            "multitask_report": str(
                (RUNS / f"{multitask_model}.json").relative_to(ROOT)
            ),
        }

    detectable = {
        seed: round(-2.0 + interval_half_widths[str(seed)], 2) for seed in SEEDS
    }

    summary = {
        "study": "R016-budget-matched-specialist",
        "status": "scored_exploratory",
        "question": (
            "Does matching the multitask arm's ~1,060 optimizer-update budget remove "
            "the SIB-200 specialist's accuracy gap versus same-seed multitask?"
        ),
        "endpoint": "D_s = acc(SP@budget, seed s) - acc(MT, seed s) on the SIB-200 test split",
        "test": {
            "file": "data/processed/sib200-km-test.jsonl",
            "rows": TEST_ROWS,
            "unique_state_groups": TEST_ROWS,
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 2.5/97.5",
            "grouping": "sha256 of NFC-normalized, whitespace-collapsed state JSON",
        },
        "per_seed": per_seed,
        "interpretation": {
            "caveat": (
                "The SIB-200 test split has only 204 rows. R008's -2 pp "
                "non-inferiority gate requires a point estimate above +3.39 pp "
                "against a ~5.4 pp minimum detectable effect at n=204, so the "
                "gate is untestable here and this study cannot resolve R008's "
                "non-inferiority question. All conclusions are exploratory and "
                "conditional on these two seeds and this test set."
            ),
            "minimum_detectable_effect_percentage_points": "~5.4",
            "point_estimate_required_to_pass_noninferiority_percentage_points": detectable,
        },
    }
    output = RUNS / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return summary


def main():
    summary = summarize()
    for seed in SEEDS:
        entry = summary["per_seed"][str(seed)]
        ci = entry["paired_group_bootstrap_ci95_percentage_points"]
        print(
            f"seed {seed}: SP {entry['specialist_accuracy'] * 100:.2f}% "
            f"MT {entry['multitask_accuracy'] * 100:.2f}% "
            f"D_s = {entry['difference_percentage_points']:+.2f} pp "
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp "
            f"updates={entry['optimizer_updates']['successful']}/"
            f"{entry['optimizer_updates']['attempts']}"
        )
    print(f"Wrote {RUNS / 'summary.json'}")


if __name__ == "__main__":
    main()
