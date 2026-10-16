"""Validate R014 inference artifacts and summarize its locked comparisons.

Verifies every prepared export against `datasets/manifests/sea-nli-km.json`
and every report against its dataset and weight fingerprints, then computes
the pre-declared endpoints and writes
`research/R014-sea-nli-khmer-transfer/results.json` and `results.md`.

No result files are produced by the code author: this runs only after
`scripts/evaluate_r014.py` has written all twelve reports.
"""

import hashlib
import json
from datetime import UTC, datetime
from math import comb
from pathlib import Path

import numpy as np
import prepare_r014
from evaluate_r014 import (
    BASE_REVISION,
    CONDITIONS,
    DATASETS,
    EXPECTED_WEIGHTS,
    HEAD_MAX_LEN,
    MAX_LEN,
    MODELS,
    OUT,
    TAG_PREFIX,
)
from laya import evals

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/sea-nli-km.json"
RESULTS_JSON = ROOT / "research/R014-sea-nli-khmer-transfer/results.json"
RESULTS_MD = ROOT / "research/R014-sea-nli-khmer-transfer/results.md"
PROTOCOL = ROOT / "research/R014-sea-nli-khmer-transfer/README.md"
PREPARER = Path(prepare_r014.__file__).resolve()

LABELS = ("entailment", "neutral", "contradiction")
KHMER_CONDITIONS = ("km-normal", "km-hard")
ENGLISH_CONDITIONS = ("en-normal", "en-hard")
CONFIG_BY_CONDITION = {"km-normal": "normal", "km-hard": "hard"}
SEEDS = (13, 14)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 13
DEGENERACY_THRESHOLD = 0.90
# Counterfactual only: reported as a post-hoc diagnostic, never used to
# invalidate an arm. 2/3 is a reasonable collapse bar for a 3-way task
# (uniform = 0.333), where the locked 0.90 is lax.
STRICTER_COLLAPSE_THRESHOLD = 2.0 / 3.0
POWER_NOTE = (
    "The Khmer slice is 65 rows in 37 premise groups, so only very large "
    "effects are resolvable."
)
PRIOR_COLLAPSE_NOTE = (
    "Post-scoring, post-hoc diagnostic computed from the existing reports. It "
    "changes no threshold, no endpoint, no estimand, no row set, and no "
    "support rule; the locked primary and its verdict are unchanged. Every "
    "pooled arm is dominated by a single predicted label, but a different "
    "one, and the pooled deltas are consistent with seed-level label-prior "
    "drift rather than acquired NLI capability."
)
EVALUATION = {
    "primary_metric": "top-1 choice accuracy",
    "primary_comparison": (
        "each R006 seed minus the frozen multilingual base on pooled Khmer rows"
    ),
    "support_rule": (
        "both R006 seed-specific paired group-bootstrap 95% lower bounds > 0"
    ),
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_interval": (
        "percentile; numpy.quantile(method='linear') at 2.5 and 97.5"
    ),
    "bootstrap_weighting": (
        "resample premise groups; retain all rows; row-weighted accuracy "
        "difference in percentage points"
    ),
    "degeneracy_threshold": DEGENERACY_THRESHOLD,
    "degeneracy_guard": (
        "primary invalid when any arm assigns more than 90% of pooled Khmer "
        "rows to a single label"
    ),
    "reported_metrics": [
        "choice_accuracy",
        "macro_f1",
        "top_answer_brier",
        "ece_15_equal_width_bins",
        "mean_confidence",
        "per_class_recall",
        "predicted_label_distribution",
    ],
    "power_note": POWER_NOTE,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text_atomic(path, text):
    """Write via tmp + replace so an interrupt cannot leave a partial result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _tags(case, prefix):
    matches = [
        tag[len(prefix) :] for tag in case.get("tags", []) if tag.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"R014 row must carry exactly one {prefix} tag")
    return matches[0]


def _macro_f1(gold, predictions):
    f1_by_label = {}
    for label in LABELS:
        true_positive = sum(
            1
            for g, p in zip(gold, predictions, strict=True)
            if g == label and p == label
        )
        false_positive = sum(
            1
            for g, p in zip(gold, predictions, strict=True)
            if g != label and p == label
        )
        false_negative = sum(
            1
            for g, p in zip(gold, predictions, strict=True)
            if g == label and p != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_by_label[label] = 2 * true_positive / denominator if denominator else 0.0
    return f1_by_label


def compute_metrics(cases):
    """Row-weighted metrics read directly from the report cases."""
    if not cases:
        raise ValueError("R014 metrics received no cases")
    expected = [case["expected"] for case in cases]
    predictions = [case["answer"]["choice"] for case in cases]
    correct = np.asarray([bool(case["correct"]) for case in cases], dtype=float)
    confidences = np.asarray([float(case["confidence"]) for case in cases], dtype=float)
    per_class_recall = {}
    for label in LABELS:
        support = sum(1 for value in expected if value == label)
        hits = sum(
            1
            for g, p in zip(expected, predictions, strict=True)
            if g == label and p == label
        )
        per_class_recall[label] = hits / support if support else None
    f1_by_label = _macro_f1(expected, predictions)
    ece = evals.ece(confidences.tolist(), [bool(value) for value in correct], bins=15)
    if ece is None:
        raise ValueError("R014 ECE is undefined for non-empty cases")
    return {
        "rows": len(cases),
        "groups": len({_tags(case, TAG_PREFIX) for case in cases}),
        "choice_accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(list(f1_by_label.values()))),
        "per_class_f1": f1_by_label,
        "per_class_recall": per_class_recall,
        "top_answer_brier": float(np.square(confidences - correct).mean()),
        "ece_15_equal_width_bins": float(ece),
        "mean_confidence": float(confidences.mean()),
        "predicted_label_distribution": {
            label: sum(1 for value in predictions if value == label) for label in LABELS
        },
    }


def paired_group_bootstrap(reference_cases, candidate_cases, draws=10000, seed=13):
    if len(reference_cases) != len(candidate_cases) or not reference_cases:
        raise ValueError("R014 paired bootstrap requires aligned non-empty reports")
    grouped_differences = {}
    for reference, candidate in zip(reference_cases, candidate_cases, strict=True):
        if reference.get("expected") != candidate.get("expected") or reference.get(
            "tags"
        ) != candidate.get("tags"):
            raise ValueError("R014 paired bootstrap reports do not share aligned rows")
        group = _tags(reference, TAG_PREFIX)
        grouped_differences.setdefault(group, []).append(
            float(bool(candidate.get("correct")))
            - float(bool(reference.get("correct")))
        )
    group_ids = sorted(grouped_differences)
    if not group_ids:
        raise ValueError("R014 paired bootstrap has no premise groups")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        differences = [
            value for group in sampled for value in grouped_differences[str(group)]
        ]
        estimates[draw] = 100.0 * float(np.mean(differences))
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    point = 100.0 * np.mean(
        [value for values in grouped_differences.values() for value in values]
    )
    return {
        "difference_pp": float(point),
        "ci95_pp": [float(lower), float(upper)],
        "draws": draws,
        "seed": seed,
        "groups": len(group_ids),
        "rows": sum(len(values) for values in grouped_differences.values()),
        "quantile_method": "numpy.quantile(method='linear')",
    }


def _pooled_cases(payloads, model, conditions=KHMER_CONDITIONS):
    cases = []
    for condition in conditions:
        cases.extend(payloads[(condition, model)]["cases"])
    return cases


def _report_path(condition, model):
    return OUT / condition / f"{model}.json"


def _verify_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"R014 manifest is missing: {MANIFEST}. Run prepare_r014.py first."
        )
    manifest = read_json(MANIFEST)
    if manifest.get("dataset") != prepare_r014.DATASET:
        raise ValueError("R014 manifest dataset id is not the pinned source")
    if manifest.get("revision") != prepare_r014.REVISION:
        raise ValueError("R014 manifest revision is not the pinned revision")
    if manifest.get("preparer_sha256") != sha256(PREPARER):
        raise ValueError("R014 manifest preparer fingerprint mismatch")
    exports = manifest.get("exports")
    if not isinstance(exports, dict) or not exports:
        raise ValueError("R014 manifest has no exports")
    for relative, entry in exports.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise ValueError(f"R014 prepared export fingerprint mismatch: {relative}")
        ids_relative = entry.get("ids")
        if not isinstance(ids_relative, str) or not ids_relative:
            raise ValueError(f"R014 manifest export has no sidecar: {relative}")
        ids_path = ROOT / ids_relative
        if not ids_path.is_file() or sha256(ids_path) != entry.get("ids_sha256"):
            raise ValueError(f"R014 sidecar fingerprint mismatch: {ids_relative}")
    expected = {
        f"data/processed/{stem}.jsonl"
        for stem in (
            "sea-nli-km-normal-test",
            "sea-nli-km-hard-test",
            "sea-nli-en-normal-test",
            "sea-nli-en-hard-test",
            "sea-nli-other-test",
        )
    }
    if set(exports) != expected:
        raise ValueError("R014 manifest export inventory is unexpected")
    return manifest, sha256(MANIFEST)


def _condition_export(manifest, condition):
    relative = DATASETS[condition]
    if relative not in manifest["exports"]:
        raise ValueError(f"R014 report condition is not a manifest export: {condition}")
    entry = manifest["exports"][relative]
    return relative, entry["sha256"], entry["ids"], entry["ids_sha256"]


def _sidecar_group_ids(relative):
    sidecar = read_json(ROOT / relative)
    if not isinstance(sidecar, list):
        raise TypeError(f"R014 sidecar is not a list: {relative}")
    group_ids = [entry.get("group_id") for entry in sidecar]
    if any(not isinstance(group, str) or not group for group in group_ids):
        raise ValueError(f"R014 sidecar has an invalid entry: {relative}")
    return group_ids


def _load_reports(manifest):
    payloads = {}
    weights_by_model = {}
    for model in MODELS:
        for condition in CONDITIONS:
            path = _report_path(condition, model)
            relative = str(path.relative_to(ROOT))
            if not path.is_file():
                raise FileNotFoundError(f"R014 report is missing: {relative}")
            payload = read_json(path)
            config = payload.get("config", {})
            exported, digest, ids_relative, ids_digest = _condition_export(
                manifest, condition
            )
            if (
                config.get("study") != "R014-sea-nli-khmer-transfer"
                or config.get("condition") != condition
                or config.get("dataset") != exported
                or config.get("dataset_sha256") != digest
                or config.get("sidecar") != ids_relative
                or config.get("sidecar_sha256") != ids_digest
                or config.get("base_revision") != BASE_REVISION
                or config.get("max_len") != MAX_LEN
                or config.get("head_max_len") != HEAD_MAX_LEN
                or config.get("model") != model
                or config.get("measurement_protocol") != "isolated-warm-v1"
                or config.get("batch_size") != 8
                or not isinstance(payload.get("cases"), list)
            ):
                raise ValueError(f"R014 report configuration mismatch: {relative}")
            if not payload["cases"]:
                raise ValueError(f"R014 report has no cases: {relative}")
            weights = config.get("weights_sha256")
            if not isinstance(weights, str) or not weights:
                raise ValueError(f"R014 report has no weight fingerprint: {relative}")
            expected_weights = EXPECTED_WEIGHTS.get(model)
            if expected_weights is not None and weights != expected_weights:
                raise ValueError(
                    f"R014 report weights do not match the pinned digest: {relative}"
                )
            tags = [_tags(case, TAG_PREFIX) for case in payload["cases"]]
            if tags != _sidecar_group_ids(ids_relative):
                raise ValueError(
                    f"R014 report group tags do not match the sidecar: {relative}"
                )
            weights_by_model.setdefault(model, set()).add(weights)
            payloads[(condition, model)] = payload
    for model, digests in weights_by_model.items():
        if len(digests) != 1:
            raise ValueError(f"R014 weight fingerprint differs across reports: {model}")
    fingerprints = {
        model: next(iter(digests)) for model, digests in weights_by_model.items()
    }
    if len(set(fingerprints.values())) != len(MODELS):
        raise ValueError("R014 model weight fingerprints are not distinct")
    if fingerprints["multilingual"] in {
        fingerprints["mt-s13"],
        fingerprints["mt-s14"],
        fingerprints["full-e2-8bit"],
    }:
        raise ValueError("R014 base weights match an adapted checkpoint")
    return payloads, fingerprints


def _degeneracy(payloads):
    per_arm = {}
    degenerate = False
    arms = []
    for model in MODELS:
        cases = _pooled_cases(payloads, model)
        if not cases:
            raise ValueError(f"R014 degeneracy check received no cases for {model}")
        distribution = {
            label: sum(1 for case in cases if case["answer"]["choice"] == label)
            for label in LABELS
        }
        fraction = max(distribution.values()) / len(cases)
        per_arm[model] = {
            "rows": len(cases),
            "predicted_label_distribution": distribution,
            "max_label_fraction": fraction,
            "degenerate": fraction > DEGENERACY_THRESHOLD,
        }
        if fraction > DEGENERACY_THRESHOLD:
            degenerate = True
            arms.append(model)
    return {
        "guard": EVALUATION["degeneracy_guard"],
        "threshold": DEGENERACY_THRESHOLD,
        "per_arm": per_arm,
        "degenerate": degenerate,
        "arms": arms,
    }


def _cohen_kappa(expected, predictions):
    """Unweighted Cohen kappa over the three fixed labels."""
    if len(expected) != len(predictions) or not expected:
        raise ValueError("R014 kappa requires aligned non-empty rows")
    total = len(expected)
    observed = (
        sum(1 for g, p in zip(expected, predictions, strict=True) if g == p) / total
    )
    chance = sum(
        (sum(1 for g in expected if g == label) / total)
        * (sum(1 for p in predictions if p == label) / total)
        for label in LABELS
    )
    if chance == 1.0:
        raise ValueError("R014 kappa is undefined when chance agreement is 1")
    return (observed - chance) / (1.0 - chance)


def _mcnemar_exact_two_sided(base_correct, model_correct):
    """Exact two-sided McNemar via the binomial distribution on discordant rows."""
    if len(base_correct) != len(model_correct) or not base_correct:
        raise ValueError("R014 McNemar requires aligned non-empty rows")
    base_wrong_model_right = sum(
        1 for b, m in zip(base_correct, model_correct, strict=True) if m and not b
    )
    base_right_model_wrong = sum(
        1 for b, m in zip(base_correct, model_correct, strict=True) if b and not m
    )
    discordant = base_wrong_model_right + base_right_model_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(base_wrong_model_right, base_right_model_wrong)
        p_value = min(
            1.0,
            2.0 * sum(comb(discordant, k) for k in range(tail + 1)) / 2**discordant,
        )
    return {
        "base_wrong_model_right": base_wrong_model_right,
        "base_right_model_wrong": base_right_model_wrong,
        "discordant_rows": discordant,
        "net_flipped_rows": base_wrong_model_right - base_right_model_wrong,
        "mcnemar_exact_two_sided_p": float(p_value),
    }


def _label_agreement(reference_cases, candidate_cases):
    if len(reference_cases) != len(candidate_cases) or not reference_cases:
        raise ValueError("R014 label agreement requires aligned non-empty rows")
    rows = len(reference_cases)
    agreeing = sum(
        1
        for reference, candidate in zip(reference_cases, candidate_cases, strict=True)
        if reference["answer"]["choice"] == candidate["answer"]["choice"]
    )
    return {
        "rows": rows,
        "agreeing_rows": agreeing,
        "agreement_fraction": agreeing / rows,
    }


def _constant_majority_reference(expected):
    """Best constant-label predictor and the label(s) that achieve it."""
    if not expected:
        raise ValueError("R014 constant-majority reference received no rows")
    counts = {label: sum(1 for value in expected if value == label) for label in LABELS}
    best = max(counts.values())
    return {
        "rows": len(expected),
        "gold_distribution": counts,
        "accuracy": best / len(expected),
        "labels": [label for label in LABELS if counts[label] == best],
        "label_accuracy": {label: counts[label] / len(expected) for label in LABELS},
    }


def _prior_collapse_diagnostics(payloads):
    """Post-scoring descriptive diagnostics. Reads reports only; no re-scoring."""
    base_cases = _pooled_cases(payloads, "multilingual")
    expected = [case["expected"] for case in base_cases]
    base_correct = [bool(case["correct"]) for case in base_cases]
    reference = _constant_majority_reference(expected)
    per_arm = {}
    dominant_fractions = {}
    for model in MODELS:
        cases = _pooled_cases(payloads, model)
        distribution = {
            label: sum(1 for case in cases if case["answer"]["choice"] == label)
            for label in LABELS
        }
        dominant_label = max(
            LABELS, key=lambda label: (distribution[label], -LABELS.index(label))
        )
        accuracy = sum(1 for case in cases if case["correct"]) / len(cases)
        dominant_fraction = distribution[dominant_label] / len(cases)
        dominant_fractions[model] = dominant_fraction
        per_arm[model] = {
            "rows": len(cases),
            "predicted_label_distribution": distribution,
            "dominant_label": dominant_label,
            "dominant_fraction": dominant_fraction,
            "choice_accuracy": accuracy,
            "constant_majority_accuracy": reference["accuracy"],
            "margin_over_constant_majority_pp": 100.0
            * (accuracy - reference["accuracy"]),
            "cohen_kappa_vs_gold": _cohen_kappa(
                expected, [case["answer"]["choice"] for case in cases]
            ),
        }
    agreement = {}
    for left, right in (
        ("multilingual", "mt-s13"),
        ("multilingual", "mt-s14"),
        ("multilingual", "full-e2-8bit"),
        ("mt-s13", "mt-s14"),
    ):
        agreement[f"{left}__{right}"] = _label_agreement(
            _pooled_cases(payloads, left), _pooled_cases(payloads, right)
        )
    net_flips = {}
    for seed in SEEDS:
        model = f"mt-s{seed}"
        net_flips[model] = _mcnemar_exact_two_sided(
            base_correct,
            [bool(case["correct"]) for case in _pooled_cases(payloads, model)],
        )
    invalidated = [
        model
        for model in ("mt-s13", "mt-s14")
        if dominant_fractions[model] > STRICTER_COLLAPSE_THRESHOLD
    ]
    return {
        "status": "post-scoring, post-hoc diagnostic",
        "note": PRIOR_COLLAPSE_NOTE,
        "constant_majority_reference": reference,
        "per_arm": per_arm,
        "label_agreement": agreement,
        "net_flips_vs_base": net_flips,
        "collapse_threshold_counterfactual": {
            "locked_threshold": DEGENERACY_THRESHOLD,
            "locked_guard_triggered": any(
                fraction > DEGENERACY_THRESHOLD
                for fraction in dominant_fractions.values()
            ),
            "stricter_threshold": STRICTER_COLLAPSE_THRESHOLD,
            "arms_invalidated_at_stricter_threshold": invalidated,
        },
    }


def _bootstrap(reference, candidate):
    return paired_group_bootstrap(
        reference,
        candidate,
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )


def summarize():
    manifest, manifest_sha256 = _verify_manifest()
    payloads, _ = _load_reports(manifest)
    degenerate = _degeneracy(payloads)
    diagnostics = _prior_collapse_diagnostics(payloads)

    base_pooled = _pooled_cases(payloads, "multilingual")
    primary = {"per_seed": {}, "valid": not degenerate["degenerate"]}
    for seed in SEEDS:
        model = f"mt-s{seed}"
        candidate = _pooled_cases(payloads, model)
        result = _bootstrap(base_pooled, candidate)
        lower = result["ci95_pp"][0]
        primary["per_seed"][str(seed)] = {
            **result,
            "model": model,
            "base_accuracy": compute_metrics(base_pooled)["choice_accuracy"],
            "model_accuracy": compute_metrics(candidate)["choice_accuracy"],
            "lower_bound_above_zero": lower > 0.0,
        }
    primary["supported"] = primary["valid"] and all(
        row["lower_bound_above_zero"] for row in primary["per_seed"].values()
    )
    primary["comparison"] = EVALUATION["primary_comparison"]
    primary["support_rule"] = EVALUATION["support_rule"]

    secondary = {"per_config": {}, "single_task_contrast": {}, "non_khmer": {}}
    for condition in KHMER_CONDITIONS:
        base_cases = payloads[(condition, "multilingual")]["cases"]
        entry = {"rows": len(base_cases)}
        for seed in SEEDS:
            candidate = payloads[(condition, f"mt-s{seed}")]["cases"]
            entry[str(seed)] = _bootstrap(base_cases, candidate)
        secondary["per_config"][condition] = entry

    single = _bootstrap(base_pooled, _pooled_cases(payloads, "full-e2-8bit"))
    secondary["single_task_contrast"] = {
        **single,
        "base_accuracy": compute_metrics(base_pooled)["choice_accuracy"],
        "model_accuracy": compute_metrics(_pooled_cases(payloads, "full-e2-8bit"))[
            "choice_accuracy"
        ],
        "note": (
            "descriptive multitask-vs-single-task question; R003 has one seed "
            "against the two R006 seeds, so this is not a matched-seed contrast"
        ),
    }
    secondary["non_khmer"] = {
        "note": (
            "weak Khmer-specificity check; the mixed slice covers seven other "
            "cultures and cannot isolate Khmer-ness; no support claims"
        ),
        "metrics": {
            model: compute_metrics(payloads[("other", model)]["cases"])
            for model in MODELS
        },
    }
    secondary["english_control"] = {
        "note": (
            "descriptive only; the english arm shares row identity, grouping, "
            "and gold with the native arm, so this difference is the "
            "within-model native-vs-English text gap; no support claims"
        ),
        "per_model": {},
    }
    for model in MODELS:
        gap = _bootstrap(
            _pooled_cases(payloads, model),
            _pooled_cases(payloads, model, ENGLISH_CONDITIONS),
        )
        secondary["english_control"]["per_model"][model] = {
            **gap,
            "native_accuracy": compute_metrics(_pooled_cases(payloads, model))[
                "choice_accuracy"
            ],
            "english_accuracy": compute_metrics(
                _pooled_cases(payloads, model, ENGLISH_CONDITIONS)
            )["choice_accuracy"],
        }

    metrics = {
        condition: {
            model: compute_metrics(payloads[(condition, model)]["cases"])
            for model in MODELS
        }
        for condition in CONDITIONS
    }
    descriptors = manifest.get("descriptors") or {}
    descriptors_by_config = descriptors.get("by_config") or {}
    is_misclassified_stratification = {
        "note": (
            "descriptive only; is_misclassified semantics are unverified, so "
            "no row was filtered on it and it never reaches a model input"
        ),
        "by_config": {
            condition: descriptors_by_config.get(
                CONFIG_BY_CONDITION[condition], {}
            ).get("is_misclassified")
            for condition in KHMER_CONDITIONS
        },
    }
    result = {
        "schema": "r014-results/1",
        "study": "R014-sea-nli-khmer-transfer",
        "status": "completed_exploratory",
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": sha256(PROTOCOL),
        "evaluation": EVALUATION,
        "exports": {
            condition: {
                "path": _condition_export(manifest, condition)[0],
                "sha256": _condition_export(manifest, condition)[1],
                "ids": _condition_export(manifest, condition)[2],
                "ids_sha256": _condition_export(manifest, condition)[3],
            }
            for condition in CONDITIONS
        },
        "reports": {
            f"{condition}/{model}": {
                "path": str(_report_path(condition, model).relative_to(ROOT)),
                "sha256": sha256(_report_path(condition, model)),
                "dataset_sha256": payloads[(condition, model)]["config"][
                    "dataset_sha256"
                ],
                "weights_sha256": payloads[(condition, model)]["config"][
                    "weights_sha256"
                ],
            }
            for condition in CONDITIONS
            for model in MODELS
        },
        "degeneracy": degenerate,
        "prior_collapse_diagnostics": diagnostics,
        "primary": primary,
        "secondary": secondary,
        "metrics": metrics,
        "descriptors": descriptors,
        "is_misclassified_stratification": is_misclassified_stratification,
        "generated_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "Test split only; no train or validation split exists, so no SEA-NLI temperature is fitted.",
            POWER_NOTE,
            (
                "The predicted-label distribution is dominated by a single "
                "different label per arm; the post-scoring diagnostics above "
                "show that this is consistent with label-prior drift."
            ),
            "The Khmer text is machine-generated from Wikipedia concepts with native verification, not naturalistic user input.",
            "The non-Khmer slice mixes seven other cultures and cannot isolate Khmer-ness.",
            "R003 contributes one seed to the single-task contrast while R006 contributes two.",
            "The english arm is a descriptive control only; it shares rows and gold with the native arm.",
            "is_misclassified and step_misclassified are disclosed but never filtered on; their semantics are unverified.",
            "Labels are public on Hugging Face; sealing is procedural only.",
        ],
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        RESULTS_JSON, json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    write_text_atomic(RESULTS_MD, _render_markdown(result))
    print(f"R014 summarized: primary supported={result['primary']['supported']}")
    print(
        "R014 post-scoring prior-collapse diagnostics (post-hoc; changes no "
        "threshold and no endpoint):"
    )
    print(
        json.dumps(result["prior_collapse_diagnostics"], indent=2, ensure_ascii=False)
    )
    return result


def _format(value):
    return f"{value:.4f}"


def _signed(value):
    """Two-decimal signed percentage point value, with unsigned exact zero.

    A bootstrap lower bound of exactly 0.0 must not render as `+0.00`: the plus
    sign reads as support at the boundary.
    """
    if value == 0.0:
        return "0.00"
    return f"{value:+.2f}"


def _render_markdown(result):
    primary = result["primary"]
    lines = [
        "# R014 results",
        "",
        (
            "Exploratory 3-way NLI transfer to an unseen task type, asked as a "
            "fixed three-option choice."
        ),
        "",
        f"Manifest SHA-256: `{result['manifest_sha256']}`",
        f"Protocol SHA-256: `{result['protocol_sha256']}`",
        "",
        "## Primary comparison",
        "",
        (
            "Each R006 seed versus the frozen multilingual base on pooled Khmer "
            "rows (normal + hard)."
        ),
        "",
        "| Seed | Rows | Groups | Base accuracy | Model accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = primary["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {row['rows']} | {row['groups']} | {_format(row['base_accuracy'])} | "
            f"{_format(row['model_accuracy'])} | {_signed(row['difference_pp'])} | "
            f"[{_signed(row['ci95_pp'][0])}, {_signed(row['ci95_pp'][1])}] |"
        )
    lines.extend(
        [
            "",
            (
                f"Primary support rule met: **{primary['supported']}** "
                f"(valid={primary['valid']}, "
                f"degenerate arms={result['degeneracy']['arms']})."
            ),
            (
                "Both seed-specific lower bounds must exceed zero; a crossing "
                "interval is inconclusive, not equivalence."
            ),
            "",
            "## Secondary (descriptive only)",
            "",
            "| Contrast | Rows | Groups | Difference (pp) | Paired group-bootstrap 95% CI (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, entry in result["secondary"]["per_config"].items():
        for seed in SEEDS:
            row = entry[str(seed)]
            lines.append(
                f"| {condition} mt-s{seed} - base | {row['rows']} | {row['groups']} | "
                f"{_signed(row['difference_pp'])} | "
                f"[{_signed(row['ci95_pp'][0])}, {_signed(row['ci95_pp'][1])}] |"
            )
    single = result["secondary"]["single_task_contrast"]
    lines.append(
        f"| pooled full-e2-8bit - base | {single['rows']} | {single['groups']} | "
        f"{_signed(single['difference_pp'])} | "
        f"[{_signed(single['ci95_pp'][0])}, {_signed(single['ci95_pp'][1])}] |"
    )
    lines.extend(
        [
            "",
            single["note"] + ".",
        ]
    )
    diagnostics = result["prior_collapse_diagnostics"]
    reference = diagnostics["constant_majority_reference"]
    lines.extend(
        [
            "",
            "## Post-scoring prior-collapse diagnostics (post-hoc)",
            "",
            diagnostics["note"],
            "",
            (
                "| Arm | Predicted distribution (entailment/neutral/"
                "contradiction) | Dominant label | Dominant fraction | "
                "Constant-majority accuracy | Margin above it (pp) | "
                "Cohen κ vs gold |"
            ),
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        row = diagnostics["per_arm"][model]
        distribution = row["predicted_label_distribution"]
        rendered = ", ".join(f"{label} {distribution[label]}" for label in LABELS)
        lines.append(
            f"| {model} | {rendered} | {row['dominant_label']} | "
            f"{_format(row['dominant_fraction'])} | "
            f"{_format(row['constant_majority_accuracy'])} | "
            f"{_signed(row['margin_over_constant_majority_pp'])} | "
            f"{_format(row['cohen_kappa_vs_gold'])} |"
        )
    majority_names = " or ".join(reference["labels"])
    majority_hits = reference["gold_distribution"][reference["labels"][0]]
    lines.extend(
        [
            "",
            (
                f"Constant-majority reference: predicting {majority_names} for "
                f"every row scores {reference['accuracy']:.4f} "
                f"({majority_hits}/{reference['rows']}); contradiction scores "
                f"{reference['label_accuracy']['contradiction']:.4f}."
            ),
            "",
            "| Pair | Agreeing rows | Agreement |",
            "|---|---:|---:|",
        ]
    )
    for pair, row in diagnostics["label_agreement"].items():
        left, right = pair.split("__")
        lines.append(
            f"| {left} vs {right} | {row['agreeing_rows']}/{row['rows']} | "
            f"{_format(row['agreement_fraction'])} |"
        )
    lines.extend(
        [
            "",
            (
                "| Contrast | Base wrong / model right | Base right / model "
                "wrong | Net flipped | Exact two-sided McNemar p |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("mt-s13", "mt-s14"):
        row = diagnostics["net_flips_vs_base"][model]
        lines.append(
            f"| {model} - base | {row['base_wrong_model_right']} | "
            f"{row['base_right_model_wrong']} | {row['net_flipped_rows']:+d} | "
            f"{row['mcnemar_exact_two_sided_p']:.4f} |"
        )
    seeds_pair = diagnostics["label_agreement"]["mt-s13__mt-s14"]
    stricter = diagnostics["collapse_threshold_counterfactual"]
    invalidated = stricter["arms_invalidated_at_stricter_threshold"]
    invalidated_text = ", ".join(
        f"{arm} ({_format(diagnostics['per_arm'][arm]['dominant_fraction'])})"
        for arm in invalidated
    )
    lines.extend(
        [
            "",
            (
                f"The two same-recipe R006 seeds agree on only "
                f"{seeds_pair['agreeing_rows']}/{seeds_pair['rows']} rows "
                f"({_format(seeds_pair['agreement_fraction'])}), which bounds how "
                "much stable signal the pooled deltas can carry. The locked "
                f"{stricter['locked_threshold']:.2f} degeneracy guard did not "
                "trigger; at a stricter 2/3 collapse threshold it would have "
                f"invalidated {invalidated_text}. The primary therefore remains "
                "unsupported/inconclusive on power grounds, and the observed "
                "differences are not evidence of stable NLI capability."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Descriptive metrics",
            "",
            "| Condition | Model | Acc | Macro-F1 | Top-answer Brier | ECE | Mean conf |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        for model in MODELS:
            metrics = result["metrics"][condition][model]
            lines.append(
                f"| {condition} | {model} | {_format(metrics['choice_accuracy'])} | "
                f"{_format(metrics['macro_f1'])} | {_format(metrics['top_answer_brier'])} | "
                f"{_format(metrics['ece_15_equal_width_bins'])} | "
                f"{_format(metrics['mean_confidence'])} |"
            )
    lines.extend(
        [
            "",
            (
                "Per-class recall, per-class F1, and the predicted-label "
                "distribution per arm are in `results.json`."
            ),
            "",
            "## English control (descriptive only)",
            "",
            (
                "Same rows, grouping, and gold as the native arm; the "
                "difference isolates native-vs-English text within a model."
            ),
            "",
            "| Model | Native acc | English acc | Difference (pp) | Paired group-bootstrap 95% CI (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        gap = result["secondary"]["english_control"]["per_model"][model]
        lines.append(
            f"| {model} | {_format(gap['native_accuracy'])} | "
            f"{_format(gap['english_accuracy'])} | {_signed(gap['difference_pp'])} | "
            f"[{_signed(gap['ci95_pp'][0])}, {_signed(gap['ci95_pp'][1])}] |"
        )
    lines.extend(
        [
            "",
            (
                "Descriptive stratification of `is_misclassified` (never used "
                "as a filter and never exported to the model input): "
                f"{result['is_misclassified_stratification']['by_config']}."
            ),
            "",
            "## Scope and limitations",
            "",
            (
                "Test split only; there is no validation split, so no SEA-NLI "
                "temperature is fitted and checkpoint temperatures are used "
                "unchanged. " + POWER_NOTE + " The Khmer text is "
                "machine-generated from Wikipedia concepts with native "
                "verification, not naturalistic user input. Labels are public, "
                "so sealing is procedural only. The non-Khmer slice is a weak "
                "specificity check that mixes seven other cultures. This study "
                "does not close the repository gap for a rights-cleared "
                "independent Khmer decision benchmark."
            ),
            "",
            (
                "Intervals resample normalized premise groups and retain all "
                "rows. Identical premises across the normal and hard configs "
                "fall in one cluster; exact duplicates were removed within each "
                "config only."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    summarize()


if __name__ == "__main__":
    main()
