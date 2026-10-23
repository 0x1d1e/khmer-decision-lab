"""Shared CER, edit-op and bootstrap helpers for the R015 document-image study."""

import re
import unicodedata

import numpy as np
import regex

BOOTSTRAP_SEED = 20260101
BOOTSTRAP_DRAWS = 10_000
HARNESS_REFERENCE_CER = 0.0526
HARNESS_TOLERANCE_PP = 1.0


def normalize_for_cer(text: str) -> str:
    """NFC normalization and whitespace collapse, matching R010."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def graphemes(text: str) -> list[str]:
    return regex.findall(r"\X", normalize_for_cer(text))


def edit_distance_with_ops(source: list, target: list) -> dict:
    """Levenshtein distance with deletion/substitution/insertion counts.

    Ties resolve deletion-first, then substitution, then insertion, so the
    composition is deterministic for identical inputs.
    """
    m, n = len(source), len(target)
    dp = [[(0, 0, 0, 0)] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = (i, i, 0, 0)
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            deletion = dp[i - 1][j]
            substitution = dp[i - 1][j - 1]
            insertion = dp[i][j - 1]
            dp[i][j] = min(
                (
                    (deletion[0] + 1, deletion[1] + 1, deletion[2], deletion[3]),
                    (
                        substitution[0] + 1,
                        substitution[1],
                        substitution[2] + 1,
                        substitution[3],
                    ),
                    (
                        insertion[0] + 1,
                        insertion[1],
                        insertion[2],
                        insertion[3] + 1,
                    ),
                ),
                key=lambda value: value[0],
            )
    distance, deletions, substitutions, insertions = dp[m][n]
    return {
        "distance": distance,
        "deletions": deletions,
        "substitutions": substitutions,
        "insertions": insertions,
    }


def cer_record(reference: str, hypothesis: str) -> dict:
    """Codepoint and grapheme-cluster CER plus edit-op composition."""
    ref_norm = normalize_for_cer(reference)
    hyp_norm = normalize_for_cer(hypothesis)
    ref_gr = graphemes(reference)
    hyp_gr = graphemes(hypothesis)

    cp_ops = edit_distance_with_ops(list(ref_norm), list(hyp_norm))
    gr_ops = edit_distance_with_ops(ref_gr, hyp_gr)

    return {
        "cer_cp": cp_ops["distance"] / len(ref_norm) if ref_norm else 0.0,
        "cer_gr": gr_ops["distance"] / len(ref_gr) if ref_gr else 0.0,
        "ref_codepoints": len(ref_norm),
        "ref_graphemes": len(ref_gr),
        "cp_edits": cp_ops["distance"],
        "gr_edits": gr_ops["distance"],
        "cp_deletions": cp_ops["deletions"],
        "cp_substitutions": cp_ops["substitutions"],
        "cp_insertions": cp_ops["insertions"],
        "gr_deletions": gr_ops["deletions"],
        "gr_substitutions": gr_ops["substitutions"],
        "gr_insertions": gr_ops["insertions"],
        "exact": ref_norm == hyp_norm,
        "blank": hyp_norm == "",
    }


def group_indices(rows: list[dict], key: str = "decree_group") -> list[list[int]]:
    """Row indices grouped by decree group, in sorted group order."""
    order = sorted({row[key] for row in rows})
    buckets = {value: [] for value in order}
    for index, row in enumerate(rows):
        buckets[row[key]].append(index)
    return [buckets[value] for value in order]


def decree_grouped_bootstrap(
    values: np.ndarray,
    groups: list[list[int]],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile interval from resampling decree groups, not lines."""
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    sums = np.array([values[index].sum() for index in groups])
    sizes = np.array([len(index) for index in groups])
    samples = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, n_groups, size=n_groups)
        samples[draw] = sums[picked].sum() / sizes[picked].sum()
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def condition_metrics(rows: list[dict], key: str) -> dict:
    """Aggregate CER, edit composition and exact/blank counts for one condition."""
    ref_cp = sum(row[f"edit_ops_{key}"]["ref_codepoints"] for row in rows)
    ref_gr = sum(row[f"edit_ops_{key}"]["ref_graphemes"] for row in rows)
    edits_cp = sum(row[f"edit_ops_{key}"]["cp_edits"] for row in rows)
    edits_gr = sum(row[f"edit_ops_{key}"]["gr_edits"] for row in rows)
    exact = sum(1 for row in rows if row[f"edit_ops_{key}"]["exact"])
    return {
        "codepoint_cer": edits_cp / ref_cp,
        "grapheme_cer": edits_gr / ref_gr,
        "cp_edits": edits_cp,
        "gr_edits": edits_gr,
        "ref_codepoints": ref_cp,
        "ref_graphemes": ref_gr,
        "exact_match": exact,
        "exact_rate": exact / len(rows),
        "blank_outputs": sum(1 for row in rows if row[f"edit_ops_{key}"]["blank"]),
        "cp_deletions": sum(row[f"edit_ops_{key}"]["cp_deletions"] for row in rows),
        "cp_substitutions": sum(
            row[f"edit_ops_{key}"]["cp_substitutions"] for row in rows
        ),
        "cp_insertions": sum(row[f"edit_ops_{key}"]["cp_insertions"] for row in rows),
    }


def build_summary(rows: list[dict], preparation: dict) -> dict:
    """Compute the paired summary from per-line rows."""
    real = condition_metrics(rows, "real")
    synth = condition_metrics(rows, "synth")
    groups = group_indices(rows)

    delta_cp = np.array([row["cer_real_cp"] - row["cer_synth_cp"] for row in rows])
    delta_gr = np.array([row["cer_real_gr"] - row["cer_synth_gr"] for row in rows])
    line_mean_cp = float(delta_cp.mean())
    line_mean_gr = float(delta_gr.mean())
    macro_cp = real["codepoint_cer"] - synth["codepoint_cer"]
    macro_gr = real["grapheme_cer"] - synth["grapheme_cer"]

    line_ci_cp = decree_grouped_bootstrap(delta_cp, groups)
    line_ci_gr = decree_grouped_bootstrap(delta_gr, groups)

    # Macro delta interval: resample decree groups, recompute each condition's
    # ratio-of-sums CER, and take the difference.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    real_cp = np.array([row["edit_ops_real"]["cp_edits"] for row in rows])
    real_gr = np.array([row["edit_ops_real"]["gr_edits"] for row in rows])
    synth_cp = np.array([row["edit_ops_synth"]["cp_edits"] for row in rows])
    synth_gr = np.array([row["edit_ops_synth"]["gr_edits"] for row in rows])
    ref_cp = np.array([row["edit_ops_real"]["ref_codepoints"] for row in rows])
    ref_gr = np.array([row["edit_ops_real"]["ref_graphemes"] for row in rows])
    g_real_cp = np.array([real_cp[index].sum() for index in groups])
    g_real_gr = np.array([real_gr[index].sum() for index in groups])
    g_synth_cp = np.array([synth_cp[index].sum() for index in groups])
    g_synth_gr = np.array([synth_gr[index].sum() for index in groups])
    g_ref_cp = np.array([ref_cp[index].sum() for index in groups])
    g_ref_gr = np.array([ref_gr[index].sum() for index in groups])
    n_groups = len(groups)
    macro_cp_samples = np.empty(BOOTSTRAP_DRAWS)
    macro_gr_samples = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, n_groups, size=n_groups)
        macro_cp_samples[draw] = (
            g_real_cp[picked].sum() - g_synth_cp[picked].sum()
        ) / g_ref_cp[picked].sum()
        macro_gr_samples[draw] = (
            g_real_gr[picked].sum() - g_synth_gr[picked].sum()
        ) / g_ref_gr[picked].sum()
    macro_ci_cp = np.percentile(macro_cp_samples, [2.5, 97.5])
    macro_ci_gr = np.percentile(macro_gr_samples, [2.5, 97.5])

    per_group = []
    for index in groups:
        subset = [rows[i] for i in index]
        group_real = condition_metrics(subset, "real")
        group_synth = condition_metrics(subset, "synth")
        per_group.append(
            {
                "decree_group": subset[0]["decree_group"],
                "n_lines": len(subset),
                "real_cp_cer": round(group_real["codepoint_cer"], 6),
                "synth_cp_cer": round(group_synth["codepoint_cer"], 6),
                "delta_cp_pp": round(
                    (group_real["codepoint_cer"] - group_synth["codepoint_cer"]) * 100,
                    2,
                ),
            }
        )
    per_group.sort(key=lambda entry: -entry["real_cp_cer"])

    return {
        "study": "R015-document-image-ocr",
        "n_lines": len(rows),
        "n_pages": preparation["test_pages_filtered"],
        "n_decree_groups": n_groups,
        "real": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in real.items()
        },
        "synthetic": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in synth.items()
        },
        "delta": {
            "line_mean_codepoint_pp": round(line_mean_cp * 100, 2),
            "line_mean_grapheme_pp": round(line_mean_gr * 100, 2),
            "macro_codepoint_pp": round(macro_cp * 100, 2),
            "macro_grapheme_pp": round(macro_gr * 100, 2),
            "macro_codepoint_95ci_pp": [
                round(float(macro_ci_cp[0]) * 100, 2),
                round(float(macro_ci_cp[1]) * 100, 2),
            ],
            "macro_grapheme_95ci_pp": [
                round(float(macro_ci_gr[0]) * 100, 2),
                round(float(macro_ci_gr[1]) * 100, 2),
            ],
            "line_mean_codepoint_95ci_pp": [
                round(line_ci_cp[0] * 100, 2),
                round(line_ci_cp[1] * 100, 2),
            ],
            "line_mean_grapheme_95ci_pp": [
                round(line_ci_gr[0] * 100, 2),
                round(line_ci_gr[1] * 100, 2),
            ],
        },
        "bootstrap": {
            "method": "decree-grouped; resample groups, keep all their lines",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "groups": n_groups,
        },
        "harness": {
            "reference_r010_codepoint_cer": HARNESS_REFERENCE_CER,
            "observed_codepoint_cer": preparation.get("harness_validation_cer"),
            "tolerance_pp": HARNESS_TOLERANCE_PP,
        },
        "per_decree_group": per_group,
        "limitations": [
            (
                "Images are PDF re-renders of Word-exported decrees, not "
                "photographed scans. Capture degradation (camera, lighting, "
                "skew, blur) is not measured."
            ),
            (
                "The pages dropped for a missing text layer include the genuine "
                "scans in this collection; they carry zero labels and cannot be "
                "scored."
            ),
            (
                "This measures OCR text error on line crops, not end-task intent "
                "accuracy; real-image intent evaluation remains blocked."
            ),
            (
                "One Tesseract version, one language configuration, one PSM/OEM "
                "setting and one render font are tested."
            ),
        ],
    }
