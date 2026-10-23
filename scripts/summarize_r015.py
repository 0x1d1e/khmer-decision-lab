"""Summarize R015 paired document-image vs synthetic OCR results.

Reads artifacts/runs/R015/per-line-results.jsonl and preparation.json, rebuilds
summary.json from the per-line rows (so the summary is always derived from the
raw records), and writes the markdown table used in the results file.
"""

import json
from collections import Counter
from pathlib import Path

from r015_metrics import build_summary

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/runs/R015"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    rows = read_jsonl(ARTIFACT_DIR / "per-line-results.jsonl")
    preparation = read_json(ARTIFACT_DIR / "preparation.json")
    if preparation.get("study") != "R015-document-image-ocr":
        raise ValueError("unexpected R015 preparation study")
    if len(rows) != preparation["test_lines_filtered"]:
        raise ValueError("R015 per-line rows do not match preparation count")

    summary = build_summary(rows, preparation)

    # Gold-character load in lines the real crop got wrong; a coarse diagnostic
    # of which Khmer characters dominate document OCR error.
    gold_counter: Counter[str] = Counter()
    for row in rows:
        if not row["edit_ops_real"]["exact"]:
            gold_counter.update(char for char in row["gold_text"] if not char.isspace())
    summary["top_gold_chars_in_real_errors"] = [
        {"char": char, "count": count} for char, count in gold_counter.most_common(15)
    ]

    write_json(ARTIFACT_DIR / "summary.json", summary)

    real = summary["real"]
    synth = summary["synthetic"]
    delta = summary["delta"]
    n = summary["n_lines"]
    table = "\n".join(
        [
            "| Condition | Codepoint CER | Grapheme CER | Exact match | Blank |",
            "|---|---:|---:|---:|---:|",
            (
                f"| A: real document crop | {real['codepoint_cer'] * 100:.2f}% "
                f"| {real['grapheme_cer'] * 100:.2f}% "
                f"| {real['exact_match']} / {n} | {real['blank_outputs']} |"
            ),
            (
                f"| B: synthetic render | {synth['codepoint_cer'] * 100:.2f}% "
                f"| {synth['grapheme_cer'] * 100:.2f}% "
                f"| {synth['exact_match']} / {n} | {synth['blank_outputs']} |"
            ),
            (
                f"| Delta (A - B) | +{delta['macro_codepoint_pp']:.2f} pp "
                f"| +{delta['macro_grapheme_pp']:.2f} pp "
                f"| {real['exact_match'] - synth['exact_match']} "
                f"| {real['blank_outputs'] - synth['blank_outputs']} |"
            ),
        ]
    )
    write_text(ARTIFACT_DIR / "results-table.md", table + "\n")

    print(f"Wrote {ARTIFACT_DIR / 'summary.json'}")
    print(f"Wrote {ARTIFACT_DIR / 'results-table.md'}")
    print(
        f"Lines: {n}, Pages: {summary['n_pages']}, "
        f"Decree groups: {summary['n_decree_groups']}"
    )
    print(
        f"Real CER:      cp={real['codepoint_cer'] * 100:.2f}%, "
        f"gr={real['grapheme_cer'] * 100:.2f}%"
    )
    print(
        f"Synthetic CER: cp={synth['codepoint_cer'] * 100:.2f}%, "
        f"gr={synth['grapheme_cer'] * 100:.2f}%"
    )
    print(
        f"Macro delta:   cp=+{delta['macro_codepoint_pp']:.2f} pp "
        f"[{delta['macro_codepoint_95ci_pp'][0]:.2f}, "
        f"{delta['macro_codepoint_95ci_pp'][1]:.2f}], "
        f"gr=+{delta['macro_grapheme_pp']:.2f} pp "
        f"[{delta['macro_grapheme_95ci_pp'][0]:.2f}, "
        f"{delta['macro_grapheme_95ci_pp'][1]:.2f}]"
    )


if __name__ == "__main__":
    main()
