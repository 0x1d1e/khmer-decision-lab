"""Paired document-image vs synthetic OCR measurement for Khmer decree lines.

Measures how much R010's synthetic OCR proxy underestimates real document-image
OCR error. For each line in the filtered test split of structure-839:

  Condition A (real): crop the line from the page PNG, run Tesseract.
  Condition B (synthetic): render the same gold text with the R010 recipe, run
  Tesseract.

Images are PDF re-renders (Word-exported decrees), NOT scans. Capture
degradation (camera, lighting, skew, blur) is not measured.
"""

import argparse
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import regex
from PIL import Image, ImageDraw, ImageFont, features
from r015_metrics import (
    BOOTSTRAP_SEED,
    build_summary,
    cer_record,
    normalize_for_cer,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/home/k/data/structure-839")
FONT_PATH = Path("/usr/share/fonts/noto/NotoSansKhmer-Regular.ttf")
TESSDATA_DIR = Path("/usr/share/tessdata")
ARTIFACT_DIR = ROOT / "artifacts/runs/R015"
MASSIVE_TEST = ROOT / "data/processed/massive-km-test-full.jsonl"

FONT_SIZE = 48
HARNESS_REFERENCE_CER = 0.0526  # R010 row-weighted MASSIVE test CER at 48 px
HARNESS_TOLERANCE_PP = 1.0

QUALITY_FILTER = {
    "text_ok": True,
    "duplicate_rect_frac_max": 0.02,
    "bad_coeng_frac_max": 0.002,
    "dep_dep_frac_max": 0.02,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_synthetic(text: str, font: ImageFont.FreeTypeFont) -> bytes:
    """R010 recipe: 48 px Noto Sans Khmer, black on white, RAQM, 16/12 padding."""
    box = font.getbbox(text)
    width = max(256, box[2] - box[0] + 32)
    height = max(60, box[3] - box[1] + 24)
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).text(
        (16 - box[0], 12 - box[1]), text, font=font, fill="black"
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def transcribe(image_bytes: bytes, tesseract_path: str) -> str:
    """Tesseract khm+eng, OEM 1, PSM 7 on PNG bytes from stdin."""
    result = subprocess.run(
        [
            tesseract_path,
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(TESSDATA_DIR),
            "-l",
            "khm+eng",
            "--oem",
            "1",
            "--psm",
            "7",
        ],
        input=image_bytes,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def crop_line(page_image: Image.Image, bbox: list[float]) -> bytes:
    x0, y0, x1, y1 = (round(value) for value in bbox)
    output = io.BytesIO()
    page_image.crop((x0, y0, x1, y1)).save(output, format="PNG")
    return output.getvalue()


def load_split_test_pages(manifest: dict, split_data: dict) -> set[str]:
    test_groups = {
        entry["group"]
        for entry in split_data["assignments"]
        if entry["split"] == "test"
    }
    return {
        page["page_id"] for page in manifest["pages"] if page["group"] in test_groups
    }


def apply_quality_filter(manifest: dict, test_page_ids: set[str]) -> list[dict]:
    """Strict pre-registered filter; yields the expected 148 test pages."""
    keep = []
    for page in manifest["pages"]:
        if page["page_id"] not in test_page_ids or not page["text_ok"]:
            continue
        if page["dup_frac"] > QUALITY_FILTER["duplicate_rect_frac_max"]:
            continue
        if page["bad_coeng_frac"] > QUALITY_FILTER["bad_coeng_frac_max"]:
            continue
        if page["dep_dep_frac"] > QUALITY_FILTER["dep_dep_frac_max"]:
            continue
        keep.append(page)
    return keep


def collect_environment(tesseract_path: str) -> dict:
    version_output = subprocess.run(
        [tesseract_path, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pillow": __import__("PIL").__version__,
        "pillow_raqm": features.version("raqm"),
        "numpy": np.__version__,
        "regex": regex.__version__,
        "tesseract_path": tesseract_path,
        "tesseract_version": version_output[0].strip(),
        "tesseract_sha256": sha256(Path(tesseract_path)),
        "tessdata_sha256": {
            language: sha256(TESSDATA_DIR / f"{language}.traineddata")
            for language in ("khm", "eng")
        },
        "font_path": str(FONT_PATH),
        "font_sha256": sha256(FONT_PATH),
        "font_size": FONT_SIZE,
    }


def validate_harness(
    font: ImageFont.FreeTypeFont,
    tesseract_path: str,
    sample_size: int | None = None,
) -> float:
    """Render MASSIVE Khmer test text, transcribe it, and return codepoint CER.

    R010 measured 5.26% row-weighted codepoint CER at 48 px over all 2,974
    MASSIVE test rows. A 300-row subsample has its own standard error of about
    0.55 pp (observed 300-row CERs span roughly 4.2-6.4% across seeds), too noisy
    to separate a broken harness from sampling noise under a 1 pp gate. The
    default therefore transcribes every row; the row-weighted aggregate
    reproduces R010's metric definition exactly.
    """
    if not MASSIVE_TEST.is_file():
        print(f"SKIP harness validation: {MASSIVE_TEST} not found")
        return -1.0

    with MASSIVE_TEST.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if sample_size is not None and sample_size < len(rows):
        picked = np.random.default_rng(BOOTSTRAP_SEED).choice(
            len(rows), size=sample_size, replace=False
        )
        rows = [rows[index] for index in picked]

    cache: dict[str, tuple[int, int]] = {}
    total_edits = 0
    total_ref = 0
    for row in rows:
        text = row["state"]["utterance"]
        if text not in cache:
            hypothesis = transcribe(render_synthetic(text, font), tesseract_path)
            reference = normalize_for_cer(text)
            from r015_metrics import edit_distance_with_ops

            ops = edit_distance_with_ops(
                list(reference), list(normalize_for_cer(hypothesis))
            )
            cache[text] = (ops["distance"], len(reference))
        edits, ref_cp = cache[text]
        total_edits += edits
        total_ref += ref_cp
    return total_edits / total_ref if total_ref else 0.0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data availability and filter counts without running OCR.",
    )
    parser.add_argument(
        "--harness-sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Transcribe only N deterministically sampled MASSIVE rows for the "
            "harness check. Default: all rows (row-weighted, matches R010)."
        ),
    )
    args = parser.parse_args()

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        print("ERROR: tesseract not found", file=sys.stderr)
        sys.exit(1)
    tesseract_path = str(Path(tesseract_path).resolve())
    if not FONT_PATH.is_file():
        print(f"ERROR: font not found: {FONT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not features.check_feature("raqm"):
        print("ERROR: Pillow RAQM support required", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(
        (DATA_DIR / "lines_gt_manifest.json").read_text(encoding="utf-8")
    )
    split_data = json.loads(
        (DATA_DIR / "lines_gt_split.json").read_text(encoding="utf-8")
    )
    test_page_ids = load_split_test_pages(manifest, split_data)
    all_test_pages = [p for p in manifest["pages"] if p["page_id"] in test_page_ids]
    filtered_pages = apply_quality_filter(manifest, test_page_ids)
    filtered_ids = {page["page_id"] for page in filtered_pages}

    total_test_lines = 0
    filtered_test_lines = 0
    for page in all_test_pages:
        gt = json.loads(
            (DATA_DIR / "lines_gt" / f"{page['page_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        total_test_lines += len(gt["lines"])
        if page["page_id"] in filtered_ids:
            filtered_test_lines += len(gt["lines"])
    no_text_pages = sum(1 for page in all_test_pages if not page["text_ok"])

    print(f"Test pages (all):      {len(all_test_pages)}")
    print(f"Test pages (no text):  {no_text_pages}")
    print(f"Test pages (filtered): {len(filtered_pages)}")
    print(f"Test lines (all):      {total_test_lines}")
    print(f"Test lines (filtered): {filtered_test_lines}")

    if args.dry_run:
        print("\n--dry-run: data checks passed, exiting.")
        return

    font = ImageFont.truetype(
        str(FONT_PATH), FONT_SIZE, layout_engine=ImageFont.Layout.RAQM
    )

    print("\n=== Harness validation (MASSIVE Khmer test, 48 px) ===")
    harness_cer = validate_harness(font, tesseract_path, args.harness_sample)
    if harness_cer >= 0:
        print(f"Harness CER: {harness_cer * 100:.2f}%")
        if abs(harness_cer - HARNESS_REFERENCE_CER) * 100 > HARNESS_TOLERANCE_PP:
            print(
                f"ERROR: harness CER {harness_cer * 100:.2f}% is more than "
                f"{HARNESS_TOLERANCE_PP:.1f} pp from R010's "
                f"{HARNESS_REFERENCE_CER * 100:.2f}%. Stopping.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Harness validation passed.")
    else:
        print("Harness validation skipped (MASSIVE test file not found).")

    print(
        f"\n=== Processing {filtered_test_lines} lines across "
        f"{len(filtered_pages)} pages ==="
    )
    page_to_group = {page["page_id"]: page["group"] for page in manifest["pages"]}
    results = []
    processed = 0
    for page in filtered_pages:
        page_id = page["page_id"]
        group = page_to_group[page_id]
        gt = json.loads(
            (DATA_DIR / "lines_gt" / f"{page_id}.json").read_text(encoding="utf-8")
        )
        with Image.open(DATA_DIR / "images" / f"{page_id}.png") as page_image:
            for line in gt["lines"]:
                gold = line["text"]
                ocr_real = transcribe(
                    crop_line(page_image, line["bbox"]), tesseract_path
                )
                ocr_synth = transcribe(render_synthetic(gold, font), tesseract_path)
                real = cer_record(gold, ocr_real)
                synth = cer_record(gold, ocr_synth)
                results.append(
                    {
                        "line_id": f"{page_id}_L{line['id']}",
                        "page": page_id,
                        "decree_group": group,
                        "gold_text": gold,
                        "ocr_real": ocr_real,
                        "ocr_synthetic": ocr_synth,
                        "cer_real_cp": real["cer_cp"],
                        "cer_real_gr": real["cer_gr"],
                        "cer_synth_cp": synth["cer_cp"],
                        "cer_synth_gr": synth["cer_gr"],
                        "edit_ops_real": _edit_ops(real),
                        "edit_ops_synth": _edit_ops(synth),
                    }
                )
                processed += 1
                if processed % 500 == 0:
                    print(f"  {processed}/{filtered_test_lines} lines", flush=True)
    print(f"  {processed}/{filtered_test_lines} lines (done)")

    write_jsonl(ARTIFACT_DIR / "per-line-results.jsonl", results)
    environment = collect_environment(tesseract_path)
    preparation = {
        "study": "R015-document-image-ocr",
        "created_utc": datetime.now(UTC).isoformat(),
        "data_source": str(DATA_DIR),
        "data_manifest_sha256": sha256(DATA_DIR / "lines_gt_manifest.json"),
        "data_split_sha256": sha256(DATA_DIR / "lines_gt_split.json"),
        "quality_filter": QUALITY_FILTER,
        "test_pages_total": len(all_test_pages),
        "test_pages_no_text_layer": no_text_pages,
        "test_pages_filtered": len(filtered_pages),
        "test_lines_total": total_test_lines,
        "test_lines_filtered": filtered_test_lines,
        "decree_groups_in_test": len({page["group"] for page in filtered_pages}),
        "harness_validation_cer": round(harness_cer, 6) if harness_cer >= 0 else None,
        "environment": environment,
        "preparer_sha256": sha256(Path(__file__)),
        "per_line_results_sha256": sha256(ARTIFACT_DIR / "per-line-results.jsonl"),
    }
    summary = build_summary(results, preparation)
    write_json(ARTIFACT_DIR / "preparation.json", preparation)
    write_json(ARTIFACT_DIR / "summary.json", summary)

    print("\n=== Results ===")
    real = summary["real"]
    synth = summary["synthetic"]
    delta = summary["delta"]
    print(
        f"Lines: {summary['n_lines']}, Pages: {summary['n_pages']}, "
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
        f"gr=+{delta['macro_grapheme_pp']:.2f} pp"
    )
    print(
        f"Real exact: {real['exact_match']}/{summary['n_lines']}, "
        f"blank: {real['blank_outputs']}"
    )
    print(
        f"Synth exact: {synth['exact_match']}/{summary['n_lines']}, "
        f"blank: {synth['blank_outputs']}"
    )
    print(
        f"Error composition (real):  del={real['cp_deletions']}, "
        f"sub={real['cp_substitutions']}, ins={real['cp_insertions']}"
    )
    print(
        f"Error composition (synth): del={synth['cp_deletions']}, "
        f"sub={synth['cp_substitutions']}, ins={synth['cp_insertions']}"
    )


def _edit_ops(record: dict) -> dict:
    keys = (
        "cp_edits",
        "cp_deletions",
        "cp_substitutions",
        "cp_insertions",
        "gr_edits",
        "gr_deletions",
        "gr_substitutions",
        "gr_insertions",
        "ref_codepoints",
        "ref_graphemes",
    )
    ops = {key: record[key] for key in keys}
    ops["exact"] = record["exact"]
    ops["blank"] = record["blank"]
    return ops


if __name__ == "__main__":
    main()
