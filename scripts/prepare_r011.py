"""Prepare de-overlapped validation text under the locked R010 OCR condition."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import ImageFont
from prepare_r010 import render_png, transcribe
from r010_identity import FONT_PATH, cer_counts, normalize_input
from r011_identity import (
    ARTIFACT_DIR,
    CALIBRATION_DATA,
    ROOT,
    TRANSCRIPT_ARTIFACT,
    calibration_selection,
    read_jsonl,
    sha256,
    verify_locked_manifest,
    write_json,
)

FONT_SIZE = 12


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    manifest, manifest_hash, r010, _r010_hash = verify_locked_manifest()
    source = manifest["source"]
    validation_path = ROOT / source["validation_file"]
    train_path = ROOT / source["train_file"]
    test_path = ROOT / source["test_file"]
    validation_rows = read_jsonl(validation_path)
    train_inputs = {
        row["state"]["utterance"]
        for row in read_jsonl(train_path)
        if row.get("state", {}).get("utterance")
    }
    test_inputs = {
        row["state"]["utterance"]
        for row in read_jsonl(test_path)
        if row.get("state", {}).get("utterance")
    }
    selected, audit = calibration_selection(
        validation_rows,
        {normalize_input(text) for text in train_inputs},
        {normalize_input(text) for text in test_inputs},
    )
    if any(audit[key] != source["audit"][key] for key in audit):
        raise ValueError(
            "R011 validation overlap audit differs from the locked manifest"
        )
    if len(validation_rows) != source["validation_rows"]:
        raise ValueError("R011 validation row count mismatch")

    tesseract = r010["environment"]["tesseract_path"]
    font = ImageFont.truetype(
        str(FONT_PATH), FONT_SIZE, layout_engine=ImageFont.Layout.RAQM
    )
    prepared_rows: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    for source_index, original in selected:
        reference = original["state"]["utterance"]
        group_key = normalize_input(reference)
        group_id = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
        canonical = json.dumps(
            original, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        source_row_id = hashlib.sha256(
            f"{source['validation_sha256']}\0{source_index}\0{canonical}".encode()
        ).hexdigest()
        image = render_png(reference, font)
        recognized = transcribe(image, tesseract)
        row = copy.deepcopy(original)
        row["state"]["utterance"] = recognized
        row["tags"] = list(original.get("tags", [])) + [
            f"r011-row:{source_row_id}",
            f"r011-group:{group_id}",
        ]
        prepared_rows.append(row)
        transcripts.append(
            {
                "source_row_index": source_index,
                "source_row_id": source_row_id,
                "group_id": group_id,
                "reference": reference,
                "text": recognized,
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "train_overlap": False,
                "test_overlap": False,
                "conflicting_label": False,
                "cer": cer_counts(reference, recognized),
            }
        )
        if len(prepared_rows) % 250 == 0:
            print(
                f"OCR validation groups {len(prepared_rows)}/{len(selected)}",
                flush=True,
            )

    calibration_path = ROOT / CALIBRATION_DATA
    transcript_path = ROOT / TRANSCRIPT_ARTIFACT
    write_jsonl(calibration_path, prepared_rows)
    write_jsonl(transcript_path, transcripts)
    codepoints = sum(item["cer"]["reference_codepoints"] for item in transcripts)
    graphemes = sum(item["cer"]["reference_graphemes"] for item in transcripts)
    preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "preparer_sha256": sha256(Path(__file__)),
        "source_validation_sha256": source["validation_sha256"],
        "source_train_sha256": source["train_sha256"],
        "source_test_sha256": source["test_sha256"],
        "validation_audit": audit,
        "calibration_file": CALIBRATION_DATA,
        "transcript_artifact": TRANSCRIPT_ARTIFACT,
        "outputs_sha256": {
            CALIBRATION_DATA: sha256(calibration_path),
            TRANSCRIPT_ARTIFACT: sha256(transcript_path),
        },
        "cer": {
            "rows": len(transcripts),
            "codepoint_edits": sum(
                item["cer"]["codepoint_edits"] for item in transcripts
            ),
            "reference_codepoints": codepoints,
            "codepoint_cer": sum(item["cer"]["codepoint_edits"] for item in transcripts)
            / codepoints,
            "grapheme_edits": sum(
                item["cer"]["grapheme_edits"] for item in transcripts
            ),
            "reference_graphemes": graphemes,
            "grapheme_cer": sum(item["cer"]["grapheme_edits"] for item in transcripts)
            / graphemes,
            "exact_rows": sum(item["cer"]["exact"] for item in transcripts),
            "blank_rows": sum(not item["text"] for item in transcripts),
        },
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(ARTIFACT_DIR / "preparation.json", preparation)
    print(json.dumps({"audit": audit, "cer": preparation["cer"]}, indent=2))


if __name__ == "__main__":
    main()
