"""Build 12 px synthetic OCR continuations of the MASSIVE training utterances."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import ImageFont
from prepare_r010 import render_png, transcribe
from r010_identity import FONT_PATH, cer_counts
from r012_identity import (
    ARTIFACT_DIR,
    PREPARED_TRAIN,
    ROOT,
    TRAIN_TRANSCRIPTS,
    read_jsonl,
    sha256,
    train_row_id,
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


def prepare():
    manifest, manifest_hash, r010, _r010_hash = verify_locked_manifest(
        test_access=False
    )
    source = manifest["source"]
    train_path = ROOT / source["train_file"]
    source_rows = read_jsonl(train_path)
    if len(source_rows) != source["training_rows"]:
        raise ValueError("R012 MASSIVE train row count differs from the lock")
    if any(
        not isinstance(row.get("state", {}).get("utterance"), str)
        or not row["state"]["utterance"]
        for row in source_rows
    ):
        raise ValueError("R012 MASSIVE training utterances must be non-empty strings")

    prepared_path = ROOT / PREPARED_TRAIN
    transcript_path = ROOT / TRAIN_TRANSCRIPTS
    preparation_path = ARTIFACT_DIR / "preparation.json"
    if any(
        path.exists() for path in (prepared_path, transcript_path, preparation_path)
    ):
        raise FileExistsError("R012 preparation outputs already exist")

    font = ImageFont.truetype(
        str(FONT_PATH), FONT_SIZE, layout_engine=ImageFont.Layout.RAQM
    )
    transcript_cache: dict[str, dict[str, str]] = {}
    prepared_rows: list[dict[str, Any]] = []
    transcript_rows: list[dict[str, Any]] = []
    for index, original in enumerate(source_rows):
        reference = original["state"]["utterance"]
        if reference not in transcript_cache:
            image = render_png(reference, font)
            transcript_cache[reference] = {
                "text": transcribe(image, r010["environment"]["tesseract_path"]),
                "image_sha256": hashlib.sha256(image).hexdigest(),
            }
        recognized = transcript_cache[reference]
        prepared = copy.deepcopy(original)
        prepared["state"]["utterance"] = recognized["text"]
        prepared_rows.append(prepared)
        transcript_rows.append(
            {
                "row_index": index,
                "row_id": train_row_id(source["train_sha256"], index, original),
                "reference": reference,
                "text": recognized["text"],
                "image_sha256": recognized["image_sha256"],
                "cer": cer_counts(reference, recognized["text"]),
            }
        )
        if (index + 1) % 500 == 0:
            print(f"R012 OCR train rows {index + 1}/{len(source_rows)}", flush=True)

    write_jsonl(prepared_path, prepared_rows)
    write_jsonl(transcript_path, transcript_rows)
    codepoint_edits = sum(row["cer"]["codepoint_edits"] for row in transcript_rows)
    reference_codepoints = sum(
        row["cer"]["reference_codepoints"] for row in transcript_rows
    )
    grapheme_edits = sum(row["cer"]["grapheme_edits"] for row in transcript_rows)
    reference_graphemes = sum(
        row["cer"]["reference_graphemes"] for row in transcript_rows
    )
    metrics = {
        "rows": len(transcript_rows),
        "unique_raw_utterances": len(transcript_cache),
        "codepoint_edits": codepoint_edits,
        "reference_codepoints": reference_codepoints,
        "codepoint_cer": codepoint_edits / reference_codepoints,
        "grapheme_edits": grapheme_edits,
        "reference_graphemes": reference_graphemes,
        "grapheme_cer": grapheme_edits / reference_graphemes,
        "exact_rows": sum(row["cer"]["exact"] for row in transcript_rows),
        "blank_rows": sum(not row["text"] for row in transcript_rows),
    }
    preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_hash,
        "preparer_sha256": sha256(Path(__file__)),
        "source_train_sha256": source["train_sha256"],
        "training_rows": len(source_rows),
        "prepared_train_file": PREPARED_TRAIN,
        "transcript_artifact": TRAIN_TRANSCRIPTS,
        "outputs_sha256": {
            PREPARED_TRAIN: sha256(prepared_path),
            TRAIN_TRANSCRIPTS: sha256(transcript_path),
        },
        "ocr_metrics": metrics,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(preparation_path, preparation)
    print(json.dumps({"ocr_metrics": metrics}, indent=2))
    return preparation


def main():
    prepare()


if __name__ == "__main__":
    main()
