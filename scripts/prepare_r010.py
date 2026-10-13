"""Render locked MASSIVE Khmer test text and transcribe it with Tesseract."""

import copy
import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from r010_identity import (
    ARTIFACT_DIR,
    CONDITIONS,
    FONT_PATH,
    ROOT,
    TESSDATA_DIR,
    canonical_json,
    cer_counts,
    normalize_input,
    sha256,
    verify_locked_manifest,
    write_json,
)

FONT_SIZES: dict[str, int] = {"ocr_48px": 48, "ocr_24px": 24, "ocr_12px": 12}


def render_png(text, font):
    box = font.getbbox(text)
    image = Image.new(
        "RGB",
        (max(256, box[2] - box[0] + 32), max(60, box[3] - box[1] + 24)),
        "white",
    )
    ImageDraw.Draw(image).text(
        (16 - box[0], 12 - box[1]), text, font=font, fill="black"
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def transcribe(image_bytes, tesseract_path):
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


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    manifest, manifest_sha256 = verify_locked_manifest()
    source = manifest["source"]
    test_path = ROOT / source["test_file"]
    train_path = ROOT / source["train_file"]
    test_rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    if len(test_rows) != source["test_rows"]:
        raise ValueError("R010 test row count differs from the locked manifest")
    if not test_rows or any(not row["state"].get("utterance") for row in test_rows):
        raise ValueError("R010 test rows must contain non-empty utterance state")

    train_inputs = {normalize_input(row["state"]["utterance"]) for row in train_rows}
    row_metadata: list[dict[str, Any]] = []
    grouped_texts: dict[str, dict[str, dict[str, str]]] = {
        condition: {} for condition in FONT_SIZES
    }
    group_labels = {}
    group_sizes = {}
    for index, row in enumerate(test_rows):
        text = row["state"]["utterance"]
        group_key = normalize_input(text)
        if not group_key:
            raise ValueError(f"R010 test row {index} has an empty normalized utterance")
        row_id = hashlib.sha256(
            f"{source['test_sha256']}\0{index}\0{canonical_json(row)}".encode()
        ).hexdigest()
        group_id = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
        row_metadata.append(
            {
                "row_index": index,
                "row_id": row_id,
                "group_id": group_id,
                "train_overlap": group_key in train_inputs,
                "reference": text,
            }
        )
        group_labels.setdefault(group_key, set()).add(row["expected"]["intent"])
        group_sizes[group_key] = group_sizes.get(group_key, 0) + 1

    conflicting_groups = {
        group for group, labels in group_labels.items() if len(labels) > 1
    }
    observed_audit = {
        "test_rows": len(test_rows),
        "normalized_test_groups": len(group_labels),
        "conflicting_label_groups": len(conflicting_groups),
        "conflicting_label_rows": sum(
            group_sizes[group] for group in conflicting_groups
        ),
        "exact_train_overlap_groups": len(
            {key for key in group_labels if key in train_inputs}
        ),
        "exact_train_overlap_rows": sum(
            metadata["train_overlap"] for metadata in row_metadata
        ),
    }
    if observed_audit != source["audit"]:
        raise ValueError("R010 test rows differ from the locked split audit")

    environment = manifest["environment"]
    fonts = {
        condition: ImageFont.truetype(
            str(FONT_PATH), size, layout_engine=ImageFont.Layout.RAQM
        )
        for condition, size in FONT_SIZES.items()
    }
    for index, metadata in enumerate(row_metadata):
        text = metadata["reference"]
        for condition, font in fonts.items():
            if text not in grouped_texts[condition]:
                image_bytes = render_png(text, font)
                grouped_texts[condition][text] = {
                    "text": transcribe(image_bytes, environment["tesseract_path"]),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
        if (index + 1) % 250 == 0:
            print(f"OCR rows {index + 1}/{len(test_rows)}", flush=True)

    prepared_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    ocr_records: list[dict[str, Any]] = []
    for source_row, metadata in zip(test_rows, row_metadata, strict=True):
        tags = list(source_row.get("tags", []))
        tags.extend(
            (
                f"r010-row:{metadata['row_id']}",
                f"r010-group:{metadata['group_id']}",
                f"r010-train-overlap:{int(metadata['train_overlap'])}",
            )
        )
        clean_row = copy.deepcopy(source_row)
        clean_row["tags"] = tags
        prepared_rows["clean"].append(clean_row)

        record = {**metadata, "ocr": {}}
        for condition in FONT_SIZES:
            recognized = grouped_texts[condition][metadata["reference"]]
            row = copy.deepcopy(source_row)
            row["state"]["utterance"] = recognized["text"]
            row["tags"] = tags
            prepared_rows[condition].append(row)
            record["ocr"][condition] = {
                **recognized,
                **cer_counts(metadata["reference"], recognized["text"]),
            }
        ocr_records.append(record)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for condition, rows in prepared_rows.items():
        relative = f"data/processed/r010-massive-km-{condition}.jsonl"
        write_jsonl(ROOT / relative, rows)
        outputs[condition] = {
            "file": relative,
            "sha256": sha256(ROOT / relative),
            "rows": len(rows),
        }
    ocr_path = ARTIFACT_DIR / "ocr-transcripts.jsonl"
    write_jsonl(ocr_path, ocr_records)
    outputs["ocr_transcripts"] = {
        "file": str(ocr_path.relative_to(ROOT)),
        "sha256": sha256(ocr_path),
        "rows": len(ocr_records),
    }

    condition_metrics = {}
    for condition in FONT_SIZES:
        records = [record["ocr"][condition] for record in ocr_records]
        codepoints = sum(record["reference_codepoints"] for record in records)
        grapheme_count = sum(record["reference_graphemes"] for record in records)
        condition_metrics[condition] = {
            "unique_utterances_transcribed": len(grouped_texts[condition]),
            "codepoint_cer": sum(r["codepoint_edits"] for r in records) / codepoints,
            "grapheme_cer": sum(r["grapheme_edits"] for r in records) / grapheme_count,
            "exact_rows": sum(record["exact"] for record in records),
            "blank_rows": sum(
                not record["ocr"][condition]["text"] for record in ocr_records
            ),
        }

    preparation = {
        "study": manifest["study"],
        "manifest_sha256": manifest_sha256,
        "preparer_sha256": sha256(Path(__file__)),
        "source_test_sha256": source["test_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "rows": len(test_rows),
        "source_audit": observed_audit,
        "unique_raw_utterances": len(
            Counter(metadata["reference"] for metadata in row_metadata)
        ),
        "unique_normalized_groups": len({r["group_id"] for r in row_metadata}),
        "train_overlap_rows": sum(r["train_overlap"] for r in row_metadata),
        "condition_metrics": condition_metrics,
        "files_sha256": {value["file"]: value["sha256"] for value in outputs.values()},
        "outputs": outputs,
    }
    write_json(ARTIFACT_DIR / "preparation.json", preparation)
    print(json.dumps(condition_metrics, indent=2))


if __name__ == "__main__":
    main()
