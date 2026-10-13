"""Integrity and Khmer text helpers for the R010 OCR cascade."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import unicodedata
from importlib.metadata import version
from pathlib import Path

import regex
import torch
from PIL import ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/r010-massive-ocr-cascade.json"
PROTOCOL = ROOT / "research/R010-massive-ocr-cascade/README.md"
ARTIFACT_DIR = ROOT / "artifacts/runs/R010"
IMPLEMENTATION_FILES = (
    "scripts/r008_identity.py",
    "scripts/r010_identity.py",
    "scripts/prepare_r010.py",
    "scripts/evaluate_r010.py",
    "scripts/summarize_r010.py",
    "tests/test_r010_ocr.py",
)
MODEL_PATHS = {
    "full-e2-8bit": "artifacts/checkpoints/R003-full-e2-8bit",
    "mt-s13": "artifacts/checkpoints/R006-mt-s13",
    "mt-s14": "artifacts/checkpoints/R006-mt-s14",
}
CONDITIONS = ("clean", "ocr_48px", "ocr_24px", "ocr_12px")
FONT_PATH = Path("/usr/share/fonts/noto/NotoSansKhmer-Regular.ttf")
TESSDATA_DIR = Path("/usr/share/tessdata")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_input(text):
    """Normalization used only for exact-input grouping and train-overlap checks."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def normalize_for_cer(text):
    """NFC and whitespace normalization for character-error metrics only."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def graphemes(text):
    return regex.findall(r"\X", normalize_for_cer(text))


def edit_distance(source, target):
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, source_unit in enumerate(source, 1):
        current = [i]
        for j, target_unit in enumerate(target, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (source_unit != target_unit),
                )
            )
        previous = current
    return previous[-1]


def cer_counts(reference, hypothesis):
    reference = normalize_for_cer(reference)
    hypothesis = normalize_for_cer(hypothesis)
    return {
        "codepoint_edits": edit_distance(list(reference), list(hypothesis)),
        "reference_codepoints": len(reference),
        "grapheme_edits": edit_distance(graphemes(reference), graphemes(hypothesis)),
        "reference_graphemes": len(graphemes(reference)),
        "exact": reference == hypothesis,
    }


def collect_environment():
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise ValueError("tesseract executable is missing")
    tesseract = str(Path(tesseract).resolve())
    version_output = subprocess.run(
        [tesseract, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    languages_output = subprocess.run(
        [tesseract, "--list-langs"], check=True, capture_output=True, text=True
    ).stdout
    if not all(
        re.search(rf"^\s*{language}\s*$", languages_output, re.MULTILINE)
        for language in ("khm", "eng")
    ):
        raise ValueError("Tesseract must have both khm and eng traineddata")
    if not features.check_feature("raqm"):
        raise ValueError("Pillow must be built with RAQM for Khmer text shaping")
    if not FONT_PATH.is_file():
        raise ValueError(f"Khmer rendering font is missing: {FONT_PATH}")
    font = ImageFont.truetype(str(FONT_PATH), 48)
    if font.layout_engine != ImageFont.Layout.RAQM:
        raise ValueError("Noto Sans Khmer must be rendered with Pillow RAQM")
    tessdata = {
        language: sha256(TESSDATA_DIR / f"{language}.traineddata")
        for language in ("khm", "eng")
    }
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_driver = None
    if torch.cuda.is_available() and nvidia_smi:
        gpu_driver = (
            subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "laya": version("laya"),
        "torch": version("torch"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "safetensors": version("safetensors"),
        "numpy": version("numpy"),
        "pillow": version("Pillow"),
        "pillow_freetype": features.version("freetype2"),
        "pillow_raqm": features.version("raqm"),
        "pillow_harfbuzz": features.version("harfbuzz"),
        "pillow_fribidi": features.version("fribidi"),
        "regex": version("regex"),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "gpu_driver": gpu_driver,
        "cuda_available": torch.cuda.is_available(),
        "tesseract_path": tesseract,
        "tesseract_version": version_output[0].strip(),
        "tesseract_sha256": sha256(tesseract),
        "tessdata_directory": str(TESSDATA_DIR),
        "tessdata_sha256": tessdata,
        "font_path": str(FONT_PATH),
        "font_sha256": sha256(FONT_PATH),
        "pillow_raqm_enabled": True,
    }


def verify_locked_manifest(path=MANIFEST):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = read_json(path)
    if manifest.get("study") != "R010-massive-ocr-cascade":
        raise ValueError("unexpected R010 manifest")
    if manifest.get("status") != "locked_exploratory":
        raise ValueError("R010 requires a locked exploratory manifest")
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise ValueError("R010 protocol fingerprint mismatch")
    implementation = manifest.get("implementation_sha256", {})
    for relative in IMPLEMENTATION_FILES:
        if implementation.get(relative) != sha256(ROOT / relative):
            raise ValueError(f"R010 implementation fingerprint mismatch: {relative}")

    environment = collect_environment()
    if manifest.get("environment") != environment:
        raise ValueError("R010 runtime environment fingerprint mismatch")

    source = manifest["source"]
    for path_key, hash_key in (
        ("test_file", "test_sha256"),
        ("train_file", "train_sha256"),
        ("massive_manifest", "massive_manifest_sha256"),
        ("r006_manifest", "r006_manifest_sha256"),
    ):
        if sha256(ROOT / source[path_key]) != source[hash_key]:
            raise ValueError(f"R010 source fingerprint mismatch: {source[path_key]}")

    from r008_identity import (
        INFERENCE_POLICY,
        checkpoint_inference_inventory,
    )

    if manifest.get("recipe", {}).get("inference") != INFERENCE_POLICY:
        raise ValueError("R010 inference precision differs from the locked policy")
    if set(manifest.get("models", {})) != set(MODEL_PATHS):
        raise ValueError("R010 model anchors differ from the locked model set")
    for model, expected in manifest["models"].items():
        if MODEL_PATHS.get(model) != expected["checkpoint"]:
            raise ValueError(f"R010 model path mismatch: {model}")
        checkpoint = ROOT / expected["checkpoint"]
        actual = checkpoint_inference_inventory(checkpoint)
        if actual != expected["inference_files_sha256"]:
            raise ValueError(f"R010 checkpoint fingerprint mismatch: {model}")
        config = read_json(checkpoint / "rl_agent_config.json")
        seed = config["training"]["laya_train"]["seed"]
        if (
            seed != expected["seed"]
            or config.get("max_len") != expected["max_len"]
            or config.get("head_max_len") != expected["head_max_len"]
            or config.get("temperature") != expected["choice_temperature"]
            or config.get("amp_dtype") != "bf16"
        ):
            raise ValueError(f"R010 checkpoint inference settings mismatch: {model}")
    return manifest, sha256(path)


def verify_prepared(manifest, manifest_sha256):
    preparation_path = ARTIFACT_DIR / "preparation.json"
    preparation = read_json(preparation_path)
    if (
        preparation.get("study") != manifest["study"]
        or preparation.get("manifest_sha256") != manifest_sha256
        or preparation.get("source_test_sha256") != manifest["source"]["test_sha256"]
        or preparation.get("source_audit") != manifest["source"]["audit"]
        or preparation.get("preparer_sha256")
        != manifest["implementation_sha256"]["scripts/prepare_r010.py"]
    ):
        raise ValueError("R010 preparation provenance does not match locked inputs")
    expected_files = {
        output["file"]: output["sha256"] for output in preparation["outputs"].values()
    }
    if preparation.get("files_sha256") != expected_files:
        raise ValueError("R010 prepared file inventory is incomplete or inconsistent")
    for relative, digest in expected_files.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"R010 prepared file fingerprint mismatch: {relative}")

    source = manifest["source"]
    with (ROOT / source["test_file"]).open(encoding="utf-8") as stream:
        source_rows = [json.loads(line) for line in stream]
    ocr_path = ROOT / preparation["outputs"]["ocr_transcripts"]["file"]
    with ocr_path.open(encoding="utf-8") as stream:
        ocr_records = [json.loads(line) for line in stream]
    train_inputs = {
        normalize_input(row["state"]["utterance"])
        for row in _iter_jsonl(ROOT / source["train_file"])
    }
    group_labels = {}
    group_sizes = {}
    overlap_groups = set()
    for row in source_rows:
        group = normalize_input(row["state"]["utterance"])
        group_labels.setdefault(group, set()).add(row["expected"]["intent"])
        group_sizes[group] = group_sizes.get(group, 0) + 1
        if group in train_inputs:
            overlap_groups.add(group)
    conflict_groups = {
        group for group, labels in group_labels.items() if len(labels) > 1
    }
    observed_audit = {
        "test_rows": len(source_rows),
        "normalized_test_groups": len(group_labels),
        "conflicting_label_groups": len(conflict_groups),
        "conflicting_label_rows": sum(group_sizes[group] for group in conflict_groups),
        "exact_train_overlap_groups": len(overlap_groups),
        "exact_train_overlap_rows": sum(
            size for group, size in group_sizes.items() if group in overlap_groups
        ),
    }
    if observed_audit != source["audit"] or len(ocr_records) != len(source_rows):
        raise ValueError("R010 source split audit or prepared row count differs")

    for condition in CONDITIONS:
        output = preparation["outputs"].get(condition)
        if not output or output["rows"] != len(source_rows):
            raise ValueError(f"R010 prepared row count is invalid: {condition}")
        with (ROOT / output["file"]).open(encoding="utf-8") as stream:
            prepared_rows = [json.loads(line) for line in stream]
        if len(prepared_rows) != len(source_rows):
            raise ValueError(f"R010 prepared file row count mismatch: {condition}")
        for index, (original, prepared, record) in enumerate(
            zip(source_rows, prepared_rows, ocr_records)
        ):
            group_id = hashlib.sha256(
                normalize_input(original["state"]["utterance"]).encode("utf-8")
            ).hexdigest()
            row_id = hashlib.sha256(
                f"{source['test_sha256']}\0{index}\0{canonical_json(original)}".encode()
            ).hexdigest()
            overlap = normalize_input(original["state"]["utterance"]) in train_inputs
            expected_tags = list(original.get("tags", [])) + [
                f"r010-row:{row_id}",
                f"r010-group:{group_id}",
                f"r010-train-overlap:{int(overlap)}",
            ]
            if (
                record.get("row_index") != index
                or record.get("row_id") != row_id
                or record.get("group_id") != group_id
                or record.get("train_overlap") != overlap
                or record.get("reference") != original["state"]["utterance"]
                or prepared.get("tags") != expected_tags
                or prepared.get("questions") != original.get("questions")
                or prepared.get("expected") != original.get("expected")
                or prepared.get("language") != original.get("language")
            ):
                raise ValueError(f"R010 row identity mismatch: {condition}/{index}")
            expected_state = original["state"]
            if condition != "clean":
                ocr_result = record["ocr"][condition]
                expected_cer = cer_counts(record["reference"], ocr_result["text"])
                if any(
                    ocr_result.get(key) != value for key, value in expected_cer.items()
                ):
                    raise ValueError(f"R010 CER audit mismatch: {condition}/{index}")
                if not re.fullmatch(
                    r"[0-9a-f]{64}", ocr_result.get("image_sha256", "")
                ):
                    raise ValueError(
                        f"R010 rendered-image hash is invalid: {condition}/{index}"
                    )
                expected_state = dict(expected_state)
                expected_state["utterance"] = ocr_result["text"]
            if prepared.get("state") != expected_state:
                raise ValueError(
                    f"R010 state transformation mismatch: {condition}/{index}"
                )
    return preparation


def _iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)
