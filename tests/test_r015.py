"""Tests for the R015 document-image OCR protocol study."""

import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

metrics = importlib.import_module("r015_metrics")
preparer = importlib.import_module("prepare_r015")

DATA_DIR = Path("/home/k/data/structure-839")


def test_cer_record_on_known_inputs():
    """Verify edit distance and operation breakdown on known pairs."""
    # Exact match
    res = metrics.cer_record("ព្រះរាជក្រឹត្យ", "ព្រះរាជក្រឹត្យ")
    assert res["exact"] is True
    assert res["blank"] is False
    assert res["cer_cp"] == 0.0
    assert res["cer_gr"] == 0.0
    assert res["cp_edits"] == 0
    assert res["gr_edits"] == 0

    # 1 deletion
    res_del = metrics.cer_record("abcd", "abc")
    assert res_del["cp_deletions"] == 1
    assert res_del["cp_substitutions"] == 0
    assert res_del["cp_insertions"] == 0
    assert res_del["cp_edits"] == 1

    # 1 substitution
    res_sub = metrics.cer_record("abcd", "abxd")
    assert res_sub["cp_deletions"] == 0
    assert res_sub["cp_substitutions"] == 1
    assert res_sub["cp_insertions"] == 0
    assert res_sub["cp_edits"] == 1

    # 1 insertion
    res_ins = metrics.cer_record("abc", "abcd")
    assert res_ins["cp_deletions"] == 0
    assert res_ins["cp_substitutions"] == 0
    assert res_ins["cp_insertions"] == 1
    assert res_ins["cp_edits"] == 1

    # Blank hypothesis
    res_blank = metrics.cer_record("abc", "")
    assert res_blank["blank"] is True
    assert res_blank["exact"] is False
    assert res_blank["cer_cp"] == 1.0


def test_cer_graphemes_distinguish_khmer_clusters():
    """ក្ខ is 3 codepoints (U+1780, U+17D2, U+1780) but 1 grapheme cluster."""
    res = metrics.cer_record("ក្ខ", "ក")
    assert res["ref_codepoints"] == 3
    assert res["ref_graphemes"] == 1
    assert res["cp_edits"] == 2  # delete U+17D2 and U+1780
    assert res["gr_edits"] == 1  # substitute cluster


def test_quality_filter_page_and_line_counts():
    """Quality filter must produce exactly 148 test pages and 4,778 lines."""
    if not DATA_DIR.is_dir():
        pytest.skip(f"Data dir {DATA_DIR} not available")

    manifest = json.loads((DATA_DIR / "lines_gt_manifest.json").read_text())
    split_data = json.loads((DATA_DIR / "lines_gt_split.json").read_text())

    test_page_ids = preparer.load_split_test_pages(manifest, split_data)
    assert len(test_page_ids) == 175

    filtered = preparer.apply_quality_filter(manifest, test_page_ids)
    assert len(filtered) == 148

    total_lines = 0
    for p in filtered:
        gt = json.loads((DATA_DIR / "lines_gt" / f"{p['page_id']}.json").read_text())
        total_lines += len(gt["lines"])
    assert total_lines == 4778


def test_synthetic_render_pipeline_dimensions():
    """Verify minimum canvas dimensions and deterministic rendering."""
    font = ImageFont.truetype(
        str(preparer.FONT_PATH), 48, layout_engine=ImageFont.Layout.RAQM
    )

    # Short string hits min dimensions: width >= 256, height >= 60
    short_png = preparer.render_synthetic("ក", font)
    img_short = Image.open(io.BytesIO(short_png))
    assert img_short.width >= 256
    assert img_short.height >= 60

    # Longer string grows in width
    long_text = "ព្រះរាជក្រឹត្យ នស/រកត/០២១៧/០៧៨ ស្តីពីរបបសន្តិសុខសង្គម"
    long_png = preparer.render_synthetic(long_text, font)
    img_long = Image.open(io.BytesIO(long_png))
    assert img_long.width > 256
    assert img_long.height >= 60

    # Determinism
    assert short_png == preparer.render_synthetic("ក", font)


def test_decree_grouped_bootstrap_is_deterministic():
    """Grouped bootstrap with a fixed seed produces identical intervals."""
    deltas = np.array([0.05, 0.10, 0.0, 0.08, -0.02, 0.12])
    groups = [[0, 1], [2, 3], [4, 5]]
    ci1 = metrics.decree_grouped_bootstrap(deltas, groups, draws=1000, seed=20260101)
    ci2 = metrics.decree_grouped_bootstrap(deltas, groups, draws=1000, seed=20260101)
    assert ci1 == ci2
    assert ci1[0] < ci1[1]
