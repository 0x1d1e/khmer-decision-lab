# R015 - Document-image OCR versus the synthetic proxy

**Status: exploratory measurement. The harness is validated before the primary run.**

## Question and hypothesis

How much does R010's synthetic OCR proxy underestimate the OCR error that a real
document image produces on the same text? R010 rendered clean task text in one
font and transcribed it; R015 pairs that synthetic condition against a
document-image crop of the *identical* text, so the difference isolates the
image-pipeline effect from the text-domain effect.

Hypothesis: real document-image line crops produce higher codepoint and grapheme
CER than the matched synthetic render of the same gold text. The primary
quantity is the paired within-text delta, `CER(real) - CER(synthetic)`.

## What this study is and is not

- **Is:** a paired within-text control. Both conditions read the same gold line
  text; only the image pipeline differs.
- **Is not:** real-image intent evaluation, real-scan evaluation, or a
  capture-degradation study. The source PNGs are PDF re-renders of
  Word-exported decrees, not photographed scans. Camera geometry, lighting,
  skew, blur, paper texture and compression artifacts are absent.
- **Consequence:** this bounds how far a synthetic proxy can fall short of *one*
  document pipeline. It does not measure how far it falls short of a real scan,
  and it cannot unblock real-image intent accuracy.

## Data source

`/home/k/data/structure-839/`:

- 836 page PNGs at 1275x1651, 150 dpi, rendered from the decree PDFs.
- `lines_gt/`: one JSON per page with each line's `bbox`, `text`, `id`,
  `reading_order_index`, `n_khmer` and `text_validity`.
- `lines_gt_split.json`: decree-grouped split (seed 20260101), grouped by decree
  number and year, never by page. Test = 175 pages / 5,565 lines across 17
  decree groups.
- `lines_gt_manifest.json`: per-page quality fields used by the filter.

### Quality filter (pre-registered)

A page is scored only if all of the following hold in `lines_gt_manifest.json`:

| Field | Requirement |
|---|---|
| `text_ok` | `true` |
| `dup_frac` | `<= 0.02` |
| `bad_coeng_frac` | `<= 0.002` |
| `dep_dep_frac` | `<= 0.02` |

This keeps **148 test pages / 4,778 lines**. The 27 dropped test pages split into
4 pages with no text layer (zero labels) and 23 pages that fail a quality
threshold. The ~38 pages dropped across the whole collection for a missing text
layer are the genuine scans in this set (stamps, signatures); they carry no
labels and therefore cannot be scored. Two of those fall in the test split.

## Protocol

For each line in the filtered test split:

### Condition A - real document-image crop

1. Open the page PNG from `images/`.
2. Crop the line using its `bbox` (integer-rounded, xyxy).
3. Run Tesseract 5.5.3 `khm+eng` with `--psm 7 --oem 1`.
4. Score codepoint and grapheme-cluster CER against the text-layer gold.

### Condition B - synthetic render (R010 recipe, matched text)

1. Render the same gold line text: Noto Sans Khmer Regular
   (`/usr/share/fonts/noto/NotoSansKhmer-Regular.ttf`), 48 px, black on white,
   Pillow with RAQM layout, 16 px horizontal and 12 px vertical padding, minimum
   canvas 256x60.
2. Run the same Tesseract command on the render.
3. Score against the same gold text.

### Harness validation

Before the primary measurement, MASSIVE Khmer test utterances
(`data/processed/massive-km-test-full.jsonl`) are rendered through the identical
synthetic pipeline and transcribed. R010 measured 5.26% row-weighted codepoint
CER at 48 px; R015 fails fast if the reproduction strays more than 1 pp.

The check runs over the full 2,974-row file rather than a 300-row slice. A
300-row subsample has its own standard error of ~0.55 pp (observed 300-row CERs
span roughly 4.2-6.4% across seeds), which is too noisy to separate a broken
harness from ordinary sampling variation under a 1 pp gate. The full-set
row-weighted aggregate reproduces R010's metric definition exactly and lands on
5.26%.

## Metrics

- **Primary:** paired delta `CER(real) - CER(synthetic)` in codepoint and
  grapheme CER, reported both macro (aggregate edits / aggregate reference units)
  and as a per-line mean.
- **Intervals:** decree-grouped bootstrap, resampling decree groups (not lines),
  10,000 draws, seed 20260101, 95% percentile interval. Grouping by decree keeps
  page- and document-level correlation inside a resampling unit.
- **Error composition:** codepoint deletions, substitutions and insertions per
  condition.
- **Secondary:** exact-match rate and blank OCR outputs per condition.
- **Reproducibility:** per-line `gold_text`, both OCR strings, both CERs and both
  edit-op records are written to JSONL.

CER matches R010: `editdistance`-style Levenshtein over codepoints after NFC
normalization and whitespace collapse, and over `regex`-derived extended
grapheme clusters (`\X`).

## Scope and deviations

- **Not a scan study.** Every image is a PDF re-render. Capture degradation is
  not measured, and the 38 genuine scans in the collection are unscorable.
- **Matched text, one pipeline each.** Condition B uses the R010 renderer; the
  real crop uses the existing page raster. No augmentation, resizing or
  compression is applied to either.
- **One OCR configuration.** Tesseract 5.5.3, `khm+eng`, `--oem 1 --psm 7`, with
  the system tessdata. No engine, PSM sweep, preprocessing or fine-tuning is
  tested.
- **Line-level, not task-level.** This measures text error on annotated line
  crops. It says nothing about intent accuracy on real images, which remains
  blocked.
- **The gold text is the PDF text layer.** It is not independently transcribed,
  so residual label error is possible; the quality filter removes the worst
  pages.
- The `fintech-ocr` recognizer is out of scope for this round.

## Reproduction

```sh
uv sync --frozen
uv run --frozen pytest -q tests/test_r015.py
uv run --frozen python scripts/prepare_r015.py --dry-run
uv run --frozen python scripts/prepare_r015.py
uv run --frozen python scripts/summarize_r015.py
```

`prepare_r015.py` writes `artifacts/runs/R015/per-line-results.jsonl`,
`preparation.json` and `summary.json`. `summarize_r015.py` rebuilds `summary.json`
from the JSONL and writes `results-table.md`. Outcomes live in `results.md` and
`results.json`.
