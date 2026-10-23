# R015 results

Document-image OCR study for the Khmer Decision Lab: real line-crops versus synthetic renders.

## Primary comparison

| Condition | Codepoint CER | Grapheme CER | Exact match | Blank |
|---|---:|---:|---:|---:|---:|---:|
| A: real document crop | 10.69% | 15.05% | 474/4778 | 3 |
| B: synthetic render | 2.07% | 2.94% | 2671/4778 | 18 |
| Delta (A - B) | +8.62 pp | +12.11 pp | -2197 | -15 |

The paired delta is the macro CER difference (aggregate edits / aggregate reference units).

### Error composition

| Condition | Deletions | Substitutions | Insertions |
|---|---:|---:|---:|---:|
| Real crop | 15777 | 8533 | 4011 |
| Synthetic render | 728 | 3053 | 1698 |

## Bootstrap 95% intervals (decree-grouped)

Codepoint delta: [11.45, 14.22] pp
Grapheme delta: [16.69, 19.13] pp

Macro delta (ratio-of-sums difference): cp=8.62 pp [7.73, 9.51], gr=12.11 pp [11.31, 12.87]

## Limitations

- Images are PDF re-renders of Word-exported decrees, not photographed scans. Capture degradation (camera, lighting, skew, blur) is not measured.
- The pages dropped for a missing text layer include the genuine scans in this collection; they carry zero labels and cannot be scored.
- This measures OCR text error on line crops, not end-task intent accuracy; real-image intent evaluation remains blocked.
- Only one Tesseract version, one language configuration, one PSM/OEM setting and one render font are tested.

## Top gold characters in real OCR errors

The ten most frequent gold characters (non-space) among lines whose OCR output differed from gold.
| Character | Count |
|---|---:|---:|
| ្ | 23748 |
| ា | 20203 |
| រ | 16299 |
| ន | 14753 |
| ក | 11751 |
| ស | 9733 |
| ប | 8931 |
| ត | 8866 |
| ិ | 8776 |
| ម | 8597 |
| ង | 7633 |
| ល | 5273 |
| ច | 4743 |
| យ | 4278 |
| ុ | 4141 |