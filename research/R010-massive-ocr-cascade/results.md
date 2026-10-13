# R010 results

Status: scored exploratory. The primary hypothesis was supported for the two fixed R006 checkpoints on this MASSIVE test split under the locked synthetic 12 px Tesseract condition. The protocol README remains unchanged; its status line records the state at lock.

## Primary comparison

| Checkpoint | Clean accuracy | OCR 12 px accuracy | Difference, 95% paired group-bootstrap CI |
|---|---:|---:|---:|
| R006 seed 13 | 84.10% | 74.21% | -9.89 pp [-11.15, -8.65] |
| R006 seed 14 | 83.69% | 73.57% | -10.12 pp [-11.43, -8.81] |

Both interval upper bounds are below zero, meeting the preregistered success rule. The intervals resample 2,786 NFKC-normalized utterance groups with all rows retained and row-weight the differences; 10,000 draws, NumPy seed 13. This supports an accuracy loss only for these checkpoints and this reused test split, not a population-level or general OCR claim.

The contextual R003 checkpoint scored 83.69% on clean text and 73.77% at 12 px (-9.92 pp, 95% CI [-11.16, -8.65]). Full paired results for all checkpoints and image sizes are in `results.json`.

## OCR and secondary results

| Render | Codepoint CER | Grapheme CER | Exact rows | Blank rows |
|---|---:|---:|---:|---:|
| 48 px | 5.26% | 8.20% | 1,766 / 2,974 | 5 |
| 24 px | 5.44% | 8.21% | 1,773 / 2,974 | 27 |
| 12 px | 11.94% | 16.13% | 895 / 2,974 | 1 |

OCR error was slightly higher at 24 px than 48 px, so font size is not a linear error scale. All three checkpoints had lower accuracy at each OCR condition than on clean text. At 12 px, R006 mean confidence fell from 84.19% to 77.45% (seed 13) and 83.27% to 77.09% (seed 14); ECE, top-answer Brier, and AURC all worsened. Temperatures were frozen, not refit for OCR. The report includes per-label recall, macro-F1, selective accuracy, train-overlap strata, and normalized exact-versus-changed-transcription strata.

## Execution and provenance

Preparation verified all 2,974 source rows, 2,786 normalized groups, 34 conflicting-label groups spanning 129 rows, and 429 train-overlap rows across 267 groups. Tesseract transcribed 2,786 distinct raw utterances per font size. The evaluator produced all 12 model-condition reports on the RTX 3050 6 GB GPU; reported peak allocated VRAM was 1,504.5 MiB per run. Report hashes, source alignment, row accuracy, and the primary bootstrap intervals were independently recomputed after scoring.

Before scoring, independent review found that a loop-local Laya agent could remain alive while the next checkpoint loaded. The evaluator now drops that reference before `Router.unload()` collects the guard-hook cycle. An integration regression test using the actual Laya Router and stub agents fails on the prior code and passes on the fix. The locked manifest was amended before any R010 model scoring; no inference or analysis rule changed.

## Limits

This is synthetic typesetting, not a real-image OCR evaluation: text was rendered in one font and transcribed by one Tesseract version/configuration. MASSIVE test data were previously scored in R001-R008, so R010 is not independent confirmation. Duplicate utterances, conflicting-label groups, and exact train overlap remain in the test and are handled or reported as specified by the locked protocol. OCR-exact strata use NFC and whitespace-normalized comparison; they are descriptive, not causal.
