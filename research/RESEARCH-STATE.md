# Research state

## Current position

- Latest completed study: R010, exploratory synthetic OCR evaluation. Results and protocol: [R010](R010-massive-ocr-cascade/results.md) and [locked procedure](R010-massive-ocr-cascade/README.md).
- R010 found 12 px Tesseract OCR reduced accuracy by 9.89 pp and 10.12 pp for fixed R006 seeds 13 and 14; both paired group-bootstrap intervals excluded zero. This is evidence for those checkpoints on the reused MASSIVE test split, not real-image robustness or independent confirmation.
- R010's test has 2,786 normalized utterance groups, 34 conflicting-label groups (129 rows), and 267 groups (429 rows) with exact training overlap. Preserve these caveats in follow-up analysis.
- Best established MASSIVE result remains R006 seed 13 at 84.10% on the benchmark test; seed 14 scored 83.69%. The evaluation split has been reused across R001-R010. Full history and qualifications are in the experiment results, especially [R006](R006-multitask/README.md), [R008](R008-specialist-controls/results.md), and [R009](R009-sib-specialist-continuation/results.md).

## In progress

R011 is selected as an exploratory OCR-domain calibration study. It will fit an OCR-specific temperature map from synthetic 12 px validation transcriptions, then compare against each R006 checkpoint's existing clean-text calibration on R010's OCR test inputs. The primary metric will be test NLL; fit and score only the specified splits. Exact normalized validation utterance groups overlapping train or test, and groups with conflicting labels, must be excluded from calibration. Test reuse makes any result exploratory. No R011 protocol is locked and no R011 calibration or test scoring has started.

## Immediate next action

Audit the exact eligible calibration rows and source hashes; write and lock an R011 protocol before rendering validation OCR or accessing R010 test reports. Then implement validation-only calibration, a pre-test readiness gate for both seeds, paired group-bootstrap scoring, and regression tests. Keep model weights unchanged and report accuracy invariance alongside NLL, Brier, ECE, and selective-risk diagnostics.

## Resume guidance

- Repository: `/home/k/Projects/khmer-decision-lab`, branch `main`.
- R010 working tree was verified with `uv run --frozen ruff check .`, `uv run --frozen ruff format --check .`, and `uv run --frozen pytest -q` (112 passed). The R010 summarizer regenerated `research/R010-massive-ocr-cascade/results.json` after validating all report hashes and paired bootstrap results.
- RTX 3050 Laptop GPU with 6 GB VRAM is available. Large run outputs belong under ignored `artifacts/` and `data/` paths.
- For every study, read its protocol before execution, pin the source and implementation hashes, keep test access behind readiness checks, and put post-lock results outside the fingerprinted protocol README.
