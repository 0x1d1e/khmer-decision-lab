# Research state

## Current position

- Latest completed study: R011, exploratory OCR-domain temperature calibration. Results: [R011](R011-ocr-domain-calibration/results.md); locked protocol: [R011 procedure](R011-ocr-domain-calibration/README.md).
- On the no-train-overlap subset of R010's synthetic 12 px OCR test, OCR-fitted temperatures lowered mean NLL by 0.00840 for R006 seed 13 (95% grouped-bootstrap CI [-0.01217, -0.00479]) and 0.00802 for seed 14 (CI [-0.01223, -0.00396]); accuracy was unchanged. The maps were fitted on 1,715 leakage-filtered MASSIVE validation transcriptions. This is not independent confirmation or real-image evidence because the test split was reused in R001-R010 and the OCR recipe is synthetic.
- R010/R011 test caveats remain: 2,786 normalized utterance groups, 34 conflicting-label groups (129 rows), and 267 exact train-overlap groups (429 rows). R011 excluded the train-overlap rows from its primary NLL analysis; conflicting-label groups remained clustered.
- Best established clean-text MASSIVE result remains R006 seed 13 at 84.10%; seed 14 scored 83.69%. Full history and qualifications are in [R006](R006-multitask/README.md), [R008](R008-specialist-controls/results.md), [R009](R009-sib-specialist-continuation/results.md), and [R010](R010-massive-ocr-cascade/results.md).

## In progress

R012 will test whether fixed-budget continuation on synthetic OCR transcripts improves R006 decision accuracy under the R010 12 px condition beyond matched clean-text continuation. The next step is to resolve training and optimizer details, specify clean-text preservation diagnostics, pin data and code, and lock before preparing OCR train inputs or using evaluation outputs. R010's test reuse means the comparison remains exploratory.

## Immediate next action

Design the paired R012 clean-versus-OCR continuation protocol. Keep training inputs restricted to MASSIVE train, use matched fixed updates from the same R006 checkpoints, forbid test-based selection, and retain the original clean and OCR evaluations as separate outcomes. Determine how exact train-overlap rows will be handled before locking.

## Resume guidance

- Repository: `/home/k/Projects/khmer-decision-lab`, branch `main`.
- R011 passed `uv run --frozen ruff check .`, `uv run --frozen ruff format --check .`, `uv run --frozen pytest -q` (120 passed), `uv run --frozen ty check ...`, and LSP diagnostics. A post-review calibration/evaluation/summarization rerun reproduced the R011 calibration payload hashes and result artifacts.
- RTX 3050 Laptop GPU with 6 GB VRAM is available. Large run outputs belong under ignored `artifacts/` and `data/` paths.
- For every study, read its protocol before execution, pin source and implementation hashes, keep test access behind readiness checks, and put post-lock results outside the fingerprinted protocol README.
