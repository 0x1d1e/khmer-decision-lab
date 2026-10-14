# Research state

## Current position

Latest completed study: R012, a locked exploratory comparison of matched clean-text and synthetic 12 px OCR continuations from the two fixed R006 checkpoints. The OCR continuation improved accuracy on the OCR input by 3.18 pp (95% grouped-bootstrap CI [1.91, 4.52]) for seed 13 and 2.43 pp ([1.27, 3.59]) for seed 14. Clean-input preservation passed the preregistered -2 pp threshold for both. Results: [R012](R012-ocr-continuation/results.md); protocol: [R012 procedure](R012-ocr-continuation/README.md).

R012 reused the MASSIVE test split evaluated in R001-R011, and its OCR examples are synthetic typesetting. Treat the result as exploratory, not independent confirmation or evidence of real-image robustness. The full experiment record, including overlap filtering, training telemetry, pre-score failures, and secondary metrics, is in `research/R012-ocr-continuation/results.json` and ignored `artifacts/runs/R012/` files.

Earlier OCR findings remain exploratory: R010 found lower accuracy under 12 px synthetic OCR, and R011 found lower NLL after validation-fitted OCR temperature calibration. R006 remains the clean-text baseline. See their result pages for methods and qualifications.

## Next research

Do not reuse the MASSIVE test split for further model selection. First identify an independent Khmer decision or intent dataset with usable labels, a defensible split, and clear access terms. Khmer OCR transcription benchmarks may inform input-noise research, but transcription labels alone do not measure downstream intent decisions. No R013 protocol is locked or data prepared.

## Resume guidance

- Repository: `/home/k/Projects/khmer-decision-lab`, branch `main`.
- Run `uv sync --frozen`; relevant checks: `uv run --frozen ruff check`, `uv run --frozen ruff format --check`, `uv run --frozen pytest -q`, and `uv run --frozen ty check`.
- RTX 3050 Laptop GPU with 6 GB VRAM is available. Large run outputs belong under ignored `data/` and `artifacts/` paths.
- Read each study's protocol before execution. Pin source and implementation hashes, keep test access behind readiness checks, and store post-lock results outside fingerprinted protocol files.
