# Research state

## Current position

Latest completed study: R013, an exploratory test of seen-only adaptation against the frozen multilingual base on one fixed set of 12 held-out MASSIVE intent option names. Accuracy improved by 25.94 pp for seed 13 (95% paired group-bootstrap CI [20.62, 31.25]) and 23.12 pp for seed 14 ([17.81, 28.44]); both preregistered lower-bound criteria passed. This supports only the selected option strings and split, not broad unseen-intent transfer. Results: [R013](R013-unseen-intent-transfer/results.md); protocol: [R013 procedure](R013-unseen-intent-transfer/README.md).

R013 used a group-held-out subset of the MASSIVE training partition, which R003/R006 had already used; foundation-model pretraining exposure is unknown. Intent labels are short English strings, not natural user requests. The matched-budget all-label contrast is descriptive because training data composition differs. Seven partial reports from an earlier aborted attempt remain archived as non-results. The completed scoring and report hashes are recorded in R013's result artifacts.

R012 reused the MASSIVE test split evaluated in R001-R011, and its OCR examples are synthetic typesetting. Treat the result as exploratory, not independent confirmation or evidence of real-image robustness. The full experiment record, including overlap filtering, training telemetry, pre-score failures, and secondary metrics, is in `research/R012-ocr-continuation/results.json` and ignored `artifacts/runs/R012/` files.

Earlier OCR findings remain exploratory: R010 found lower accuracy under 12 px synthetic OCR, and R011 found lower NLL after validation-fitted OCR temperature calibration. R006 remains the clean-text baseline. See their result pages for methods and qualifications.

## Next research

No screened source provides an independent Khmer decision benchmark with verified rights, suitable labels, and a defensible untouched split. Candidates fail different gates:

- [Khmer Emotional Text](https://huggingface.co/datasets/phonsobon/Text_Emotional_kh) declares Apache-2.0 but requires access acceptance; its independent split is unverified, and emotion labels do not measure service intent.
- [Pisethan Khmer Instruct](https://huggingface.co/datasets/Pisethan/khmer-instruct-dataset) declares MIT and has 500 prompt/response pairs in one train split. It has no intent labels or held-out partition.
- [SeyhaLite Khmer Q-A-M](https://huggingface.co/datasets/SeyhaLite/Q-A-M-All-Khmer-Conversations-200k) declares Apache-2.0, but has Q&A rather than intent labels, no documented evaluation split, and unclear source provenance. Its card and listing also disagree on size.
- [nphearum Khmer Raw Text](https://huggingface.co/datasets/nphearum/khmer-raw-text-3M) declares Apache-2.0 and has topic/task-type metadata, but the card says it has no human supervised labels; it has one train split and limited source detail.
- [CADT-IDRI Khmer News](https://huggingface.co/datasets/CADT-IDRI/Khmer_News_classification) lacks a stated license and its source family was used in R004. [GKLMIP news](https://huggingface.co/datasets/SEACrowd/gklmip_newsclass) lists its license as unknown and overlaps prior project work. The CC BY-NC [Khmer ASAG benchmark](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) was used in R007. The MIT-licensed [KHOB OCR benchmark](https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset) has transcription labels, not decision labels. A [commercial Khmer call-center dataset](https://datatoorder.com/datasets/khmer-call-center-speech/) is a possible acquisition lead, but its terms, provenance, naturalistic status, and annotation quality are unverified. The [Khmer Polarity paper](https://arxiv.org/abs/2511.09313) has not established a usable rights-cleared dataset or independent split in this screen.

A later confirmatory path is frozen evaluation on naturally occurring Khmer service-intent requests from a rights-cleared first-party source, comparing both fixed R006 seeds with the matching multilingual base. This needs a source and fluent Khmer annotators; neither is secured.

Before collection, secure written research, privacy, retention, and publication rights; record source and collection period; remove direct identifiers; and exclude sensitive records unless separately approved. Recruit fluent native Khmer annotators. Define one request per item and a taxonomy for primary, multiple, ambiguous, and out-of-scope intent. Pilot and freeze the guide, double-annotate without model outputs, adjudicate disagreements, report agreement, and set an agreement gate and independent-group target. Keep the pilot out of the locked test.

Group by user, case/thread, intake channel or source batch, template family, and normalized near-duplicates. Assign whole groups to chronological train/validation/test periods. This proposed confirmatory comparison is frozen evaluation: no target training, calibration, or checkpoint selection. Keep the test labels and scores sealed until source terms, taxonomy, group/time split, model identities, and analysis are locked. Do not use model predictions as gold labels or describe generated/translated examples as naturalistic user data. If user or source IDs are unavailable, do not claim those forms of independence.

No partner source or native annotators are secured. Reassess source terms and label quality before pursuing the confirmatory collection path. R013's completed MASSIVE-based study is exploratory evidence and does not satisfy the independent-benchmark requirement.

## Resume guidance

- Repository: `/home/k/Projects/khmer-decision-lab`, branch `main`.
- Run `uv sync --frozen`; relevant checks: `uv run --frozen ruff check`, `uv run --frozen ruff format --check`, `uv run --frozen pytest -q`, and `uv run --frozen ty check`.
- RTX 3050 Laptop GPU with 6 GB VRAM is available. Large run outputs belong under ignored `data/` and `artifacts/` paths.
- Read each study's protocol before execution. Pin source and implementation hashes, keep test access behind readiness checks, and store post-lock results outside fingerprinted protocol files.
