# R013 - Transfer to unseen MASSIVE intents

**Protocol status: locked exploratory.** Lock: `datasets/manifests/r013-unseen-intent-transfer.json`. Execution results belong in `results.md` and `results.json`.

## Question and hypotheses

Does full-encoder adaptation on 48 MASSIVE Khmer intents improve choice accuracy on 12 held-out intent labels, when their names are presented as text options, compared with the same frozen multilingual base?

- **Primary hypothesis:** for each fixed seed (13 and 14), the seen-only model's paired 95% group-bootstrap lower confidence bound for accuracy minus the frozen base on the fixed held-out-intent test is above zero. Both seeds must pass to support positive transfer to these held-out option names. An interval crossing zero is inconclusive, not evidence of equivalence.
- **Secondary comparison:** compare a matched-example-budget model trained on all 60 intents with the seen-only model on the same held-out-intent test. This describes the effect of including supervised examples from the held-out labels under a fixed training budget; it is not a pure causal effect of label exposure because the training data composition changes.
- **Known-intent control:** report accuracy and other metrics on test groups from the 48 seen labels. Do not use validation or test outcomes for checkpoint, recipe, class-set, or threshold selection.

## Locked design

### Source and split

Use only the pinned `km` training partition of `mteb/amazon_massive_intent` at revision `940fd47a81eaa7f2cc7b129674d945d618ac38c2`, exported as `data/processed/massive-km-train-full.jsonl`. The pinned [MTEB dataset card](https://huggingface.co/datasets/mteb/amazon_massive_intent/tree/940fd47a81eaa7f2cc7b129674d945d618ac38c2) declares Apache-2.0; upstream [Amazon MASSIVE](https://github.com/alexa/massive/blob/main/NOTICE.md) is CC BY 4.0. Preserve source attribution and verify terms before redistributing derived data. Do not read or use the official validation or test partitions. R003 and R006 previously trained on the same source training partition; R013 starts every trained model from the pinned multilingual base, never from those checkpoints. This is an internal exploratory split, not independent confirmation.

Normalize utterances with NFKC, casefolding, and whitespace collapse. Exclude every normalized group with more than one intent label. Keep one representative row per remaining group, chosen by lowest source row index. Select 12 held-out intents with `random.Random(20261014).sample` from labels with at least 40 eligible groups; the manifest records the exact labels. Within each label, shuffle group IDs with `random.Random("20261014:<label>")`; assign `max(1, round(0.20*n))` groups to test, then `max(1, round(0.10*n))` to validation, and the remainder to training, where `n` is the eligible group count for that label. Python's built-in `round` semantics apply. The held-out test consists only of the 12 selected intents. The 48 other intents form the known-intent test.

Prepare two training arms from the training groups:

1. **seen-only:** all groups from the 48 non-held-out intents. Each row contains its gold label and 19 distractors sampled from those same 48 labels. Held-out labels never appear as gold labels or candidate options.
2. **all-label matched-budget:** a deterministic, label-proportional sample from all 60 intents, using largest-remainder allocation to match the seen-only arm's exact example count. Allocate floors first, then remaining rows by descending fractional remainder, breaking ties by alphabetical intent name. For each intent, sort training group IDs and sample without replacement using `random.Random("20261014:all-label-sample:<intent>")`. This arm must include training examples from every intent.

For each non-fixed candidate set, sort the label pool, remove the gold label, shuffle the negatives with `random.Random("20261014:<split>:<group-id>:candidate-selection")`, and take the first 19. Then shuffle the gold label and selected negatives with an independent `random.Random("20261014:<split>:<group-id>:display-order")`. For fixed held-out option sets, use the same display-order rule. The split names are `train-seen-only`, `train-all-label`, `validation-unseen`, `validation-seen`, `test-unseen`, and `test-seen`; `group-id` is the SHA-256 of the normalized utterance.

The test files are derived from the source training partition during preparation. The scoring gate prevents the training and evaluation code from reading test files or outcomes until all four trained outputs pass readiness checks.

### Choices and outcomes

Use short English intent names with underscores replaced by spaces, matching R000/R005. The primary test uses 20 options per row: all 12 held-out labels plus 8 seen-label distractors chosen once by the locked seed. Randomize display order independently by source group. The known-intent test uses the gold label plus 19 seeded distractors from the 48 seen labels, with independent display shuffling. Every model sees byte-identical test rows and option order.

This is an open-vocabulary, option-conditioned transfer test, not a fixed classifier-head test. In Laya 0.4.1, each choice option is rendered as input text and receives a logit at a shared option marker (`laya.common.render_options`, `build_head`, and `DecisionModel.forward`); there is no output neuron tied to each intent name. The seen-only model can therefore score a held-out label name at inference even though neither its examples nor that option text occurred in its training rows. The test asks whether adaptation on seen labels improves selection among these exact novel text options. It does not show that the model learned the held-out intent from supervision, nor that short English identifiers fully express the class meaning.

The primary outcome is top-1 choice accuracy on the fixed 12 selected held-out intents. For each seed, compare the seen-only model with the frozen base on identical rows and option order. Use 10,000 paired bootstrap draws: sample normalized utterance groups with replacement using `numpy.random.default_rng(13)`, retain every row in each sampled group, and calculate the row-weighted accuracy difference in percentage points. The 95% interval is the 2.5th and 97.5th percentiles using `numpy.quantile(..., method="linear")`. The support rule requires both seed-specific lower bounds above zero. This interval covers utterance-group sampling within this fixed intent set; it does not estimate variation across possible unseen intents. A crossing interval is inconclusive, not evidence of equivalence.

Report per-intent recall over gold labels, macro-F1 over gold labels (a prediction to a distractor is a false negative for its gold class), and macro one-vs-rest AUROC descriptively. For AUROC, use only rows whose candidate set contains that label; average labels with both positive and negative rows. Also report mean multiclass NLL, categorical Brier (mean row-wise sum of squared option-probability errors), top-answer Brier, 15-bin equal-width ECE, mean answer confidence, AURC, and selective accuracy at 50% and 80% coverage. Keep seeds separate.

### Training and inference

Train each arm from the pinned `convaiinnovations/laya` multilingual base revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, with seeds 13 and 14. Both arms use the same exact row count and two epochs: micro-batch 1, accumulation 32, max lengths 512/256, encoder/head learning rates `2.5e-5` / `1e-4`, minimum learning rate `1e-6`, weight decay `0.01`, gradient clip `1.0`, soft-CE, full encoder, choice-option shuffling, CUDA AMP, gradient checkpointing, and fresh PagedAdamW8bit state. Expected budget is 376 optimizer attempts per run. Do not calibrate, evaluate validation during training, early-stop, or select checkpoints.

Use `laya_training_guard()` so the cosine scheduler advances only after successful optimizer updates. Record attempts, successful/skipped updates, scheduler steps, optimizer identity, runtime, peak VRAM, hardware, package versions, failures, and weight hashes. Require at least 95% successful updates and at most 5 successful-update difference between arms for each seed. Every output must have changed weights, identity calibration, no training-time evaluation, and authenticated input/configuration provenance before any test file is opened.

Evaluate the frozen multilingual base and all four trained outputs with the same R010-compatible CUDA BF16 policy, per-call `max_len=512` and `head_max_len=256`, explicit batch size 1, and model-unload-before-load discipline. Laya's evaluator batches only consecutive rows with identical ordered question schemas; the prepared test rows randomize option order independently, and the first attempt recorded every row as a singleton (`rows_grouped=0`, `max_chunk=1`). Pin batch size 1 so the report matches the effective call shape. The shared context limits apply even when a checkpoint's saved config permits a longer context. Do not fit or tune calibration. Verify output temperatures are identity. Record inference runtime and peak allocated VRAM separately from training.

Before scoring, an inference-only amendment set common context limits because the frozen base config allows `max_len=1024`, while trained outputs use 512. It changes no training recipe, data, weights, estimand, or thresholds. Existing preparation and training artifacts remain bound to the predecessor manifest and protocol hashes; the amended manifest records that lineage. The predecessor manifest is archived at `datasets/manifests/r013-unseen-intent-transfer-pre-amendment.json`.

Scoring under that amendment left seven per-model reports from failed attempts, and one attempt stopped at all-label seed 13, unseen-test row 225 with a probability-validation error. The diagnostic did not identify which probability invariant failed. Treat every partial report and printed metric as a non-result; the files are preserved under `artifacts/runs/R013/pre-score-failures/partial-scoring-ac704/`.

A second inference-only amendment pins the known Laya rewrite of trained `tokenizer/tokenizer_config.json` and the recovery path. Before test rows are read, verify each `checkpoint_latest` inventory against training provenance, require the output root to differ only by this metadata file and only by the locked runtime hash, restore its exact bytes from `checkpoint_latest`, then rerun strict readiness checks. Apply the same guarded restoration immediately after each trained checkpoint loads. This amendment uses batch size 1 and validates the rounded probability sum against the 0.001 bound with a `1e-12` floating-point allowance. Record restoration hashes. These changes preserve training runs, weights, prepared rows, inference policy, estimand, and support thresholds. The second amendment is bound to the first inference-amended manifest `datasets/manifests/r013-unseen-intent-transfer-pre-recovery.json`; existing preparation and training outputs retain their earlier lineage.

Independent review then found that the second amendment's `0.0001` per-value tolerance could accept a report probability or answer confidence one four-decimal step away from correct rounding. Before any scoring retry, a third inference-only amendment now computes the choice softmax using the captured float32 logits and each loaded checkpoint's choice temperature, and requires every reported four-decimal probability and answer confidence to match exactly. Its rounded-sum bound is `0.00005` per option plus the measured float32 softmax normalization residual and `1e-12`. It retains the tokenizer recovery, singleton batching, training outputs, inference policy, estimand, and thresholds. The amendment descends from `datasets/manifests/r013-unseen-intent-transfer-pre-rounding-review.json`. At the time this amendment was locked, no scoring retry had occurred; earlier partial reports remained non-results. Final scoring later completed under this amendment.

After scoring, report validation was amended to accept the evaluation index and reports bound to the immediate predecessor manifest and to verify the recorded singleton-call metadata (`batch_size=1`, `max_chunk=1`). This reporting-only change did not rerun inference or change scored artifacts, metrics, bootstrap computation, estimand, or support thresholds. See `results.md` for completed outcomes and artifact identities.

## Scope and limitations

This estimates performance on one fixed, randomly selected set of 12 English-named MASSIVE intent options under a 20-choice prompt. It does not estimate transfer to a population of unseen intents, unseen domains, customer-service requests, or naturalistic deployment. The source training partition was reused by earlier project checkpoints, though R013 starts from the multilingual base; foundation-model pretraining exposure to MASSIVE is unknown. Exact normalized duplicates are grouped, but paraphrase leakage is not measured. English label wording may expose useful semantics, and some short identifiers may be ambiguous or opaque. There is one class split and two fixed training seeds. Treat results as exploratory and do not use them to tune later models.

## Reproduction

```sh
uv sync --frozen
uv run --frozen pytest -q tests/test_r013_unseen_intent.py
uv run --frozen python scripts/prepare_r013.py
uv run --frozen python scripts/train_r013.py 13 seen-only
uv run --frozen python scripts/train_r013.py 13 all-label
uv run --frozen python scripts/train_r013.py 14 seen-only
uv run --frozen python scripts/train_r013.py 14 all-label
uv run --frozen python scripts/evaluate_r013.py
uv run --frozen python scripts/summarize_r013.py
```
