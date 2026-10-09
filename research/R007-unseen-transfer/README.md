# R007 - Frozen-checkpoint transfer

**Status: exploratory protocol locked; frozen-checkpoint scoring complete.**
The primary MKQA hypothesis was not supported under its prespecified rule;
results are summarized below and stored in [`results.json`](results.json).
ASAG and MKQA replace the conditional SEA-NLI target. No publisher access
approval is needed for these public sources.

Question: do the frozen R006 checkpoints transfer better than MASSIVE-only
R003 to tasks absent from their adaptation data? This does not establish
absence from Laya pretraining or superiority over task-specific training.

## Targets and scope

| Role | Derived target | Audit rows/groups | Test rows/groups | Budget |
|---|---|---:|---:|---:|
| Primary, exploratory | MKQA Khmer questions with English entity choices | 694 / 545 | 2,093 / 1,723 | 512 / 256 |
| Secondary, diagnostic | ASAG consistent, unique-input teacher-credit slice | 143 / 9 | 358 / 23 | 2,048 / 256 |

Budgets are `max_len / head_max_len`, identical across checkpoints within
each target. Do not pool the tasks' scores or compare their efficiency as
if their inputs and budgets were identical.

The executable preparation rules and exact hashes/counts are in
[`scripts/prepare_r007.py`](../../scripts/prepare_r007.py) and the
[manifest](../../datasets/manifests/r007-unseen-transfer.json).
Row-level source IDs, groups and exclusions remain in git-ignored `data/`.
The [dataset audit](dataset-audit.json) preserves the earlier rejected/gated
candidates and records the amended selection.

### MKQA derived entity choices

Source: [Apple MKQA](https://github.com/apple/ml-mkqa/tree/f964fe1bac4d580cee3d4928572df1f676f8e656),
a 10,000-query, 26-language open-ended QA release. Dataset license:
**CC BY-SA 3.0**; retain attribution and share-alike terms for derived data.
The publisher describes human-translated queries and passage-independent
answers. This experiment uses the original `km` question, not a new translation.

Include questions whose English and Khmer annotations agree on one entity
QID. Exclude other answer types, ambiguous/mismatched QIDs, queries without
Khmer characters and entities without an uncapped, unambiguous display name.
Use one name vocabulary from the English entity annotations for **both**
correct and incorrect options. A reserved preview revealed Thai-script
names in the original `km` answers; uniform English annotation names remove
that uncontrolled answer-script variation. This is a cross-lingual choice
task, not fully Khmer answer generation.

Each question has its accepted entity plus three seeded random other
entities. Exclude the gold QID and known English/Khmer answer aliases.
Shuffle answer position with a separate deterministic random stream.
The whole-release entity vocabulary is part of benchmark construction;
it is not a target-trained model or a test-performance-based selection.
Entity IDs, English queries and gold mappings never enter model state.
All four options are shown uniformly.

**These are not standard MKQA scores.** Random distractors may be easy,
semantically mismatched or still ambiguous beyond the recorded aliases.
Some questions/answers are dated or awkwardly translated. Performance
measures agreement with this derived entity-choice release, not open-ended
QA, retrieval, native-language answer production or broad reasoning.

### ASAG teacher agreement

Source: [phorknorak/khmer-asag-benchmark](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark/tree/5fadaef3422c5784458d662394b5d891e7ebb659),
**CC BY-NC 4.0**. Use `dataset_no_10c_biology.csv`, the publisher's 909-row
curated export from real school exams. The full release contains 1,184
answers from 203 students, 41 questions, four subjects and one grader.
Do not re-identify or contact students, teachers or schools; do not deploy
this as standalone high-stakes grading. Keep source attribution and the
noncommercial restriction when sharing derived data.

Input only question, reference and student answer. Omit school, class,
student, year, scores and all other metadata. Gold bins are
`round(4 * Student Score / Max Score)`, using half-even rounding, representing
nearest 0/25/50/75/100% teacher credit. This measures **teacher agreement**,
not universal correctness or a validated grading rubric.

The audit found 118 rows with conflicting binned grades for identical
normalized inputs. Exclude all such rows rather than choose a preferred
label. Collapse 286 duplicate input/label rows; exclude one invalid score
and three inputs that still truncate. This selected slice does not represent
the original student-response frequency distribution. The exclusions use
released labels before scoring, not checkpoint performance.

A page preview exposed Biology Q7-Q10 before reservation. Their entire
connected groups are forced into audit; never describe ASAG as an untouched
holdout. Inputs have a median 602 tokens before the question head, so the
original 512-token protocol was infeasible. The pre-scoring amendment uses
2,048 tokens, retaining complete inputs only. Source encoders support 8,192
positions; actual long-context inference memory remains unmeasured.

Students cross question groups: 72 retained student hashes occur in both
partitions. The diagnostic is not unseen-student generalization. Source
references and partial-credit judgments may themselves be imperfect; the
small spot-check does not certify teacher-label quality or native-language
expert review. Rare bins and only 23 test groups preclude precise gain claims.

## Split, inspection and overlap rules

Both releases lack an independent audit partition. Before manual review,
form connected components from normalized duplicate questions. ASAG also
joins identical references/answers of at least 60 normalized characters;
MKQA also joins target QIDs and duplicate English queries. This catches
spacing/punctuation variants, not every paraphrase or shared topic.

Sort group IDs, shuffle with `random.Random(20261009)` and reserve
`ceil(0.2 * groups)`. Union previously exposed ASAG Biology groups, the eight
MKQA source IDs inspected under the superseded answer rendering, and 87 MKQA
test rows accidentally emitted by an internal shell search before protocol
lock. The 87 rows and every final connected component containing them are
forced into audit; the exact source IDs are disclosed in the
[dataset audit](dataset-audit.json) and persisted in the manifest. The
remainder is a **derived test partition**, not an untouched published test.
Audit/test group intersection is zero. The manifest persists all reserved
source IDs, including
subsequently excluded rows. Amendments force their new components into audit;
preparation refuses to overwrite a locked manifest.

Check all adaptation rows, including calibration, against manifest-verified
MASSIVE, News and SIB-200 sources. NFKC letters/numbers/marks plus casefold
and bidirectional substring matching require at least 60 retained characters.
No matching groups were found under this rule. This does not exclude shorter,
partial, semantic, near-duplicate or pretraining exposure. Whole-passage
matching previously missed Belebele's embedded SIB sentences.

Retained rows have complete instructions, state and options under the
production Laya renderer, equal token sequences across the four checkpoint
tokenizers and distinct option spans. Label-quality inspection stays in
reserved audit data. No prompt search, calibration or seed selection.

## Hypothesis and analysis

H1, exploratory: **both** frozen R006 seeds improve accuracy over R003 on
the primary derived MKQA task. Report paired differences in percentage
points. A positive pattern requires each paired 95% interval's lower bound
to exceed zero; intervals crossing zero are inconclusive, not equivalence.
Do not select the better seed or use ASAG to rescue H1.

Use 10,000 paired group-bootstrap resamples, seed 13, with the same sampled
entity/query groups for all models. Draw groups uniformly with replacement;
retain every row of each drawn group and divide correct totals by sampled row
counts. Use percentile 2.5/97.5 bounds, not equally weighted group means.
Intervals are conditional on these fixed
checkpoints and the grouping assumption. Related entities/questions may
remain dependent. R003 has one seed; training exposure, method and seed
effects are not causally isolated.

Before predictions, MKQA's group-size-weighted effective count is about
1,330. Under independent groups, distribution-free 95% Hoeffding half-widths
are 3.72 points for accuracy and 7.45 for a paired difference. These are
conservative **design bounds**, not observed confidence intervals or a
power guarantee. ASAG's effective count is about 16; shared students also
violate that independence assumption. Keep ASAG descriptive; any question-
bootstrap intervals must disclose the unmodeled crossed student dependence.

For both tasks, report accuracy, ECE (15 bins), top-answer Brier, AURC and
selective accuracy at 50%/80% coverage. Include entire confidence-tie groups
and report actual coverage. Add macro-F1 over all five bins and per-grade recall
for ASAG; report its paired differences descriptively without inferential CIs. For MKQA, A-D recall describes answer-position bias, not four semantic
classes. Report base comparisons and audit-fitted majority/position baselines
separately; fit constant labels only on audit counts, breaking ties
lexicographically. Never present a test-selected majority label as a fitted baseline.

## Preparation and scoring gate

```bash
uv run python scripts/prepare_r007.py
```

Requires `curl`, existing local checkpoint/tokenizer files and unchanged
R006 adaptation exports. Pins both source revisions, verifies raw hashes,
checks all four tokenizers and records checkpoint weight, agent/encoder config
and tokenizer fingerprints. Does not train, calibrate, load model weights for inference or
predict. Raw data, exports and row provenance stay outside Git.

This protocol, preparer and manifest are committed together with manifest
status `locked_exploratory`. That lock is required before scoring. Evaluation
with `--manifest` verifies the frozen exports/provenance, checkpoint
files, preparer identity, model order and target budgets before any prediction.
Use `--verify-only` to check without inference. Canonical `r007-*` exports
require the manifest. Preserve the recorded checkpoint choice
temperatures and all weights. Do not fit target temperatures or abstention
thresholds in R007. Reserve audit data for quality review, not model selection.

After that gate only:

```bash
for spec in mkqa-entity-choice:512 asag:2048; do
  task=${spec%:*}; budget=${spec#*:}
  MAX_LEN=$budget HEAD_MAX_LEN=256 uv run python scripts/evaluate.py \
    data/processed/r007-$task-test.jsonl artifacts/runs/r007-$task-frozen \
    control-base-maxlen512,full-e2-8bit,mt-s13,mt-s14 \
    --manifest datasets/manifests/r007-unseen-transfer.json
done
uv run python scripts/summarize_r007.py
```

Evaluation carries a SHA-256 tag per report case, binding the exact exported
row to its source and group IDs. The R007 summarizer checks those tags before
pairing rows, then validates rounded probability normalization, selected-choice
argmax and top-probability confidence against the fixed Laya checkpoint
semantics. It recomputes accuracy/calibration/selective metrics and writes
aggregate `results.json` in this experiment directory. Full reports remain in
git-ignored artifacts.
It does not modify historical `research/results.json` or use its row bootstrap.

The base alias retains `512` in its name; the explicit override makes ASAG
2,048 for **every** checkpoint. Do not mistake that alias for the actual budget.
Record RTX 3050 6 GB hardware, software, failures and effective batch grouping.
MKQA's row-varying choices may reduce requested batch size 8 to single rows.
Use `isolated-warm-v1` timing/memory; do not overwrite historical reports or
compare them directly with the earlier cold-load measurement protocol.

No new training is planned. OCR robustness and matched single-task controls
remain separate follow-ups. Resolve the recorded AMP scheduler issue before
any later training experiment.

## Observed results

The primary derived MKQA test has 2,093 rows in 1,723 groups. R003
(`full-e2-8bit`) reached 43.2% accuracy; R006 `mt-s13` reached 41.8%
(difference -1.43 pp, paired 95% CI [-3.48, 0.63]) and `mt-s14` reached
45.1% (+1.86 pp, CI [-0.10, 3.88]). Both intervals include zero, so the
prespecified requirement that both lower bounds exceed zero is not met.
This is inconclusive evidence for the positive hypothesis, not evidence of
equivalence. These are results on the derived English-entity-choice task,
not standard MKQA.

ASAG is descriptive only: on 358 rows across 23 question groups, accuracy
was 15.4% for R003, 16.5% for `mt-s13` and 20.1% for `mt-s14`. The audit-fit
constant-label baseline reached 42.7%. Treat this as teacher-agreement
diagnostic evidence, not a valid standalone grading result; students cross
partitions and the study has one teacher. ASAG does not alter the primary
MKQA decision. All evaluated checkpoints remained frozen; no target training
or recalibration was done.
