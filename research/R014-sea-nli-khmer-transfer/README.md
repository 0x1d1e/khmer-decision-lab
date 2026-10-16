# R014 - SEA-NLI Khmer NLI transfer

**Protocol status: locked exploratory.** Lock manifest is written by
`scripts/prepare_r014.py` to `datasets/manifests/sea-nli-km.json` at run time
(it depends on gated data) and **must be committed together with the results**,
the same as every other study manifest under `datasets/manifests/`. The
manifest holds only ids, hashes, counts, and rules - no gated text - so a result
reported without its committed manifest is not reportable. Execution outcomes
belong in `results.md` and `results.json`, which no code author produces.

## Question and hypotheses

Does full-encoder Khmer adaptation beat the matched frozen multilingual base on
a genuinely unseen task type - 3-way natural language inference - scored in the
lab's candidate-scoring format?

- **Primary hypothesis:** for each fixed R006 seed (13 and 14), the adapted
  model's paired 95% group-bootstrap lower confidence bound for accuracy minus
  the frozen multilingual base on pooled Khmer SEA-NLI rows is above zero. Both
  seeds must pass to support positive transfer. An interval that crosses zero is
  inconclusive, not evidence of equivalence. This is a pre-registered support
  rule, not a significance claim.
- **Secondary, descriptive only (no support claims):** Khmer `normal` alone,
  Khmer `hard` alone, the same bootstrap for the `full-e2-8bit` versus base
  contrast, the non-Khmer slice, and the matched english control arm.

**Power statement, fixed before scoring.** The Khmer slice is 65 rows in 37
premise groups (normal: 42 rows / 14 groups; hard: 23 rows / 23 groups; the two
configs share no `concept_id`). Only very large accuracy differences are
resolvable at that size; a wide interval is an expected outcome, not a failed
run. A zero-crossing interval is inconclusive, not evidence of equivalence, and
no effect size is claimed as "no difference".

## Locked design

### Source and split

Use the test split only of `aisingapore/SEA-NLI` at revision
`f3a73ccb4213ff995ef2bf8bcb903eaf53195713`, configs `normal` and `hard`
(`normal/test-00000-of-00001.parquet` and `hard/test-00000-of-00001.parquet`).
The dataset is gated with `manual` access, so preparation and scoring require an
authenticated account that has accepted the terms. It declares MIT; the Khmer
text is machine-generated from Wikipedia concepts with native verification, and
the Wikipedia seeds carry CC BY-SA. The MIT declaration coexists with those
CC BY-SA seeds. Only two files exist: there is **no train or validation split**.
No SEA-NLI temperature is fitted; checkpoint temperatures are used unchanged.

The labels are public on Hugging Face, so sealing is **procedural only**: this
protocol fixes the design before scoring, but a reader cannot be prevented from
having seen the answers.

### Task mapping

One question, qid `relationship`, `type: "choice"`, with exactly three criteria
in this fixed order: `entailment`, `neutral`, `contradiction`. Each criterion's
description is the empty string, so `laya.common.render_options` renders exactly
the bare labels. A preparation test asserts those rendered strings. The option
order is identical for every row and every model so Laya's consecutive-row
batching stays coherent. The state is `{"premise": <text>, "hypothesis": <text>}`; the model-visible
keys are fixed at `premise`/`hypothesis` and the instructions name both. The
native arm reads `premise_native`/`hypothesis_native`; the english control arm
reads `premise_english`/`hypothesis_english` (GPT translation) and is matched
row for row. Gold comes from the dataset's `true_label` field, not from
`entailment_type`: the two disagree on 4/42 normal and 7/23 hard Khmer rows
(11/65). Source text is kept byte-identical to the source; no normalization is
applied to model-visible text.

### Row selection and hygiene

- **Khmer selection is by `culture`.** A row is Khmer when its `culture` value
  casefolded equals `"cambodian"` (values are lowercase culture adjectives:
  singaporean, indonesian, thai, filipino, vietnamese, myanmar, malaysian,
  cambodian). The manifest records the rule and the per-config selected and
  non-selected counts. There is no script-based selection mechanism.
- **Khmer-script validation, not selection.** Every selected row's
  `premise_native` and `hypothesis_native` must contain a codepoint in
  U+1780-U+17FF, else preparation raises and names the offending `entry_id`.
  The manifest records the rule and the outcome (checked rows, passed).
- **Field allowlist only.** `premise_native`, `hypothesis_native`,
  `premise_english`, `hypothesis_english`, and `true_label`; `culture` is used
  for selection only and never exported. Every other column is dropped,
  including `is_misclassified`, `step_misclassified`, `improved_*`, and
  `entailment_type`. No metadata reaches model input; a test asserts the emitted
  rows carry no unexpected keys.
- **Exact-duplicate removal** per export: drop rows whose sha256 of (normalized
  native premise, normalized native hypothesis, `true_label`) was already seen,
  keeping the lowest source row index. `normal` and `hard` are never deduped
  together: they are different items. Both arms dedup on the native identity, so
  they drop the same rows.
- **Grouping, bootstrap only.** `group_id = sha256(NFKC + casefold +
  collapse-whitespace + strip U+200B/U+200C/U+200D/U+FEFF on the native
  premise)`. ZWSP (U+200B) occurs in 3 hard Khmer premises but in no normal
  premise, so the key strips it. On this slice no two selected premises collapse
  under stripping (14 normal and 23 hard groups either way); the stripping keeps
  the key canonical rather than merging rows here. Grouping normalization never
  touches model-visible text; a test pins both properties. Both arms carry
  identical grouping and sidecars.
- **Unfiltered descriptors.** The manifest records per-config row/group counts,
  the `true_label` distribution, `entailment_type` vs `true_label` disagreement
  counts, the `is_misclassified` distribution, the rows-per-concept histogram,
  and the concept_id overlap between configs. Rows are **never filtered** on
  `is_misclassified` or `step_misclassified`: their exact semantics are
  unverified, and filtering would drop all 23 hard rows plus 14 normal rows.
  The summary reports `is_misclassified` as a descriptive stratification only.

### Exports

- `data/processed/sea-nli-km-normal-test.jsonl` and
  `data/processed/sea-nli-km-hard-test.jsonl` (Khmer, native arm).
- `data/processed/sea-nli-en-normal-test.jsonl` and
  `data/processed/sea-nli-en-hard-test.jsonl` (matched english control arm:
  identical rows, grouping, sidecar, and gold; only the state text differs).
- `data/processed/sea-nli-other-test.jsonl` (all non-Khmer rows, same format,
  descriptive only; it mixes the seven other cultures).
- A sidecar row-index -> group_id map per file (`*-ids.json`), so the evaluator
  attaches `r014-group:<group_id>` tags without re-deriving them. The native and
  english sidecars are byte-identical.
- `datasets/manifests/sea-nli-km.json`, written at run time, recording the
  dataset id, revision, file list with sha256, the selection rule and counts,
  the Khmer-script validation outcome, the native/english arms and their shared
  identity, per-file row/group counts, duplicate counts, exclusion counts by
  reason, rows sharing a premise, per-config label distribution, the
  descriptors above, the sha256 of every export and its sidecar, and the
  preparer's own sha256. It **must be committed with the results**; a result
  without it is not reportable.

### Endpoints

- **Primary:** pooled Khmer rows from both configs. For each R006 seed, a paired
  group-bootstrap over `group_id` clusters: 10,000 draws,
  `numpy.random.default_rng(13)`, row-weighted accuracy difference in percentage
  points, 95% interval from `numpy.quantile(..., [0.025, 0.975],
  method="linear")`. Support for positive transfer requires both seeds' lower
  bounds above zero.
- **Secondary, descriptive only:** Khmer `normal` alone, Khmer `hard` alone, the
  same bootstrap for `full-e2-8bit` versus base (the multitask-vs-single-task
  question left open in `research/R006-multitask/README.md:168` - one R003 seed
  against two R006 seeds, a disclosed mismatch), the non-Khmer slice, and the
  matched native-vs-english control. The non-Khmer slice is a weak
  Khmer-specificity check: it mixes seven other cultures and cannot isolate
  Khmer-ness. The english control separates "Khmer text understanding" from
  "NLI reasoning" and gives a within-model native-vs-english gap; it shares row
  identity, grouping, and gold with the native arm, so the only difference is
  the text.
- **Also reported descriptively:** macro-F1, top-answer Brier, 15-bin
  equal-width ECE, mean confidence, per-class recall, and the predicted-label
  distribution per arm. "Brier" here is the top-answer Brier, not multiclass
  Brier.
- **Degeneracy guard, pre-declared:** if any arm assigns more than 90% of pooled
  Khmer rows to a single label, the primary is marked invalid and results are
  reported descriptive-only.

### Models and inference

Four models, no others: the frozen multilingual base
`convaiinnovations/laya` at revision `7b928d828b7b0e022f929d9bd2e44165aa270148`,
the R006 multitask seeds `mt-s13` and `mt-s14`, and the R003 single-task
checkpoint `full-e2-8bit`. `scripts/evaluate_r014.py` is study-local and leaves
`scripts/evaluate.py` untouched. It uses batch size 8, pinned `max_len=512` and
`head_max_len=256` on every call so the base config (which may permit 1024) and
the adapted checkpoints share one context window, unload-before-load (model
unloading is explicit so peak VRAM counts one resident model), one warm-up pass
outside timing and peak measurement, and reports the `isolated-warm-v1` fields.
Every example carries the target model name before its checkpoint loads, so the
warm-up exercises the checkpoint being scored. Each example is tagged with its
sidecar group id before evaluation so the tag survives into `cases[].tags`. The
report `config` records the batch size, the model list, the dataset and weights
sha256, the sidecar sha256, group counts, and the pinned inference limits. No
calibration is fitted; no training occurs.

The three adapted checkpoints are additionally locked to the sha256 of their
`model.safetensors`, captured at staging time from the frozen local R003/R006
artifacts. A mismatch is a **hard failure, not a warning**: a swapped or
corrupted checkpoint is never scored under the R006/R003 name.

| Model | Checkpoint | `model.safetensors` sha256 |
| --- | --- | --- |
| `mt-s13` | `artifacts/checkpoints/R006-mt-s13` | `ae4042ab1eb5748d5618f3996e222fd696c617d293a430717c79a1bbdb492da9` |
| `mt-s14` | `artifacts/checkpoints/R006-mt-s14` | `e2ff8c7d7149af6692e653e35ed625781919789c35ac6a6017ee01a5062e5d92` |
| `full-e2-8bit` | `artifacts/checkpoints/R003-full-e2-8bit` | `b5c682c025001fdd2d7f5dcbfeef7a7022edc1bfe7e87fbafc08a8907b587f93` |

## Decisions and reasons

- **Fixed option order, one question.** Laya batches only consecutive rows with
  identical ordered question schemas. NLI is a fixed 3-way task and no
  distraction sampling is needed, so one constant option order maximizes
  batching without ordering confounds.
- **Empty criterion descriptions.** The lab scores candidate option strings;
  bare labels are the natural fixed option set. A test pins the rendering.
- **Gold is `true_label`, not `entailment_type`, and no label repair.**
  `entailment_type` and `true_label` disagree on 11/65 Khmer rows; the exported
  gold is `true_label` unchanged. No row was relabelled, repaired, dropped, or
  filtered on any disagreement or quality flag (`is_misclassified`,
  `step_misclassified`), whose exact semantics are unverified. `is_misclassified`
  is reported as a descriptive stratification only and never reaches a model
  input.
- **No audit/held-out carve-out.** R007 reserved an audit slice because fitting
  occurred. Here zero fitting occurs: no training, no threshold, no class
  selection, and no temperature refit. With about 65 Khmer rows, power is the
  binding constraint, so reserving rows would destroy the endpoint for no
  leakage benefit. The whole Khmer slice is scored.
- **Machine-generated text.** The Khmer SEA-NLI items are generated from
  Wikipedia concepts with native verification. They are not naturalistic user
  requests, so results do not estimate deployment performance.
- **Missing RV/audio.** SEA-NLI has no right-voice or audio component; R014 uses
  text only and adds no speech evaluation.

## Scope and limitations

This tests transfer to one fixed 3-way NLI label set on one gated source. It does
not estimate performance on a population of unseen tasks, natural Khmer service
traffic, or deleted/held-out source partitions. The Khmer slice is 65 rows in 37
premise groups, and power is the binding constraint: only very large effects are
resolvable, so a zero-crossing interval is inconclusive, not equivalence. The
source pretraining exposure of the base is unknown, and the R006 and R003
checkpoints already used Khmer data from other studies. The non-Khmer slice is
descriptive and multi-culture. R003 contributes one seed to the single-task
contrast, and the english control is descriptive with no support claims. No
filtering or label repair was applied; `entailment_type` and `true_label`
disagree on 11/65 rows and `true_label` is used as published. Treat results as
exploratory and do not use them to tune later models.

**This study does not close the repository's gap for a rights-cleared
independent Khmer decision benchmark.** SEA-NLI is an NLI resource, not an
intent decision benchmark, and its labels are public.

## Reproduction

`hf auth login` (or `HF_TOKEN`) is a hard prerequisite: SEA-NLI is gated.

```sh
uv sync --frozen
hf auth login                                            # required
uv run --frozen pytest -q tests/test_r014_sea_nli.py
uv run --frozen python scripts/prepare_r014.py
uv run --frozen python scripts/evaluate_r014.py
uv run --frozen python scripts/summarize_r014.py
```

## Post-scoring reporting amendment

This amendment is **reporting-only**. It re-ran `scripts/summarize_r014.py`
only; `scripts/evaluate_r014.py` was not re-run and no report, checkpoint, or
export was touched. It changed no inference, no weights, no rows, no threshold,
no estimand, and no support rule. The locked primary and its verdict are
unchanged: pooled Khmer, model minus base, paired group-bootstrap over 37
premise groups (65 rows), seed 13 `+3.08 pp [-14.93, +20.63]`, seed 14
`+6.15 pp [-4.48, +17.39]`, support rule not met, so the verdict remains
**unsupported / inconclusive**. The locked 0.90 degeneracy guard was not
triggered and that outcome stands.

The amendment adds a `prior_collapse_diagnostics` block to `results.json` and a
matching subsection to `results.md`. Those diagnostics are **explicitly
post-hoc**: they are computed after scoring from the existing reports, they
change no threshold and no endpoint, and they are descriptive only. They record
that every arm is dominated by a different single predicted label, that the two
same-recipe R006 seeds agree on only 26/65 rows, and that a stricter 2/3
collapse threshold would have invalidated both seeds while the locked 0.90
guard did not. This is consistent with label-prior drift rather than acquired
NLI capability and is not used to revise any locked quantity. The amendment also
corrects two reporting defects: a bootstrap lower bound of exactly 0.0 is no
longer rendered with a positive sign, and the limitations prose now states the
concrete 37 premise groups instead of "a few dozen".
