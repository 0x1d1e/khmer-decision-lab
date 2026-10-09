# R005 - Decision complexity

Question: how do the base, R002 and R003 checkpoints behave as the decision
gets harder or is presented differently? Three factors: candidate count,
option order, and option text.

## Hypotheses (written before running)

- **H1 candidate count.** Accuracy falls monotonically with k for every model.
  R003 (trained only at k=20) beats base at every k, with test CI excluding 0.
  At k=60 the default head budget truncates options (below), and the long-head
  control beats the default for R003 (test CI excluding 0).
- **H2 option order.** R002/R003 were trained with shuffled options, so they
  are less order-sensitive than base: lower per-row flip rate across 5
  permutations on every task and split.
- **H3 option text.** Raw label ids are no better than the short text on all
  tasks and models (paired test CI upper bound ≤ +1 point). On SIB-200 and
  Khmer News, base does better with descriptive text than with short text
  (test CI excluding 0).

## Protocol

Evaluation only: no training, temperature refit, or text iteration. Variants
come from `scripts/prepare_r005.py` (seed `20261009`), derived from the
R000/R004 files. Validation and test are both reported; neither is used for
selection.

| Factor | Sets | Variants |
|---|---|---|
| Candidate count | MASSIVE | k = 2, 5, 7, 20, 60, nested per row (k=60 = all intents) |
| Option order | MASSIVE (orig. k=20 draw), SIB-200, Khmer News | 5 seeded per-row permutations |
| Option text | MASSIVE, SIB-200, Khmer News | raw label id, short text (reference), description (SIB/News) |

- Models: base at `max_len` 512 (`control-base-maxlen512`, see R004), R002,
  R003. On MASSIVE and SIB-200 nothing exceeds 512 tokens, so base@512 gets
  the same inputs as base; this is checked on the reference files.
- The nested k draw is new, so k=20 here is a different sample of negatives
  from R001-R003. The original draw is kept as a reference.
- References for order and text are the R001-R004 files (base@512 runs
  added).
- Descriptions were written from the label names only, without reading any
  examples (`DESCRIPTIONS` in `scripts/prepare_r005.py`).
- Statistics: paired bootstrap 95% CI on accuracy differences (10k
  resamples, seed 13). Flip rate is the share of rows whose predicted label is
  not the same across all 5 permutations.

### Head budget at k=60 (found before running)

The checkpoints use `head_max_len` 256. Laya truncates every option to
`(256-16)//k` tokens once the options overflow it. All 60 intents need 524
head tokens. At 256 each option gets 4 tokens (its `[MASK]` plus 3), and only
48 of the 60 intents remain distinct token spans. k≤20 is not truncated
(k=20 head ≤ 202 tokens).

Pre-registered control `k60-long`: the same k=60 file with
`HEAD_MAX_LEN=576 MAX_LEN=768` (all options intact; MASSIVE utterances are
at most 185 tokens). These budgets are beyond what R002/R003 were trained at.

### Description confound

Descriptions lengthen the head from 47 to 96-105 tokens. That does not touch
SIB-200 (no truncation), but Khmer News articles lose about 50 of their about
460 state tokens. News description effects therefore mix option text with
context length.

A 40-row functionality check of the budget override ran on MASSIVE k=60 test
rows before this file was written. It was used only to confirm the override
reaches the model, not for any decision.

## Results

Pending.

## Reproduction

```
uv run python scripts/prepare_r005.py
bash -c 'M=control-base-maxlen512,head-e2,full-e2-8bit
for s in validation test; do
  for v in k2 k5 k7 k20 k60 perm0 perm1 perm2 perm3 perm4 raw; do
    uv run python scripts/evaluate.py data/processed/massive-km-$v-$s.jsonl artifacts/runs/massive-km-$v-$s $M
  done
  MAX_LEN=768 HEAD_MAX_LEN=576 uv run python scripts/evaluate.py \
    data/processed/massive-km-k60-$s.jsonl artifacts/runs/massive-km-k60-long-$s $M
  for t in sib200-km khmer-news; do for v in perm0 perm1 perm2 perm3 perm4 raw desc; do
    uv run python scripts/evaluate.py data/processed/$t-$v-$s.jsonl artifacts/runs/$t-$v-$s $M
  done; done
  for t in massive-km sib200-km; do
    f=data/processed/$t-$s.jsonl; [ $t = massive-km ] && f=data/processed/massive-km-$s-full.jsonl
    uv run python scripts/evaluate.py $f artifacts/runs/$t-$s control-base-maxlen512
  done
done'
uv run python scripts/summarize_runs.py
```
