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

## Deviation: nested draw fixed after the first run

The first run's k<60 files reused the negative-selection order as the display
order, which put the gold option last in 98.6% of k=2 rows and 67% of k=20
rows. The protocol above specifies a uniform gold position. A sanity check
caught this: nested k=20 disagreed with the original k=20 draw for base and
R002 but not R003. `prepare_r005.py` now shuffles the display order
independently (gold-last share 0.504 / 0.204 / 0.057 / 0.016 for k = 2 / 5 /
20 / 60). The k conditions were rerun, and only those reruns are reported
below. Order and text files were unaffected (byte-identical). The flawed
reports are kept in `artifacts/runs/_r005-flawed-gold-position/`.

The flawed draw did show something by accident. Test accuracy, flawed ->
fixed:

| k | base | R002 | R003 |
|---|---|---|---|
| 5 | 43.1 -> 36.9 | 53.9 -> 46.5 | 93.3 -> 92.6 |
| 7 | 40.3 -> 29.5 | 51.6 -> 39.3 | 91.0 -> 90.7 |
| 20 | 16.2 -> 12.9 | 27.1 -> 21.0 | 83.5 -> 83.4 |

## Results

Paired bootstrap 95% CI, 10k resamples, seed 13. Full numbers in
`research/results.json` (`tasks` per variant, `R005` for the cross-variant
comparisons).

Reference check: base@512 and base gave the same prediction on every MASSIVE
and SIB-200 row (max probability difference 0.027 on one row, bf16 noise).

### Candidate count (MASSIVE, nested, uniform gold position)

Accuracy %, validation / test. Chance = 1/k.

| k | base@512 | R002 | R003 | R003 - base (test) |
|---:|---|---|---|---|
| 2 | 66.9 / 64.0 | 74.7 / 72.9 | 96.7 / 97.1 | +33.1 [+31.3, +34.8] |
| 5 | 38.2 / 36.9 | 50.9 / 46.5 | 92.8 / 92.6 | +55.7 [+53.8, +57.6] |
| 7 | 30.6 / 29.5 | 42.6 / 39.3 | 92.0 / 90.7 | +61.2 [+59.4, +63.0] |
| 20 | 12.4 / 12.9 | 23.1 / 21.0 | 83.4 / 83.4 | +70.5 [+68.8, +72.2] |
| 60 | 6.2 / 5.3 | 6.8 / 5.7 | 55.7 / 52.7 | +47.4 [+45.6, +49.2] |
| 60, long head | 3.9 / 4.7 | 9.9 / 9.5 | 69.5 / 69.4 | +64.6 [+62.9, +66.4] |

- Nested k=20 agrees with the original draw (base 13.6 / 13.3, R002 23.6 /
  22.6, R003 83.1 / 83.7).
- Long head vs default at k=60 (test): R003 +16.7 [+14.9, +18.5], R002 +3.9
  [+2.8, +4.9], base -0.6 [-1.5, +0.3] (validation -2.3 [-3.3, -1.2]).
- Truncation damage is concentrated on the 15 intents that collide at 256
  head tokens (`audio_volume_*`, `iot_hue_*`, `iot_wemo_*`, `email_query*`,
  `iot_cleaning`/`iot_coffee`). R003 test accuracy on those rows (n=427):
  16.9% default -> 77.8% long head. On the other rows: 58.7% -> 68.0%.
- Calibration: R003 ECE ≤ 0.04 and R002 ≤ 0.06 at every k. Base ECE grows
  from 0.12-0.14 (k=2) to 0.49-0.50 (k=20) and 0.64 at k=60 long head
  (confidence 0.68 at about 4% accuracy).
- Cost per decision p50: 12 ms (k=2) -> 17.6 ms (k=20) -> 22 ms (k=60) ->
  36 ms (k=60 long head, test; base on validation 57 ms). Peak allocated VRAM 1.5 GB for the first model in a
  process (the 2.5 GB recorded for R002/R003 includes the previous model, see
  R004).

### Option order (5 permutations)

| Set | Model | Acc range (val / test) | Flip rate (val / test) | Flip - base (test) |
|---|---|---|---|---|
| MASSIVE k=20 | base@512 | 1.5 / 0.7 | 86.1% / 85.0% | |
| | R002 | 1.3 / 2.1 | 84.1% / 83.6% | -1.5 [-3.0, +0.1] |
| | R003 | 0.7 / 0.3 | 10.8% / 10.4% | -74.6 [-76.3, -72.9] |
| SIB-200 | base@512 | 3.0 / 2.0 | 18.2% / 18.6% | |
| | R002 | 4.0 / 1.0 | 21.2% / 19.6% | +1.0 [-2.0, +3.9] |
| | R003 | 1.0 / 2.5 | 13.1% / 16.7% | -2.0 [-8.8, +4.9] |
| Khmer News | base@512 | 1.2 / 0.7 | 13.8% / 12.1% | |
| | R002 | 1.5 / 0.8 | 16.2% / 14.5% | +2.5 [+1.0, +4.0] |
| | R003 | 1.1 / 0.3 | 9.2% / 10.0% | -2.1 [-4.0, -0.1] |

Validation flip-rate differences have the same sign. Significant ones: R003
on MASSIVE -75.3 and News -4.5; R002 on MASSIVE -2.0 and News +2.4.

Position preference (exploratory, not pre-registered; test, pooled over the 5
permutations, so gold position is uniform). On MASSIVE R002 picks the last
slot 9.2% of the time (uniform 5%). Its accuracy is 32.6% when the gold is
last and 21.9% when it is in the middle. Base slightly favours both ends
(7.3% first, 6.1% last), and R003 is flat (5.2% / 5.0%). On SIB-200 and News,
every model is within 2 points of uniform at both ends.

### Option text

Accuracy difference vs the short text reference, test (validation in
brackets).

| Set | Model | raw id - short | desc - short |
|---|---|---|---|
| MASSIVE | base@512 | -0.9 [-2.0, +0.1] (val -0.3) | - |
| | R002 | -1.5 [-2.8, -0.0] (val -1.1) | - |
| | R003 | -2.6 [-3.5, -1.7] (val -1.3) | - |
| SIB-200 | base@512 | -0.5 [-2.9, +2.0] (val 0.0) | -1.5 [-5.9, +2.9] (val -10.1 [-17.2, -4.0]) |
| | R002 | -2.0 [-4.9, +1.0] (val -1.0) | -2.0 [-6.4, +2.5] (val -10.1 [-17.2, -4.0]) |
| | R003 | -0.5 [-1.5, 0.0] (val +1.0 [0.0, +3.0]) | -2.5 [-6.9, +2.0] (val -7.1 [-13.1, -2.0]) |
| Khmer News | base@512 | -0.8 [-1.7, 0.0] (val -0.9) | -0.7 [-2.3, +1.0] (val +0.6) |
| | R002 | -0.7 [-1.6, +0.3] (val -1.7) | -0.5 [-2.1, +1.2] (val -0.6) |
| | R003 | -0.8 [-1.5, -0.1] (val -1.2) | +0.7 [-0.8, +2.3] (val +1.4) |

## Verdicts

- **H1 supported.** Accuracy falls monotonically with k for every model on
  both splits. R003 beats base at every k, and the long head lifts R003 at
  k=60 by 16.7 points. R003's lead is largest at k=20, its training k, and it
  stays ≥ +33 points at every k.
- **H2 not supported as stated** (it required both models on every task
  and split). The R003 half holds on two of three tasks: R003 is much less
  order-sensitive than base on MASSIVE and less on Khmer News (both splits);
  SIB-200 cannot resolve it.
  R002 is not: it is more order-sensitive than base on News and only
  marginally less on MASSIVE. Head-only training with shuffled options did
  not buy order robustness. Caveat: on MASSIVE base is near chance (13%), and
  near-random predictions flip by construction. The News comparison
  (accuracies 52-54%) is the cleaner one.
- **H3a supported on MASSIVE and Khmer News, unresolved on SIB-200.** Raw
  ids are never better than short text: the largest point estimate is +1.0,
  and on MASSIVE and News every test CI upper bound is ≤ +0.3. On SIB-200 the CIs
  are too wide for the ≤ +1 criterion (upper bounds up to +3.0). Raw ids are
  significantly worse for R003 on MASSIVE (-2.6, the text it was trained on)
  and on News test.
- **H3b rejected.** Descriptions
  did not help base: no significant effect on News or SIB test, and a
  significant -10.1 drop on SIB validation for base and R002 (-7.1 R003).

## Findings

- **Fine-tuning generalizes across k.** R003, trained only at k=20, reaches
  97% at k=2 and 69% over all 60 intents with the long head. It is also well
  calibrated at every k in-domain. Base is poorly calibrated, and more so as
  k grows.
- **The default head budget is a silent failure mode at large k.** With
  `head_max_len` 256, 15 of 60 intents become identical token spans, and R003
  loses 17 points. Laya reports `options_distinct` but does not raise. The
  fix costs about 1.6x per-decision latency (22 -> 36 ms). Applications with
  more than about 25 short options should raise `head_max_len` (and
  `max_len`).
- **R002 has a recency bias** on MASSIVE (last-slot preference), visible both
  in the flawed draw and in the properly randomized permutation runs. R003
  does not.
- **Order and wording are second-order for R003.** Its accuracy range across
  5 permutations is ≤ 1.1 points on MASSIVE and News, and raw-id effects are
  ≤ 2.6 points. The exception is descriptions on SIB validation (-7.1). For base, descriptions are not a free win. The SIB
  validation drop is large but not replicated on test.

## Limitations

- Descriptions are one hand-written English set with no Khmer variant. The
  negative SIB validation result may be specific to this wording. News
  descriptions also cost about 50 context tokens.
- Flip rate depends on accuracy, so it compares cleanly only between models
  of similar accuracy.
- The k=60 long-head budgets exceed what R002/R003 were trained with. Base
  gets worse with the long head, so its k=60 numbers are not a clean
  zero-shot ceiling.
- k uses MASSIVE only, so it covers short utterances. SIB-200 (n=99/204) is
  too small for order and text effects of a few points.
- The position-preference analysis is post hoc.
- Single training seed for R002/R003; one negative draw per k.

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
