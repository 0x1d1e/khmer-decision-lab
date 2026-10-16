# R014 results

Exploratory 3-way NLI transfer to an unseen task type, asked as a fixed three-option choice.

Manifest SHA-256: `7f3b49f0590f7b75477fb41b390867ac4f6ce0a314e5467283a8f642c56544ba`
Protocol SHA-256: `e6c6af64cd73f657d8a3107fa87bcaca14380194a3859de31dd5f82053621165`

## Primary comparison

Each R006 seed versus the frozen multilingual base on pooled Khmer rows (normal + hard).

| Seed | Rows | Groups | Base accuracy | Model accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 65 | 37 | 0.4769 | 0.5077 | +3.08 | [-14.93, +20.63] |
| 14 | 65 | 37 | 0.4769 | 0.5385 | +6.15 | [-4.48, +17.39] |

Primary support rule met: **False** (valid=True, degenerate arms=[]).
Both seed-specific lower bounds must exceed zero; a crossing interval is inconclusive, not equivalence.

## Secondary (descriptive only)

| Contrast | Rows | Groups | Difference (pp) | Paired group-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|
| km-normal mt-s13 - base | 42 | 14 | +9.52 | [-9.52, +28.57] |
| km-normal mt-s14 - base | 42 | 14 | +7.14 | [-9.52, +23.81] |
| km-hard mt-s13 - base | 23 | 23 | -8.70 | [-43.48, +26.09] |
| km-hard mt-s14 - base | 23 | 23 | +4.35 | [0.00, +13.04] |
| pooled full-e2-8bit - base | 65 | 37 | +6.15 | [-2.82, +15.87] |

descriptive multitask-vs-single-task question; R003 has one seed against the two R006 seeds, so this is not a matched-seed contrast.

## Post-scoring prior-collapse diagnostics (post-hoc)

Post-scoring, post-hoc diagnostic computed from the existing reports. It changes no threshold, no endpoint, no estimand, no row set, and no support rule; the locked primary and its verdict are unchanged. Every pooled arm is dominated by a single predicted label, but a different one, and the pooled deltas are consistent with seed-level label-prior drift rather than acquired NLI capability.

| Arm | Predicted distribution (entailment/neutral/contradiction) | Dominant label | Dominant fraction | Constant-majority accuracy | Margin above it (pp) | Cohen κ vs gold |
|---|---|---|---:|---:|---:|---:|
| multilingual | entailment 39, neutral 8, contradiction 18 | entailment | 0.6000 | 0.3846 | +9.23 | 0.2050 |
| mt-s13 | entailment 8, neutral 47, contradiction 10 | neutral | 0.7231 | 0.3846 | +12.31 | 0.2296 |
| mt-s14 | entailment 44, neutral 14, contradiction 7 | entailment | 0.6769 | 0.3846 | +15.38 | 0.2697 |
| full-e2-8bit | entailment 37, neutral 16, contradiction 12 | entailment | 0.5692 | 0.3846 | +15.38 | 0.2831 |

Constant-majority reference: predicting entailment or neutral for every row scores 0.3846 (25/65); contradiction scores 0.2308.

| Pair | Agreeing rows | Agreement |
|---|---:|---:|
| multilingual vs mt-s13 | 24/65 | 0.3692 |
| multilingual vs mt-s14 | 44/65 | 0.6769 |
| multilingual vs full-e2-8bit | 48/65 | 0.7385 |
| mt-s13 vs mt-s14 | 26/65 | 0.4000 |

| Contrast | Base wrong / model right | Base right / model wrong | Net flipped | Exact two-sided McNemar p |
|---|---:|---:|---:|---:|
| mt-s13 - base | 19 | 17 | +2 | 0.8679 |
| mt-s14 - base | 8 | 4 | +4 | 0.3877 |

The two same-recipe R006 seeds agree on only 26/65 rows (0.4000), which bounds how much stable signal the pooled deltas can carry. The locked 0.90 degeneracy guard did not trigger; at a stricter 2/3 collapse threshold it would have invalidated mt-s13 (0.7231), mt-s14 (0.6769). The primary therefore remains unsupported/inconclusive on power grounds, and the observed differences are not evidence of stable NLI capability.

## Descriptive metrics

| Condition | Model | Acc | Macro-F1 | Top-answer Brier | ECE | Mean conf |
|---|---|---:|---:|---:|---:|---:|
| km-normal | multilingual | 0.4524 | 0.4375 | 0.3699 | 0.3619 | 0.8095 |
| km-normal | mt-s13 | 0.5476 | 0.5278 | 0.3629 | 0.3428 | 0.8733 |
| km-normal | mt-s14 | 0.5238 | 0.5165 | 0.2756 | 0.2786 | 0.8024 |
| km-normal | full-e2-8bit | 0.5238 | 0.5347 | 0.3553 | 0.3625 | 0.8023 |
| km-hard | multilingual | 0.5217 | 0.3625 | 0.4196 | 0.4573 | 0.9326 |
| km-hard | mt-s13 | 0.4348 | 0.4048 | 0.4921 | 0.4916 | 0.8777 |
| km-hard | mt-s14 | 0.5652 | 0.5120 | 0.2848 | 0.3217 | 0.8316 |
| km-hard | full-e2-8bit | 0.5652 | 0.4575 | 0.3582 | 0.3565 | 0.9217 |
| en-normal | multilingual | 0.5714 | 0.5368 | 0.3439 | 0.3715 | 0.8934 |
| en-normal | mt-s13 | 0.6429 | 0.6156 | 0.2792 | 0.2850 | 0.8800 |
| en-normal | mt-s14 | 0.6190 | 0.6169 | 0.3030 | 0.3303 | 0.8702 |
| en-normal | full-e2-8bit | 0.6190 | 0.6037 | 0.3196 | 0.3038 | 0.8659 |
| en-hard | multilingual | 0.4783 | 0.4453 | 0.4338 | 0.4723 | 0.8820 |
| en-hard | mt-s13 | 0.4783 | 0.4569 | 0.3899 | 0.4002 | 0.8785 |
| en-hard | mt-s14 | 0.6087 | 0.5387 | 0.3346 | 0.3441 | 0.7980 |
| en-hard | full-e2-8bit | 0.4783 | 0.4339 | 0.3362 | 0.3619 | 0.7761 |
| other | multilingual | 0.5661 | 0.5607 | 0.3459 | 0.3239 | 0.8897 |
| other | mt-s13 | 0.5413 | 0.5462 | 0.3527 | 0.3350 | 0.8763 |
| other | mt-s14 | 0.6038 | 0.5982 | 0.2925 | 0.2502 | 0.8540 |
| other | full-e2-8bit | 0.5690 | 0.5661 | 0.3221 | 0.2928 | 0.8596 |

Per-class recall, per-class F1, and the predicted-label distribution per arm are in `results.json`.

## English control (descriptive only)

Same rows, grouping, and gold as the native arm; the difference isolates native-vs-English text within a model.

| Model | Native acc | English acc | Difference (pp) | Paired group-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|
| multilingual | 0.4769 | 0.5385 | +6.15 | [-11.27, +22.67] |
| mt-s13 | 0.5077 | 0.5846 | +7.69 | [-7.05, +22.22] |
| mt-s14 | 0.5385 | 0.6154 | +7.69 | [-9.52, +24.66] |
| full-e2-8bit | 0.5385 | 0.5692 | +3.08 | [-14.29, +19.40] |

Descriptive stratification of `is_misclassified` (never used as a filter and never exported to the model input): {'km-normal': {'False': 28, 'True': 14}, 'km-hard': {'True': 23}}.

## Scope and limitations

Test split only; there is no validation split, so no SEA-NLI temperature is fitted and checkpoint temperatures are used unchanged. The Khmer slice is 65 rows in 37 premise groups, so only very large effects are resolvable. The Khmer text is machine-generated from Wikipedia concepts with native verification, not naturalistic user input. Labels are public, so sealing is procedural only. The non-Khmer slice is a weak specificity check that mixes seven other cultures. This study does not close the repository gap for a rights-cleared independent Khmer decision benchmark.

Intervals resample normalized premise groups and retain all rows. Identical premises across the normal and hard configs fall in one cluster; exact duplicates were removed within each config only.
