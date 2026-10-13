# R008 results

**Status:** Exploratory, protocol locked, scoring complete. Joint non-inferiority was not supported because SIB-200 seed 13 was inconclusive.

Test accuracy, multitask (MT) vs matching specialist (SP); paired difference in percentage points with 95% paired group-bootstrap interval. Full metrics are in [`results.json`](results.json).

| Task | Seed | MT | SP | MT - SP (pp) | 95% interval | Gate (> -2) |
|---|---|---:|---:|---:|---|---|
| Khmer News | 13 | 92.68 | 92.29 | +0.39 | [-0.52, +1.30] | pass |
| Khmer News | 14 | 93.00 | 92.74 | +0.26 | [-0.78, +1.30] | pass |
| SIB-200 | 13 | 72.55 | 70.59 | +1.96 | [-3.43, +7.35] | fail (inconclusive) |
| SIB-200 | 14 | 75.49 | 66.67 | +8.82 | [+2.45, +15.20] | pass |

On Khmer News, adding MASSIVE and SIB-200 data did not measurably cost accuracy. On SIB-200, neither seed shows a negative point difference; seed 14 favors multitask, while seed 13 is imprecise with n=204. The specialist varied from 66.7% to 70.6% across seeds after only 44 optimizer attempts per seed, so the difference may reflect specialist under-training rather than cross-task benefit. Joint non-inferiority is not supported, but not refuted.
