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

## Post-scoring power diagnostic (post-hoc)

Post-scoring, post-hoc diagnostic computed from the existing reports in [`results.json`](results.json). It changes no threshold, no endpoint, no estimand, no row set, and no support rule; the locked criterion and its recorded verdict are unchanged. The -2 pp gate remains what R008 actually applied, and `joint_noninferiority_supported` stays `false`.

The diagnostic asks whether that gate was decidable at the SIB-200 test size. Non-inferiority passes when the interval's lower bound exceeds -2 pp, which is equivalent to requiring the point estimate to exceed `-2 + half-width`. The half-width is fixed by n and the paired disagreement rate, so the interval, not the point estimate alone, sets what the gate can detect.

| Task | Seed | Rows | MT - SP (pp) | 95% interval | Half-width (pp) | Point estimate required to pass (pp) | Observed |
|---|---|---:|---:|---|---:|---:|---|
| Khmer News | 13 | 1543 | +0.39 | [-0.52, +1.30] | 0.91 | > -1.09 | passes |
| Khmer News | 14 | 1543 | +0.26 | [-0.78, +1.30] | 1.04 | > -0.96 | passes |
| SIB-200 | 13 | 204 | +1.96 | [-3.43, +7.35] | 5.39 | > +3.39 | fails (inconclusive) |
| SIB-200 | 14 | 204 | +8.82 | [+2.45, +15.20] | 6.37 | > +4.37 | passes |

On SIB-200 the interval half-width is 5.39 pp (seed 13) and 6.37 pp (seed 14), so a non-inferiority pass required a point estimate above roughly +3.4 and +4.4 pp. The gaps actually observed are +1.96 and +8.82 pp. The minimum detectable difference at this n is therefore about 5.4 pp per seed, well above the ~2 pp difference the gate was meant to test.

Seed 14 did clear the gate, but for the opposite reason the criterion was designed to test: its lower bound is +2.45, so it demonstrates that multitask was *superior* to the specialist on SIB-200, not that non-inferiority was established. A pass in that direction carries no information about a 2 pp tolerance.

The accurate restatement of the SIB-200 outcome is **non-inferiority untestable at n=204**, rather than "joint criterion not supported" as a statement about multitask versus specialist behaviour. The recorded verdict stays as the locked rule produced it: the joint criterion was not supported, because the study required all four comparisons to pass and one did not. What changes is the interpretation of that failure, which is a statement about the test's power on 204 rows, not evidence about multitask learning.

No remedy is proposed or executed here. Extending the SIB-200 test set or adding a budget or seed sweep would be new work; that was considered and deferred, and nothing in this diagnostic authorizes it.
