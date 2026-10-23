# R016 results

**Status:** Scored, exploratory. Budget-matched specialist training complete for both
seeds; all results below are exploratory and do not resolve R008's non-inferiority
question (n=204 makes the -2 pp gate untestable; see caveat).

Test accuracy, specialist-at-budget (SP@budget) vs same-seed multitask (MT);
paired difference `D_s = acc(SP@budget, s) - acc(MT, s)` in percentage points with
95% paired group-bootstrap interval (10,000 draws, seed 13; group = exact
normalized input state). Full metrics are in `artifacts/runs/R016/{model}.json`;
the recomputed summary is `artifacts/runs/R016/summary.json`.

| Seed | SP@budget | MT | D_s = SP - MT (pp) | 95% interval (pp) | SP updates | MT comparator |
|---:|---:|---:|---:|---|---:|---:|
| 13 | 72.06 | 72.55 | -0.49 | [-5.88, +5.39] | 1051 / 1056 | R008-mt-s13 |
| 14 | 70.59 | 75.49 | -4.90 | [-10.29, +0.49] | 1050 / 1056 | R008-mt-s14 |

## Training

- Seed 13: 1,056 optimizer attempts, 1,051 successful, 5 skipped, 1,051 scheduler
  steps; mean epoch loss 1.519 -> 0.000 over 48 epochs; 2,942.8 s (~49 min);
  peak 3,432.7 MiB allocated.
- Seed 14: 1,056 attempts, 1,050 successful, 6 skipped, 1,050 scheduler steps;
  mean epoch loss 1.383 -> 0.000 over 48 epochs; 2,924.0 s (~49 min);
  peak 3,433.0 MiB allocated.
- Both seeds ran serially under `flock /tmp/khmer-gpu.lock` on the RTX 3050;
  `artifacts/checkpoints/R008-*` were not touched.

Per-seed training records (weights, telemetry, report, provenance hashes) are in
`artifacts/runs/R016/preparation.json` and each checkpoint directory.

## Interpretation

Matching the multitask arm's ~1,060-update budget did **not** close the SIB-200
specialist's gap versus same-seed multitask: `D_s` is negative for both seeds
(-0.49 pp seed 13, -4.90 pp seed 14), and both intervals include zero. Specialist
training at the matched budget underperforms multitask on point estimate in both
seeds, so specialist under-training at 44 updates (R008's specialist budget) is
not supported by this study as the explanation for the R008 gap; if anything the
budget-matched specialist is further behind on seed 14.

Both intervals are very wide at n=204 (half-widths 5.64 pp seed 13, 5.39 pp seed
14). Passing R008's -2 pp non-inferiority gate would require `D_s > +3.64` pp
(seed 13) and `D_s > +3.39` pp (seed 14); observed estimates are -0.49 and -4.90.
The minimum detectable effect is ~5.4 pp per seed. The accurate restatement is
that non-inferiority is untestable at n=204; this study cannot resolve R008's
non-inferiority question and establishes no claim about multitask versus
specialist training beyond these two seeds, this recipe, and this test set.

## Reproduction

```sh
uv sync --frozen
uv run --frozen pytest -q
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 13'
flock /tmp/khmer-gpu.lock -c 'uv run --frozen python scripts/train_r016_budget_specialist.py 14'
uv run --frozen python scripts/evaluate_r016_budget_specialist.py
uv run --frozen python scripts/summarize_r016.py
```
