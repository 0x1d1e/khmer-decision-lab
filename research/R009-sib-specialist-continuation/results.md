# R009 results

Status: scored exploratory. The preregistered improvement hypothesis was not supported; estimates are inconclusive or mixed. The protocol README's status line records the pre-training lock state; this file is the current execution record.

## Primary comparison

Continued specialist accuracy was lower than the matched R008 specialist in both seeds. Both paired 95% group-bootstrap intervals include zero, so the protocol's joint improvement criterion was not met.

| Seed | Continued specialist | R008 specialist | Difference, 95% CI | R008 multitask difference, 95% CI |
|---|---:|---:|---:|---:|
| 13 | 68.63% | 70.59% | -1.96 pp [-6.86, +2.94] | -3.92 pp [-8.82, +0.98] |
| 14 | 65.69% | 66.67% | -0.98 pp [-6.86, +4.90] | -9.80 pp [-15.69, -4.41] |

The primary result does not show improvement, harm, or equivalence. For the secondary multitask comparison, both point estimates favor R008 multitask; only seed 14's interval excludes zero.

| Seed | Model | Macro-F1 | ECE | Top-answer Brier | Macro OVR AUROC | AURC |
|---|---|---:|---:|---:|---:|---:|
| 13 | Continued specialist | 0.6833 | 0.0916 | 0.1743 | 0.9050 | 0.1594 |
| 13 | R008 specialist | 0.6944 | 0.0559 | 0.1714 | 0.9130 | 0.1417 |
| 14 | Continued specialist | 0.6384 | 0.1726 | 0.1994 | 0.8777 | 0.1968 |
| 14 | R008 specialist | 0.6691 | 0.0751 | 0.1848 | 0.9025 | 0.1685 |

## Training and calibration

Each seed completed 132 optimizer attempts on 681 training items over six fixed epochs. Seed 13 had 129 successful and 3 skipped updates; seed 14 had 128 successful and 4 skipped updates. Checkpoint weights changed for both seeds. Validation diagnostics confirmed all 99 items were held out; no validation-based checkpoint selection occurred.

| Seed | Validation accuracy, before → after | Validation loss, before → after | ECE, before → after | Time | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| 13 | 70.71% → 71.72% | 0.9631 → 2.8677 | 0.1348 → 0.2791 | 374.7 s | 3,422,438,400 bytes |
| 14 | 67.68% → 63.64% | 1.0425 → 4.2329 | 0.1513 → 0.3463 | 368.0 s | 3,422,438,400 bytes |

Temperature fitting used all 99 validation rows per seed and returned `[5.0, 1.0, 1.0]`; no option-bucket temperatures were fitted. These validation diagnostics do not estimate test performance.

## Lock and provenance amendments

Training started under manifest SHA-256 `27648e73d99b19f3c509780f448a76dc2730981406d84c1ce22ed66c4dc1810f`. After training, telemetry showed the optimizer class as `bitsandbytes.optim.adamw.PagedAdamW8bit`, while the manifest used the package re-export name `bitsandbytes.optim.PagedAdamW8bit`. A Python identity check confirmed these are the same class object. The manifest was corrected to the canonical telemetry path; its intermediate SHA-256 was `a78b89d4a74285b39e321140a25ec23fc5470a08fbd44e870a263a1930e90d52`.

The first calibration load triggered Laya 0.4.1's `_fix_tokenizer_config` compatibility rewrite on each R009 checkpoint. It changed only `tokenizer/tokenizer_config.json`; model weights were unchanged. The calibration script now verifies the exact rewrite against `checkpoint_latest`, updates the inference inventory and training-provenance hash, and binds the calibration payload to that hash. Tests reject any unrelated inference-file change. Both calibrations were rerun under this code before test scoring.

The final manifest SHA-256 is `b63a224292e0eda8b80685f25d02b1f9c3ae4a97cc8a04b43161e4d6af0af354`. These post-lock amendments correct metadata and account for pinned Laya behavior; they do not change the optimizer class, training runs, weights, validation use, or evaluation rules. The original training manifest hash and both amendments are retained here. The README records the earlier pre-lock test-file hash and one-row schema check.

## Limits

The test split has 204 rows and was reused in R008, whose results motivated R009. This is an exploratory follow-up, not independent confirmation. The two seeds and one task cannot establish population-level improvement, general task superiority, or unseen-task transfer. Full per-case reports and metrics are in `results.json` and ignored `artifacts/runs/R009/` files.
