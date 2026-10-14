# R013 results

Exploratory open-vocabulary transfer to one fixed set of held-out MASSIVE intent option names.

Manifest SHA-256: `f78f66d492b1b34729de7e5b52f3f9b3d9645a3a4ba5a137236f21dab3d37ecd`
Protocol SHA-256: `a11149f67e1fdaf56c27eb099e79a50c01065035697fbdce6ec39ec535582145`
Evaluation manifest SHA-256: `93725d7d914b0c10f678cc649b3a57c348ae3fffeb7b94efdaf1e2ad97ee5d41`
Evaluation protocol SHA-256: `04044d5ae4e2cb34f39c1d84b448e7e6d05da205a21b41117a45db84892599f3`
Preparation SHA-256: `5c641922102bf2aa294b8af56bd62001e4547b274d38155594c74d14748aa264`
Evaluation SHA-256: `6bb721a5ba0f2bfdc462980fb502906a6c1ff69acfb886455c07cd6061a4a322`

## Execution record

The frozen base's saved `max_len=1024` caused the first gated scoring attempt to stop before warmup or a model forward; no prediction or outcome was produced. The inference-only amendment applied `max_len=512` and `head_max_len=256` to every checkpoint. A later probability-validation failure left seven partial reports, preserved as non-results under `artifacts/runs/R013/pre-score-failures/partial-scoring-ac704/`. Final scoring completed under the exact-probability amendment. A post-scoring reporting amendment corrected predecessor-lock and singleton-call validation only; it did not rerun inference or change reports, logits, metrics, bootstrap computation, or support thresholds. Preparation and training artifacts retain their predecessor-lock provenance.

## Primary comparison

Seen-only adaptation versus the frozen multilingual base on identical held-out-intent rows.

| Seed | Rows | Base accuracy | Seen-only accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |
|---:|---:|---:|---:|---:|---:|
| 13 | 320 | 0.1187 | 0.3781 | +25.94 | [+20.62, +31.25] |
| 14 | 320 | 0.1187 | 0.3500 | +23.12 | [+17.81, +28.44] |

Primary support rule met: **True**. Both seed-specific lower bounds must exceed zero.

## All-label control

The matched-budget all-label versus seen-only comparison is descriptive, not a pure causal effect because the training data composition differs.

| Seed | Seen-only accuracy | All-label accuracy | Difference (pp) | Paired group-bootstrap 95% CI (pp) |
|---:|---:|---:|---:|---:|
| 13 | 0.3781 | 0.7906 | +41.25 | [+35.00, +47.19] |
| 14 | 0.3500 | 0.7781 | +42.81 | [+36.88, +48.75] |

## Descriptive metrics

Accuracy, macro-F1, macro OvR AUROC, multiclass NLL, categorical Brier, top-answer Brier, ECE, AURC, and selective accuracy are in `results.json`; values remain separated by condition and model.

The study tests transfer to these exact English option strings, not a population of unseen intents. MASSIVE training data were used in earlier project checkpoints; foundation-model pretraining exposure is unknown. Short intent names may be opaque, only one intent split and two fixed seeds were used, and the dataset is not natural service traffic.

Intervals resample normalized source-utterance groups and retain all rows. Because the preparation keeps one representative row per normalized group, each test group currently contributes one row.
