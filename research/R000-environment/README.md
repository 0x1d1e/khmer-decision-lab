# R000 - Environment and data

Question: can MASSIVE Khmer be prepared reproducibly as a Laya choice task on
the local workstation?

## Environment

- GPU: RTX 3050 Laptop, 6 GB (`gpu.txt`; another process held 5.4 GB at capture time)
- Software: `software.txt` (Python 3.11.15, PyTorch 2.14.1+cu130, Laya 0.4.1)
- Also used: bitsandbytes 0.50.2, datasets 5.1.0, transformers 5.19.0; full pins in `uv.lock`

## Data

`scripts/prepare_massive.py` writes `data/processed/` from
`mteb/amazon_massive_intent`, config `km`, revision pinned in
`datasets/manifests/massive-km.json`.

- Each example: one utterance, 20 candidate intents (gold + 19 random
  negatives from the 60 train labels), order shuffled. Seed 13, fresh RNG per
  split.
- Gold is always among the candidates.
- Validation and test each contain 59 of the 60 intents.

| File | Rows | SHA-256 |
|---|---:|---|
| `massive-km-train-full.jsonl` | 11,514 | `0d9a37e0...0ab5` |
| `massive-km-validation-full.jsonl` | 2,033 | `35cf7541...64c4` |
| `massive-km-test-full.jsonl` | 2,974 | `a7ba2189...73e` |
| `massive-km-validation-100.jsonl` (R001 pilot only) | 100 | `a23b0ded...dedd` |

Laya training holds out a calibration slice from the train file
(`calib_max=400`, `calib_seed=20260922`), leaving 11,114 optimization rows.

## Verification (2026-10-09)

- Re-running the script offline from the cached dataset into a clean directory
  reproduced train/validation/test byte-for-byte.
- The script was edited (validation 100 -> full) and re-run after R002 was
  trained. Each split uses its own seeded RNG, so the train file R002 saw is
  the same as the current one.

## Finding: utterance overlap inside MASSIVE

After NFC + whitespace + lowercase normalization, utterances also present
verbatim in train: validation 289/2,033 (14.2%), test 429/2,974 (14.4%).
Test utterances also in validation: 210. This is a property of the source
dataset, not a preparation bug. Laya's leakage check misses it because each row
has its own candidate set. `research/results.json` reports accuracy on both
subsets. Overlapping rows show no accuracy advantage (R003 test: 82.98% seen, n=429;
83.81% unseen, n=2,545). Single seed, no significance test.
