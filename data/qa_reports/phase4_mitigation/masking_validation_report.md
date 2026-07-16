# Phase-4 M3.5 — Masking-Correctness Validation

*Generated: 2026-07-16T19:43:26Z*

- **Verdict:** PASS
- **Commit:** 7d67ebc
- **ultralytics:** 8.4.96
- **torch:** 2.13.0+cpu

## Verdict

**PASS** — masking correctness proven; M4/M5 may proceed.

## (a) Correctness unit suite

✅ `[32m============================= [32m[1m31 passed[0m[32m in 4.13s[0m[32m ==============================[0m` (bit-identity vs stock v8DetectionLoss, exact-zero masked gradients, criterion injection, golden train-kwargs regression).

## (b) Mask spot-checks on the real artifact

188 images checked against configs/dataset_sources.yaml expectations (nc=23).

| Policy     | Images | Trusted/nc | Expected | OK |
| ---------- | ------ | ---------- | -------- | -- |
| coco       | 53     | 10/23      | 10       | ✅  |
| negatives  | 18     | 23/23      | 23       | ✅  |
| openimages | 57     | 3/23       | 3        | ✅  |
| wider_face | 60     | 1/23       | 1        | ✅  |

## (c) Mitigated 1-epoch run — ✅ PASS

- exit code: 0
- weights written: True
- finite losses: True
- runtime: 21.2 s
- preflight ran: True
- mitigation announced: True
- mask stats logged: True
- metrics.json mitigation block: True

## (d) Disabled 1-epoch run — ✅ PASS

- exit code: 0
- weights written: True
- finite losses: True
- runtime: 21.2 s
- stock path clean (no preflight/trainer/mitigation traces): True
