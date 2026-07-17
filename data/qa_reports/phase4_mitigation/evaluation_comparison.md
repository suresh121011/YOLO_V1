# Baseline vs Mitigated — Evaluation Report

*Generated: 2026-07-17T01:41:05Z*

- **Runs:** baseline, mitigated
- **Commit:** fd5f3cc
- **ultralytics:** 8.4.96

## Aggregate metrics

| Run       | P      | R      | F1     | mAP50  | mAP50-95 |
| --------- | ------ | ------ | ------ | ------ | -------- |
| baseline  | 0.0066 | 0.1417 | 0.0125 | 0.1298 | 0.089    |
| mitigated | 0.0103 | 0.1382 | 0.0192 | 0.0309 | 0.0179   |

## Mitigated − baseline (per class)

Public-source validation labels are partially annotated; absolute metrics underestimate untrusted classes. Deltas on the same split are meaningful; unbiased numbers require the custom eval set.

| Class    | ΔP     | ΔR      | ΔF1    | ΔmAP50 | ΔmAP50-95 |
| -------- | ------ | ------- | ------ | ------ | --------- |
| cupboard | 0.0002 | 0.0     | 0.0005 | -0.396 | -0.2871   |
| door     | 0.0    | 0.0     | 0.0    | 0.0    | 0.0       |
| face     | 0.0148 | -0.0144 | 0.0097 | 0.0001 | 0.0027    |
| stove    | 0.0    | 0.0     | 0.0    | 0.0    | 0.0       |
