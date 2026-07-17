# Missing-Annotation Mitigation — Benchmark Report

*Generated: 2026-07-17T01:38:49Z*

- **Verdict:** PASS
- **Config:** 3 epochs @ 320px, batch 8, 3× repeats, cpu

## Verdict

**PASS** — every performance budget met. Smoke-scale benchmark (188 images): metric numbers are indicative only and NOT generalizable; deterministic seeds make metric variance across repeats ≈0 by design. Budgets are re-validated at full scale in Phase 5.

## Arms (mean ± std over repeats)

| Arm       | Wall s         | Peak RSS MB       | P     | R     | F1    | mAP50 | mAP50-95 |
| --------- | -------------- | ----------------- | ----- | ----- | ----- | ----- | -------- |
| baseline  | 63.867 ± 2.73  | 1073.033 ± 14.049 | 0.007 | 0.142 | 0.013 | 0.13  | 0.089    |
| mitigated | 47.767 ± 2.843 | 1078.933 ± 8.121  | 0.01  | 0.138 | 0.019 | 0.031 | 0.018    |

## Performance budgets

| Budget                                 | Measured      | Limit | Unit | Status |
| -------------------------------------- | ------------- | ----- | ---- | ------ |
| Training wall-time overhead per run    | -25.21        | 5.0   | %    | PASS   |
| Peak CPU RSS delta                     | 0.55          | 5.0   | %    | PASS   |
| Peak CPU RSS delta (absolute)          | 5.9           | 200.0 | MB   | PASS   |
| Peak GPU memory delta                  | n/a (no CUDA) | 5.0   | %    | N/A    |
| Loss-forward overhead per call (bs=16) | 0.077         | 1.0   | ms   | PASS   |
| Per-batch mask build (bs=16)           | 0.0504        | 1.0   | ms   | PASS   |
| CompletenessLookup.load wall time      | 0.002         | 2.0   | s    | PASS   |

## Microbenchmarks

Loss forward (bs=16, interleaved median of 9 rounds): stock 4.047 ms vs masked 4.124 ms — +0.077 ms/call (1.91% of the isolated loss call; ≈0% of a full training step, see the wall-time budget). Mask build: 0.0504 ms/batch; lookup load 0.002 s for 188 images.
