"""
Performance tests conftest.

Performance tests validate the per-module latency budget defined in pyproject.toml.

Budget targets (Pi 5 CPU-only — worst case hardware):
    Frame capture + resize     ≤  5ms
    YOLO11n inference (640²)   ≤ 80ms
    Detection post-processing  ≤  2ms
    Memory update              ≤  1ms
    Rule engine evaluation     ≤  3ms
    SmolVLM2 (when invoked)    ≤ 2000ms
    Alert queue enqueue        ≤  0.1ms
    Total per frame (no VLM)   ≤ 92ms (~10 FPS)

All performance tests require model weights and real hardware.
Skip on CI with: @pytest.mark.skipif(os.environ.get("CI"), reason="No model on CI")
"""
from __future__ import annotations

import pytest
