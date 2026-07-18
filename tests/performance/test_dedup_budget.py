"""
Performance budget: vectorized dedup at 30k-image scale (plan §Performance
budgets — "dedup (vectorized) | 30k images | ≤ 10 min").

Bypasses real image I/O (compute_image_hashes is monkeypatched to return a
pre-generated random 64-bit hash instantly) so the test isolates the actual
bottleneck the vectorization targets: the O(n) Hamming-distance scan against
the growing kept-set, repeated n times (O(n²) worst case at full scale).
Real image hashing (PIL decode + resize) is a separate, already-fast cost
this budget is not about.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import pytest

from src.dataset.dedup import DedupIndex, ImageHashes
from src.dataset.sources_config import DedupSettings

pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
    pytest.mark.skipif(bool(os.environ.get("CI")), reason="No model/scale budget check on CI"),
]

_BUDGET_SECONDS = 600  # 10 minutes
_SCALE = 30_000


def test_vectorized_dedup_scales_to_30k_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(0)  # noqa: S311 — deterministic perf test, not crypto

    def _fake_hashes(path: Path, check_flip: bool = True) -> ImageHashes:
        # Random-but-deterministic 64-bit hash — near-zero real collision
        # rate, so almost every call falls through to the "kept" branch,
        # exercising the worst-case O(n) scan on every single insertion.
        return ImageHashes(ahash=f"{rng.getrandbits(64):016x}", flip_ahash=None)

    monkeypatch.setattr("src.dataset.dedup.compute_image_hashes", _fake_hashes)

    index = DedupIndex(settings=DedupSettings(hamming_threshold=5, check_flips=False))
    start = time.perf_counter()
    for i in range(_SCALE):
        index.check_and_add(Path(f"img_{i}.jpg"))
    elapsed = time.perf_counter() - start

    assert index.kept_count >= _SCALE * 0.99  # near-zero random-hash collisions expected
    assert elapsed <= _BUDGET_SECONDS, (
        f"vectorized dedup took {elapsed:.1f}s for {_SCALE} images " f"(budget {_BUDGET_SECONDS}s)"
    )
