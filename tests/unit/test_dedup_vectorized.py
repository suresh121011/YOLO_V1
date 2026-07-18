"""
Property test: the vectorized Hamming scan makes IDENTICAL keep/drop
decisions to the naive Python scan (ADR-P5-12) — 500 random hash sets plus
targeted edge cases (near-duplicate, flip-match, empty kept-set).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.dataset.dedup import DedupIndex
from src.dataset.sources_config import DedupSettings

pytestmark = pytest.mark.unit


def _make_index(kept: dict[Path, tuple[int, int | None]]) -> DedupIndex:
    index = DedupIndex(settings=DedupSettings(hamming_threshold=5, check_flips=True))
    for path, (a_int, flip_int) in kept.items():
        index._kept[path] = (a_int, flip_int)
        index._kept_paths.append(path)
    return index


class TestVectorizedMatchesNaive:
    def test_500_random_hash_sets_agree(self) -> None:
        rng = random.Random(42)  # noqa: S311 — deterministic property test, not crypto
        for trial in range(500):
            n_kept = rng.randint(0, 50)
            kept = {
                Path(f"kept_{trial}_{i}.jpg"): (
                    rng.getrandbits(64),
                    rng.getrandbits(64) if rng.random() < 0.5 else None,
                )
                for i in range(n_kept)
            }
            index = _make_index(kept)
            threshold = rng.choice([1, 3, 5, 8, 10])
            a_int = rng.getrandbits(64)
            flip_int = rng.getrandbits(64) if rng.random() < 0.5 else None

            naive = index._check_naive(a_int, flip_int, threshold)
            vectorized = index._check_vectorized(a_int, flip_int, threshold)
            assert naive == vectorized, (
                f"trial {trial} (n_kept={n_kept}, threshold={threshold}): "
                f"naive={naive} vectorized={vectorized}"
            )

    def test_empty_kept_set_agrees(self) -> None:
        index = _make_index({})
        assert index._check_naive(123, None, threshold=5) is None
        assert index._check_vectorized(123, None, threshold=5) is None

    def test_near_duplicate_detected_by_both_paths(self) -> None:
        base = 0x0F0F0F0F0F0F0F0F
        index = _make_index({Path("a.jpg"): (base, None)})
        near = base ^ 0b1  # Hamming distance 1
        assert index._check_naive(near, None, threshold=5) == Path("a.jpg")
        assert index._check_vectorized(near, None, threshold=5) == Path("a.jpg")

    def test_flip_match_detected_by_both_paths(self) -> None:
        # incoming a_int is far from the kept hash, but its flip is identical.
        index = _make_index({Path("a.jpg"): (0xFFFFFFFFFFFFFFFF, None)})
        a_int = 0x00000000FFFFFFFF
        flip_int = 0xFFFFFFFFFFFFFFFF
        assert index._check_naive(a_int, flip_int, threshold=1) == Path("a.jpg")
        assert index._check_vectorized(a_int, flip_int, threshold=1) == Path("a.jpg")

    def test_first_match_wins_on_both_paths(self) -> None:
        # Two kept hashes both within threshold of the incoming hash — both
        # paths must pick the SAME (first-inserted) one.
        base = 0x00000000000000FF
        index = _make_index(
            {
                Path("first.jpg"): (base, None),
                Path("second.jpg"): (base ^ 0b1, None),
            }
        )
        assert index._check_naive(base, None, threshold=5) == Path("first.jpg")
        assert index._check_vectorized(base, None, threshold=5) == Path("first.jpg")

    def test_vectorization_threshold_boundary_still_agrees(self) -> None:
        """Same decision right at VECTORIZE_THRESHOLD, not just far above it."""
        from src.dataset.dedup import VECTORIZE_THRESHOLD

        rng = random.Random(7)  # noqa: S311 — deterministic property test, not crypto
        kept: dict[Path, tuple[int, int | None]] = {
            Path(f"k{i}.jpg"): (rng.getrandbits(64), None) for i in range(VECTORIZE_THRESHOLD)
        }
        index = _make_index(kept)
        a_int = rng.getrandbits(64)
        assert index._check_naive(a_int, None, threshold=3) == index._check_vectorized(
            a_int, None, threshold=3
        )
