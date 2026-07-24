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


def _make_index(kept: dict[Path, tuple[int, int | None]], hash_size: int = 8) -> DedupIndex:
    index = DedupIndex(
        settings=DedupSettings(hamming_threshold=5, check_flips=True, hash_size=hash_size)
    )
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

    def test_earlier_flip_match_beats_later_direct_match(self) -> None:
        """Adversarial case (M7 audit finding): item0 matches ONLY via flip,
        item1 (inserted after) matches ONLY via direct hash. Naive checks
        item0's a_int-then-flip_int before ever reaching item1, so it must
        return item0 — a vectorized scan that checked a_int across ALL
        items before flip_int across ALL items would wrongly prefer item1.
        """
        far = 0xFFFFFFFFFFFFFFFF
        index = _make_index(
            {
                Path("item0.jpg"): (far, None),  # far from a_int; flip_int hits it
                Path("item1.jpg"): (0x0000000000000000, None),  # a_int hits it directly
            }
        )
        a_int = 0x0000000000000000  # distance 0 from item1, far from item0
        flip_int = 0xFFFFFFFFFFFFFFFF  # distance 0 from item0

        assert index._check_naive(a_int, flip_int, threshold=1) == Path("item0.jpg")
        assert index._check_vectorized(a_int, flip_int, threshold=1) == Path("item0.jpg")

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


class TestVectorizedWideHash:
    """P7: hash_size=16 → 256-bit hashes (4 uint64 lanes). The naive
    int.bit_count scan and the lane-summed vectorized scan must still make
    byte-identical keep/drop decisions."""

    def test_500_random_256bit_hash_sets_agree(self) -> None:
        rng = random.Random(1234)  # noqa: S311 — deterministic property test, not crypto
        for trial in range(500):
            n_kept = rng.randint(0, 40)
            kept = {
                Path(f"kept_{trial}_{i}.jpg"): (
                    rng.getrandbits(256),
                    rng.getrandbits(256) if rng.random() < 0.5 else None,
                )
                for i in range(n_kept)
            }
            index = _make_index(kept, hash_size=16)
            threshold = rng.choice([1, 5, 12, 20, 40])
            a_int = rng.getrandbits(256)
            flip_int = rng.getrandbits(256) if rng.random() < 0.5 else None

            naive = index._check_naive(a_int, flip_int, threshold)
            vectorized = index._check_vectorized(a_int, flip_int, threshold)
            assert naive == vectorized, (
                f"trial {trial} (n_kept={n_kept}, threshold={threshold}): "
                f"naive={naive} vectorized={vectorized}"
            )

    def test_wide_hash_flip_and_first_match(self) -> None:
        base = (0x0F0F0F0F0F0F0F0F << 192) | 0xABCD
        index = _make_index({Path("a.jpg"): (base, None)}, hash_size=16)
        near = base ^ 0b101  # Hamming distance 2 in the low lane
        assert index._check_naive(near, None, threshold=5) == Path("a.jpg")
        assert index._check_vectorized(near, None, threshold=5) == Path("a.jpg")
        far = base ^ ((1 << 200) - 1)  # many bits differ across lanes → no match
        assert index._check_naive(far, None, threshold=5) is None
        assert index._check_vectorized(far, None, threshold=5) is None
