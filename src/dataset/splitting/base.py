"""
src.dataset.splitting.base — Strategy Contract
==============================================

Defines the inputs every split strategy receives (:class:`SplitContext`)
and the protocol strategies implement. Strategies assign GROUP KEYS (not
individual files) to splits — group integrity is the leakage-prevention
invariant of the whole system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass
class SplitContext:
    """Everything a split strategy may need.

    Attributes:
        groups:      Capture-group key → list of image Paths.
        train_ratio: Target train fraction.
        val_ratio:   Target val fraction.
        test_ratio:  Target test fraction.
        seed:        Determinism seed.
        labels_dir:  Root of YOLO labels (required by stratified
                     strategies for per-group class histograms).
        num_classes: Taxonomy size (for histogram sizing).
    """

    groups: dict[str, list[Path]] = field(default_factory=dict)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    labels_dir: Path | None = None
    num_classes: int = 23

    def validate(self) -> None:
        """Raise ValueError if ratios do not sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )


class SplitStrategy(Protocol):
    """A split strategy assigns group keys to train/val/test."""

    name: str

    def assign(self, context: SplitContext) -> dict[str, list[str]]:
        """Return split name → list of group keys (all groups covered once)."""
        ...
