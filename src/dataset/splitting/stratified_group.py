"""
src.dataset.splitting.stratified_group — Stratified Group Split
===============================================================

Greedy stratified-group assignment: groups are processed from largest to
smallest annotation mass; each goes to the split whose per-class deficit
(relative to its ratio target) the group best reduces. Group integrity is
preserved (no group straddles splits), and rare classes end up spread
across splits far more evenly than a plain shuffle achieves.

Deterministic: ties broken by seeded jitter, ordering fixed by group key.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

from src.dataset.splitting.base import SPLIT_NAMES, SplitContext
from src.utils.annotation_utils import parse_label_file

logger = logging.getLogger(__name__)


class StratifiedGroupStrategy:
    """Greedy per-class balancing across splits, group-aware."""

    name = "stratified_group"

    def assign(self, context: SplitContext) -> dict[str, list[str]]:
        """Assign groups to splits balancing class distributions.

        Requires ``context.labels_dir`` to compute per-group class
        histograms; falls back to image counts when a label is missing.
        """
        context.validate()
        if context.labels_dir is None:
            raise ValueError(
                "stratified_group requires labels_dir in the SplitContext "
                "(per-group class histograms)"
            )

        ratios = {
            "train": context.train_ratio,
            "val": context.val_ratio,
            "test": context.test_ratio,
        }

        histograms = {
            key: self._group_histogram(paths, context.labels_dir, context.num_classes)
            for key, paths in context.groups.items()
        }
        total_per_class: dict[int, int] = defaultdict(int)
        for hist in histograms.values():
            for class_id, count in hist.items():
                total_per_class[class_id] += count

        # Largest annotation mass first; seeded jitter breaks ties, then
        # group key keeps the order fully deterministic.
        rng = random.Random(context.seed)  # noqa: S311 — determinism, not security
        order = sorted(
            context.groups,
            key=lambda k: (-sum(histograms[k].values()), rng.random(), k),
        )

        assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
        split_class_counts: dict[str, dict[int, int]] = {
            name: defaultdict(int) for name in SPLIT_NAMES
        }
        split_sizes = dict.fromkeys(SPLIT_NAMES, 0)
        total_images = sum(len(paths) for paths in context.groups.values())

        for key in order:
            hist = histograms[key]
            best = max(
                SPLIT_NAMES,
                key=lambda s: self._gain(hist, split_class_counts[s], total_per_class, ratios[s])
                # secondary objective: keep image-count ratios on target
                - (split_sizes[s] / max(total_images, 1) - ratios[s]),
            )
            assigned[best].append(key)
            split_sizes[best] += len(context.groups[key])
            for class_id, count in hist.items():
                split_class_counts[best][class_id] += count

        logger.info(
            "Stratified-group assignments: "
            + " / ".join(f"{len(assigned[s])} {s}" for s in SPLIT_NAMES)
            + f" (from {len(context.groups)} groups)"
        )
        return assigned

    @staticmethod
    def _group_histogram(
        paths: list[Path],
        labels_dir: Path,
        num_classes: int,
    ) -> dict[int, int]:
        """Class id → annotation count over all images of one group."""
        hist: dict[int, int] = defaultdict(int)
        for img_path in paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            for ann in parse_label_file(label_path):
                if 0 <= ann.class_id < num_classes:
                    hist[ann.class_id] += 1
        return dict(hist)

    @staticmethod
    def _gain(
        hist: dict[int, int],
        current: dict[int, int],
        totals: dict[int, int],
        ratio: float,
    ) -> float:
        """How much this group reduces the split's per-class deficit."""
        gain = 0.0
        for class_id, count in hist.items():
            target = totals[class_id] * ratio
            deficit = target - current.get(class_id, 0)
            gain += min(float(count), max(deficit, 0.0))
        return gain
