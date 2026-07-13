"""
src.dataset.splitting.group_aware — Default Group-Aware Split
=============================================================

The original algorithm from ``scripts/dataset/split_dataset.py`` (which now
delegates here): shuffle group keys with a fixed seed, slice by rounded
ratio targets, give the remainder to test. All frames of a group land in
one split.
"""

from __future__ import annotations

import logging
import random

from src.dataset.splitting.base import SplitContext

logger = logging.getLogger(__name__)


class GroupAwareStrategy:
    """Seeded shuffle of groups, sliced by ratio (default strategy)."""

    name = "group_aware"

    def assign(self, context: SplitContext) -> dict[str, list[str]]:
        """Assign capture groups to splits while respecting ratio targets."""
        context.validate()

        group_keys = list(context.groups.keys())
        # Deterministic shuffle for reproducible splits — not a security context.
        rng = random.Random(context.seed)  # noqa: S311
        rng.shuffle(group_keys)

        n = len(group_keys)
        n_train = round(n * context.train_ratio)
        n_val = round(n * context.val_ratio)
        n_test = n - n_train - n_val  # remainder → test, so all groups covered

        assignments = {
            "train": group_keys[:n_train],
            "val": group_keys[n_train : n_train + n_val],
            "test": group_keys[n_train + n_val :],
        }

        logger.info(
            f"Group assignments: {n_train} train / {n_val} val / {n_test} test "
            f"(from {n} total groups)"
        )
        return assignments
