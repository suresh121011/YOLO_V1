"""
src.dataset.splitting.leave_one_house_out — House-Level Split
=============================================================

Group-aware split whose leakage unit is the HOUSE, not the capture
session: all sessions from one house land in the same split. This is the
honest evaluation setup for Phase-3 custom captures — a model validated
on unseen rooms of a *seen* house still overfits that home's furniture,
lighting and layouts.

House derivation: ``SplitContext.house_pattern`` (one capture group) is
searched in each group key. Merged custom-capture keys look like
``custom_captures_h01_kitchen_s001`` → house ``h01``. Group keys without
a match (public-dataset images) form their own single-group super-group,
so mixed public+custom datasets split exactly like ``group_aware`` for
the public part.

``SplitContext.holdout_houses`` forces named houses entirely into test
(the locked-eval workflow); the remaining super-groups get the seeded
shuffle/slice.
"""

from __future__ import annotations

import logging
import random
import re

from src.dataset.splitting.base import SplitContext

logger = logging.getLogger(__name__)


class LeaveOneHouseOutStrategy:
    """House-level super-groups, seeded shuffle, holdout support."""

    name = "leave_one_house_out"

    def assign(self, context: SplitContext) -> dict[str, list[str]]:
        """Assign capture groups to splits with house integrity."""
        context.validate()
        pattern = re.compile(context.house_pattern)

        # Build super-groups: house ID → group keys (or the key itself).
        super_groups: dict[str, list[str]] = {}
        for key in context.groups:
            match = pattern.search(key)
            super_key = f"house:{match.group(1)}" if match else f"solo:{key}"
            super_groups.setdefault(super_key, []).append(key)

        houses = [k for k in super_groups if k.startswith("house:")]
        logger.info(
            f"leave_one_house_out: {len(super_groups)} super-groups "
            f"({len(houses)} houses: {sorted(h.split(':', 1)[1] for h in houses)})"
        )

        holdout = {f"house:{house}" for house in context.holdout_houses}
        missing = sorted(h.split(":", 1)[1] for h in holdout if h not in super_groups)
        if missing:
            logger.warning(f"holdout houses not present in the data: {missing}")

        test_keys: list[str] = []
        pool: list[str] = []
        for super_key in super_groups:
            (test_keys if super_key in holdout else pool).append(super_key)

        # Deterministic shuffle for reproducible splits — not a security context.
        rng = random.Random(context.seed)  # noqa: S311
        pool.sort()
        rng.shuffle(pool)

        n = len(pool)
        n_train = round(n * context.train_ratio)
        n_val = round(n * context.val_ratio)

        assignments_super = {
            "train": pool[:n_train],
            "val": pool[n_train : n_train + n_val],
            "test": pool[n_train + n_val :] + test_keys,
        }

        assignments = {
            split: [key for super_key in super_keys for key in super_groups[super_key]]
            for split, super_keys in assignments_super.items()
        }
        logger.info(
            "House-integral assignments: "
            + " / ".join(f"{len(assignments[s])} {s} groups" for s in ("train", "val", "test"))
            + (f" (holdout → test: {sorted(context.holdout_houses)})" if holdout else "")
        )
        return assignments
