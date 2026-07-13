"""Unit tests for src.dataset.splitting — strategy registry and strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.splitting import SplitContext, available_strategies, get_strategy
from src.dataset.splitting.group_aware import GroupAwareStrategy
from src.dataset.splitting.stratified_group import StratifiedGroupStrategy


def _groups(n_groups: int = 20, per_group: int = 3) -> dict[str, list[Path]]:
    return {
        f"group{i:03d}": [Path(f"group{i:03d}_frame_{j}.jpg") for j in range(per_group)]
        for i in range(n_groups)
    }


@pytest.mark.unit
class TestRegistry:
    """Strategy lookup behavior."""

    def test_implemented_strategies_resolve(self) -> None:
        assert isinstance(get_strategy("group_aware"), GroupAwareStrategy)
        assert isinstance(get_strategy("stratified_group"), StratifiedGroupStrategy)

    def test_reserved_strategies_raise_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="future phase"):
            get_strategy("kfold")
        with pytest.raises(NotImplementedError, match="house_id"):
            get_strategy("leave_one_house_out")

    def test_unknown_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown split strategy"):
            get_strategy("alphabetical")

    def test_available_lists_all(self) -> None:
        names = available_strategies()
        for expected in ("group_aware", "stratified_group", "kfold", "leave_one_house_out"):
            assert expected in names


@pytest.mark.unit
class TestGroupAwareStrategy:
    """Default strategy invariants."""

    def test_all_groups_assigned_exactly_once(self) -> None:
        groups = _groups(20)
        assignments = GroupAwareStrategy().assign(SplitContext(groups=groups))
        assigned = [k for keys in assignments.values() for k in keys]
        assert sorted(assigned) == sorted(groups)

    def test_deterministic_for_seed(self) -> None:
        groups = _groups(30)
        first = GroupAwareStrategy().assign(SplitContext(groups=groups, seed=7))
        second = GroupAwareStrategy().assign(SplitContext(groups=groups, seed=7))
        assert first == second
        different = GroupAwareStrategy().assign(SplitContext(groups=groups, seed=8))
        assert different != first

    def test_bad_ratios_raise(self) -> None:
        with pytest.raises(ValueError):
            GroupAwareStrategy().assign(
                SplitContext(groups=_groups(5), train_ratio=0.9, val_ratio=0.2)
            )


@pytest.mark.unit
class TestStratifiedGroupStrategy:
    """Stratified strategy: group integrity + class balance."""

    @staticmethod
    def _make_labeled_dataset(
        tmp_path: Path, n_groups: int = 30
    ) -> tuple[dict[str, list[Path]], Path]:
        """Groups whose images have labels; a rare class appears in 1/3 of groups."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        groups: dict[str, list[Path]] = {}
        for i in range(n_groups):
            key = f"group{i:03d}"
            paths = []
            for j in range(3):
                img = images_dir / f"{key}_frame_{j}.jpg"
                img.write_bytes(b"fake")
                lines = ["0 0.5 0.5 0.2 0.2"]  # common class in every image
                if i % 3 == 0:
                    lines.append("5 0.3 0.3 0.1 0.1")  # rare class: 1/3 of groups
                (labels_dir / f"{key}_frame_{j}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                paths.append(img)
            groups[key] = paths
        return groups, labels_dir

    def test_requires_labels_dir(self) -> None:
        with pytest.raises(ValueError, match="labels_dir"):
            StratifiedGroupStrategy().assign(SplitContext(groups=_groups(5)))

    def test_all_groups_assigned_exactly_once(self, tmp_path: Path) -> None:
        groups, labels_dir = self._make_labeled_dataset(tmp_path)
        assignments = StratifiedGroupStrategy().assign(
            SplitContext(groups=groups, labels_dir=labels_dir)
        )
        assigned = [k for keys in assignments.values() for k in keys]
        assert sorted(assigned) == sorted(groups)

    def test_rare_class_present_in_every_split(self, tmp_path: Path) -> None:
        groups, labels_dir = self._make_labeled_dataset(tmp_path, n_groups=30)
        assignments = StratifiedGroupStrategy().assign(
            SplitContext(groups=groups, labels_dir=labels_dir)
        )
        # Rare class lives in groups where i % 3 == 0 (10 of 30 groups) —
        # stratification must spread them so each split gets at least one.
        rare_groups = {f"group{i:03d}" for i in range(30) if i % 3 == 0}
        for split, keys in assignments.items():
            assert rare_groups & set(keys), f"split '{split}' got no rare-class group"

    def test_deterministic(self, tmp_path: Path) -> None:
        groups, labels_dir = self._make_labeled_dataset(tmp_path)
        ctx = SplitContext(groups=groups, labels_dir=labels_dir, seed=42)
        assert StratifiedGroupStrategy().assign(ctx) == StratifiedGroupStrategy().assign(ctx)
