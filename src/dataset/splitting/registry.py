"""
src.dataset.splitting.registry — Strategy Lookup
================================================

Maps ``split.strategy`` config values to strategy implementations.
``kfold`` is registered as a guided stub: deliberately deferred (no CV
folds are needed yet) but its config name is reserved so enabling it
later is non-breaking. ``leave_one_house_out`` shipped with Phase-3
(capture sessions carry house_id provenance).
"""

from __future__ import annotations

from src.dataset.splitting.base import SplitStrategy
from src.dataset.splitting.group_aware import GroupAwareStrategy
from src.dataset.splitting.leave_one_house_out import LeaveOneHouseOutStrategy
from src.dataset.splitting.stratified_group import StratifiedGroupStrategy

_RESERVED: dict[str, str] = {
    "kfold": (
        "k-fold cross-validation is reserved for a future phase — "
        "implement KFoldStrategy in src/dataset/splitting/ when CV is needed"
    ),
}


def available_strategies() -> list[str]:
    """Names accepted by get_strategy (implemented ones first)."""
    return ["group_aware", "stratified_group", "leave_one_house_out", *_RESERVED]


def get_strategy(name: str) -> SplitStrategy:
    """Resolve a strategy name from dataset_split_config.yaml.

    Raises:
        NotImplementedError: For reserved-but-unimplemented strategies.
        ValueError:          For unknown names.
    """
    if name == "group_aware":
        return GroupAwareStrategy()
    if name == "stratified_group":
        return StratifiedGroupStrategy()
    if name == "leave_one_house_out":
        return LeaveOneHouseOutStrategy()
    if name in _RESERVED:
        raise NotImplementedError(f"Split strategy '{name}': {_RESERVED[name]}")
    raise ValueError(f"Unknown split strategy '{name}'. Available: {available_strategies()}")
