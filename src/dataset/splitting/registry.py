"""
src.dataset.splitting.registry — Strategy Lookup
================================================

Maps ``split.strategy`` config values to strategy implementations.
``kfold`` and ``leave_one_house_out`` are registered as guided stubs: they
were deliberately deferred (no CV folds are needed yet and zero houses have
been captured — see docs/04_dataset_engineering/README.md §5) but reserve
their config names so enabling them later is non-breaking.
"""

from __future__ import annotations

from src.dataset.splitting.base import SplitStrategy
from src.dataset.splitting.group_aware import GroupAwareStrategy
from src.dataset.splitting.stratified_group import StratifiedGroupStrategy

_RESERVED: dict[str, str] = {
    "kfold": (
        "k-fold cross-validation is reserved for a future phase — "
        "implement KFoldStrategy in src/dataset/splitting/ when CV is needed"
    ),
    "leave_one_house_out": (
        "leave-one-house-out requires Phase-3 capture sessions with house_id "
        "provenance (CaptureSessionManifest); no houses exist yet"
    ),
}


def available_strategies() -> list[str]:
    """Names accepted by get_strategy (implemented ones first)."""
    return ["group_aware", "stratified_group", *_RESERVED]


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
    if name in _RESERVED:
        raise NotImplementedError(f"Split strategy '{name}': {_RESERVED[name]}")
    raise ValueError(f"Unknown split strategy '{name}'. Available: {available_strategies()}")
