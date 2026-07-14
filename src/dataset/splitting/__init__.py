"""
src.dataset.splitting — Configurable Split Strategies
=====================================================

Strategy-pattern split system driven by ``configs/dataset_split_config.yaml``
(``split.strategy``). Every strategy is group-aware: a capture group (video,
burst, future Phase-3 capture session) never straddles splits.

Available strategies (see registry.py):
    group_aware         — seeded shuffle of groups, sliced by ratio (default)
    stratified_group    — greedy per-class balancing across splits
    leave_one_house_out — house-level integrity for Phase-3 custom captures
                          (all sessions of a house share a split; supports
                          holdout_houses → test)
    kfold               — reserved; raises NotImplementedError with guidance
"""

from src.dataset.splitting.base import SplitContext, SplitStrategy
from src.dataset.splitting.registry import available_strategies, get_strategy

__all__ = ["SplitContext", "SplitStrategy", "available_strategies", "get_strategy"]
