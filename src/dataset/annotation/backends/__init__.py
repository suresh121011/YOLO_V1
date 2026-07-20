"""
src.dataset.annotation.backends — Built-in Auto-Annotator Backends
==================================================================

Importing this package registers the built-in backends (registry side
effect, mirroring how completeness policies self-register). The modules
themselves stay import-light: torch/ultralytics/transformers are imported
lazily inside ``load()`` so this package is importable in every environment
(CI has no annotation extra — ADR-P5-11).

Built-ins: ``yolo_world`` (primary, ultralytics-native), ``cross_dataset``
(L3 near-dup candidates, no ML — ADR-P5-08). ``grounding_dino`` (optional
second opinion) registers here when M8 first recommends enabling it.
"""

from src.dataset.annotation.backends import cross_dataset, yolo_world

__all__ = ["cross_dataset", "yolo_world"]
