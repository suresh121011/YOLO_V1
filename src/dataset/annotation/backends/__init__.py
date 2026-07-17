"""
src.dataset.annotation.backends — Built-in Auto-Annotator Backends
==================================================================

Importing this package registers the built-in backends (registry side
effect, mirroring how completeness policies self-register). The modules
themselves stay import-light: torch/ultralytics/transformers are imported
lazily inside ``load()`` so this package is importable in every environment
(CI has no annotation extra — ADR-P5-11).

Built-ins: ``yolo_world`` (primary, ultralytics-native). ``grounding_dino``
(optional second opinion) and ``cross_dataset`` (L3) register here when
their milestones land (M8 decision / L3 merge-salvage work respectively).
"""

from src.dataset.annotation.backends import yolo_world

__all__ = ["yolo_world"]
