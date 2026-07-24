"""
src.dataset.annotation.backends — Built-in Auto-Annotator Backends
==================================================================

Importing this package registers the built-in backends (registry side
effect, mirroring how completeness policies self-register). The modules
themselves stay import-light: torch/ultralytics/transformers are imported
lazily inside ``load()`` so this package is importable in every environment
(CI has no annotation extra — ADR-P5-11).

Built-ins: ``yolo_world`` (primary, ultralytics-native), ``yoloe`` (seeder
upgrade with batched inference — P5), ``cross_dataset`` (L3 near-dup
candidates, no ML — ADR-P5-08), ``grounding_dino`` (optional second opinion,
HF transformers — ADR-P5-02). Registering a backend here is
NOT the same as enabling it — ``configs/annotation.yaml``'s per-backend
``enabled`` flag (grounding_dino defaults false) decides what actually
runs; registration only means ``get_annotator()`` can resolve the name
instead of hard-failing if a human or M8's own
``grounding_dino_decision()`` ever asks for it.
"""

from src.dataset.annotation.backends import cross_dataset, grounding_dino, yolo_world, yoloe

__all__ = ["cross_dataset", "grounding_dino", "yoloe", "yolo_world"]
