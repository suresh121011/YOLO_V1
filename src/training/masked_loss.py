"""
src.training.masked_loss — Masked BCE Classification Loss (Phase-4)
===================================================================

Missing-annotation mitigation at the loss level: the classification BCE map
is multiplied per image by a {0,1} class mask (from the completeness
artifact) before summation, removing the false "push to background" signal
for classes a source does not annotate. Box and DFL losses are untouched.

This module deliberately imports torch/ultralytics inside functions and
class bodies — importing it is free, so the preflight gates can use
:func:`assert_ultralytics_compat` in torch-less environments and the
disabled training path never pays the import cost.

Compatibility contract: designed against the ``v8DetectionLoss`` surface of
ultralytics >=8.3,<9.0 (developed and validated on 8.4.96). The canary below
fails loudly if an upstream release moves the seams this module relies on.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

#: The exact loss-internal seams the masked loss relies on. Checked by
#: assert_ultralytics_compat() against the installed ultralytics source.
_REQUIRED_LOSS_SOURCE_MARKERS = (
    # v8DetectionLoss.__init__ builds the BCE module the wrapper replaces.
    'BCEWithLogitsLoss(reduction="none")',
    # __call__ computes the cls loss through self.bce(...) — the wrap point.
    "self.bce(",
)


def assert_ultralytics_compat() -> None:
    """Verify the installed Ultralytics still exposes the seams we rely on.

    Checks that ``v8DetectionLoss`` exists, constructs an elementwise
    ``BCEWithLogitsLoss``, and routes the classification loss through
    ``self.bce(...)`` — the exact wrap point of the masking mechanism.

    Raises:
        RuntimeError: If ultralytics is missing or its loss internals moved,
                      with the installed version and remediation in the message.
    """
    try:
        import ultralytics
        from ultralytics.utils.loss import v8DetectionLoss
    except ImportError as e:
        raise RuntimeError(
            f"ultralytics is not importable ({e}) — install requirements.txt before "
            f"enabling missing-annotation mitigation."
        ) from e

    version = getattr(ultralytics, "__version__", "unknown")
    try:
        source = inspect.getsource(v8DetectionLoss)
    except (OSError, TypeError) as e:  # pragma: no cover - source always ships
        raise RuntimeError(
            f"Cannot inspect ultralytics {version} v8DetectionLoss source ({e}); "
            f"masked-loss compatibility is unverifiable. Pin ultralytics to a "
            f"version validated in docs/06_training_engineering/."
        ) from e

    missing = [marker for marker in _REQUIRED_LOSS_SOURCE_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"ultralytics {version} changed v8DetectionLoss internals — missing "
            f"marker(s): {missing}. The masked BCE wrapper cannot attach safely. "
            f"Pin ultralytics to a validated version (see "
            f"docs/06_training_engineering/masked_loss_architecture.md) or update "
            f"src/training/masked_loss.py for the new internals."
        )
    logger.debug(f"ultralytics {version} loss-surface compat canary passed")
