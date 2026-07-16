"""
src.training.masked_loss — Masked BCE Classification Loss (Phase-4)
===================================================================

Missing-annotation mitigation at the loss level: the classification BCE map
is multiplied per image by a {0,1} class mask (from the completeness
artifact) before summation, removing the false "push to background" signal
for classes a source does not annotate. Box and DFL losses are untouched.

Mechanism (ADR-P4-01/02): ``v8DetectionLoss`` computes the classification
loss as ``self.bce(pred_scores, target_scores).sum() / target_scores_sum``
with ``self.bce = nn.BCEWithLogitsLoss(reduction="none")``. Instead of
copying that ~80-line method, :class:`MaskedDetectionLoss` swaps ``self.bce``
for a :class:`_MaskingBCE` wrapper and sets a per-batch ``(bs, 1, nc)`` mask
built from ``batch["im_file"]`` before delegating to the stock ``__call__``.

Identity guarantee: multiplying by an all-ones mask is exact in IEEE-754, and
``target_scores_sum`` derives from the (untouched) assigner targets, so with
every class trusted the loss is bit-identical to stock — the unit tests
assert this. Box/DFL losses depend only on assigner outputs and are never
touched.

Import policy: this module stays torch-free so the preflight gates (G5) and
their tests can import :func:`assert_ultralytics_compat` in any environment.
The loss classes live in src/training/_masked_loss_impl.py (torch/ultralytics
at module level, imported only on the mitigation-enabled path).

Compatibility contract: designed against the ``v8DetectionLoss`` surface of
ultralytics >=8.3,<9.0 (developed and validated on 8.4.96). The canary
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
