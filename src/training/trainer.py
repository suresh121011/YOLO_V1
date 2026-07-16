"""
src.training.trainer — Trainer Injection for Missing-Annotation Mitigation
==========================================================================

Provides the ``DetectionTrainer`` subclass passed to
``YOLO(...).train(trainer=...)`` when mitigation is enabled. The trainer
does NOT change how the model is built or saved; it attaches a
:class:`~src.training._masked_loss_impl.MaskedDetectionLoss` to the training
and EMA models at ``on_train_start`` — after ``_setup_train`` has attached
hyperparameters (``model.args``) and created the EMA copy, and strictly
before the first batch, so it deterministically pre-empts the lazy
``init_criterion()`` in ``BaseModel.loss``.

Why not a DetectionModel subclass (ADR-P4-02): Ultralytics checkpoints
pickle the EMA model object by class reference. A subclassed (or
closure-built) model class would make every ``best.pt``/``last.pt`` depend
on this repository being importable, breaking checkpoint portability and
Phase-5/7 export. Attaching only the ``criterion`` attribute leaves the
model class stock; ``save_model`` strips ``criterion`` before serializing
(verified against 8.4.96), so checkpoints stay clean. Resume works
unchanged because train_yolo.py passes the trainer class again whenever
mitigation is enabled.

This module imports ultralytics/torch at module level and must only be
imported on the mitigation-enabled path (train_yolo.py imports it locally).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from src.training._masked_loss_impl import MaskedDetectionLoss
from src.training.completeness_lookup import CompletenessLookup
from src.training.mitigation_config import MitigationConfig

logger = logging.getLogger(__name__)


def _attach_masked_criterion(trainer: Any) -> None:
    """``on_train_start`` callback: install the masked loss on train + EMA models.

    Args:
        trainer: The running MaskedDetectionTrainer instance.

    Raises:
        RuntimeError: If a criterion already exists at this point — that means
                      the Ultralytics trainer flow changed and masking can no
                      longer be guaranteed, so training must not proceed.
    """
    config = trainer.mitigation_config
    lookup = trainer.mitigation_lookup
    train_model = unwrap_model(trainer.model)
    targets: list[tuple[str, Any]] = [("train", train_model)]
    ema = getattr(trainer, "ema", None)
    if ema is not None and getattr(ema, "ema", None) is not None:
        # The validator computes val loss through the EMA model, so it gets
        # its own masked criterion — val loss curves stay comparable.
        targets.append(("ema", ema.ema))

    for name, module in targets:
        if getattr(module, "criterion", None) is not None:
            raise RuntimeError(
                f"The {name} model already has a loss criterion at on_train_start — "
                f"the Ultralytics trainer flow changed and the masked loss can no "
                f"longer pre-empt it safely. See assert_ultralytics_compat() and "
                f"docs/06_training_engineering/masked_loss_architecture.md."
            )
        if getattr(module, "args", None) is None:  # pragma: no cover - defensive
            module.args = train_model.args
        module.criterion = MaskedDetectionLoss(module, lookup=lookup, config=config)

    logger.info(
        f"Missing-annotation mitigation ACTIVE: masked BCE attached to "
        f"{[name for name, _ in targets]} model(s) — {len(lookup)} images, "
        f"nc={lookup.nc}, artifact={lookup.source_path.as_posix()}"
    )


def _log_mask_stats(trainer: Any) -> None:
    """``on_train_epoch_end`` callback: log masking statistics for the epoch."""
    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if not isinstance(criterion, MaskedDetectionLoss):  # pragma: no cover - defensive
        return
    stats = criterion.pop_mask_stats()
    if stats["batches"]:
        logger.info(
            f"Mask stats epoch {trainer.epoch}: {stats['images']} images, "
            f"{stats['masked_cells']}/{stats['total_cells']} (image, class) cells "
            f"masked ({stats['masked_fraction']:.1%})"
        )


class MaskedDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that trains with the masked BCE classification loss.

    Never use this class directly — :func:`build_masked_trainer` returns a
    configured subclass carrying the mitigation settings, because Ultralytics
    instantiates the trainer class itself and rejects unknown train kwargs,
    so constructor injection is impossible.
    """

    mitigation_config: ClassVar[MitigationConfig | None] = None
    mitigation_lookup: ClassVar[CompletenessLookup | None] = None

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: Any = None,
    ) -> None:
        """Initialize the trainer and register the mitigation callbacks.

        Raises:
            RuntimeError: If used without build_masked_trainer, or under DDP
                          (per-image masks are not supported across ranks).
        """
        if self.mitigation_config is None or self.mitigation_lookup is None:
            raise RuntimeError(
                "MaskedDetectionTrainer requires mitigation settings — build the "
                "trainer class via src.training.trainer.build_masked_trainer(...)."
            )
        if RANK != -1:
            raise RuntimeError(
                "Missing-annotation mitigation does not support DDP training: the "
                "configured trainer class cannot be reconstructed in DDP worker "
                "processes. Train on a single device."
            )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_start", _attach_masked_criterion)
        if self.mitigation_config.log_mask_stats:
            self.add_callback("on_train_epoch_end", _log_mask_stats)


def build_masked_trainer(
    config: MitigationConfig, lookup: CompletenessLookup
) -> type[MaskedDetectionTrainer]:
    """Build a trainer class carrying the given mitigation settings.

    A fresh subclass is created per call (settings as class attributes) so
    concurrent configurations never share mutable class state. Trainer
    classes are never pickled by Ultralytics, so the dynamic class is safe.

    Args:
        config: Validated mitigation settings (must be enabled).
        lookup: Loaded completeness lookup.

    Returns:
        A MaskedDetectionTrainer subclass for ``model.train(trainer=...)``.

    Raises:
        ValueError: If called with mitigation disabled — the disabled path
                    must never construct a custom trainer.
    """
    if not config.enabled:
        raise ValueError(
            "build_masked_trainer called with mitigation disabled — the disabled "
            "path must use the stock Ultralytics trainer."
        )
    configured = type(
        "ConfiguredMaskedDetectionTrainer",
        (MaskedDetectionTrainer,),
        {"mitigation_config": config, "mitigation_lookup": lookup},
    )
    return cast(type[MaskedDetectionTrainer], configured)
