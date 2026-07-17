"""
src.training._masked_loss_impl — Masked-Loss Classes (torch-dependent)
======================================================================

Implementation half of src/training/masked_loss.py: this module imports
torch/ultralytics at module level (classes must inherit their bases at
definition time) and is therefore only imported on the mitigation-enabled
path — never when mitigation is disabled and never by the preflight gates.

Classes are module-level (not closures) so instances survive
``copy.deepcopy`` in Ultralytics' EMA/checkpoint machinery; checkpoints
themselves never contain them (``save_model`` strips ``criterion`` before
serializing — verified against 8.4.96), so .pt files stay portable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss

from src.training.completeness_lookup import CompletenessLookup, UnknownImageError
from src.training.masked_loss import assert_ultralytics_compat
from src.training.mitigation_config import MitigationConfig

logger = logging.getLogger(__name__)


class _MaskingBCE(nn.Module):
    """Elementwise BCE wrapper that multiplies by the current batch mask.

    ``v8DetectionLoss`` computes ``self.bce(pred_scores, target_scores)``
    exactly once per loss evaluation with shape ``(bs, anchors, nc)``.
    Wrapping ``self.bce`` masks that map without copying any upstream math:
    the owner sets a ``(bs, 1, nc)`` {0,1} mask before delegating to the
    stock ``__call__`` and clears it afterwards. With no mask set (e.g. a
    non-mitigated caller) the wrapper is transparent.
    """

    def __init__(self, inner: nn.Module) -> None:
        """Wrap the stock ``BCEWithLogitsLoss(reduction="none")`` module.

        Args:
            inner: The elementwise BCE module being wrapped.
        """
        super().__init__()
        self.inner = inner
        self._mask: torch.Tensor | None = None

    def set_mask(self, mask: torch.Tensor | None) -> None:
        """Install the (bs, 1, nc) mask for the imminent loss evaluation."""
        self._mask = mask

    def clear_mask(self) -> None:
        """Remove the current mask (transparent passthrough again)."""
        self._mask = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return elementwise BCE, masked when a batch mask is installed.

        Raises:
            RuntimeError: If a mask is installed but the input shape is not
                          the expected (bs, anchors, nc) — masking silently
                          skipping would corrupt supervision, so it fails loud.
        """
        out: torch.Tensor = self.inner(pred, target)
        mask = self._mask
        if mask is None:
            return out
        if out.dim() != 3 or out.shape[0] != mask.shape[0] or out.shape[-1] != mask.shape[-1]:
            raise RuntimeError(
                f"Masked BCE expected (bs={mask.shape[0]}, anchors, nc={mask.shape[-1]}) "
                f"logits, got {tuple(out.shape)} — the Ultralytics loss surface changed; "
                f"see assert_ultralytics_compat()."
            )
        # In-place is autograd-safe here: BCEWithLogitsLoss backward recomputes
        # from logits/targets, never from its output — and it avoids allocating
        # a second (bs, anchors, nc) map (bit-identity test covers this).
        return out.mul_(mask.to(dtype=out.dtype))


class MaskedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with per-image class masking of the BCE cls loss.

    For every image in the batch, classes NOT trusted by its completeness
    policy contribute zero classification loss (no false "push to
    background"), while trusted classes — and the box/DFL losses, which
    depend only on assigner outputs — behave exactly as stock. An all-ones
    mask is bit-identical to stock v8DetectionLoss (unit-tested).
    """

    def __init__(
        self,
        model: Any,
        lookup: CompletenessLookup,
        config: MitigationConfig,
        tal_topk: int = 10,
    ) -> None:
        """Build the masked criterion for a (de-paralleled) detection model.

        Args:
            model:    The DetectionModel instance (must have ``args`` attached,
                      i.e. constructed through the trainer flow).
            lookup:   Completeness lookup (validated at load).
            config:   Mitigation settings (unknown-image policy, stats).
            tal_topk: Passed through to the stock loss.

        Raises:
            RuntimeError: On end2end models (different loss family), taxonomy
                          size mismatch, or upstream-surface drift.
        """
        assert_ultralytics_compat()
        if getattr(model, "end2end", False):
            raise RuntimeError(
                "Missing-annotation mitigation supports the standard detection loss "
                "only — end2end (E2ELoss) models are not supported."
            )
        super().__init__(model, tal_topk=tal_topk)
        if lookup.nc != self.nc:
            raise RuntimeError(
                f"Completeness artifact covers nc={lookup.nc} classes but the model "
                f"head has nc={self.nc} — regenerate the artifact against the "
                f"taxonomy this model trains on (dvc repro generate_completeness)."
            )
        # Typed handle for set/clear; self.bce is what the stock loss calls.
        self._masking_bce = _MaskingBCE(self.bce)
        self.bce = self._masking_bce  # type: ignore[assignment]
        self._lookup = lookup
        self._config = config
        self._warned_unknown: set[str] = set()
        # Few unique rows exist (one per policy) — cache their tensors so the
        # per-batch build is a stack of cached (nc,) tensors, not re-parsing.
        self._row_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._stats_batches = 0
        self._stats_images = 0
        self._stats_total_cells = 0
        self._stats_masked_cells = 0

    def _build_batch_mask(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Build the (bs, 1, nc) trusted-class mask for one batch.

        Args:
            batch: Ultralytics train/val batch dict (uses ``im_file``).

        Returns:
            Float mask tensor on the loss device, or None when the batch has
            no ``im_file`` and the config allows full supervision.

        Raises:
            RuntimeError:      If ``im_file`` is absent under strict policy.
            UnknownImageError: If an image has no completeness record under
                               strict policy (``on_unknown_image: error``).
        """
        im_files = batch.get("im_file")
        if im_files is None:
            message = (
                "training batch carries no 'im_file' — completeness masks cannot "
                "be resolved for it"
            )
            if self._config.on_unknown_image == "error":
                raise RuntimeError(
                    f"{message}. Set missing_annotation_mitigation.on_unknown_image: "
                    f"warn_full_supervision to train such batches unmasked."
                )
            if "<no-im-file>" not in self._warned_unknown:
                self._warned_unknown.add("<no-im-file>")
                logger.warning(f"{message} — proceeding with full supervision")
            return None

        rows: list[tuple[int, ...]] = []
        for file in im_files:
            try:
                rows.append(self._lookup.mask_row(str(file)))
            except UnknownImageError as e:
                if self._config.on_unknown_image == "error":
                    raise
                name = Path(str(file)).name
                if name not in self._warned_unknown:
                    self._warned_unknown.add(name)
                    logger.warning(f"{e} — training '{name}' with full supervision")
                rows.append((1,) * self.nc)
        tensors = []
        for row in rows:
            cached = self._row_tensor_cache.get(row)
            if cached is None:
                cached = torch.tensor(row, dtype=torch.float32, device=self.device)
                self._row_tensor_cache[row] = cached
            tensors.append(cached)
        return torch.stack(tensors).unsqueeze(1)

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the stock loss with the per-batch class mask installed."""
        mask = self._build_batch_mask(batch)
        self._masking_bce.set_mask(mask)
        try:
            result: tuple[torch.Tensor, torch.Tensor] = super().__call__(preds, batch)
        finally:
            self._masking_bce.clear_mask()
        if mask is not None and self._config.log_mask_stats:
            self._stats_batches += 1
            self._stats_images += int(mask.shape[0])
            self._stats_total_cells += int(mask.shape[0]) * self.nc
            self._stats_masked_cells += int((mask == 0).sum().item())
        return result

    def pop_mask_stats(self) -> dict[str, Any]:
        """Return accumulated masking statistics and reset the counters.

        Returns:
            Dict with batches, images, masked/total (image, class) cells and
            the masked fraction since the last pop.
        """
        stats: dict[str, Any] = {
            "batches": self._stats_batches,
            "images": self._stats_images,
            "masked_cells": self._stats_masked_cells,
            "total_cells": self._stats_total_cells,
            "masked_fraction": (
                round(self._stats_masked_cells / self._stats_total_cells, 4)
                if self._stats_total_cells
                else 0.0
            ),
        }
        self._stats_batches = 0
        self._stats_images = 0
        self._stats_total_cells = 0
        self._stats_masked_cells = 0
        return stats
