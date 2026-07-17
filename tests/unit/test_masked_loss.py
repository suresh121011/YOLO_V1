"""Unit tests for src.training._masked_loss_impl — masked BCE correctness.

These tests require torch + ultralytics (installed in CI via requirements.txt)
and are skipped cleanly elsewhere. They are the Phase-4 correctness core:
bit-identity with stock loss under an all-ones mask, exact zero gradients for
masked classes, and the ultralytics drift canary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from torch import nn  # noqa: E402

from src.training._masked_loss_impl import MaskedDetectionLoss, _MaskingBCE  # noqa: E402
from src.training.completeness_lookup import CompletenessLookup, UnknownImageError  # noqa: E402
from src.training.masked_loss import assert_ultralytics_compat  # noqa: E402
from src.training.mitigation_config import MitigationConfig  # noqa: E402

_NC = 5


def make_lookup(
    mask_rows: dict[str, tuple[int, ...]] | None = None,
    images: dict[str, str] | None = None,
    nc: int = _NC,
) -> CompletenessLookup:
    """Build an in-memory lookup without touching the filesystem."""
    if mask_rows is None:
        mask_rows = {"full": (1,) * nc, "coco": (1, 0, 1, 0, 0)}
    if images is None:
        images = {"a.jpg": "full", "b.jpg": "coco"}
    return CompletenessLookup(
        nc=nc,
        fingerprint="sha256:test",
        source_path=Path("in-memory"),
        _policy_by_image=images,
        _mask_row_by_policy=mask_rows,
    )


def build_model(nc: int = _NC):
    """Build a tiny DetectionModel offline (no weight download) with args attached."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(cfg="yolo11n.yaml", nc=nc, verbose=False)
    model.args = get_cfg()  # hyperparameter namespace (box/cls/dfl gains)
    return model


def make_batch(bs: int, im_files: list[str] | None, imgsz: int = 64) -> dict:
    """Synthetic detection batch: one GT box of class 0 per image."""
    batch: dict = {
        "img": torch.rand(bs, 3, imgsz, imgsz),
        "batch_idx": torch.arange(bs, dtype=torch.float32),
        "cls": torch.zeros(bs, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]).repeat(bs, 1),
    }
    if im_files is not None:
        batch["im_file"] = im_files
    return batch


@pytest.mark.unit
class TestCompatCanary:
    """The CI drift canary: installed ultralytics must expose our seams."""

    def test_installed_version_passes_canary(self) -> None:
        assert_ultralytics_compat()  # raises RuntimeError on drift

    def test_version_in_supported_window(self) -> None:
        major, minor = (int(p) for p in ultralytics.__version__.split(".")[:2])
        assert (8, 3) <= (major, minor) < (9, 0)


@pytest.mark.unit
class TestMaskingBCE:
    """The wrapper module in isolation."""

    def test_all_ones_mask_is_bit_identical(self) -> None:
        inner = nn.BCEWithLogitsLoss(reduction="none")
        wrapper = _MaskingBCE(inner)
        pred = torch.randn(4, 64, _NC)
        target = torch.rand(4, 64, _NC)
        wrapper.set_mask(torch.ones(4, 1, _NC))
        assert torch.equal(wrapper(pred, target), inner(pred, target))

    def test_zero_mask_zeroes_exactly_those_cells(self) -> None:
        inner = nn.BCEWithLogitsLoss(reduction="none")
        wrapper = _MaskingBCE(inner)
        pred = torch.randn(2, 8, _NC)
        target = torch.rand(2, 8, _NC)
        mask = torch.ones(2, 1, _NC)
        mask[0, 0, 1] = 0.0  # image 0, class 1 untrusted
        wrapper.set_mask(mask)
        out = wrapper(pred, target)
        stock = inner(pred, target)
        assert torch.all(out[0, :, 1] == 0)
        assert torch.equal(out[1], stock[1])  # image 1 untouched
        assert torch.equal(out[0, :, 0], stock[0, :, 0])  # trusted class untouched

    def test_no_mask_is_transparent(self) -> None:
        inner = nn.BCEWithLogitsLoss(reduction="none")
        wrapper = _MaskingBCE(inner)
        pred = torch.randn(2, 8, _NC)
        target = torch.rand(2, 8, _NC)
        assert torch.equal(wrapper(pred, target), inner(pred, target))

    def test_clear_mask_restores_transparency(self) -> None:
        inner = nn.BCEWithLogitsLoss(reduction="none")
        wrapper = _MaskingBCE(inner)
        wrapper.set_mask(torch.zeros(2, 1, _NC))
        wrapper.clear_mask()
        pred = torch.randn(2, 8, _NC)
        target = torch.rand(2, 8, _NC)
        assert torch.equal(wrapper(pred, target), inner(pred, target))

    def test_unexpected_shape_with_mask_fails_loud(self) -> None:
        wrapper = _MaskingBCE(nn.BCEWithLogitsLoss(reduction="none"))
        wrapper.set_mask(torch.ones(2, 1, _NC))
        with pytest.raises(RuntimeError, match="loss surface changed"):
            wrapper(torch.randn(3, 8, _NC), torch.rand(3, 8, _NC))  # wrong bs

    def test_masked_class_gradients_are_exactly_zero(self) -> None:
        inner = nn.BCEWithLogitsLoss(reduction="none")
        wrapper = _MaskingBCE(inner)
        pred = torch.randn(2, 8, _NC, requires_grad=True)
        target = torch.rand(2, 8, _NC)
        mask = torch.ones(2, 1, _NC)
        mask[:, 0, 2] = 0.0  # class 2 untrusted everywhere
        wrapper.set_mask(mask)
        wrapper(pred, target).sum().backward()
        assert pred.grad is not None
        assert torch.all(pred.grad[:, :, 2] == 0)  # no supervision signal
        assert torch.any(pred.grad[:, :, 0] != 0)  # trusted classes still learn


@pytest.mark.unit
class TestMaskedDetectionLoss:
    """Full criterion against a real (tiny) DetectionModel."""

    def test_all_ones_mask_matches_stock_loss_bitwise(self) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        model = build_model()
        model.eval()  # deterministic BN behavior for the shared forward
        batch = make_batch(2, ["a.jpg", "a2.jpg"])
        lookup = make_lookup(images={"a.jpg": "full", "a2.jpg": "full"})
        with torch.no_grad():
            preds = model(batch["img"])

        stock = v8DetectionLoss(model)
        masked = MaskedDetectionLoss(model, lookup=lookup, config=MitigationConfig(enabled=True))
        loss_stock, items_stock = stock(preds, batch)
        loss_masked, items_masked = masked(preds, batch)
        assert torch.equal(loss_stock, loss_masked)
        assert torch.equal(items_stock, items_masked)

    def test_untrusted_classes_reduce_cls_loss_only(self) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        model = build_model()
        model.eval()
        batch = make_batch(2, ["a.jpg", "b.jpg"])  # b.jpg masks classes 1, 3, 4
        lookup = make_lookup()
        with torch.no_grad():
            preds = model(batch["img"])

        stock = v8DetectionLoss(model)
        masked = MaskedDetectionLoss(model, lookup=lookup, config=MitigationConfig(enabled=True))
        _, items_stock = stock(preds, batch)  # [box, cls, dfl]
        _, items_masked = masked(preds, batch)
        assert items_masked[1] < items_stock[1]  # cls loss strictly reduced
        assert torch.equal(items_masked[0], items_stock[0])  # box identical
        assert torch.equal(items_masked[2], items_stock[2])  # dfl identical

    def test_unknown_image_error_policy_raises(self) -> None:
        model = build_model()
        model.eval()
        batch = make_batch(1, ["mystery.jpg"])
        masked = MaskedDetectionLoss(
            model, lookup=make_lookup(), config=MitigationConfig(enabled=True)
        )
        with torch.no_grad():
            preds = model(batch["img"])
        with pytest.raises(UnknownImageError):
            masked(preds, batch)

    def test_unknown_image_warn_policy_trains_unmasked(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        model = build_model()
        model.eval()
        batch = make_batch(1, ["mystery.jpg"])
        config = MitigationConfig(enabled=True, on_unknown_image="warn_full_supervision")
        masked = MaskedDetectionLoss(model, lookup=make_lookup(), config=config)
        stock = v8DetectionLoss(model)
        with torch.no_grad():
            preds = model(batch["img"])
        with caplog.at_level("WARNING"):
            loss_masked, _ = masked(preds, batch)
        loss_stock, _ = stock(preds, batch)
        assert torch.equal(loss_masked, loss_stock)  # full supervision fallback
        assert any("mystery.jpg" in r.message for r in caplog.records)

    def test_nc_mismatch_rejected(self) -> None:
        model = build_model(nc=_NC)
        with pytest.raises(RuntimeError, match="nc="):
            MaskedDetectionLoss(
                model, lookup=make_lookup(nc=7), config=MitigationConfig(enabled=True)
            )

    def test_mask_stats_accumulate_and_reset(self) -> None:
        model = build_model()
        model.eval()
        batch = make_batch(2, ["a.jpg", "b.jpg"])
        masked = MaskedDetectionLoss(
            model, lookup=make_lookup(), config=MitigationConfig(enabled=True)
        )
        with torch.no_grad():
            preds = model(batch["img"])
        masked(preds, batch)
        stats = masked.pop_mask_stats()
        assert stats["batches"] == 1
        assert stats["images"] == 2
        assert stats["total_cells"] == 2 * _NC
        assert stats["masked_cells"] == 3  # b.jpg masks classes 1, 3, 4
        assert stats["masked_fraction"] == round(3 / (2 * _NC), 4)
        assert masked.pop_mask_stats()["batches"] == 0  # reset
