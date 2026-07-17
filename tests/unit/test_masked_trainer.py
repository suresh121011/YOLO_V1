"""Unit tests for src.training.trainer — trainer factory and criterion attach."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from src.training import trainer as trainer_mod  # noqa: E402
from src.training._masked_loss_impl import MaskedDetectionLoss  # noqa: E402
from src.training.completeness_lookup import CompletenessLookup  # noqa: E402
from src.training.mitigation_config import MitigationConfig  # noqa: E402
from src.training.trainer import (  # noqa: E402
    MaskedDetectionTrainer,
    _attach_masked_criterion,
    build_masked_trainer,
)

_NC = 5


def make_lookup() -> CompletenessLookup:
    """In-memory lookup for a tiny taxonomy."""
    return CompletenessLookup(
        nc=_NC,
        fingerprint="sha256:test",
        source_path=Path("in-memory"),
        _policy_by_image={"a.jpg": "full"},
        _mask_row_by_policy={"full": (1,) * _NC},
    )


def build_model():
    """Tiny offline DetectionModel with hyperparameters attached."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(cfg="yolo11n.yaml", nc=_NC, verbose=False)
    model.args = get_cfg()
    return model


@pytest.mark.unit
class TestBuildMaskedTrainer:
    """Factory contract."""

    def test_returns_configured_subclass(self) -> None:
        config = MitigationConfig(enabled=True)
        lookup = make_lookup()
        trainer_cls = build_masked_trainer(config, lookup)
        assert issubclass(trainer_cls, MaskedDetectionTrainer)
        assert trainer_cls.mitigation_config is config
        assert trainer_cls.mitigation_lookup is lookup

    def test_two_factories_do_not_share_state(self) -> None:
        config_a = MitigationConfig(enabled=True)
        config_b = MitigationConfig(enabled=True, log_mask_stats=False)
        cls_a = build_masked_trainer(config_a, make_lookup())
        cls_b = build_masked_trainer(config_b, make_lookup())
        assert cls_a.mitigation_config is config_a
        assert cls_b.mitigation_config is config_b
        assert MaskedDetectionTrainer.mitigation_config is None  # base untouched

    def test_disabled_config_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="disabled"):
            build_masked_trainer(MitigationConfig(enabled=False), make_lookup())

    def test_base_class_unusable_without_factory(self) -> None:
        with pytest.raises(RuntimeError, match="build_masked_trainer"):
            MaskedDetectionTrainer()

    def test_ddp_rank_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer_cls = build_masked_trainer(MitigationConfig(enabled=True), make_lookup())
        monkeypatch.setattr(trainer_mod, "RANK", 0)
        with pytest.raises(RuntimeError, match="DDP"):
            trainer_cls()


@pytest.mark.unit
class TestAttachMaskedCriterion:
    """The on_train_start callback against real (tiny) models."""

    def _fake_trainer(self, model, ema_model=None):
        ema = SimpleNamespace(ema=ema_model) if ema_model is not None else None
        return SimpleNamespace(
            model=model,
            ema=ema,
            mitigation_config=MitigationConfig(enabled=True),
            mitigation_lookup=make_lookup(),
        )

    def test_attaches_masked_criterion_to_train_model(self) -> None:
        model = build_model()
        _attach_masked_criterion(self._fake_trainer(model))
        assert isinstance(model.criterion, MaskedDetectionLoss)

    def test_attaches_separate_criterion_to_ema_model(self) -> None:
        model = build_model()
        ema_model = build_model()
        _attach_masked_criterion(self._fake_trainer(model, ema_model))
        assert isinstance(model.criterion, MaskedDetectionLoss)
        assert isinstance(ema_model.criterion, MaskedDetectionLoss)
        assert model.criterion is not ema_model.criterion

    def test_preexisting_criterion_fails_loud(self) -> None:
        model = build_model()
        model.criterion = object()  # trainer flow changed — must not proceed
        with pytest.raises(RuntimeError, match="already has a loss criterion"):
            _attach_masked_criterion(self._fake_trainer(model))

    def test_lazy_init_criterion_is_preempted(self) -> None:
        # After attach, BaseModel.loss() must use our criterion, not create
        # a stock one via init_criterion().
        model = build_model()
        model.eval()
        _attach_masked_criterion(self._fake_trainer(model))
        attached = model.criterion
        batch = {
            "img": torch.rand(1, 3, 64, 64),
            "batch_idx": torch.zeros(1),
            "cls": torch.zeros(1, 1),
            "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "im_file": ["a.jpg"],
        }
        with torch.no_grad():
            model.loss(batch)
        assert model.criterion is attached  # lazy guard never replaced it
