"""Unit tests for the auto-annotator registry and the base contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.base import (
    AnnotationError,
    AutoAnnotator,
    BackendConfig,
    Detection,
    ModelFingerprint,
    prompt_fingerprint,
)
from src.dataset.annotation.registry import (
    available_annotators,
    get_annotator,
    register_annotator,
)
from unit.annotation_fakes import FakeAnnotator

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> BackendConfig:
    base: dict[str, object] = {
        "enabled": True,
        "weights": "",
        "weights_sha256": "",
        "imgsz": 640,
        "conf_floor": 0.05,
        "max_det": 100,
        "prompts": {"charger": ["phone charger"], "wire": []},
        "thresholds": {"default": 0.25, "charger": 0.2},
    }
    base.update(overrides)
    return BackendConfig.from_annotation_config("fake", base)


class TestRegistry:
    def test_fake_annotator_is_registered(self) -> None:
        assert "fake" in available_annotators()

    def test_get_annotator_instantiates(self) -> None:
        assert isinstance(get_annotator("fake"), FakeAnnotator)

    def test_unknown_name_error_lists_registered(self) -> None:
        with pytest.raises(AnnotationError, match="fake"):
            get_annotator("nope")

    def test_duplicate_registration_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @register_annotator("fake")
            class Duplicate(AutoAnnotator):  # pragma: no cover - never used
                def load(self, config: BackendConfig, device: str) -> None: ...

                def annotate(
                    self, image_path: Path, target_class_ids: tuple[int, ...]
                ) -> list[Detection]:
                    return []

                def fingerprint(self) -> ModelFingerprint:
                    raise NotImplementedError

    def test_registration_extends_generation_without_core_changes(self) -> None:
        @register_annotator("custom_for_test")
        class CustomAnnotator(FakeAnnotator):
            pass

        try:
            assert isinstance(get_annotator("custom_for_test"), CustomAnnotator)
        finally:
            # keep the module-level registry clean for other tests
            from src.dataset.annotation import registry

            del registry._ANNOTATORS["custom_for_test"]


class TestBackendConfig:
    def test_from_annotation_config_parses_prompts_and_thresholds(self) -> None:
        config = _config()
        assert config.prompts["charger"] == ("phone charger",)
        assert config.prompts["wire"] == ()
        assert config.threshold_for("charger") == 0.2
        assert config.threshold_for("unlisted") == 0.25

    def test_null_prompt_entry_becomes_empty(self) -> None:
        config = _config(prompts={"charger": None})
        assert config.prompts["charger"] == ()

    def test_malformed_prompts_raise(self) -> None:
        with pytest.raises(AnnotationError, match="list of strings"):
            _config(prompts={"charger": "not-a-list"})

    def test_extra_keys_preserved(self) -> None:
        config = _config(hf_model="org/model")
        assert config.extra["hf_model"] == "org/model"

    def test_validate_clean(self) -> None:
        assert _config().validate({"charger": 10, "wire": 11}) == []

    def test_validate_flags_out_of_range_and_missing_default(self) -> None:
        config = _config(conf_floor=1.5, thresholds={"charger": 2.0})
        problems = config.validate()
        assert any("conf_floor" in p for p in problems)
        assert any("'default'" in p for p in problems)
        assert any("charger" in p for p in problems)

    def test_validate_flags_unknown_taxonomy_names(self) -> None:
        problems = _config().validate({"person": 0})
        assert any("not in the taxonomy" in p for p in problems)


class TestPromptFingerprint:
    def test_stable_and_order_independent(self) -> None:
        a = prompt_fingerprint({"b": ("y",), "a": ("x",)}, {"default": 0.25})
        b = prompt_fingerprint({"a": ("x",), "b": ("y",)}, {"default": 0.25})
        assert a == b
        assert a.startswith("sha256:")

    def test_changes_with_prompt_text_and_threshold(self) -> None:
        base = prompt_fingerprint({"a": ("x",)}, {"default": 0.25})
        assert prompt_fingerprint({"a": ("z",)}, {"default": 0.25}) != base
        assert prompt_fingerprint({"a": ("x",)}, {"default": 0.30}) != base


class TestFakeAnnotator:
    def test_deterministic_detections_for_targeted_classes(self, tmp_path: Path) -> None:
        annotator = get_annotator("fake")
        annotator.load(_config(), device="cpu")
        detections = annotator.annotate(tmp_path / "img.jpg", (10, 11))
        assert [d.class_id for d in detections] == [10, 11]
        assert [d.conf for d in detections] == [0.9, 0.8]
        assert all(d.origin == "fake" for d in detections)
        # deterministic: same call → same result
        assert annotator.annotate(tmp_path / "img.jpg", (10, 11)) == detections

    def test_requires_load_first(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="load"):
            FakeAnnotator().annotate(tmp_path / "img.jpg", (0,))

    def test_fingerprint_reflects_prompts(self) -> None:
        annotator = get_annotator("fake")
        config = _config()
        annotator.load(config, device="cpu")
        fp = annotator.fingerprint()
        assert fp.backend == "fake"
        assert fp.prompt_fingerprint == prompt_fingerprint(config.prompts, config.thresholds)
