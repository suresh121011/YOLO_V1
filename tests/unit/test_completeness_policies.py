"""Unit tests for src.dataset.completeness_policies — policy provider registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.annotation.ledger import LedgerView, new_ledger, record_verdict
from src.dataset.completeness_policies import (
    _PROVIDERS,
    CompletenessError,
    CompletenessPolicyProvider,
    LedgerLike,
    PerSessionPolicy,
    PolicyContext,
    TrustedListPolicy,
    TrustedListWithLedgerPolicy,
    VerifiedAbsenceAllPolicy,
    get_policy_provider,
    register_policy_provider,
    registered_policy_modes,
)
from src.dataset.manifest import CaptureSessionManifest

_NAMES = {"person": 0, "face": 1, "knife": 2, "door": 3, "stove": 4}
_NC = 5


def make_ctx(
    source: str = "coco",
    manifest_trusted: tuple[str, ...] | None = ("person", "knife"),
    config_trusted: tuple[str, ...] | None = ("person", "knife"),
    capture_manifests_dir: Path | None = None,
    verification_ledger: LedgerLike | None = None,
) -> PolicyContext:
    """Build a PolicyContext with small-taxonomy defaults."""
    return PolicyContext(
        source=source,
        manifest_trusted_classes=manifest_trusted,
        config_trusted_classes=config_trusted,
        class_ids_by_name=_NAMES,
        nc=_NC,
        capture_manifests_dir=capture_manifests_dir,
        verification_ledger=verification_ledger,
    )


def write_session_manifest(
    manifests_dir: Path,
    session_id: str,
    trusted_classes: list[str],
    annotation_status: str = "finalized",
) -> Path:
    """Write one capture session manifest and return its path."""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = CaptureSessionManifest(
        source="custom_captures",
        session_id=session_id,
        annotation_status=annotation_status,
        trusted_classes=trusted_classes,
    )
    path = manifests_dir / f"{session_id}.json"
    manifest.save(path)
    return path


@pytest.mark.unit
class TestRegistry:
    """Provider registry behavior."""

    def test_builtin_modes_registered(self) -> None:
        assert registered_policy_modes() == [
            "per_session",
            "trusted_list",
            "trusted_list_with_ledger",
            "verified_absence_all",
        ]

    def test_get_provider_returns_fresh_instances(self) -> None:
        a = get_policy_provider("per_session")
        b = get_policy_provider("per_session")
        assert isinstance(a, PerSessionPolicy)
        assert a is not b

    def test_unknown_mode_error_lists_registered_modes(self) -> None:
        with pytest.raises(CompletenessError, match="per_session.*trusted_list"):
            get_policy_provider("nonsense_mode")

    def test_custom_provider_registration_extends_registry(self) -> None:
        @register_policy_provider("test_only_mode")
        class TestOnlyPolicy(CompletenessPolicyProvider):
            def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
                return {ctx.source: (0,)}

        try:
            provider = get_policy_provider("test_only_mode")
            assert provider.mode == "test_only_mode"
            assert provider.resolve_policies(make_ctx()) == {"coco": (0,)}
        finally:
            del _PROVIDERS["test_only_mode"]

    def test_duplicate_mode_registration_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_policy_provider("trusted_list")(TrustedListPolicy)


@pytest.mark.unit
class TestTrustedListPolicy:
    """trusted_list: manifest is data-of-record, config is cross-check."""

    def test_resolves_sorted_unique_ids(self) -> None:
        ctx = make_ctx(manifest_trusted=("knife", "person"), config_trusted=("person", "knife"))
        assert TrustedListPolicy().resolve_policies(ctx) == {"coco": (0, 2)}

    def test_policy_key_is_source(self) -> None:
        assert TrustedListPolicy().policy_key_for_image(make_ctx(), "coco_0001.jpg") == "coco"

    def test_missing_manifest_entry_is_error(self) -> None:
        ctx = make_ctx(manifest_trusted=None)
        with pytest.raises(CompletenessError, match="no trusted classes"):
            TrustedListPolicy().resolve_policies(ctx)

    def test_empty_trusted_list_is_error(self) -> None:
        ctx = make_ctx(manifest_trusted=())
        with pytest.raises(CompletenessError, match="verified_absence_all"):
            TrustedListPolicy().resolve_policies(ctx)

    def test_config_manifest_drift_is_error(self) -> None:
        ctx = make_ctx(manifest_trusted=("person",), config_trusted=("person", "knife"))
        with pytest.raises(CompletenessError, match="drift"):
            TrustedListPolicy().resolve_policies(ctx)

    def test_absent_config_key_skips_cross_check(self) -> None:
        ctx = make_ctx(manifest_trusted=("door",), config_trusted=None)
        assert TrustedListPolicy().resolve_policies(ctx) == {"coco": (3,)}

    def test_unknown_class_name_is_error(self) -> None:
        ctx = make_ctx(manifest_trusted=("person", "unicorn"), config_trusted=None)
        with pytest.raises(CompletenessError, match="unicorn"):
            TrustedListPolicy().resolve_policies(ctx)


def _ledger_with(
    entries: list[tuple[str, str, str, list[tuple[float, float, float, float]]]]
) -> LedgerView:
    """Build a LedgerView from (filename, source, class_name, boxes) tuples.

    Empty ``boxes`` records a ``verified_absent`` verdict; non-empty records
    ``present_labeled``.
    """
    ledger = new_ledger()
    for i, (filename, source, class_name, boxes) in enumerate(entries):
        status = "present_labeled" if boxes else "verified_absent"
        record_verdict(
            ledger, filename, source, class_name, status, boxes, f"vb{i:03d}", "anno_1", "cvat", ""
        )
    return LedgerView(raw=ledger)


@pytest.mark.unit
class TestTrustedListWithLedgerPolicy:
    """trusted_list_with_ledger: base trusted_list, expanded by the ledger."""

    def test_no_ledger_is_byte_identical_to_base(self) -> None:
        ctx = make_ctx(verification_ledger=None)
        assert TrustedListWithLedgerPolicy().resolve_policies(ctx) == {"coco": (0, 2)}

    def test_empty_ledger_is_byte_identical_to_base(self) -> None:
        ctx = make_ctx(verification_ledger=LedgerView(raw=new_ledger()))
        assert TrustedListWithLedgerPolicy().resolve_policies(ctx) == {"coco": (0, 2)}

    def test_ledger_verified_image_gets_expanded_policy(self) -> None:
        ledger = _ledger_with([("coco_1.jpg", "coco", "face", [(0.5, 0.5, 0.1, 0.1)])])
        ctx = make_ctx(verification_ledger=ledger)
        policy = TrustedListWithLedgerPolicy()
        resolved = policy.resolve_policies(ctx)
        assert resolved["coco"] == (0, 2)  # base untouched
        ledger_keys = [k for k in resolved if k.startswith("coco/ledger/")]
        assert len(ledger_keys) == 1
        assert resolved[ledger_keys[0]] == (0, 1, 2)  # base (person, knife) + face

    def test_policy_key_for_image_routes_ledger_images(self) -> None:
        ledger = _ledger_with([("coco_1.jpg", "coco", "face", [(0.5, 0.5, 0.1, 0.1)])])
        ctx = make_ctx(verification_ledger=ledger)
        policy = TrustedListWithLedgerPolicy()
        resolved = policy.resolve_policies(ctx)
        ledger_key = next(k for k in resolved if k.startswith("coco/ledger/"))
        assert policy.policy_key_for_image(ctx, "coco_1.jpg") == ledger_key
        assert policy.policy_key_for_image(ctx, "coco_2.jpg") == "coco"

    def test_other_source_ledger_entries_ignored(self) -> None:
        ledger = _ledger_with([("oi_1.jpg", "openimages", "face", [(0.5, 0.5, 0.1, 0.1)])])
        ctx = make_ctx(verification_ledger=ledger)
        resolved = TrustedListWithLedgerPolicy().resolve_policies(ctx)
        assert resolved == {"coco": (0, 2)}

    def test_verified_absent_also_expands_policy(self) -> None:
        ledger = _ledger_with([("coco_1.jpg", "coco", "door", [])])
        ctx = make_ctx(verification_ledger=ledger)
        resolved = TrustedListWithLedgerPolicy().resolve_policies(ctx)
        ledger_keys = [k for k in resolved if k.startswith("coco/ledger/")]
        assert resolved[ledger_keys[0]] == (0, 2, 3)  # base + door

    def test_fully_redundant_verification_stays_on_base_policy(self) -> None:
        # 'person' is already in the base trusted list — nothing to expand.
        ledger = _ledger_with([("coco_1.jpg", "coco", "person", [(0.5, 0.5, 0.1, 0.1)])])
        ctx = make_ctx(verification_ledger=ledger)
        policy = TrustedListWithLedgerPolicy()
        resolved = policy.resolve_policies(ctx)
        assert list(resolved) == ["coco"]  # no extra policy created
        assert policy.policy_key_for_image(ctx, "coco_1.jpg") == "coco"

    def test_unknown_ledger_class_name_is_error(self) -> None:
        ledger = _ledger_with([("coco_1.jpg", "coco", "unicorn", [(0.5, 0.5, 0.1, 0.1)])])
        ctx = make_ctx(verification_ledger=ledger)
        with pytest.raises(CompletenessError, match="unicorn"):
            TrustedListWithLedgerPolicy().resolve_policies(ctx)

    def test_two_images_same_effective_set_share_one_key(self) -> None:
        ledger = _ledger_with(
            [
                ("coco_1.jpg", "coco", "face", [(0.5, 0.5, 0.1, 0.1)]),
                ("coco_2.jpg", "coco", "face", [(0.3, 0.3, 0.1, 0.1)]),
            ]
        )
        ctx = make_ctx(verification_ledger=ledger)
        policy = TrustedListWithLedgerPolicy()
        resolved = policy.resolve_policies(ctx)
        ledger_keys = [k for k in resolved if k.startswith("coco/ledger/")]
        assert len(ledger_keys) == 1
        assert policy.policy_key_for_image(ctx, "coco_1.jpg") == policy.policy_key_for_image(
            ctx, "coco_2.jpg"
        )


@pytest.mark.unit
class TestVerifiedAbsenceAllPolicy:
    """verified_absence_all: negatives trust ALL classes (all-ones mask)."""

    def test_resolves_all_class_ids(self) -> None:
        ctx = make_ctx(source="negatives", manifest_trusted=(), config_trusted=())
        assert VerifiedAbsenceAllPolicy().resolve_policies(ctx) == {"negatives": (0, 1, 2, 3, 4)}

    def test_declared_trusted_classes_contradict_mode(self) -> None:
        ctx = make_ctx(source="negatives", manifest_trusted=("person",), config_trusted=())
        with pytest.raises(CompletenessError, match="verified_absence_all"):
            VerifiedAbsenceAllPolicy().resolve_policies(ctx)


@pytest.mark.unit
class TestPerSessionPolicy:
    """per_session: one policy per finalized capture session manifest."""

    def test_resolves_one_policy_per_session(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife", "stove"])
        write_session_manifest(manifests, "h02_hall_s001", ["door"])
        ctx = make_ctx(
            source="custom_captures",
            manifest_trusted=None,
            config_trusted=(),
            capture_manifests_dir=manifests,
        )
        policies = PerSessionPolicy().resolve_policies(ctx)
        assert policies == {
            "custom_captures/h01_kitchen_s001": (2, 4),
            "custom_captures/h02_hall_s001": (3,),
        }

    def test_image_maps_to_session_by_prefix(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife"])
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        provider = PerSessionPolicy()
        provider.resolve_policies(ctx)
        key = provider.policy_key_for_image(ctx, "custom_captures_h01_kitchen_s001_0001.jpg")
        assert key == "custom_captures/h01_kitchen_s001"

    def test_longest_session_id_prefix_wins(self, tmp_path: Path) -> None:
        # "h01_kitchen_s001" is a string prefix of "h01_kitchen_s001b_..." only
        # when the separator underscore matches — the longer id must win.
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife"])
        write_session_manifest(manifests, "h01_kitchen_s001_retake", ["door"])
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        provider = PerSessionPolicy()
        provider.resolve_policies(ctx)
        key = provider.policy_key_for_image(ctx, "custom_captures_h01_kitchen_s001_retake_0001.jpg")
        assert key == "custom_captures/h01_kitchen_s001_retake"

    def test_unfinalized_session_is_error(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife"], annotation_status="staged")
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        with pytest.raises(CompletenessError, match="finalized"):
            PerSessionPolicy().resolve_policies(ctx)

    def test_session_without_trusted_classes_is_error(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", [])
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        with pytest.raises(CompletenessError, match="trusted_classes"):
            PerSessionPolicy().resolve_policies(ctx)

    def test_missing_manifests_dir_resolves_empty(self, tmp_path: Path) -> None:
        ctx = make_ctx(
            source="custom_captures",
            manifest_trusted=None,
            capture_manifests_dir=tmp_path / "nope",
        )
        assert PerSessionPolicy().resolve_policies(ctx) == {}

    def test_unmatched_image_is_error(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife"])
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        provider = PerSessionPolicy()
        provider.resolve_policies(ctx)
        with pytest.raises(CompletenessError, match="no finalized capture session"):
            provider.policy_key_for_image(ctx, "custom_captures_h99_bath_s001_0001.jpg")

    def test_foreign_prefix_is_error(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_session_manifest(manifests, "h01_kitchen_s001", ["knife"])
        ctx = make_ctx(
            source="custom_captures", manifest_trusted=None, capture_manifests_dir=manifests
        )
        provider = PerSessionPolicy()
        provider.resolve_policies(ctx)
        with pytest.raises(CompletenessError, match="merge prefix"):
            provider.policy_key_for_image(ctx, "coco_h01_kitchen_s001_0001.jpg")
