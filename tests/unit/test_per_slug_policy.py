"""Per-slug trust resolution (ADR-P5-15).

`local_captures` ingests one archive per slug, each annotating its own class set
exhaustively. The merge stage records one `label_completeness` entry per
*source* — the union of all 23 slugs — so a source-level policy declared ~21.7
classes of 23 falsely trusted across 69.5% of the `dataset-v0.6.0` build, and
told `targeting.py` those images were already supervised so auto-annotation
never looked at them.

These tests replace `test_completeness_policy_granularity.py`, which pinned the
defect. That pin's own docstring required its removal once the fix landed.

The two resolution paths — completeness masks and annotation targets — must
agree exactly. If they drift, training masks and candidate generation describe
different datasets, which is the failure ADR-P5-15 exists to prevent, so the
agreement is asserted directly rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.annotation.base import AnnotationError
from src.dataset.annotation.targeting import build_targets
from src.dataset.completeness_policies import (
    CompletenessError,
    PolicyContext,
    load_slug_index,
    slug_for_merged_filename,
)
from src.dataset.manifest import MergedManifest

pytestmark = pytest.mark.unit

SOURCE = "local_captures"
IDS_BY_NAME = {"person": 0, "chair": 1, "bed": 2, "sink": 3, "charger": 4, "book": 5}


def _write_slug(root: Path, slug: str, trusted: list[str], image_keys: list[str]) -> None:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "source": f"{SOURCE}/{slug}",
                "trusted_classes": trusted,
                "image_hashes": {k: "0" * 40 for k in image_keys},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def slug_root(tmp_path: Path) -> Path:
    """Two slugs mirroring the real shape: one multi-class, one single-class."""
    root = tmp_path / "local_captures"
    _write_slug(root, "bed", ["bed", "chair", "person", "sink"], ["bed__a.jpg", "bed__b.jpg"])
    _write_slug(root, "charger", ["charger"], ["charger__c.jpg"])
    return root


def _ctx(slug_root: Path, **over: object) -> PolicyContext:
    base = {
        "source": SOURCE,
        "manifest_trusted_classes": ("bed", "chair", "person", "sink", "charger"),
        "config_trusted_classes": None,
        "class_ids_by_name": IDS_BY_NAME,
        "nc": len(IDS_BY_NAME),
        "capture_manifests_dir": None,
        "verification_ledger": None,
        "source_raw_root": slug_root,
    }
    base.update(over)
    return PolicyContext(**base)  # type: ignore[arg-type]


class TestSlugIndex:
    def test_reads_each_slugs_own_trusted_classes(self, slug_root: Path) -> None:
        index = load_slug_index(slug_root, SOURCE)
        assert index.trusted_by_slug["bed"] == ("bed", "chair", "person", "sink")
        assert index.trusted_by_slug["charger"] == ("charger",)

    def test_resolves_merged_filename_via_recorded_image_keys(self, slug_root: Path) -> None:
        """Not filename parsing — a lookup in what ingest recorded."""
        index = load_slug_index(slug_root, SOURCE)
        assert slug_for_merged_filename(index, SOURCE, "local_captures_bed__a.jpg") == "bed"
        assert slug_for_merged_filename(index, SOURCE, "local_captures_charger__c.jpg") == "charger"

    def test_unknown_image_raises_rather_than_defaulting(self, slug_root: Path) -> None:
        """The load-bearing guarantee: no silent fallback.

        Falling back to the source-level union would look exactly like success
        while re-introducing the defect for that image.
        """
        index = load_slug_index(slug_root, SOURCE)
        with pytest.raises(CompletenessError, match="matches no slug manifest"):
            slug_for_merged_filename(index, SOURCE, "local_captures_ghost__z.jpg")

    def test_missing_merge_prefix_raises(self, slug_root: Path) -> None:
        index = load_slug_index(slug_root, SOURCE)
        with pytest.raises(CompletenessError, match="merge prefix"):
            slug_for_merged_filename(index, SOURCE, "bed__a.jpg")

    def test_absent_root_raises_rather_than_returning_empty(self, tmp_path: Path) -> None:
        with pytest.raises(CompletenessError, match="does not exist"):
            load_slug_index(tmp_path / "nope", SOURCE)

    def test_slug_without_trusted_classes_raises(self, slug_root: Path) -> None:
        _write_slug(slug_root, "mystery", [], ["mystery__x.jpg"])
        with pytest.raises(CompletenessError, match="declares no trusted_classes"):
            load_slug_index(slug_root, SOURCE)

    def test_cross_slug_key_collision_raises(self, slug_root: Path) -> None:
        _write_slug(slug_root, "clash", ["book"], ["bed__a.jpg"])
        with pytest.raises(CompletenessError, match="claimed by slugs"):
            load_slug_index(slug_root, SOURCE)


class TestPerSlugWithLedgerPolicy:
    @staticmethod
    def _provider():
        from src.dataset.completeness_policies import get_policy_provider

        return get_policy_provider("per_slug_with_ledger")

    def test_emits_one_policy_per_slug_not_one_per_source(self, slug_root: Path) -> None:
        policies = self._provider().resolve_policies(_ctx(slug_root))
        assert set(policies) == {"local_captures/bed", "local_captures/charger"}
        assert SOURCE not in policies, "a source-level key would restore the union"

    def test_each_slug_trusts_only_its_own_classes(self, slug_root: Path) -> None:
        policies = self._provider().resolve_policies(_ctx(slug_root))
        assert policies["local_captures/bed"] == (0, 1, 2, 3)  # person, chair, bed, sink
        assert policies["local_captures/charger"] == (4,)

    def test_charger_image_does_not_inherit_bed_trust(self, slug_root: Path) -> None:
        """The defect, stated as its own test."""
        provider = self._provider()
        policies = provider.resolve_policies(_ctx(slug_root))
        key = provider.policy_key_for_image(_ctx(slug_root), "local_captures_charger__c.jpg")
        assert key == "local_captures/charger"
        assert 2 not in policies[key], "bed must not be trusted in a charger frame"

    def test_union_mismatch_with_merged_manifest_raises(self, slug_root: Path) -> None:
        """Ingest manifests and the merge must describe the same build."""
        ctx = _ctx(slug_root, manifest_trusted_classes=("bed", "chair"))
        with pytest.raises(CompletenessError, match="does not match the merged manifest"):
            self._provider().resolve_policies(ctx)

    def test_missing_raw_root_raises(self, slug_root: Path) -> None:
        with pytest.raises(CompletenessError, match="no raw root was supplied"):
            self._provider().resolve_policies(_ctx(slug_root, source_raw_root=None))

    def test_unattributable_image_raises_at_key_lookup(self, slug_root: Path) -> None:
        provider = self._provider()
        provider.resolve_policies(_ctx(slug_root))
        with pytest.raises(CompletenessError):
            provider.policy_key_for_image(_ctx(slug_root), "local_captures_ghost__z.jpg")


class _FakeLedger:
    """Minimal LedgerLike over one verified image."""

    def __init__(self, filename: str, classes: frozenset[str]) -> None:
        self._f, self._c = filename, classes

    def all_images(self) -> frozenset[str]:
        return frozenset({self._f})

    def verified_class_names(self, filename: str) -> frozenset[str]:
        return self._c if filename == self._f else frozenset()

    def entry_source(self, filename: str) -> str | None:
        return SOURCE if filename == self._f else None

    def taxonomy_fingerprint(self) -> str:
        return ""


class TestLedgerExpansionOverPerSlugBase:
    """Verification unmasks cells — over the slug's base, not the union."""

    @staticmethod
    def _provider():
        from src.dataset.completeness_policies import get_policy_provider

        return get_policy_provider("per_slug_with_ledger")

    def test_verified_class_expands_only_that_image(self, slug_root: Path) -> None:
        ledger = _FakeLedger("local_captures_charger__c.jpg", frozenset({"book"}))
        ctx = _ctx(slug_root, verification_ledger=ledger)
        provider = self._provider()
        policies = provider.resolve_policies(ctx)

        key = provider.policy_key_for_image(ctx, "local_captures_charger__c.jpg")
        assert key.startswith("local_captures/charger/ledger/")
        assert policies[key] == (4, 5)  # charger + verified book
        # The unverified sibling stays on the plain slug policy.
        other = provider.policy_key_for_image(ctx, "local_captures_bed__a.jpg")
        assert other == "local_captures/bed"

    def test_redundant_verification_does_not_fork_a_policy(self, slug_root: Path) -> None:
        """Verifying a class the slug already trusts changes nothing."""
        ledger = _FakeLedger("local_captures_bed__a.jpg", frozenset({"bed"}))
        ctx = _ctx(slug_root, verification_ledger=ledger)
        provider = self._provider()
        provider.resolve_policies(ctx)
        assert (
            provider.policy_key_for_image(ctx, "local_captures_bed__a.jpg") == "local_captures/bed"
        )

    def test_empty_ledger_is_byte_identical_to_no_ledger(self, slug_root: Path) -> None:
        provider = self._provider()
        without = provider.resolve_policies(_ctx(slug_root))
        with_empty = self._provider().resolve_policies(
            _ctx(slug_root, verification_ledger=_FakeLedger("nobody.jpg", frozenset()))
        )
        assert without == with_empty


class TestTargetingUsesTheSameTrust:
    """`targeting.py` must resolve trust exactly as the policy layer does."""

    @staticmethod
    def _manifest(tmp_path: Path) -> MergedManifest:
        path = tmp_path / "merged_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": "2026-07-29T00:00:00Z",
                    "sources": [],
                    "image_provenance": {
                        "local_captures_bed__a.jpg": SOURCE,
                        "local_captures_charger__c.jpg": SOURCE,
                    },
                    "duplicates_removed": 0,
                    "filtered_out": 0,
                    "class_counts": {},
                    "label_completeness": {SOURCE: ["bed", "chair", "person", "sink", "charger"]},
                    "notes": [],
                }
            ),
            encoding="utf-8",
        )
        return MergedManifest.load(path)

    def test_per_slug_source_targets_classes_the_slug_does_not_annotate(
        self, tmp_path: Path, slug_root: Path
    ) -> None:
        """The bug: under the union these images were targeted for nothing."""
        manifest = self._manifest(tmp_path)
        policies = {SOURCE: "per_slug_with_ledger"}
        promptable = (0, 1, 2, 3, 4, 5)  # every class promptable
        index = load_slug_index(slug_root, SOURCE)

        targets = build_targets(manifest, policies, promptable, IDS_BY_NAME, None, {SOURCE: index})

        # charger slug trusts only charger(4) -> everything else is targetable
        assert targets["local_captures_charger__c.jpg"] == (0, 1, 2, 3, 5)
        # bed slug trusts person/chair/bed/sink -> charger + book remain
        assert targets["local_captures_bed__a.jpg"] == (4, 5)

    def test_union_would_have_targeted_nothing(self, tmp_path: Path) -> None:
        """Regression witness for the ADR-P5-15 finding.

        Same manifest under the OLD source-level mode: the union covers every
        annotated class, so no cell is ever offered to auto-annotation.
        """
        manifest = self._manifest(tmp_path)
        targets = build_targets(
            manifest,
            {SOURCE: "trusted_list_with_ledger"},
            (0, 1, 2, 3, 4),  # promptable = exactly the union
            IDS_BY_NAME,
            None,
        )
        assert targets == {}, "the union suppressed all candidate generation"

    def test_per_slug_source_without_index_is_an_error_not_a_fallback(self, tmp_path: Path) -> None:
        manifest = self._manifest(tmp_path)
        with pytest.raises(AnnotationError, match="no slug index was supplied"):
            build_targets(
                manifest, {SOURCE: "per_slug_with_ledger"}, (0, 1), IDS_BY_NAME, None, None
            )

    def test_unresolvable_image_is_an_error(self, tmp_path: Path, slug_root: Path) -> None:
        path = tmp_path / "m2.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": "2026-07-29T00:00:00Z",
                    "sources": [],
                    "image_provenance": {"local_captures_ghost__z.jpg": SOURCE},
                    "duplicates_removed": 0,
                    "filtered_out": 0,
                    "class_counts": {},
                    "label_completeness": {SOURCE: ["bed", "chair", "person", "sink", "charger"]},
                    "notes": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(AnnotationError, match="matches no slug manifest"):
            build_targets(
                MergedManifest.load(path),
                {SOURCE: "per_slug_with_ledger"},
                (0, 1),
                IDS_BY_NAME,
                None,
                {SOURCE: load_slug_index(slug_root, SOURCE)},
            )
