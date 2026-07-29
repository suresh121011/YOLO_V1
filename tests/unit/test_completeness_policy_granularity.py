"""`local_captures` declares more trust than its slug manifests support.

This is a **pin, not a gate** (ADR-P5-14). The defect is real and known: merge
records one source-level `label_completeness` entry holding the union of all 23
slugs' trusted classes, so every `local_captures` image — 69.5% of the
`dataset-v0.6.0` release — inherits ~21.7 falsely-trusted classes, and absent
labels for those classes are supervised as background rather than masked.

It is deferred to `dataset-v0.7.0` because it changes no image, no label and no
split; it is consumed only at train time, which `dataset-v0.6.0` does not
certify.

What these tests exist to prevent is the *silent* part. They fail if the
over-claim gets worse, and they fail if someone fixes it without revisiting
ADR-P5-14 — a fixed pipeline should not keep a test asserting the old shape.
Prose in a changelog cannot do either.

They skip when the raw local_captures tree is absent (a checkout that has not
run `dvc pull`), because the evidence lives in `data/raw/`, not in git.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SLUG_ROOT = REPO_ROOT / "data" / "raw" / "local_captures"
MERGED_MANIFEST = REPO_ROOT / "data" / "merged" / "merged_manifest.json"

#: Measured 2026-07-29 on the dataset-v0.6.0 build. Recorded in ADR-P5-14.
MEASURED_UNION_SIZE = 23
MEASURED_SLUG_COUNT = 23
MEASURED_NARROWER_SLUGS = 23
MEASURED_MEAN_OVERCLAIM = 21.7


def _slug_trusted_sets() -> dict[str, set[str]]:
    """Each slug's own trusted classes, as ingest recorded them."""
    out: dict[str, set[str]] = {}
    for d in sorted(p for p in SLUG_ROOT.iterdir() if p.is_dir()):
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out[d.name] = set(data.get("trusted_classes") or [])
    return {k: v for k, v in out.items() if v}


def _merged_union() -> set[str]:
    entry = json.loads(MERGED_MANIFEST.read_text(encoding="utf-8"))["label_completeness"][
        "local_captures"
    ]
    classes = entry.get("trusted_classes", entry) if isinstance(entry, dict) else entry
    return set(classes)


needs_raw = pytest.mark.skipif(
    not SLUG_ROOT.is_dir() or not MERGED_MANIFEST.exists(),
    reason="local_captures raw tree / merged manifest absent (run `dvc pull`)",
)


@needs_raw
class TestLocalCapturesTrustOverClaim:
    def test_slug_manifests_carry_their_own_trusted_sets(self) -> None:
        """The fix has a data source — this is what makes v0.7.0 cheap.

        The backlog note that raised this assumed the slugs were single-class
        and that per-slug trust would have to be invented. It is already on disk.
        """
        slugs = _slug_trusted_sets()
        assert len(slugs) == MEASURED_SLUG_COUNT, (
            f"expected {MEASURED_SLUG_COUNT} slugs with recorded trusted_classes, "
            f"found {len(slugs)} — re-measure and update ADR-P5-14"
        )

    def test_not_every_slug_is_single_class(self) -> None:
        """Guards against the premise the backlog note got wrong.

        `bed` trusts bed/chair/person/sink; `medicine_bottle` 4; `cupboard` 3.
        A fix that assumes one class per slug would re-introduce the defect in
        the opposite direction — under-claiming trust and masking real
        supervision.
        """
        multi = {s: t for s, t in _slug_trusted_sets().items() if len(t) > 1}
        assert multi, "expected at least one multi-class slug (bed, medicine_bottle, cupboard)"

    def test_merge_publishes_the_union_of_every_slug(self) -> None:
        """The defect itself, stated as an assertion."""
        union = _merged_union()
        slugs = _slug_trusted_sets()
        computed = set().union(*slugs.values())
        assert union == computed, (
            "merged label_completeness['local_captures'] is no longer the plain union "
            "of the slug manifests — if per-slug policies landed, update ADR-P5-14 "
            "and delete this pin"
        )
        assert len(union) == MEASURED_UNION_SIZE

    def test_over_claim_has_not_worsened(self) -> None:
        """Mean falsely-trusted classes per image, weighted by slug image count.

        Fails in both directions on purpose: a larger number means the defect
        grew, a materially smaller one means it was fixed and this pin plus
        ADR-P5-14 are now stale.
        """
        union = _merged_union()
        total_images = 0
        weighted_overclaim = 0
        for slug, trusted in _slug_trusted_sets().items():
            manifest = json.loads((SLUG_ROOT / slug / "manifest.json").read_text(encoding="utf-8"))
            n = int(manifest.get("image_count", 0))
            total_images += n
            weighted_overclaim += n * len(union - trusted)
        assert total_images > 0
        mean = weighted_overclaim / total_images
        assert mean == pytest.approx(MEASURED_MEAN_OVERCLAIM, abs=0.5), (
            f"mean falsely-trusted classes per image is {mean:.1f}, measured "
            f"{MEASURED_MEAN_OVERCLAIM} — see ADR-P5-14"
        )

    def test_every_slug_trusts_less_than_the_published_union(self) -> None:
        """No slug justifies the source-level claim on its own."""
        union = _merged_union()
        narrower = [s for s, t in _slug_trusted_sets().items() if t < union]
        assert len(narrower) == MEASURED_NARROWER_SLUGS, (
            f"{len(narrower)} slugs trust strictly fewer classes than the published "
            f"union; measured {MEASURED_NARROWER_SLUGS}"
        )


@needs_raw
class TestCompletenessArtifactReflectsTheOverClaim:
    """The published artifact, not just the inputs — this is what ships."""

    @staticmethod
    def _completeness() -> dict:
        p = REPO_ROOT / "data" / "processed" / "completeness.json"
        if not p.exists():
            pytest.skip("completeness.json absent (run `dvc pull`)")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_all_local_captures_images_share_one_policy(self) -> None:
        by_policy = self._completeness()["stats"]["by_policy"]
        assert "local_captures" in by_policy, (
            "per-slug policy keys appear to have landed — update ADR-P5-14 and " "remove this pin"
        )
        assert not any(k.startswith("local_captures/") for k in by_policy)

    def test_masked_fraction_is_the_under_masked_one(self) -> None:
        """~0.19 as shipped; per-slug evidence supports ~0.84 (ADR-P5-14).

        The aggregate lives in `dataset_quality_report.json`'s
        `completeness_summary`, not in `completeness.json` — L5 reports
        summarise, they do not recompute.
        """
        p = REPO_ROOT / "data" / "qa_reports" / "dataset_quality_report.json"
        if not p.exists():
            pytest.skip("dataset_quality_report.json absent (run `dvc pull`)")
        summary = json.loads(p.read_text(encoding="utf-8"))["completeness_summary"]
        assert summary["masked_cell_fraction"] == pytest.approx(0.186, abs=0.02)
        assert summary["mean_trusted_classes_per_image"] == pytest.approx(18.7, abs=0.5)
