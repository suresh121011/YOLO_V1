"""
src.dataset.manifest — Dataset Provenance Manifests
===================================================

Every dataset artifact in the pipeline carries a manifest describing where
it came from, under what license, and what it contains. Manifests are the
lineage backbone of the dataset factory:

    data/raw/<source>/manifest.json          ← SourceManifest (downloaders;
                                               for custom_captures this is the
                                               per-source AGGREGATE rebuilt by
                                               src/dataset/capture/ingest.py)
    data/raw/custom_captures/manifests/<session_id>.json
                                             ← CaptureSessionManifest (Phase-3)
    data/merged/merged_manifest.json         ← MergedManifest (merge stage)

Privacy rule: manifests must never embed PII. Capture sessions reference an
externally stored consent record by ID only (``consent_reference``).

Schema evolution: new fields are added with defaults and ``load()`` ignores
unknown keys, so additive changes do not bump ``SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
MERGED_MANIFEST_FILENAME = "merged_manifest.json"

_T = TypeVar("_T", bound="_JsonManifest")

#: Annotation lifecycle of a capture session (see src/dataset/capture/).
VALID_ANNOTATION_STATUSES = ("unannotated", "staged", "finalized")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _JsonManifest:
    """Shared JSON persistence for all manifest types."""

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain JSON-serializable dict."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the manifest as pretty-printed UTF-8 JSON.

        Args:
            path: Destination file path. Parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls: type[_T], path: Path) -> _T:
        """Load a manifest from JSON, ignoring unknown keys (forward compat).

        Args:
            path: Path to a manifest JSON file.

        Returns:
            A manifest instance of the calling class.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError:        If the file is not valid JSON or not an object.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid manifest JSON in {path}: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest root must be a JSON object: {path}")

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class SourceManifest(_JsonManifest):
    """Provenance record for one acquired data source.

    Attributes:
        source:          Source identifier (e.g. "coco", "openimages",
                         "roboflow", "wider_face", "negatives").
        license:         Human-readable license string. Sources marked
                         non-commercial are gated by ``allow_noncommercial``
                         in configs/dataset_sources.yaml.
        url:             Origin URL (or API/slug description) of the data.
        retrieved_at:    ISO-8601 UTC timestamp of the download.
        query:           Parameters used for acquisition (class filters,
                         per-class caps, limit, mode) for reproducibility.
        image_count:     Number of images stored for this source.
        class_counts:    Class name → annotation instance count.
        trusted_classes: Classes this source labels EXHAUSTIVELY. Classes
                         outside this list may appear unlabeled in images
                         (label-incompleteness policy; see governance doc).
        image_hashes:    Image filename → SHA-256 (resume + lineage).
        notes:           Free-text remarks (e.g. "smoke-scale run").
    """

    source: str = ""
    license: str = ""
    url: str = ""
    retrieved_at: str = field(default_factory=utc_now_iso)
    query: dict[str, Any] = field(default_factory=dict)
    image_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    trusted_classes: list[str] = field(default_factory=list)
    image_hashes: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION


@dataclass
class CaptureSessionManifest(SourceManifest):
    """Provenance record for one custom Indian-home capture session (Phase-3).

    A capture session is the leakage-prevention grouping unit: all images
    from one session must land in the same dataset split.

    Attributes:
        session_id:        Session identifier, ``h{NN}_{room}_s{NNN}``
                           (grammar in configs/capture_config.yaml).
        house_id:          Pseudonymous house identifier (never an address).
        room:              Room type (kitchen, bedroom, bathroom, hall, …).
        capture_device:    Camera/phone model used.
        lighting:          Lighting condition (day, night, low-light, mixed).
        captured_at:       ISO-8601 date of the session.
        consent_reference: ID of the externally stored, signed consent
                           record. PII must never be embedded here.
        annotation_status: One of :data:`VALID_ANNOTATION_STATUSES`.
        annotators:        Pseudonymous annotator handles that produced
                           staged/finalized labels for this session.
        iaa_agreement:     Overall dual-annotator agreement from the last
                           ``09_import_annotations.py --compare`` run;
                           -1.0 until measured.
    """

    session_id: str = ""
    house_id: str = ""
    room: str = ""
    capture_device: str = ""
    lighting: str = ""
    captured_at: str = ""
    consent_reference: str = ""
    annotation_status: str = "unannotated"
    annotators: list[str] = field(default_factory=list)
    iaa_agreement: float = -1.0


@dataclass
class MergedManifest(_JsonManifest):
    """Lineage record for the merged dataset (output of the merge stage).

    Attributes:
        created_at:         ISO-8601 UTC timestamp of the merge.
        sources:            Per-source acceptance stats, each entry:
                            {"source", "total", "accepted", "duplicates",
                             "filtered_out", "manifest_path"}.
        image_provenance:   Merged image filename → source identifier.
        duplicates_removed: Total images dropped by perceptual dedup.
        filtered_out:       Total images dropped by quality/indoor filters.
        class_counts:       Class name → instance count after merge.
        label_completeness: Source identifier → trusted (exhaustively
                            labeled) class names, propagated from the
                            per-source manifests for training-time use.
        notes:              Free-text remarks.
    """

    created_at: str = field(default_factory=utc_now_iso)
    sources: list[dict[str, Any]] = field(default_factory=list)
    image_provenance: dict[str, str] = field(default_factory=dict)
    duplicates_removed: int = 0
    filtered_out: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    label_completeness: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""
    # L3 label salvage (ADR-P5-08, D7) — counts additive; 0 on manifests
    # written before this landed (forward-compat default).
    labels_salvaged: int = 0
    cross_dataset_candidates_linked: int = 0
    schema_version: int = SCHEMA_VERSION
