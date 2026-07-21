"""
src.dataset.annotation.candidates — Candidate-Label Artifact
============================================================

Build/save/load/validate for ``data/annotation/candidates/<backend>/
candidates.json`` — the ONLY place auto-generated detections live until a
human verifies them (ADR-P5-01). One current artifact per backend: the
directory is a DVC stage out (wholly regenerated on rerun; history lives in
the DVC cache) and verification-batch manifests pin the ``candidates_sha256``
they consumed, so a regenerated artifact never silently changes an in-flight
batch.

Determinism contract (ADR-P5-02): ``run_id`` derives from the backend name,
git commit, and prompt/threshold fingerprint — never wall-clock time — so an
unchanged (code, config, data) triple reproduces the same id. The
``generated_at`` timestamp is metadata only and excluded from any equality
comparison (house normalized-comparison convention).

Failure philosophy: :class:`AnnotationError` on any ambiguity, mirroring
src/dataset/completeness.py.

Schema evolution follows the manifest convention: consumers ignore unknown
keys; additive fields do not bump ``CANDIDATES_SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.annotation.base import AnnotationError, Detection, ModelFingerprint
from src.utils.report_utils import git_commit_short, timestamp_str

logger = logging.getLogger(__name__)

CANDIDATES_SCHEMA_VERSION = 1
CANDIDATES_FILENAME = "candidates.json"
GENERATOR_SCRIPT = "scripts/dataset/12_auto_annotate.py"


@dataclass(frozen=True)
class ImageCandidates:
    """In-memory candidates for one image (artifact ``images`` entry).

    Attributes:
        targeted_class_ids: Class ids the backend was asked to look for
                            (untrusted + unverified cells only).
        detections:         Candidate detections, all with
                            ``class_id ∈ targeted_class_ids``.
    """

    targeted_class_ids: tuple[int, ...]
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable artifact entry."""
        return {
            "targeted_class_ids": list(self.targeted_class_ids),
            "detections": [d.to_dict() for d in self.detections],
        }


def compute_run_id(backend: str, prompt_fp: str, git_commit: str) -> str:
    """Deterministic run identifier (no wall clock — resume/idempotency safe).

    Args:
        backend:    Registered backend name.
        prompt_fp:  ``sha256:<hex>`` prompt/threshold fingerprint.
        git_commit: Short git commit hash.
    """
    fp_hex = prompt_fp.removeprefix("sha256:")[:8]
    return f"{backend}_{git_commit}_{fp_hex}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` failing on duplicate JSON object keys."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AnnotationError(
                f"Duplicate JSON key '{key}' in candidates artifact — the file is "
                f"corrupt or was hand-edited; regenerate it via "
                f"`dvc repro auto_annotate`."
            )
        seen[key] = value
    return seen


def build_candidates_artifact(
    backend: str,
    model: ModelFingerprint,
    taxonomy_fp: str,
    inputs: Mapping[str, Any],
    determinism: Mapping[str, Any],
    images: Mapping[str, ImageCandidates],
    runtime_s: float,
    class_names_by_id: Mapping[int, str],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Assemble the artifact dict (validated by the caller before save).

    Args:
        backend:           Registered backend name.
        model:             Fingerprint of the loaded model.
        taxonomy_fp:       ``taxonomy_fingerprint(nc, names)`` of the live
                           configs/data.yaml.
        inputs:            Input lineage (images_root, merged_manifest_sha256,
                           ledger_sha256 or "absent").
        determinism:       Determinism settings actually used (seed,
                           deterministic_algorithms, image_order).
        images:            Filename → :class:`ImageCandidates` (sorted on
                           save for stable output).
        runtime_s:         Wall-clock generation time (stats only).
        class_names_by_id: Taxonomy id → name (per-class stats keys).
        git_commit:        Short hash override (default: from git).
    """
    commit = git_commit if git_commit is not None else git_commit_short()
    per_class: Counter[str] = Counter()
    detections_total = 0
    for entry in images.values():
        for det in entry.detections:
            per_class[class_names_by_id.get(det.class_id, str(det.class_id))] += 1
            detections_total += 1

    return {
        "schema_version": CANDIDATES_SCHEMA_VERSION,
        "generated_at": timestamp_str(),
        "generator": {"script": GENERATOR_SCRIPT, "git_commit": commit},
        "run_id": compute_run_id(backend, model.prompt_fingerprint, commit),
        "backend": backend,
        "model": model.to_dict(),
        "taxonomy_fingerprint": taxonomy_fp,
        "inputs": dict(inputs),
        "determinism": dict(determinism),
        "images": {name: images[name].to_dict() for name in sorted(images)},
        "stats": {
            "images_processed": len(images),
            "detections_total": detections_total,
            "per_class": dict(sorted(per_class.items())),
            "runtime_s": round(runtime_s, 3),
        },
    }


def save_candidates(artifact: Mapping[str, Any], path: Path) -> None:
    """Write the artifact as pretty-printed UTF-8 JSON (manifest convention).

    Args:
        artifact: Artifact dict (validate first — this does not re-validate).
        path:     Destination; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"Candidates artifact written: {path} "
        f"({artifact.get('stats', {}).get('detections_total', 0)} detections)"
    )


def load_candidates(path: Path) -> dict[str, Any]:
    """Load an artifact, rejecting duplicate keys and unsupported schemas.

    Unknown top-level keys are tolerated (forward compatibility).

    Args:
        path: Artifact file path.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        AnnotationError:   On invalid JSON, duplicate keys, or an unsupported
                           schema version.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise AnnotationError(f"Invalid candidates JSON in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise AnnotationError(f"Candidates artifact root must be a JSON object: {path}")
    version = raw.get("schema_version")
    if version != CANDIDATES_SCHEMA_VERSION:
        raise AnnotationError(
            f"Unsupported candidates schema_version {version!r} in {path} "
            f"(supported: {CANDIDATES_SCHEMA_VERSION}). Regenerate via "
            f"`dvc repro auto_annotate`."
        )
    return raw


def validate_candidates(
    artifact: Mapping[str, Any],
    nc: int,
    expected_taxonomy_fp: str | None = None,
) -> list[str]:
    """Exhaustively validate an artifact. Returns problems (empty = valid).

    Checks (house artifact-validation rigor): required keys, class-id ranges,
    detections confined to their image's targeted set, confidence/bbox ranges,
    degenerate boxes, stats consistency, taxonomy-fingerprint drift.

    Args:
        artifact:             Artifact dict (from :func:`load_candidates`).
        nc:                   Taxonomy class count.
        expected_taxonomy_fp: Live taxonomy fingerprint to compare against
                              (None skips the drift check).
    """
    problems: list[str] = []

    for key in ("run_id", "backend", "model", "taxonomy_fingerprint", "images", "stats"):
        if key not in artifact:
            problems.append(f"missing required key '{key}'")
    if problems:
        return problems  # structural failures make field checks meaningless

    if expected_taxonomy_fp is not None and artifact["taxonomy_fingerprint"] != (
        expected_taxonomy_fp
    ):
        problems.append(
            f"taxonomy fingerprint drift: artifact {artifact['taxonomy_fingerprint']} "
            f"vs live {expected_taxonomy_fp} — regenerate after taxonomy changes"
        )

    images = artifact["images"]
    if not isinstance(images, Mapping):
        return problems + ["'images' must be an object"]

    detections_total = 0
    for name, entry in images.items():
        if not isinstance(entry, Mapping):
            problems.append(f"image '{name}': entry must be an object")
            continue
        targeted = entry.get("targeted_class_ids", [])
        if not isinstance(targeted, list) or not all(isinstance(c, int) for c in targeted):
            problems.append(f"image '{name}': targeted_class_ids must be a list of ints")
            continue
        bad_targets = [c for c in targeted if not 0 <= c < nc]
        if bad_targets:
            problems.append(
                f"image '{name}': targeted class ids out of range [0, {nc}): " f"{bad_targets}"
            )
        targeted_set = set(targeted)
        for i, det in enumerate(entry.get("detections", [])):
            where = f"image '{name}' detection[{i}]"
            if not isinstance(det, Mapping):
                problems.append(f"{where}: must be an object")
                continue
            class_id = det.get("class_id")
            if not isinstance(class_id, int) or not 0 <= class_id < nc:
                problems.append(f"{where}: class_id {class_id!r} out of range [0, {nc})")
            elif class_id not in targeted_set:
                problems.append(
                    f"{where}: class_id {class_id} not in the image's targeted set "
                    f"{sorted(targeted_set)} — generator bug (untargeted candidates "
                    f"waste verification time)"
                )
            conf = det.get("conf")
            if not isinstance(conf, int | float) or not 0.0 <= float(conf) <= 1.0:
                problems.append(f"{where}: conf {conf!r} outside [0, 1]")
            bbox = det.get("bbox_xywhn")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, int | float) for v in bbox)
            ):
                problems.append(f"{where}: bbox_xywhn must be 4 numbers, got {bbox!r}")
            else:
                x, y, w, h = (float(v) for v in bbox)
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    problems.append(f"{where}: bbox_xywhn values outside [0, 1]: {bbox}")
                if w <= 0.0 or h <= 0.0:
                    problems.append(f"{where}: degenerate box (w={w}, h={h})")
            detections_total += 1

    stats = artifact["stats"]
    if isinstance(stats, Mapping):
        if stats.get("images_processed") != len(images):
            problems.append(
                f"stats.images_processed {stats.get('images_processed')!r} != "
                f"actual image count {len(images)}"
            )
        if stats.get("detections_total") != detections_total:
            problems.append(
                f"stats.detections_total {stats.get('detections_total')!r} != "
                f"actual detection count {detections_total}"
            )
    else:
        problems.append("'stats' must be an object")

    return problems
