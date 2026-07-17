"""
src.dataset.annotation.targeting — Untrusted-Cell Targeting
===========================================================

Decides, per merged image, which taxonomy classes an auto-annotation backend
should look for: the classes that are promptable (non-empty prompts in the
backend config), NOT already trusted by the image's source completeness
policy, and NOT already human-verified in the ledger. Cells outside this set
are either already supervised (trusted/verified) or outside L2 scope — asking
a model about them wastes GPU time and, worse, human verification time (R30).

Policy-mode handling (mirrors src/dataset/completeness_policies.py):
    trusted_list / trusted_list_with_ledger — target promptable minus trusted
    verified_absence_all — skip entirely (absence of every class is verified;
                           nothing is missing by definition)
    per_session          — skip in v1: capture sessions are freshly
                           human-annotated under Phase-3 IAA gates; L2 on
                           custom images is revisited after release v0.9

Ledger access: until the M2 ledger module lands, callers pass verified cells
as a plain mapping (the CLI reads the bootstrap ledger JSON directly); M2's
LedgerView becomes the single reader and this signature stays unchanged.

Failure philosophy: unknown provenance source, missing policy entry, or an
untranslatable trusted-class name raises :class:`AnnotationError` — guessing
would target the wrong cells and poison the candidate artifact.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from src.dataset.annotation.base import AnnotationError, BackendConfig
from src.dataset.manifest import MergedManifest

logger = logging.getLogger(__name__)

#: Policy modes whose images are targetable by L2 auto-annotation.
TARGETABLE_MODES = ("trusted_list", "trusted_list_with_ledger")

#: Policy modes whose images are skipped, with the reason recorded.
SKIP_MODES: dict[str, str] = {
    "verified_absence_all": "verified-negative source — absence of every class is trusted",
    "per_session": "capture sessions are human-annotated under Phase-3 IAA gates",
}


def promptable_class_ids(config: BackendConfig, ids_by_name: Mapping[str, int]) -> tuple[int, ...]:
    """Class ids the backend can actually be asked about (non-empty prompts).

    Args:
        config:      Backend configuration (prompts already validated
                     against the taxonomy by ``BackendConfig.validate``).
        ids_by_name: Taxonomy class name → id.

    Raises:
        AnnotationError: If a prompted class name is not in the taxonomy
                         (defense in depth — validate() should have caught it).
    """
    unknown = sorted(name for name, p in config.prompts.items() if p and name not in ids_by_name)
    if unknown:
        raise AnnotationError(
            f"Backend '{config.name}': prompted classes not in the taxonomy: {unknown}. "
            f"Fix configs/annotation.yaml prompts or configs/data.yaml."
        )
    return tuple(sorted(ids_by_name[name] for name, p in config.prompts.items() if p))


def build_targets(
    merged_manifest: MergedManifest,
    policies: Mapping[str, str],
    promptable: tuple[int, ...],
    ids_by_name: Mapping[str, int],
    verified_cells: Mapping[str, frozenset[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Compute filename → sorted targeted class ids (empty sets omitted).

    Args:
        merged_manifest: The merge stage's manifest (``image_provenance`` +
                         ``label_completeness`` are the data-of-record).
        policies:        Source → completeness policy mode, from the
                         top-level ``completeness.policies`` section of
                         configs/dataset_sources.yaml.
        promptable:      Output of :func:`promptable_class_ids`.
        ids_by_name:     Taxonomy class name → id.
        verified_cells:  Filename → class ids already human-verified in the
                         ledger (those cells are supervised; never re-target).

    Raises:
        AnnotationError: On an image without provenance policy coverage or an
                         untranslatable trusted-class name.
    """
    verified = verified_cells or {}
    promptable_set = frozenset(promptable)

    trusted_ids_by_source: dict[str, frozenset[int]] = {}
    skipped_sources: dict[str, str] = {}
    for source, trusted_names in merged_manifest.label_completeness.items():
        mode = policies.get(source)
        if mode is None:
            raise AnnotationError(
                f"Source '{source}' appears in the merged manifest but has no entry "
                f"under completeness.policies in configs/dataset_sources.yaml — "
                f"declare its policy mode before auto-annotating."
            )
        if mode in SKIP_MODES:
            skipped_sources[source] = SKIP_MODES[mode]
            continue
        if mode not in TARGETABLE_MODES:
            raise AnnotationError(
                f"Source '{source}' declares unknown completeness policy mode "
                f"'{mode}' — known: {sorted((*TARGETABLE_MODES, *SKIP_MODES))}."
            )
        unknown = sorted(set(trusted_names) - set(ids_by_name))
        if unknown:
            raise AnnotationError(
                f"Source '{source}': merged-manifest trusted classes not in the "
                f"taxonomy: {unknown} — re-run the merge stage or fix configs/data.yaml."
            )
        trusted_ids_by_source[source] = frozenset(ids_by_name[n] for n in trusted_names)

    for source, reason in skipped_sources.items():
        logger.info(f"Targeting: skipping source '{source}' ({reason})")

    targets: dict[str, tuple[int, ...]] = {}
    for filename, source in merged_manifest.image_provenance.items():
        if source in skipped_sources:
            continue
        if source not in trusted_ids_by_source:
            raise AnnotationError(
                f"Image '{filename}' is attributed to source '{source}' which has no "
                f"resolved trusted-class set — provenance and label_completeness "
                f"disagree; re-run the merge stage."
            )
        remaining = (
            promptable_set - trusted_ids_by_source[source] - verified.get(filename, frozenset())
        )
        if remaining:
            targets[filename] = tuple(sorted(remaining))
    return targets
