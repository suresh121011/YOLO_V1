"""
src.dataset.completeness_policies — Pluggable Label-Completeness Policies
=========================================================================

Phase-4: providers that resolve, per data source, which taxonomy classes are
exhaustively annotated ("trusted") — the raw material for per-image class
masks used by the missing-annotation mitigation loss.

Why a registry: ``trusted_classes: []`` in configs/dataset_sources.yaml is
ambiguous — for ``negatives`` it means "verified absence of ALL classes"
(every class is trusted-absent → all-ones mask) while for ``custom_captures``
it means "declared per capture session". The completeness generator therefore
never infers semantics from a source's name or its bare ``trusted_classes``
value; every source must declare an explicit policy mode under the top-level
``completeness:`` section, resolved through this registry (mirroring the
split-strategy registry in src/dataset/splitting).

Adding a policy for a future dataset type: subclass
:class:`CompletenessPolicyProvider`, decorate with
``@register_policy_provider("my_mode")``, and reference ``my_mode`` in
``completeness.policies`` — the core generator needs no changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from src.dataset.manifest import CaptureSessionManifest

logger = logging.getLogger(__name__)


@runtime_checkable
class LedgerLike(Protocol):
    """Structural read surface :class:`TrustedListWithLedgerPolicy` needs.

    Defined here (not imported from ``src.dataset.annotation``) to keep this
    core policy layer free of any dependency on the annotation subpackage
    that is layered on top of it (ADR-P5-04 layering note) —
    ``src.dataset.annotation.ledger.LedgerView`` satisfies this Protocol
    structurally, with zero import needed on either side.
    """

    def all_images(self) -> frozenset[str]:
        """Every filename with at least one verified cell."""

    def verified_class_names(self, filename: str) -> frozenset[str]:
        """Classes verified (either verdict) for one image."""

    def entry_source(self, filename: str) -> str | None:
        """The provenance source a ledger entry attributes an image to."""

    def taxonomy_fingerprint(self) -> str:
        """Fingerprint recorded at last import (``""`` if never imported)."""


class CompletenessError(ValueError):
    """Raised when completeness metadata cannot be resolved unambiguously.

    Every ambiguity is a hard error by design: a wrong mask silently corrupts
    training supervision, so the generator refuses to guess.
    """


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy provider may need to resolve one source.

    Attributes:
        source:                  Provenance source identifier (e.g. "coco").
        manifest_trusted_classes: Trusted class names recorded for this source
                                 in the merged manifest's ``label_completeness``
                                 (data-of-record), or None if absent.
        config_trusted_classes:  ``trusted_classes`` declared for this source in
                                 configs/dataset_sources.yaml (cross-check), or
                                 None if the source has no such key.
        class_ids_by_name:       Taxonomy class name → integer id.
        nc:                      Number of taxonomy classes.
        capture_manifests_dir:   Directory of per-session capture manifests
                                 (data/raw/custom_captures/manifests), or None.
        verification_ledger:    M2's human-verification ledger (read view), or
                                 None. Additive trailing field (both existing
                                 construction sites use keyword args, so this
                                 is back-compatible) — only
                                 :class:`TrustedListWithLedgerPolicy` reads it.
        source_raw_root:         This source's raw acquisition directory
                                 (``sources.<source>.output_dir``), e.g.
                                 ``data/raw/local_captures``, or None. Read by
                                 :class:`PerSlugWithLedgerPolicy` to load the
                                 per-slug ingest manifests. Additive trailing
                                 field, same back-compatibility argument as
                                 ``verification_ledger``.
    """

    source: str
    manifest_trusted_classes: tuple[str, ...] | None
    config_trusted_classes: tuple[str, ...] | None
    class_ids_by_name: Mapping[str, int]
    nc: int
    capture_manifests_dir: Path | None
    verification_ledger: LedgerLike | None = None
    source_raw_root: Path | None = None


class CompletenessPolicyProvider(ABC):
    """Resolves trusted-class policies for one data source.

    A provider returns one or more named policies (policy key → sorted tuple
    of trusted taxonomy class ids) and maps each merged image filename to the
    policy key that governs it. Simple sources yield exactly one policy keyed
    by the source name; per-session sources yield one per capture session.
    """

    #: Config value under ``completeness.policies.<source>`` selecting this
    #: provider. Set by :func:`register_policy_provider`.
    mode: ClassVar[str] = ""

    @abstractmethod
    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        """Return policy key → sorted trusted class ids for this source.

        Args:
            ctx: Resolution context for the source.

        Raises:
            CompletenessError: On any ambiguity or config/manifest mismatch.
        """

    def policy_key_for_image(self, ctx: PolicyContext, merged_filename: str) -> str:
        """Return the policy key governing one merged image filename.

        Default: the single source-level policy. Providers that emit multiple
        policies (e.g. per-session) must override this.

        Args:
            ctx:             Resolution context for the source.
            merged_filename: Basename of the image in data/processed
                             (source-prefixed by the merge stage).

        Raises:
            CompletenessError: If the image cannot be attributed to a policy.
        """
        return ctx.source

    def _class_ids(
        self, ctx: PolicyContext, names: tuple[str, ...], origin: str
    ) -> tuple[int, ...]:
        """Map class names to sorted unique taxonomy ids, failing loudly.

        Args:
            ctx:    Resolution context (for taxonomy and error messages).
            names:  Class names to map.
            origin: Human-readable description of where the names came from.

        Raises:
            CompletenessError: If any name is not in the taxonomy.
        """
        unknown = sorted(set(names) - set(ctx.class_ids_by_name))
        if unknown:
            raise CompletenessError(
                f"Source '{ctx.source}': {origin} references class names not in the "
                f"taxonomy (configs/data.yaml): {unknown}. "
                f"Valid names: {sorted(ctx.class_ids_by_name)}"
            )
        return tuple(sorted({ctx.class_ids_by_name[n] for n in names}))


# ─── Registry ─────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[CompletenessPolicyProvider]] = {}


def register_policy_provider(
    mode: str,
) -> Callable[[type[CompletenessPolicyProvider]], type[CompletenessPolicyProvider]]:
    """Class decorator registering a provider under a policy mode name.

    Args:
        mode: The ``completeness.policies`` config value for this provider.

    Raises:
        ValueError: If the mode name is already registered.
    """

    def _register(cls: type[CompletenessPolicyProvider]) -> type[CompletenessPolicyProvider]:
        if mode in _PROVIDERS:
            raise ValueError(
                f"Completeness policy mode '{mode}' already registered by "
                f"{_PROVIDERS[mode].__name__}"
            )
        cls.mode = mode
        _PROVIDERS[mode] = cls
        return cls

    return _register


def registered_policy_modes() -> list[str]:
    """Return the sorted list of registered policy mode names."""
    return sorted(_PROVIDERS)


def get_policy_provider(mode: str) -> CompletenessPolicyProvider:
    """Instantiate the provider registered for a policy mode.

    Args:
        mode: Policy mode name from ``completeness.policies.<source>``.

    Raises:
        CompletenessError: If the mode is unknown, listing registered modes.
    """
    if mode not in _PROVIDERS:
        raise CompletenessError(
            f"Unknown completeness policy mode '{mode}'. "
            f"Registered modes: {registered_policy_modes()}"
        )
    return _PROVIDERS[mode]()


# ─── Built-in providers ───────────────────────────────────────────────────────


@register_policy_provider("trusted_list")
class TrustedListPolicy(CompletenessPolicyProvider):
    """Source labels exactly its declared ``trusted_classes`` exhaustively.

    The merged manifest's ``label_completeness`` entry is the data-of-record
    (it reflects what was actually built); the config's ``trusted_classes``
    is cross-checked against it so silent drift between a rebuilt dataset and
    the config fails loudly.
    """

    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        if not ctx.manifest_trusted_classes:
            raise CompletenessError(
                f"Source '{ctx.source}' uses policy 'trusted_list' but the merged "
                f"manifest records no trusted classes for it (label_completeness). "
                f"An empty trusted list would silently mask every class — if the "
                f"source is a verified negative set, declare "
                f"'verified_absence_all' instead; otherwise rebuild the merge stage."
            )
        if ctx.config_trusted_classes is not None and set(ctx.config_trusted_classes) != set(
            ctx.manifest_trusted_classes
        ):
            raise CompletenessError(
                f"Source '{ctx.source}': trusted_classes drift between "
                f"configs/dataset_sources.yaml {sorted(ctx.config_trusted_classes)} and "
                f"the merged manifest {sorted(ctx.manifest_trusted_classes)}. "
                f"Re-run the merge stage (dvc repro merge_datasets) or reconcile the config."
            )
        ids = self._class_ids(
            ctx, ctx.manifest_trusted_classes, "merged-manifest label_completeness"
        )
        return {ctx.source: ids}


@register_policy_provider("trusted_list_with_ledger")
class TrustedListWithLedgerPolicy(TrustedListPolicy):
    """``trusted_list``, expanded by human-verified ledger cells (ADR-P5-04).

    Composes the base ``trusted_list`` policy for this source with one
    additional policy per distinct EFFECTIVE trusted-class set among
    ledger-verified images attributed to this source — masking shrinks
    exactly as the ledger grows (D3, plan §"Verified completeness
    expansion"). Ledger images sharing the same effective set (base trusted
    ∪ their own verified classes) share one policy key,
    ``"{source}/ledger/{8-hex-hash-of-sorted-ids}"``; images with no ledger
    entry stay on the plain base-source policy.

    An empty ledger (the M1 git-bootstrapped state) or ``ctx.verification_ledger
    is None`` resolves to exactly the base policy — byte-identical to plain
    ``trusted_list`` behavior (the empty-ledger passthrough regression this
    milestone's acceptance test pins).
    """

    def __init__(self) -> None:
        self._image_to_key: dict[str, str] = {}

    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        base_policies = super().resolve_policies(ctx)
        base_ids = frozenset(base_policies[ctx.source])
        self._image_to_key = {}

        ledger = ctx.verification_ledger
        if ledger is None:
            return base_policies

        policies = dict(base_policies)
        for filename in sorted(ledger.all_images()):
            entry_source = ledger.entry_source(filename)
            if entry_source != ctx.source:
                continue  # belongs to a different source; its own resolve_policies call handles it
            verified_names = ledger.verified_class_names(filename)
            if not verified_names:
                continue
            verified_ids = self._class_ids(
                ctx, tuple(verified_names), f"ledger verification for '{filename}'"
            )
            verified_ids_set = set(verified_ids)
            redundant = verified_ids_set & base_ids
            if redundant:
                redundant_names = sorted(
                    n for n, i in ctx.class_ids_by_name.items() if i in redundant
                )
                logger.warning(
                    f"Source '{ctx.source}': ledger verifies class(es) {redundant_names} "
                    f"for '{filename}' already in the base trusted list — no-op."
                )
            effective_ids = base_ids | verified_ids_set
            if effective_ids == base_ids:
                continue  # nothing beyond the base trust was verified — stays on the base policy
            effective = tuple(sorted(effective_ids))
            digest = hashlib.sha256(
                json.dumps(effective, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:8]
            key = f"{ctx.source}/ledger/{digest}"
            policies[key] = effective
            self._image_to_key[filename] = key

        return policies

    def policy_key_for_image(self, ctx: PolicyContext, merged_filename: str) -> str:
        return self._image_to_key.get(merged_filename, ctx.source)


@dataclass(frozen=True)
class SlugIndex:
    """Per-slug trusted classes and the merged-filename → slug map.

    Built from the ingest manifests a multi-slug source writes, one
    sub-directory per slug (``<root>/<slug>/manifest.json``). Each records
    ``trusted_classes`` for that slug and ``image_hashes`` keyed
    ``<slug>__<original>`` — and the merge stage names the image
    ``<source>_`` + that key, so the mapping is *recorded provenance*, not a
    filename convention being reverse-engineered (ADR-P5-15).

    Shared with :mod:`src.dataset.annotation.targeting`, which must resolve
    trust identically or auto-annotation targets a different set of cells than
    training masks — the defect ADR-P5-15 records.
    """

    trusted_by_slug: Mapping[str, tuple[str, ...]]
    slug_by_image_key: Mapping[str, str]

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(sorted(self.trusted_by_slug))


def load_slug_index(root: Path, source: str) -> SlugIndex:
    """Read every ``<root>/<slug>/manifest.json`` into a :class:`SlugIndex`.

    Args:
        root:   The source's raw acquisition directory.
        source: Source name, for error messages only.

    Raises:
        CompletenessError: If the directory is missing, a manifest is
            unreadable, a slug declares no trusted classes, or two slugs claim
            the same image key. Every one of these is a hard error: silently
            falling back to a source-level trusted set is exactly the defect
            being fixed, and it would look identical to success.
    """
    if not root.is_dir():
        raise CompletenessError(
            f"Source '{source}' uses a per-slug policy but its raw directory "
            f"{root} does not exist. Per-slug trust is read from "
            f"<slug>/manifest.json; run the ingest stage (or `dvc pull`) first. "
            f"Refusing to fall back to a source-level trusted set."
        )

    trusted: dict[str, tuple[str, ...]] = {}
    slug_by_key: dict[str, str] = {}
    for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = slug_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompletenessError(f"Cannot read slug manifest {manifest_path}: {exc}") from exc

        slug = slug_dir.name
        classes = tuple(data.get("trusted_classes") or ())
        if not classes:
            raise CompletenessError(
                f"Slug manifest {manifest_path} declares no trusted_classes. A slug "
                f"whose annotated classes are unknown cannot be masked correctly — "
                f"record them at ingest, or exclude the slug from the merge."
            )
        trusted[slug] = classes

        for image_key in data.get("image_hashes") or {}:
            previous = slug_by_key.get(image_key)
            if previous is not None and previous != slug:
                raise CompletenessError(
                    f"Image key '{image_key}' is claimed by slugs '{previous}' and "
                    f"'{slug}' — slug attribution would be ambiguous."
                )
            slug_by_key[image_key] = slug

    if not trusted:
        raise CompletenessError(
            f"Source '{source}': no slug manifests found under {root}. "
            f"Expected <slug>/manifest.json per ingested archive."
        )
    return SlugIndex(trusted_by_slug=trusted, slug_by_image_key=slug_by_key)


def slug_for_merged_filename(index: SlugIndex, source: str, merged_filename: str) -> str:
    """Resolve one merged filename to its slug, or raise.

    The merge stage prefixes ``<source>_``; the remainder is the key recorded
    in the slug's ``image_hashes``. Measured on the ``dataset-v0.6.0`` build:
    16,926/16,926 images resolve, 0 cross-slug collisions over 26,339 keys.

    Raises:
        CompletenessError: If the prefix is absent or the key is unknown.
    """
    prefix = f"{source}_"
    if not merged_filename.startswith(prefix):
        raise CompletenessError(
            f"Image '{merged_filename}' is attributed to source '{source}' but does "
            f"not carry its merge prefix '{prefix}' — provenance and filename "
            f"disagree; re-run the merge stage."
        )
    key = merged_filename[len(prefix) :]
    slug = index.slug_by_image_key.get(key)
    if slug is None:
        raise CompletenessError(
            f"Image '{merged_filename}' (key '{key}') matches no slug manifest under "
            f"source '{source}'. Its trusted classes are therefore unknown, and "
            f"guessing would re-introduce the over-claim ADR-P5-15 fixes. Re-run the "
            f"ingest stage so every merged image is covered by a slug manifest."
        )
    return slug


@register_policy_provider("per_slug_with_ledger")
class PerSlugWithLedgerPolicy(CompletenessPolicyProvider):
    """One policy per *slug*, expanded by the verification ledger.

    ``local_captures`` ingests one archive per sub-directory, and each archive
    annotates its own class set exhaustively — `bed` labels bed/chair/person/sink,
    `charger` labels only charger. The merge stage records a single
    ``label_completeness`` entry per *source*, so a source-level policy takes the
    **union** of all slugs and declares it trusted for every image.

    That is not a rounding error. On the ``dataset-v0.6.0`` build it marked ~21.7
    classes of 23 falsely trusted across 16,926 images — 69.5% of the release —
    turning every unlabelled person in a `bed` frame into supervised background
    (ADR-P5-14 finding, ADR-P5-15 remedy).

    Emits ``<source>/<slug>``, plus ``<source>/<slug>/ledger/<digest>`` for
    ledger-verified images whose effective trust exceeds their slug's — the same
    composition :class:`TrustedListWithLedgerPolicy` performs, but over a
    per-slug base rather than one source-wide base.
    """

    def __init__(self) -> None:
        self._index: SlugIndex | None = None
        self._image_to_key: dict[str, str] = {}

    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        if ctx.source_raw_root is None:
            raise CompletenessError(
                f"Source '{ctx.source}' uses policy 'per_slug_with_ledger' but no raw "
                f"root was supplied (sources.{ctx.source}.output_dir). The per-slug "
                f"trusted sets live there and cannot be inferred."
            )
        index = load_slug_index(ctx.source_raw_root, ctx.source)
        self._index = index
        self._image_to_key = {}

        policies: dict[str, tuple[int, ...]] = {}
        base_by_slug: dict[str, frozenset[int]] = {}
        for slug in index.slugs:
            ids = self._class_ids(
                ctx, index.trusted_by_slug[slug], f"slug manifest '{slug}/manifest.json'"
            )
            policies[f"{ctx.source}/{slug}"] = ids
            base_by_slug[slug] = frozenset(ids)

        # Cross-check the union against the merged manifest. A mismatch means the
        # merge and the ingest manifests disagree about this source — the lock is
        # then describing a build these manifests did not produce.
        if ctx.manifest_trusted_classes is not None:
            union = {n for names in index.trusted_by_slug.values() for n in names}
            recorded = set(ctx.manifest_trusted_classes)
            if union != recorded:
                raise CompletenessError(
                    f"Source '{ctx.source}': the union of per-slug trusted classes "
                    f"{sorted(union)} does not match the merged manifest's "
                    f"label_completeness {sorted(recorded)}. Re-run the merge stage "
                    f"(dvc repro merge_datasets) or reconcile the ingest manifests."
                )

        ledger = ctx.verification_ledger
        if ledger is None:
            return policies

        for filename in sorted(ledger.all_images()):
            if ledger.entry_source(filename) != ctx.source:
                continue  # another source's resolve_policies call owns it
            verified_names = ledger.verified_class_names(filename)
            if not verified_names:
                continue
            slug = slug_for_merged_filename(index, ctx.source, filename)
            base_ids = base_by_slug[slug]
            verified_ids = set(
                self._class_ids(ctx, tuple(verified_names), f"ledger verification for '{filename}'")
            )
            effective_ids = base_ids | verified_ids
            if effective_ids == base_ids:
                continue  # nothing beyond this slug's own trust was verified
            effective = tuple(sorted(effective_ids))
            digest = hashlib.sha256(
                json.dumps(effective, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:8]
            key = f"{ctx.source}/{slug}/ledger/{digest}"
            policies[key] = effective
            self._image_to_key[filename] = key

        return policies

    def policy_key_for_image(self, ctx: PolicyContext, merged_filename: str) -> str:
        ledger_key = self._image_to_key.get(merged_filename)
        if ledger_key is not None:
            return ledger_key
        if self._index is None:  # pragma: no cover — resolve_policies always runs first
            raise CompletenessError(
                f"Source '{ctx.source}': policy_key_for_image called before "
                f"resolve_policies; the slug index is not loaded."
            )
        return f"{ctx.source}/{slug_for_merged_filename(self._index, ctx.source, merged_filename)}"


@register_policy_provider("verified_absence_all")
class VerifiedAbsenceAllPolicy(CompletenessPolicyProvider):
    """Images were verified to contain NO taxonomy class at all.

    Background/negative sources: absence of every class is a verified fact,
    so every class is trusted (all-ones mask) — the empty label files are
    genuine supervision, not missing annotations.
    """

    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        declared = ctx.manifest_trusted_classes or ctx.config_trusted_classes
        if declared:
            raise CompletenessError(
                f"Source '{ctx.source}' uses policy 'verified_absence_all' but declares "
                f"trusted_classes {sorted(declared)}. A verified-negative source must "
                f"declare an empty trusted_classes list — use 'trusted_list' if this "
                f"source labels specific classes."
            )
        return {ctx.source: tuple(range(ctx.nc))}


@register_policy_provider("per_session")
class PerSessionPolicy(CompletenessPolicyProvider):
    """Trusted classes are declared per capture session (Phase-3 manifests).

    Each finalized session manifest under ``capture_manifests_dir`` yields one
    policy keyed ``<source>/<session_id>``. Merged capture filenames are
    ``<source>_<session_id>_<seq><ext>`` (ingest names files
    ``{session_id}_{seq}{ext}``; merge prefixes the source), so images are
    attributed to sessions by longest-session-id prefix match.
    """

    def __init__(self) -> None:
        self._session_ids: list[str] = []

    def resolve_policies(self, ctx: PolicyContext) -> dict[str, tuple[int, ...]]:
        manifests_dir = ctx.capture_manifests_dir
        if manifests_dir is None or not manifests_dir.exists():
            logger.info(
                f"Source '{ctx.source}': no capture manifests directory "
                f"({manifests_dir}) — resolving zero per-session policies."
            )
            return {}

        policies: dict[str, tuple[int, ...]] = {}
        for manifest_path in sorted(manifests_dir.glob("*.json")):
            session = CaptureSessionManifest.load(manifest_path)
            if not session.session_id:
                raise CompletenessError(
                    f"Capture manifest {manifest_path} has no session_id — "
                    f"cannot attribute images to a completeness policy."
                )
            if session.annotation_status != "finalized":
                raise CompletenessError(
                    f"Capture session '{session.session_id}' ({manifest_path}) has "
                    f"annotation_status='{session.annotation_status}' — only finalized "
                    f"sessions may feed training. Finalize it via "
                    f"scripts/dataset/09_import_annotations.py --finalize or remove it "
                    f"from the merge."
                )
            if not session.trusted_classes:
                raise CompletenessError(
                    f"Capture session '{session.session_id}' ({manifest_path}) declares "
                    f"no trusted_classes — a finalized session must state which classes "
                    f"were annotated exhaustively."
                )
            key = f"{ctx.source}/{session.session_id}"
            if key in policies:
                raise CompletenessError(
                    f"Duplicate capture session id '{session.session_id}' in "
                    f"{manifests_dir} — session ids must be unique."
                )
            policies[key] = self._class_ids(
                ctx,
                tuple(session.trusted_classes),
                f"capture session manifest {manifest_path.name}",
            )

        # Longest-first so overlapping ids (h01_kitchen_s001 vs h01_kitchen_s001b)
        # resolve to the most specific session.
        self._session_ids = sorted((k.split("/", 1)[1] for k in policies), key=len, reverse=True)
        return policies

    def policy_key_for_image(self, ctx: PolicyContext, merged_filename: str) -> str:
        prefix = f"{ctx.source}_"
        if not merged_filename.startswith(prefix):
            raise CompletenessError(
                f"Image '{merged_filename}' is attributed to source '{ctx.source}' but "
                f"does not carry its merge prefix '{prefix}' — provenance and filename "
                f"disagree; re-run the merge stage."
            )
        remainder = merged_filename[len(prefix) :]
        for session_id in self._session_ids:
            if remainder.startswith(f"{session_id}_"):
                return f"{ctx.source}/{session_id}"
        raise CompletenessError(
            f"Image '{merged_filename}' (source '{ctx.source}') matches no finalized "
            f"capture session. Known sessions: {sorted(self._session_ids)}. "
            f"Ingest/finalize the session manifest before generating completeness."
        )
