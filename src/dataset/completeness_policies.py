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

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.dataset.manifest import CaptureSessionManifest

logger = logging.getLogger(__name__)


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
    """

    source: str
    manifest_trusted_classes: tuple[str, ...] | None
    config_trusted_classes: tuple[str, ...] | None
    class_ids_by_name: Mapping[str, int]
    nc: int
    capture_manifests_dir: Path | None


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
