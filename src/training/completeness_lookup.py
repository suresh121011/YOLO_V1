"""
src.training.completeness_lookup — Runtime Completeness Reader
==============================================================

Pure-Python (no torch) reader of ``data/processed/completeness.json`` for
train-time mask lookups. The masked loss converts the integer rows returned
here into tensors; keeping this module torch-free lets the preflight gates
and unit tests run in any environment.

Lookups are keyed by image basename (Ultralytics exposes full paths in
``batch["im_file"]``; basenames are unique across splits — enforced by the
generator and validated at load).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.completeness import load_completeness, validate_completeness

logger = logging.getLogger(__name__)


class UnknownImageError(KeyError):
    """Raised when an image has no completeness record (strict mode)."""


@dataclass(frozen=True)
class CompletenessLookup:
    """Immutable in-memory view of a completeness artifact.

    Attributes:
        nc:          Number of taxonomy classes.
        fingerprint: Taxonomy fingerprint recorded in the artifact.
        source_path: Where the artifact was loaded from (diagnostics).
    """

    nc: int
    fingerprint: str
    source_path: Path
    _policy_by_image: dict[str, str] = field(repr=False)
    _mask_row_by_policy: dict[str, tuple[int, ...]] = field(repr=False)

    @classmethod
    def load(cls, path: Path, expected_fingerprint: str | None = None) -> CompletenessLookup:
        """Load and index a completeness artifact.

        Args:
            path:                 Path to completeness.json.
            expected_fingerprint: When given, the artifact's taxonomy
                                  fingerprint must match exactly.

        Returns:
            An immutable lookup with per-policy {0,1} mask rows precomputed.

        Raises:
            FileNotFoundError:  If the artifact does not exist.
            CompletenessError:  If the artifact is invalid (via the shared
                                validator) — same failure the preflight gives.
            ValueError:         On fingerprint mismatch.
        """
        artifact = load_completeness(path)
        errors = validate_completeness(artifact)
        if errors:
            raise ValueError(
                f"Completeness artifact {path} failed validation "
                f"({len(errors)} error(s)); first: {errors[0]} — "
                f"re-run `dvc repro generate_completeness`."
            )

        taxonomy = artifact["taxonomy"]
        nc = int(taxonomy["nc"])
        fingerprint = str(taxonomy["fingerprint"])
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                f"Completeness artifact {path} was generated for taxonomy "
                f"{fingerprint} but training expects {expected_fingerprint} — "
                f"re-run `dvc repro generate_completeness`."
            )

        mask_rows: dict[str, tuple[int, ...]] = {}
        for key, policy in artifact["policies"].items():
            row = [0] * nc
            for cid in policy["trusted_class_ids"]:
                row[cid] = 1
            mask_rows[key] = tuple(row)

        policy_by_image = {name: str(entry["policy"]) for name, entry in artifact["images"].items()}
        logger.info(
            f"Completeness lookup loaded: {len(policy_by_image)} images, "
            f"{len(mask_rows)} policies, nc={nc} ({path})"
        )
        return cls(
            nc=nc,
            fingerprint=fingerprint,
            source_path=path,
            _policy_by_image=policy_by_image,
            _mask_row_by_policy=mask_rows,
        )

    def __len__(self) -> int:
        """Number of images with completeness records."""
        return len(self._policy_by_image)

    def policy_for(self, filename: str) -> str:
        """Return the policy key governing an image.

        Args:
            filename: Image basename (or any path — the basename is used).

        Raises:
            UnknownImageError: If the image has no completeness record.
        """
        name = Path(filename).name
        try:
            return self._policy_by_image[name]
        except KeyError:
            raise UnknownImageError(
                f"Image '{name}' has no completeness record in {self.source_path} — "
                f"the artifact is stale relative to data/processed; "
                f"re-run `dvc repro generate_completeness`."
            ) from None

    def mask_row(self, filename: str) -> tuple[int, ...]:
        """Return the {0,1} trusted-class row (length nc) for an image.

        Args:
            filename: Image basename (or any path — the basename is used).

        Raises:
            UnknownImageError: If the image has no completeness record.
        """
        return self._mask_row_by_policy[self.policy_for(filename)]

    def coverage(self, filenames: Iterable[str]) -> tuple[list[str], list[str]]:
        """Compare a set of images against the artifact's records.

        Args:
            filenames: Image basenames/paths to check (e.g. everything under
                       data/processed/images/{train,val}).

        Returns:
            (missing, stale): ``missing`` = inputs without a record (breaks
            training); ``stale`` = recorded images not in the inputs
            (harmless, indicates a stale artifact).
        """
        wanted = {Path(f).name for f in filenames}
        recorded = set(self._policy_by_image)
        missing = sorted(wanted - recorded)
        stale = sorted(recorded - wanted)
        return missing, stale
