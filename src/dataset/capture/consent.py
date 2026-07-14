"""
src.dataset.capture.consent — Consent Record Verification
==========================================================

PII-free consent handling for custom capture sessions (Phase-3).

Privacy model (docs/04_dataset_engineering/ §4, DPDP):
    - Signed consent forms live OFFLINE with the collection lead. They are
      the only artifacts that carry personal information.
    - The local consent registry (``data/consent/consent_registry.yaml``,
      gitignored, never a DVC out) maps a consent ID to a pseudonymous
      house ID plus grant/withdrawal state — no names, no addresses.
    - The repo and the DVC remote only ever see ``consent_reference``
      strings inside capture-session manifests.

On machines without the registry (CI, other developers) verification
degrades to a format-only check with a warning — ingest is a
collection-machine activity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.dataset.capture.config import ConsentSettings
from src.utils.config_helpers import load_yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsentRecord:
    """One registry entry. Holds no PII — only pseudonymous identifiers.

    Attributes:
        consent_id: Registry key, e.g. "CONSENT-h01-2026-001".
        house_id:   Pseudonymous house identifier the consent covers ("h01").
        granted_on: ISO-8601 date the form was signed.
        scope:      What the consent covers (default "dataset-training").
        withdrawn:  True once the participant withdraws; affected sessions
                    are surfaced by the progress report for removal.
    """

    consent_id: str
    house_id: str
    granted_on: str = ""
    scope: str = "dataset-training"
    withdrawn: bool = False


def load_consent_registry(path: Path) -> dict[str, ConsentRecord]:
    """Load the local consent registry.

    A missing file returns an empty registry (format-only verification);
    a malformed file raises so a broken registry is never silently ignored.

    Args:
        path: Registry YAML path (mapping consent_id → record fields).

    Returns:
        Mapping of consent ID to :class:`ConsentRecord`.

    Raises:
        ValueError: If the file exists but is not a mapping of mappings,
                    or an entry lacks ``house_id``.
    """
    try:
        raw = load_yaml(path)
    except FileNotFoundError:
        logger.info(f"No consent registry at {path} — format-only consent checks")
        return {}

    if not isinstance(raw, dict):
        raise ValueError(f"Consent registry {path} must be a mapping of consent_id → record")

    registry: dict[str, ConsentRecord] = {}
    for consent_id, entry in raw.items():
        if not isinstance(entry, dict) or not entry.get("house_id"):
            raise ValueError(f"Consent registry entry '{consent_id}' in {path} needs a house_id")
        registry[str(consent_id)] = ConsentRecord(
            consent_id=str(consent_id),
            house_id=str(entry["house_id"]),
            granted_on=str(entry.get("granted_on", "")),
            scope=str(entry.get("scope", "dataset-training")),
            withdrawn=bool(entry.get("withdrawn", False)),
        )
    logger.info(f"Consent registry loaded from {path}: {len(registry)} records")
    return registry


def verify_consent(
    reference: str,
    house_id: str,
    settings: ConsentSettings,
    registry: dict[str, ConsentRecord],
) -> list[str]:
    """Verify a session's consent reference before ingest.

    Checks, in order: presence (when ``settings.required``), format against
    ``settings.reference_pattern``, then — only when a registry is available —
    resolution, withdrawal state and house match. An empty registry (file
    absent on this machine) downgrades to format-only with a warning.

    Args:
        reference: The session's ``consent_reference`` (may be empty).
        house_id:  Pseudonymous house the session belongs to.
        settings:  Consent settings from the capture config.
        registry:  Loaded registry (possibly empty).

    Returns:
        List of problems; empty means the consent reference is acceptable.
    """
    if not reference:
        if settings.required:
            return ["consent reference is required (consent.required: true)"]
        return []

    problems: list[str] = []
    if not re.match(settings.reference_pattern, reference):
        problems.append(
            f"consent reference '{reference}' does not match pattern "
            f"{settings.reference_pattern}"
        )

    if not registry:
        logger.warning(
            f"Consent reference '{reference}' format-checked only — " f"no registry on this machine"
        )
        return problems

    record = registry.get(reference)
    if record is None:
        problems.append(f"consent reference '{reference}' not found in registry")
        return problems
    if record.withdrawn:
        problems.append(f"consent '{reference}' has been WITHDRAWN — do not ingest")
    if record.house_id != house_id:
        problems.append(
            f"consent '{reference}' covers house '{record.house_id}', "
            f"but the session belongs to '{house_id}'"
        )
    return problems


def find_withdrawn_consents(
    session_refs: dict[str, str],
    registry: dict[str, ConsentRecord],
) -> dict[str, str]:
    """Find ingested sessions whose consent has since been withdrawn.

    Args:
        session_refs: Mapping of session identifier → consent_reference
                      (built from the on-disk capture-session manifests).
        registry:     Loaded consent registry.

    Returns:
        Mapping of session identifier → withdrawn consent reference.
        Empty when the registry is unavailable.
    """
    withdrawn: dict[str, str] = {}
    for session_id, reference in session_refs.items():
        record = registry.get(reference)
        if record is not None and record.withdrawn:
            withdrawn[session_id] = reference
    if withdrawn:
        logger.warning(
            f"{len(withdrawn)} session(s) reference withdrawn consent: "
            f"{sorted(withdrawn)} — remove their data and cut a patch release"
        )
    return withdrawn
