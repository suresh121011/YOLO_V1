"""Unit tests for src.dataset.capture.consent — PII-free consent verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.capture.config import ConsentSettings
from src.dataset.capture.consent import (
    ConsentRecord,
    find_withdrawn_consents,
    load_consent_registry,
    verify_consent,
)

_SETTINGS = ConsentSettings(reference_pattern=r"^CONSENT-h\d{2}-\d{4}-\d{3}$", required=True)


def _write_registry(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "consent_registry.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _registry() -> dict[str, ConsentRecord]:
    return {
        "CONSENT-h01-2026-001": ConsentRecord(
            consent_id="CONSENT-h01-2026-001", house_id="h01", granted_on="2026-07-20"
        ),
        "CONSENT-h02-2026-002": ConsentRecord(
            consent_id="CONSENT-h02-2026-002", house_id="h02", withdrawn=True
        ),
    }


@pytest.mark.unit
class TestLoadConsentRegistry:
    """Registry loading and shape validation."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path,
            """
CONSENT-h01-2026-001:
  house_id: h01
  granted_on: "2026-07-20"
  scope: dataset-training
  withdrawn: false
CONSENT-h02-2026-002:
  house_id: h02
  withdrawn: true
""",
        )
        registry = load_consent_registry(path)
        assert len(registry) == 2
        record = registry["CONSENT-h01-2026-001"]
        assert record.house_id == "h01"
        assert record.granted_on == "2026-07-20"
        assert record.withdrawn is False
        assert registry["CONSENT-h02-2026-002"].withdrawn is True

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_consent_registry(tmp_path / "missing.yaml") == {}

    def test_entry_without_house_id_raises(self, tmp_path: Path) -> None:
        path = _write_registry(tmp_path, "CONSENT-h01-2026-001:\n  granted_on: '2026-01-01'\n")
        with pytest.raises(ValueError, match="house_id"):
            load_consent_registry(path)

    def test_non_mapping_registry_raises(self, tmp_path: Path) -> None:
        path = _write_registry(tmp_path, "- CONSENT-h01-2026-001\n")
        with pytest.raises(ValueError, match="mapping"):
            load_consent_registry(path)


@pytest.mark.unit
class TestVerifyConsent:
    """Reference verification against settings + registry."""

    def test_happy_path(self) -> None:
        problems = verify_consent("CONSENT-h01-2026-001", "h01", _SETTINGS, _registry())
        assert problems == []

    def test_empty_reference_required(self) -> None:
        problems = verify_consent("", "h01", _SETTINGS, _registry())
        assert problems and "required" in problems[0]

    def test_empty_reference_optional(self) -> None:
        settings = ConsentSettings(required=False)
        assert verify_consent("", "h01", settings, {}) == []

    def test_bad_format(self) -> None:
        problems = verify_consent("consent-1", "h01", _SETTINGS, _registry())
        assert problems and "pattern" in problems[0]

    def test_unknown_reference(self) -> None:
        problems = verify_consent("CONSENT-h09-2026-009", "h09", _SETTINGS, _registry())
        assert problems and "not found" in problems[0]

    def test_withdrawn_reference(self) -> None:
        problems = verify_consent("CONSENT-h02-2026-002", "h02", _SETTINGS, _registry())
        assert problems and "WITHDRAWN" in problems[0]

    def test_house_mismatch(self) -> None:
        problems = verify_consent("CONSENT-h01-2026-001", "h03", _SETTINGS, _registry())
        assert problems and "h03" in problems[0]

    def test_empty_registry_is_format_only(self) -> None:
        # No registry on this machine: a well-formed reference passes.
        assert verify_consent("CONSENT-h05-2026-005", "h05", _SETTINGS, {}) == []


@pytest.mark.unit
class TestFindWithdrawnConsents:
    """Withdrawal sweep over ingested sessions."""

    def test_flags_withdrawn_sessions(self) -> None:
        session_refs = {
            "h01_kitchen_s001": "CONSENT-h01-2026-001",
            "h02_hall_s001": "CONSENT-h02-2026-002",
            "h02_hall_s002": "CONSENT-h02-2026-002",
        }
        withdrawn = find_withdrawn_consents(session_refs, _registry())
        assert withdrawn == {
            "h02_hall_s001": "CONSENT-h02-2026-002",
            "h02_hall_s002": "CONSENT-h02-2026-002",
        }

    def test_empty_registry_flags_nothing(self) -> None:
        assert find_withdrawn_consents({"h01_kitchen_s001": "CONSENT-h01-2026-001"}, {}) == {}
