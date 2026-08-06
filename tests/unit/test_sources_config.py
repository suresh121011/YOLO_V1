"""Unit tests for src.dataset.sources_config — acquisition configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.sources_config import load_sources_config
from src.utils.config_helpers import get_class_names_from_data_yaml, load_data_config

MINIMAL_CONFIG = """
mode: smoke
smoke:
  limit_per_source: 10
allow_noncommercial: false
dedup:
  hamming_threshold: 3
  check_flips: true
sources:
  coco:
    enabled: true
    output_dir: data/raw/coco_filtered
    license: "CC BY 4.0"
    remap_table: coco
    trusted_classes: [person]
    class_caps: {person: 800}
  wider_face:
    enabled: true
    noncommercial: true
    license: "research-only"
  disabled_source:
    enabled: false
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dataset_sources.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadSourcesConfig:
    """YAML loading, defaults and validation."""

    def test_loads_minimal_config(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.mode == "smoke"
        assert config.limit == 10
        assert config.dedup.hamming_threshold == 3
        assert config.sources["coco"].remap_table == "coco"
        # Non-field keys land in options
        assert config.sources["coco"].options["class_caps"] == {"person": 800}

    def test_full_mode_has_no_limit(self, tmp_path: Path) -> None:
        config = load_sources_config(
            _write(tmp_path, MINIMAL_CONFIG.replace("mode: smoke", "mode: full"))
        )
        assert config.limit is None

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_sources_config(
                _write(tmp_path, MINIMAL_CONFIG.replace("mode: smoke", "mode: turbo"))
            )

    def test_missing_sources_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_sources_config(_write(tmp_path, "mode: smoke\n"))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_sources_config(tmp_path / "missing.yaml")

    def test_repo_config_is_loadable(self) -> None:
        repo_config = Path(__file__).resolve().parents[2] / "configs" / "dataset_sources.yaml"
        config = load_sources_config(repo_config)
        assert config.mode in ("smoke", "full")
        for expected in ("coco", "openimages", "roboflow", "wider_face", "negatives"):
            assert expected in config.sources, f"missing source '{expected}'"


@pytest.mark.unit
class TestRepoRoboflowDatasets:
    """The repo config's `sources.roboflow.datasets` entries (H-B).

    These are consumed by the downloader (slug/version), RG7's license
    recording (license), and 05_remap_classes.build_table (classes). A typo
    in any of them surfaces only after a multi-GB download or, worse, as a
    silently dropped class at remap time — so validate them statically.
    """

    def _entries(self) -> list[dict]:
        repo_root = Path(__file__).resolve().parents[2]
        config = load_sources_config(repo_root / "configs" / "dataset_sources.yaml")
        return list(config.sources["roboflow"].options.get("datasets") or [])

    def test_every_entry_is_well_formed(self) -> None:
        for entry in self._entries():
            slug = entry.get("slug", "")
            assert slug.count("/") == 1 and all(
                slug.split("/")
            ), f"slug must be 'workspace/project', got {slug!r}"
            assert int(entry.get("version", 0)) >= 1, f"{slug}: version must be >= 1"
            # RG7 (rg7_license_gate) fails a release when Roboflow contributed
            # images but no per-slug license is recorded.
            assert str(entry.get("license", "")).strip(), f"{slug}: license must be recorded"
            assert entry.get("classes"), f"{slug}: needs a class alias map"

    def test_class_aliases_resolve_to_real_taxonomy_names(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        names = get_class_names_from_data_yaml(
            load_data_config(repo_root / "configs" / "data.yaml")
        )
        taxonomy = set(names.values())
        for entry in self._entries():
            for alias, taxonomy_name in entry["classes"].items():
                assert taxonomy_name in taxonomy, (
                    f"{entry['slug']}: alias {alias!r} maps to {taxonomy_name!r}, "
                    f"which is not a class in configs/data.yaml"
                )

    def test_aliases_match_the_real_export_when_one_is_present(self) -> None:
        """The alias must equal the class string the export actually uses.

        Remap lookup is exact-string, so a near-miss alias silently drops
        every annotation instead of failing. This happened: aliases were
        first derived from each Universe page's prose ("Wire LWH6") rather
        than the export's own `names:` ("wire-lWH6"), and the remap stage
        would have reported "0 kept, 1939 dropped" without erroring.

        Skips when the export cache is absent (CI, fresh clone) — this is a
        consistency check against real downloaded data, not a network test.
        """
        yaml = pytest.importorskip("yaml")
        repo_root = Path(__file__).resolve().parents[2]
        cache = repo_root / "data" / "downloads_cache" / "roboflow"
        if not cache.is_dir():
            pytest.skip("no roboflow export cache on this machine")

        checked = 0
        for entry in self._entries():
            export_dir = cache / f"{entry['slug'].replace('/', '_')}_v{entry['version']}"
            data_yaml = export_dir / "data.yaml"
            if not data_yaml.is_file():
                continue
            exported = set(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"])
            for alias in entry["classes"]:
                assert alias in exported, (
                    f"{entry['slug']}: configured alias {alias!r} is not a class in the "
                    f"export ({sorted(exported)}) — remap would drop every annotation"
                )
                checked += 1
        if checked == 0:
            pytest.skip("no matching exports found in the cache")

    def test_declared_trusted_classes_all_have_a_source_dataset(self) -> None:
        """Every class roboflow claims to label exhaustively must actually be
        covered by a configured dataset — otherwise `trusted_classes` promises
        supervision the source cannot deliver."""
        repo_root = Path(__file__).resolve().parents[2]
        config = load_sources_config(repo_root / "configs" / "dataset_sources.yaml")
        covered = {
            taxonomy_name
            for entry in self._entries()
            for taxonomy_name in entry["classes"].values()
        }
        missing = sorted(set(config.sources["roboflow"].trusted_classes) - covered)
        assert not missing, f"trusted classes with no configured dataset: {missing}"


@pytest.mark.unit
class TestLicenseGate:
    """allow_noncommercial governance gate."""

    def test_noncommercial_source_blocked_when_gate_closed(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.allow_noncommercial is False
        assert config.is_source_allowed("wider_face") is False
        assert config.is_source_allowed("coco") is True

    def test_noncommercial_source_allowed_when_gate_open(self, tmp_path: Path) -> None:
        body = MINIMAL_CONFIG.replace("allow_noncommercial: false", "allow_noncommercial: true")
        config = load_sources_config(_write(tmp_path, body))
        assert config.is_source_allowed("wider_face") is True

    def test_disabled_and_unknown_sources_not_allowed(self, tmp_path: Path) -> None:
        config = load_sources_config(_write(tmp_path, MINIMAL_CONFIG))
        assert config.is_source_allowed("disabled_source") is False
        assert config.is_source_allowed("does_not_exist") is False
