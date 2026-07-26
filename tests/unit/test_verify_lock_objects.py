"""Unit tests for the dvc.lock object-integrity sweep.

The sweep exists because ``dvc status -c`` cannot see an object that is missing
from the cache *and* the remote: with nothing pending to push it answers "in
sync", so RG6 goes green over a lock file citing unretrievable data. These tests
pin that behaviour, including the false-green scenario itself.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Loaded by path because scripts/qa/ is not an importable package. The module
# must be registered in sys.modules *before* exec_module: @dataclass resolves
# its own module via sys.modules[cls.__module__], which is None for an
# unregistered spec-loaded module.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "qa" / "verify_lock_objects.py"
_spec = importlib.util.spec_from_file_location("verify_lock_objects", _MODULE_PATH)
assert _spec and _spec.loader
verify_lock_objects = importlib.util.module_from_spec(_spec)
sys.modules["verify_lock_objects"] = verify_lock_objects
_spec.loader.exec_module(verify_lock_objects)

object_path = verify_lock_objects.object_path
sweep = verify_lock_objects.sweep


def _put(store: Path, md5: str, payload: bytes = b"x") -> Path:
    dest = object_path(store, md5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def _write_lock(path: Path, stages: dict) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"stages": stages}), encoding="utf-8")
    return path


class TestObjectPath:
    def test_splits_md5_into_two_char_prefix(self, tmp_path: Path) -> None:
        p = object_path(tmp_path, "abcdef1234")
        assert p == tmp_path / "files" / "md5" / "ab" / "cdef1234"

    def test_dir_suffix_preserved_outside_the_split(self, tmp_path: Path) -> None:
        p = object_path(tmp_path, "abcdef1234.dir")
        assert p == tmp_path / "files" / "md5" / "ab" / "cdef1234.dir"


class TestSweep:
    def test_all_present_passes(self, tmp_path: Path) -> None:
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        for store in (cache, remote):
            _put(store, "aaaaaaaaaaaaaaaa")
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"s1": {"outs": [{"path": "data/x", "md5": "aaaaaaaaaaaaaaaa"}]}},
        )
        result = sweep(lock, cache, remote, deep=False)
        assert result.ok
        assert result.checked == 1

    def test_missing_from_both_is_the_false_green_case(self, tmp_path: Path) -> None:
        """The exact scenario `dvc status -c` reports as 'in sync'."""
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        cache.mkdir()
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"coverage_report": {"outs": [{"path": "cov.json", "md5": "7970437206"}]}},
        )
        result = sweep(lock, cache, remote, deep=False)
        assert not result.ok
        assert len(result.missing) == 1
        missing = result.missing[0]
        assert missing.md5 == "7970437206"
        assert missing.in_cache is False
        assert missing.in_remote is False

    def test_in_cache_but_not_remote_is_reported(self, tmp_path: Path) -> None:
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        _put(cache, "bbbbbbbbbbbb")
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"qa": {"outs": [{"path": "qa.json", "md5": "bbbbbbbbbbbb"}]}},
        )
        result = sweep(lock, cache, remote, deep=False)
        assert not result.ok
        assert result.missing[0].in_cache is True
        assert result.missing[0].in_remote is False

    def test_deps_are_ignored(self, tmp_path: Path) -> None:
        """Source-file deps live in git, never the cache — flagging them is noise."""
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        cache.mkdir()
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"s1": {"deps": [{"path": "src/dataset/remap.py", "md5": "deadbeef"}]}},
        )
        result = sweep(lock, cache, remote, deep=False)
        assert result.ok
        assert result.checked == 0

    def test_metrics_section_is_checked(self, tmp_path: Path) -> None:
        """annotation_qa_report.json is declared under `metrics`, not `outs`."""
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        cache.mkdir()
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"qa_check": {"metrics": [{"path": "qa.json", "md5": "cccccccccccc"}]}},
        )
        result = sweep(lock, cache, remote, deep=False)
        assert not result.ok
        assert result.missing[0].section == "metrics"

    def test_deep_expands_dir_members(self, tmp_path: Path) -> None:
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        listing = json.dumps([{"md5": "1111111111", "relpath": "a.txt"}]).encode()
        for store in (cache, remote):
            _put(store, "ffffffffff.dir", listing)
        # member object deliberately absent from both stores
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"merge": {"outs": [{"path": "data/merged", "md5": "ffffffffff.dir"}]}},
        )
        shallow = sweep(lock, cache, remote, deep=False)
        assert shallow.ok, "shallow sweep only checks the .dir object itself"

        deep = sweep(lock, cache, remote, deep=True)
        assert not deep.ok
        assert deep.missing[0].md5 == "1111111111"

    def test_remote_none_checks_cache_only(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        _put(cache, "dddddddddddd")
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"s1": {"outs": [{"path": "d", "md5": "dddddddddddd"}]}},
        )
        assert sweep(lock, cache, None, deep=False).ok

    def test_duplicate_hashes_counted_once(self, tmp_path: Path) -> None:
        """The same object referenced by several stages must not double-report."""
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        cache.mkdir()
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {
                "a": {"outs": [{"path": "x", "md5": "eeeeeeeeeeee"}]},
                "b": {"outs": [{"path": "x", "md5": "eeeeeeeeeeee"}]},
            },
        )
        result = sweep(lock, cache, remote, deep=False)
        assert result.checked == 1
        assert len(result.missing) == 1

    def test_report_dict_is_serializable(self, tmp_path: Path) -> None:
        cache, remote = tmp_path / "cache", tmp_path / "remote"
        cache.mkdir()
        remote.mkdir()
        lock = _write_lock(
            tmp_path / "dvc.lock",
            {"s": {"outs": [{"path": "p", "md5": "aaaabbbbcccc"}]}},
        )
        payload = sweep(lock, cache, remote, deep=False).to_dict()
        assert payload["status"] == "fail"
        assert payload["missing_count"] == 1
        json.dumps(payload)  # must not raise
