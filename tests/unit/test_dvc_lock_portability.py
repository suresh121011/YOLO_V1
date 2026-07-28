"""``dvc.lock`` dependency hashes must be reproducible from a clean checkout.

Phase-F2/F3 caught `scripts/dataset/01_download_coco_subset.py` recorded in
``dvc.lock`` under the md5 of its **CRLF** form. ``.gitattributes`` declares
``*.py text eol=lf``, so every checkout — Linux *and* Windows — materialises
that file with LF and hashes it differently. The lock therefore named a hash no
clean checkout on any platform could produce, and `dvc repro` anywhere else
considered ``download_coco`` stale, which is a 2.2-hour re-download that
cascades into merge/split/qa.

It survived four months because nothing pointed at it: git normalises on add,
so ``git status`` reported the drifted working copy as clean, and the machine
that wrote the lock was the one machine whose bytes matched it.

The check is deliberately narrow — it fails only when a recorded hash is the
CRLF variant of the committed blob. An ordinary in-progress edit (dep changed,
stage not yet re-run) is normal development state and must not fail here; that
is `dvc status`'s job, not a test's.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "dvc.lock"


def _git_blob(path: str) -> bytes | None:
    """Bytes of ``path`` as committed at HEAD, or None if not tracked."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _lock_file_deps() -> list[tuple[str, str, str]]:
    """(stage, path, recorded_md5) for every file dependency in dvc.lock."""
    if not LOCK_PATH.exists():
        return []
    stages = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8")).get("stages") or {}
    deps: list[tuple[str, str, str]] = []
    for stage, spec in stages.items():
        for dep in spec.get("deps") or []:
            md5 = dep.get("md5")
            if md5 and not md5.endswith(".dir"):
                deps.append((stage, str(dep["path"]), md5))
    return deps


class TestDvcLockPortability:
    def test_no_dependency_is_recorded_under_its_crlf_hash(self) -> None:
        """The regression: a lock entry no clean checkout can ever match."""
        offenders: list[str] = []
        for stage, path, recorded in _lock_file_deps():
            blob = _git_blob(path)
            if blob is None or b"\r\n" in blob:
                continue  # untracked, or genuinely a CRLF file in git
            crlf_variant = hashlib.md5(
                blob.replace(b"\n", b"\r\n"), usedforsecurity=False
            ).hexdigest()
            if recorded == crlf_variant:
                offenders.append(
                    f"{stage}: {path} recorded as CRLF ({recorded}); "
                    f"a clean checkout produces "
                    f"{hashlib.md5(blob, usedforsecurity=False).hexdigest()}"
                )
        assert not offenders, "dvc.lock records unreproducible CRLF hashes:\n" + "\n".join(
            offenders
        )

    def test_lock_has_file_dependencies_to_check(self) -> None:
        """Guards the guard: an empty sweep would pass vacuously."""
        assert len(_lock_file_deps()) > 20, "expected dvc.lock to declare many file deps"


class TestGitattributesPinsLineEndings:
    """The mechanism the fix relies on — assert it is actually declared."""

    @staticmethod
    def _gitattributes() -> str:
        path = REPO_ROOT / ".gitattributes"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @pytest.mark.parametrize("pattern", ["*.py text eol=lf", "* text=auto eol=lf"])
    def test_declares_lf_normalisation(self, pattern: str) -> None:
        assert pattern in self._gitattributes()
