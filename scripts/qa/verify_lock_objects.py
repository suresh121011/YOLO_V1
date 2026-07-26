#!/usr/bin/env python
"""Verify every hash in ``dvc.lock`` resolves to a real object in cache/remote.

Why this exists
---------------
``dvc status -c`` compares the *cache* against the *remote*. An object that is
missing from **both** is therefore invisible to it — there is nothing pending to
push, so the check reports "in sync" and RG6 goes green over a lock file that
points at data nobody can retrieve.

That is not hypothetical. The Phase-5 audit found ``dvc.lock`` citing
``79704372…`` (coverage_report.json) and ``db86dbb6…`` (completeness_report.json)
while both were absent from ``C:\\dvc_cache`` *and* ``C:\\dvc_remote`` — most
likely lost when the cache was relocated off OneDrive. ``dvc status -c`` still
answered "Cache and remote are in sync". A false green over an unrecoverable
lock file is worse than a red one, because it silently certifies a dataset that
cannot be reconstituted.

This sweep closes that gap: it walks every ``md5`` recorded in ``dvc.lock`` and
asserts the corresponding object exists. Directory hashes (``*.dir``) are
resolved one level deep so the files they list are checked too.

Usage:
    python scripts/qa/verify_lock_objects.py
    python scripts/qa/verify_lock_objects.py --remote C:\\dvc_remote --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DIR_SUFFIX = ".dir"


@dataclass
class Missing:
    """One lock entry whose backing object could not be found."""

    stage: str
    section: str
    path: str
    md5: str
    in_cache: bool
    in_remote: bool


@dataclass
class SweepResult:
    checked: int = 0
    missing: list[Missing] = field(default_factory=list)
    unreadable_dirs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unreadable_dirs

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "missing_count": len(self.missing),
            "missing": [vars(m) for m in self.missing],
            "unreadable_dir_objects": self.unreadable_dirs,
            "status": "pass" if self.ok else "fail",
        }


def object_path(store: Path, md5: str) -> Path:
    """DVC 3.x content-addressed layout: ``files/md5/<2 chars>/<rest>``."""
    stem = md5[: -len(DIR_SUFFIX)] if md5.endswith(DIR_SUFFIX) else md5
    suffix = DIR_SUFFIX if md5.endswith(DIR_SUFFIX) else ""
    return store / "files" / "md5" / stem[:2] / (stem[2:] + suffix)


# Only these sections are DVC's to store. `deps` also carry md5s, but a dep is
# either a git-tracked source file (never cached — DVC hashes it purely to detect
# change) or another stage's out, which is already covered where it is produced.
# Checking deps would report every .py and .yaml in the pipeline as "missing".
STORED_SECTIONS = ("outs", "metrics", "plots")


def _iter_lock_entries(
    lock: dict[str, Any], sections: tuple[str, ...] = STORED_SECTIONS
) -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(stage, section, path, md5)`` for every hashed entry."""
    for stage, body in (lock.get("stages") or {}).items():
        for section in sections:
            for entry in body.get(section) or []:
                md5 = entry.get("md5")
                if md5:
                    yield stage, section, entry.get("path", "?"), md5


def _dir_member_hashes(cache: Path, md5: str) -> list[str] | None:
    """Read a ``.dir`` object and return the md5s of the files it lists."""
    obj = object_path(cache, md5)
    if not obj.exists():
        return None
    try:
        listing = json.loads(obj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return [item["md5"] for item in listing if isinstance(item, dict) and "md5" in item]


def sweep(lock_path: Path, cache: Path, remote: Path | None, deep: bool) -> SweepResult:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    result = SweepResult()
    seen: set[str] = set()

    for stage, section, path, md5 in _iter_lock_entries(lock):
        if md5 in seen:
            continue
        seen.add(md5)
        result.checked += 1

        in_cache = object_path(cache, md5).exists()
        in_remote = object_path(remote, md5).exists() if remote else True
        if not in_cache or not in_remote:
            result.missing.append(Missing(stage, section, path, md5, in_cache, in_remote))
            continue

        if deep and md5.endswith(DIR_SUFFIX):
            members = _dir_member_hashes(cache, md5)
            if members is None:
                result.unreadable_dirs.append(f"{path} ({md5})")
                continue
            for member in members:
                if member in seen:
                    continue
                seen.add(member)
                result.checked += 1
                m_cache = object_path(cache, member).exists()
                m_remote = object_path(remote, member).exists() if remote else True
                if not m_cache or not m_remote:
                    result.missing.append(
                        Missing(stage, section, f"{path}/*", member, m_cache, m_remote)
                    )
    return result


def _default_cache() -> Path:
    """Honour ``.dvc/config.local``'s ``cache.dir`` before the default location."""
    local = REPO_ROOT / ".dvc" / "config.local"
    if local.exists():
        for line in local.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("dir"):
                _, _, value = stripped.partition("=")
                if value.strip():
                    return Path(value.strip())
    return REPO_ROOT / ".dvc" / "cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "dvc.lock")
    parser.add_argument("--cache", type=Path, default=None, help="defaults to config.local")
    parser.add_argument(
        "--remote",
        type=Path,
        default=None,
        help="local-remote root; omit to check the cache only",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="skip expanding .dir objects into their member files",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    cache = args.cache or _default_cache()
    if not args.lock.exists():
        print(f"FAIL: {args.lock} not found", file=sys.stderr)
        return 2

    result = sweep(args.lock, cache, args.remote, deep=not args.no_deep)

    print(f"dvc.lock object sweep — {args.lock}")
    print(f"  cache : {cache}")
    print(f"  remote: {args.remote or '(not checked)'}")
    print(f"  objects checked: {result.checked}")

    if result.missing:
        print(f"\n  MISSING OBJECTS: {len(result.missing)}")
        for m in result.missing:
            where = []
            if not m.in_cache:
                where.append("cache")
            if not m.in_remote:
                where.append("remote")
            print(f"    [{m.stage}/{m.section}] {m.path}")
            print(f"      md5={m.md5}  absent from: {', '.join(where)}")
    if result.unreadable_dirs:
        print(f"\n  UNREADABLE .dir OBJECTS: {len(result.unreadable_dirs)}")
        for d in result.unreadable_dirs:
            print(f"    {d}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n  report → {args.json}")

    if result.ok:
        print("\nPASS: every dvc.lock hash resolves to a real object.")
        return 0
    print("\nFAIL: dvc.lock references objects that cannot be retrieved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
