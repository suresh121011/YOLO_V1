# ADR-P5-09 — Local-Drive DVC Remote as Default Now; S3 Secondary (Migration Documented)

**Status:** Accepted (Phase 5, 2026-07)
**Deciders:** Phase-5 engineering; council PASS 2026-07-17

## Context

Audit risk C-1 required the first-ever `dvc push` before any large dataset
exists. No cloud bucket is provisioned yet, and the project directory sits inside
an OneDrive-synced tree (risk R34: sync corrupting the DVC cache).

## Decision

The default DVC remote (`localstore`) points at `C:\dvc_remote` — a path OUTSIDE
the OneDrive tree — and the first `dvc push` was executed and verified at M0
(closes C-1). An S3 remote (`storage`) is kept as a named secondary for later
migration. Migration is a copy of the folder plus one config line, documented in
the README's operational prerequisites.

## Alternatives considered

1. **Provision S3 first and push there.** Rejected: no bucket/credentials yet,
   and C-1 must close before large data accumulates — a local disk closes it now.
2. **Keep the cache inside the OneDrive tree.** Rejected: R34 — OneDrive sync can
   corrupt large binary caches; the remote and cache are relocated off it.

## Consequences

- Positive: C-1 closed immediately; cache/remote off the OneDrive hazard;
  `cache.type hardlink,copy`.
- Negative: a single local disk is not off-site backup — pushing the branch to
  `origin` and later migrating the remote to a second physical disk / S3 remain
  operational follow-ups.

Related: [ADR-P5-07](ADR-P5-07-releases-as-code.md)
