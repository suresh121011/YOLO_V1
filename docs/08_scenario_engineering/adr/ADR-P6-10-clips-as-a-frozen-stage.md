# ADR-P6-10 — Scenario Clips Are a Frozen DVC Stage with Split Outs and Their Own Consent Scope

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; extends the ADR-P6-01…09 set ratified 2026-08-07

## Context

The scenario set is a set of assertions about behaviour. The alert-volume simulation
(`src/scenario_engine/simulate.py`) projects those assertions against a synthetic occupancy profile,
which is a model of a day rather than a day. To know whether `SC-KIT-001` actually fires within 30
seconds on real footage, the project needs **clips**: short videos, each carrying an expectation.

Clips are unlike every other input this repository handles:

- **They cannot be regenerated.** A downloader stage can re-fetch COCO. Nobody can re-shoot a
  household's kitchen from a script.
- **They cannot be curated after the fact.** A photograph is reviewed frame by frame before it is
  kept; thirty seconds of continuous video contains whatever was in the room for that half-minute.
- **The image privacy stack does not cover them.** `src/dataset/capture/exif.py` strips EXIF. MP4
  carries GPS in the `moov/udta` `©xyz` atom, which an EXIF stripper never touches. The S3 remote has
  versioning enabled, so a leak is not undone by deleting the object.
- **They go stale silently.** A clip asserts an outcome for a scenario. When that scenario's
  `rule_hash` changes, the clip asserts something that no longer exists — and a suite passing green
  against a stale expectation is worse than one that fails.

## Decision

Four decisions, taken together.

**1. The ingest stage is `frozen: true`.** It mirrors `ingest_custom_captures` exactly, for the same
reason: `dvc repro` on a fresh machine must never overwrite human-collected footage with an empty
re-run. The stage exists so clips are tracked in `dvc.lock` like every other output. Its `cmd` is
`--verify-all`, so deliberately reproducing it re-checks what is on disk rather than re-ingesting.

**2. The outs are split by review value, not by convenience.**

| Out | Caching | Why |
|:--|:--|:--|
| `data/scenario_engine/clips/video` | cached | Large binary, no review value in a diff, and the one Phase-6 artifact whose confidentiality rests on the remote rather than on git |
| `data/scenario_engine/clips/manifests` | `cache: false` | A manifest is an **assertion** — changing what the suite claims must appear in a pull request |

**3. Metadata stripping is a gate with a read-back assertion, and a missing toolchain is an error.**
`ffmpeg -map_metadata -1 -map_chapters -1 -c copy`, then ffprobe the output and refuse the clip if any
forbidden key survived. If ffmpeg is absent, ingest fails rather than proceeding. A privacy control
that silently degrades is not a control.

**4. Clips require their own consent scope, `scenario-video`.** `dataset-training` consent does not
cover video and is rejected at ingest. Unlike image ingest, a **missing consent registry is fatal**
rather than a downgrade to format-only checking: scope is precisely what cannot be verified without
the registry.

Two supporting rules follow from the same reasoning: clip IDs extend the Phase-3 session grammar
(`{session}_c{NNN}`) so extracted frames group as one leakage unit under the existing
`src/utils/dataset_utils.py` pattern with no new code; and at least 30 % of a clip set must be
negatives, each declaring a `negative_kind`.

## Alternatives considered

1. **Put clips under `src/dataset/capture/` with the images.** Rejected on layering — clips belong to
   the knowledge layer and `src/scenario_engine` may not import `src.dataset`
   ([ADR-P6-04](ADR-P6-04-rule-engine-injection.md)). It is not merely a rule being obeyed: the two
   paths share no code, because the containers genuinely have nothing in common. The cost is a small
   duplicated consent reader, which the module documents at its definition.
2. **Widen `capture.image.allowed_extensions` to include `.mp4`.** Rejected, and guarded by a test.
   That list feeds a per-image `min_dim` gate that opens each file with PIL; a video entry would reject
   or crash on every clip. This is the tempting one-line change and it silently breaks image intake.
3. **A non-frozen stage that re-ingests from the inbox.** Rejected: the inbox is transient and
   gitignored, so a fresh checkout would reproduce the stage into an empty directory and DVC would
   record the deletion as legitimate.
4. **Both outs cached.** Rejected: it hides the expectations. The reason the compiled scenario artifact
   is `cache: false` is that a rule set's correctness is a property of the set and must be reviewable
   at a tag by someone without remote credentials. A clip suite's claims are the same kind of thing.
5. **Both outs `cache: false`.** Rejected outright: it puts participant video in git history, where it
   cannot be removed.
6. **Treat `dataset-training` consent as covering video.** Rejected. It is the option that saves the
   most effort and it is the one that would matter most in a DPDP complaint — the household agreed to
   photographs.
7. **Skip stripping when ffmpeg is unavailable, and warn.** Rejected. The warning would be seen once,
   on the machine where it did not matter, and the clip would reach the versioned bucket anyway.
8. **Strip EXIF from clips with the existing image stripper.** Rejected as worse than doing nothing:
   it would report success while leaving every GPS atom intact.

## Consequences

- Positive: leakage prevention, session parsing, and the withdrawal SOP are all inherited from the
  Phase-3 workflow rather than reimplemented.
- Positive: the manifests are diffable, so "we changed what the suite asserts" cannot happen quietly.
- Positive: a clip whose scenario has moved on is detectable — `--verify-all` reports staleness against
  the compiled artifact's `rule_hash`.
- Constraint: **ffmpeg becomes a hard prerequisite of the collection machine.** It is not installed on
  the current development box, so no real clip has been ingested yet. This is the design working.
- Constraint: households contributing both media need two consent records, and the collection lead has
  to explain why. That conversation is the point.
- Constraint: clips can only be ingested where the consent registry lives. CI and other developers can
  verify structure but never ingest.

Related: [ADR-P6-04](ADR-P6-04-rule-engine-injection.md),
[ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md),
[ADR-P6-08](ADR-P6-08-rejected-scenario-negative-register.md),
[clip_capture_protocol.md](../clip_capture_protocol.md)
