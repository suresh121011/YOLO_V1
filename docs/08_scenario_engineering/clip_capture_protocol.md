# Scenario Clip Capture Protocol (Phase 6)

## Purpose

A scenario clip is a short video that turns a scenario into a **test**: *this clip must fire
`SC-KIT-001` at CRITICAL within 30 seconds*, or *this clip must fire nothing at all*. Without clips
the scenario set is a set of assertions nobody has checked; the alert-volume simulation
(`src/scenario_engine/simulate.py`) projects behaviour from a synthetic occupancy profile, which is a
model of a day, not a day.

This document extends the existing capture protocol rather than replacing it. Read these first — the
consent model, session grammar, room and lighting vocabularies, and the withdrawal SOP are all
unchanged and are **not** restated here:

- `configs/capture_config.yaml` — the config both workflows share
- `../04_dataset_engineering/capture_annotation_runbook.md` — §1 consent, §2 capture, §3 ingest, §7 DVC
- `../01_executive_implementation_plan/validation_strategy.md` §"System Testing Scenarios" — the
  10-scenario matrix (rooms, lighting, three negatives) this protocol collects against

---

## 1. What is different from photographs

Four things, and each one is a gate in code rather than a paragraph here.

| | Photographs | Scenario clips |
|:--|:--|:--|
| Config block | `capture:` | `clips:` (a sibling, not a child) |
| Loader | `src/dataset/capture/config.py` | `src/scenario_engine/clips.py` |
| Metadata stripping | EXIF, `src/dataset/capture/exif.py` | **ffmpeg container strip + read-back**, see §3 |
| Consent scope | `dataset-training` | **`scenario-video`** — not implied by the above, see §2 |
| Missing consent registry | Format-only check, warning | **Refused** |
| Missing toolchain | n/a | **Refused** |

The `clips:` block is deliberately a sibling of `capture:` and **video extensions are never added to
`capture.image.allowed_extensions`**. That list feeds a per-image `min_dim` gate that opens each file
with PIL; an `.mp4` entry there would reject or crash on every clip. `tests/unit/scenario_engine/
test_clips.py::test_image_extensions_were_not_widened_to_video` fails if anyone tries.

## 2. Consent — video needs its own

**A household that consented to photographs has not consented to video.** A still can be reviewed and
discarded frame by frame before it is kept. Thirty seconds of continuous video cannot: it will contain
incidental faces, whatever was audible in the room, and whatever else happened to be in shot for that
half-minute. So scenario clips require a separate signed form and a separate registry record with
`scope: scenario-video`.

```yaml
# data/consent/consent_registry.yaml — local only, never git, never DVC
CONSENT-h01-2026-002:
  house_id: h01
  granted_on: "2026-08-15"
  scope: scenario-video
  withdrawn: false
```

`verify_clip_consent()` refuses any other scope, and refuses to run at all when the registry is
absent. Image ingest degrades to a format-only check in that case; clips do not, because scope is
exactly what cannot be checked without the registry — downgrading would silently drop the one control
this section exists to apply. **Clips are only ever ingested on the collection machine.**

Withdrawal follows the existing SOP in the Phase-3 runbook: set `withdrawn: true`, remove the clip and
its manifest, `dvc commit -f ingest_scenario_clips`, and cut a patch release.

## 3. Metadata stripping is a hard gate

`src/dataset/capture/exif.py` strips EXIF from images. **It does not cover video, and could not.** MP4
is a different container: GPS rides in the `moov/udta` `©xyz` atom next to creation time, device make
and model. An EXIF stripper never touches any of it.

A clip that reaches the S3 remote with its GPS atom intact is a participant's home address in a
**versioned** bucket — deleting the object afterwards does not remove prior versions. So:

```
ffmpeg -y -i <src> -map_metadata -1 -map_chapters -1 -c copy <dst>
```

then **read the output back with ffprobe and assert nothing forbidden survived**
(`FORBIDDEN_METADATA_KEYS`). Trusting `-map_metadata` to have worked is precisely the unfalsifiable
gate this project has been bitten by before. `-c copy` means the video is never re-encoded: sanitising
must not degrade the footage a scenario is validated against.

If ffmpeg or ffprobe is not on PATH, ingest **fails**. It is not skipped, not warned about, not
deferred. A privacy control that silently degrades is not a control.

> **Current blocker:** ffmpeg is not installed on the development machine, so no real clip has been
> ingested. This is the module behaving correctly, not a defect. Install ffmpeg on the collection
> machine before the first session.

## 4. Naming and grammar

```
{house}_{room}_{session}_c{NNN}      h01_kitchen_s001_c001
```

The prefix is the Phase-3 session grammar unchanged, so `parse_clip_id()` yields the same
`(house_id, room, session_id)` the image workflow already uses. Extracted frames are named
`{clip_id}_frame_{NNNNN}.jpg`, which `src/utils/dataset_utils.py`'s group-pattern extractor already
recognises — so **leakage prevention comes for free**: frames from one clip can never be split across
train and val, with no new code.

## 5. Intake gates

Set in `configs/capture_config.yaml` under `clips.video:`.

| Gate | Default | Why |
|:--|:--|:--|
| `allowed_extensions` | `.mp4`, `.mov` | What phones produce and ffmpeg remuxes losslessly |
| `min_duration_s` | 10 | A clip shorter than a dwell threshold cannot exercise it |
| `max_duration_s` | 120 | Review cost, and the longer the clip the more it captures incidentally |
| `min_fps` | 15 | The runtime's `target_fps`. Temporal predicates are counted in **frames**, so a 10 fps clip silently stretches every dwell it is meant to be testing |
| `max_file_mb` | 500 | Catches an accidental 4K 60fps upload before ffmpeg is spent on it |

`min_fps` is a rejection rather than a warning for the reason in the table: a clip that rescales the
thing it is validating produces a green result that means nothing.

## 6. Positives and negatives

Every clip declares a `polarity`. A **negative** must additionally state `negative_kind` — a negative
without a stated reason cannot be reviewed for coverage:

| `negative_kind` | Meaning | Example from the repo's confuser list |
|:--|:--|:--|
| `confuser` | Targets a known false-positive source | Shiny floor read as wet |
| `absence` | The plain no-hazard case | Empty room, 60+ seconds |
| `assistive` | An object present that must **not** alarm | Walking stick near a person |
| `out_of_taxonomy` | Detector-level hard negative | Plastic bottle mistaken for a medicine strip |

**At least 30 % of a clip set must be negatives** (`clips.min_negative_fraction`). A suite of positives
measures sensitivity and is blind to the false-positive rate — which is the failure mode that actually
gets the device unplugged, and unplugging is a 100 % false-negative rate. The 10-scenario matrix in
`validation_strategy.md` already carries exactly three negatives, so the default is the ratio the
project had already chosen.

Note this is a different thing from `src/dataset/negatives.py`, which means *images containing none of
the 23 classes*, and from a scenario's trigger-level negation. All three are legitimate; they are not
interchangeable.

## 7. Session procedure

1. **Consent first, camera second.** Signed `scenario-video` form; registry record created; consent
   reference in hand. No exceptions, and nothing is filmed "just to see".
2. **Brief the household on the specific scene.** Scenario clips are staged, not surveillance — a
   30-second recording of a deliberately-set hazard, with everyone present knowing the camera is on.
   Never record an actual unsafe situation to document it; correct it.
3. **Fixed camera, room-appropriate mounting.** One camera per room is the deployment assumption
   (ADR-P6-07), so a hand-held pan produces a clip the runtime will never see.
4. **Cover the lighting variants** the matrix names for that room. Evening and dim are where the
   detector is weakest and are the ones most often skipped.
5. **Shoot the negative in the same session as its positive**, same room and lighting. A confuser shot
   in different conditions tests the lighting, not the confuser.
6. **Record what the clip should do before ingesting it** — scenario id, expected severity, and the
   deadline. Deciding the expectation after watching the engine's output is not a test.

## 8. Ingest

```bash
# Once per machine
python scripts/scenarios/32_ingest_scenario_clips.py --init

# A positive
python scripts/scenarios/32_ingest_scenario_clips.py \
    --clip-id h01_kitchen_s001_c001 --source kitchen_unattended.mp4 \
    --scenario-id SC-KIT-001 --lighting evening \
    --consent-ref CONSENT-h01-2026-002 \
    --expect fires --severity CRITICAL --first-alert-within 30 \
    --annotator initials

# A negative (the shiny-floor confuser)
python scripts/scenarios/32_ingest_scenario_clips.py \
    --clip-id h01_hall_s002_c003 --source shiny_floor.mp4 \
    --scenario-id SC-BTH-001 --lighting daylight \
    --consent-ref CONSENT-h01-2026-002 \
    --expect no-alert --negative-kind confuser

# Re-verify everything on disk
python scripts/scenarios/32_ingest_scenario_clips.py --verify-all
```

Ingest performs, in order: clip-id grammar → already-ingested check → ffmpeg availability → consent
scope → file gates → ffprobe duration/fps → metadata strip with read-back → `rule_hash` lookup →
manifest write. Any failure removes the partially written output and exits non-zero.

Re-ingesting an existing clip id is **refused** rather than overwritten. A clip id is what a test
result refers to; silently replacing the footage behind one changes what a green suite means. Delete
the manifest and video deliberately, or use the next clip number.

The manifest records `rule_hash_at_label_time`. When the scenario's `rule_hash` later moves, the clip
is **stale**: it asserts an expectation that no longer exists, and a suite passing green against a
stale expectation is worse than one that fails. `--verify-all` reports it.

## 9. DVC

```yaml
ingest_scenario_clips:
  frozen: true
  cmd: python scripts/scenarios/32_ingest_scenario_clips.py --verify-all
  outs:
    - data/scenario_engine/clips/video
    - data/scenario_engine/clips/manifests:
        cache: false
```

**Frozen**, mirroring `ingest_custom_captures`, for the same reason: `dvc repro` on a fresh machine
must never overwrite human-collected footage with an empty re-run. Humans ingest; then
`dvc commit -f ingest_scenario_clips`.

The two outs are split on purpose. `video/` is cached — large binary with no review value in a diff,
and the one Phase-6 artifact whose confidentiality rests on the remote rather than on git.
`manifests/` is `cache: false` and git-committed, because a manifest is an **assertion**, and changing
what the test suite claims must be visible in a pull request. Same rationale as the compiled scenario
artifact.

## 10. Checklist

- [ ] ffmpeg and ffprobe installed on the collection machine
- [ ] `scenario-video` consent signed and registered for every house being filmed
- [ ] `python scripts/scenarios/32_ingest_scenario_clips.py --init`
- [ ] Clips cover the 10-scenario matrix's rooms and lighting variants
- [ ] ≥ 30 % negatives, each with a `negative_kind`
- [ ] Every positive states a `first_alert_within_s`
- [ ] `--verify-all` exits 0
- [ ] `dvc commit -f ingest_scenario_clips` and `dvc push`
- [ ] Manifests reviewed in the pull request — they are what the suite claims

---

Related: [ADR-P6-10](adr/ADR-P6-10-clips-as-a-frozen-stage.md) ·
[scenario_taxonomy.md](scenario_taxonomy.md) · [negative_register.md](negative_register.md)
