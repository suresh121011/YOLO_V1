# data/scenario_engine/clips — Scenario Clips (DVC-tracked)

Ingested via `scripts/scenarios/32_ingest_scenario_clips.py`. Every clip here
has had its container metadata stripped and verified by read-back; every
manifest carries a `consent_reference` whose registry scope is `scenario-video`.

    video/      sanitised .mp4/.mov, named {clip_id}{ext}
    manifests/  one {clip_id}.json per clip -- the expectation it asserts

Do not add files by hand. Do not re-run the DVC stage to "rebuild" this
directory: it is frozen precisely because its contents cannot be regenerated.

Protocol: docs/08_scenario_engineering/clip_capture_protocol.md
