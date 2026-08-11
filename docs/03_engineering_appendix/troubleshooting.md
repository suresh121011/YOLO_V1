# Troubleshooting Guide & Disaster Recovery

## Purpose

Common issues with resolutions, SQLite log analysis commands, and disaster recovery procedures.

## Dependencies

Reads:
- sample_logs.md
- api_reference.md

Used By: None (terminal operational document)

Related:
- ../02_technical_architecture_specification/error_handling.md

---

## 15.1 Common Issues

| Symptom | Likely Cause | Resolution |
|:--------|:-------------|:-----------|
| "Model hash mismatch" on startup | Corrupted model file | Re-download or `dvc checkout` the model |
| FPS drops below 5 | Thermal throttling or CPU contention | Check device temperature; reduce `process_every_n_frames` |
| No alerts generated | Rule cooldowns active or config issue | See §15.3 "starts but never alerts"; call `last_decisions()` |
| `TaxonomyMismatchError` on startup | Weights trained on a different class list | `python scripts/qa/model_landing_check.py` names the difference. Fix it in training/export — do **not** add a rename map |
| TTS silent (no audio) | Speaker muted, Piper crash, or `tts_output: false` | Check the flag first, then device volume and the TTS health check |
| One scenario is never spoken but others are | `patient_facing: false` — by design for 7 of the 9 | It went to `logs/caregiver.jsonl` instead. Not a fault |
| `NOBODY HAS BEEN CALLED` in the log | A `push_and_call` alert fired; no telephony exists | Expected. Read `logs/caregiver.jsonl` and escalate by hand |
| A feature flag appears to do nothing | It may genuinely be inert | Flags read by no code are labelled `NOT WIRED` in `configs/feature_flags.yaml` |
| "Camera source unavailable" | Camera permission or hardware | Check OS camera permissions; try different camera index |
| High false positive rate | Low confidence threshold | Increase `conf_threshold` for affected class |
| High false negative rate | High confidence threshold or poor training data | Decrease threshold; collect more training data |
| SQLite "database is locked" | Concurrent write from multiple threads | Verify single logger instance; check threading |
| Out of memory error | VLM model too large for device | Disable VLM (`smolvlm_analysis: false`); use a smaller variant |
| `yaml.scanner.ScannerError` on config load | YAML syntax error | Validate YAML with `yamllint`; check indentation |
| `piper: command not found` | Piper TTS not installed | Install Piper: `pip install piper-tts` or download binary |
| Alert repeating too frequently | Cooldown too short for that scenario | Raise `cooldown_seconds` in `configs/scenarios/<id>.yaml`, then recompile. `max_repeats` is capped at 3 by the schema |
| VLM timeout errors in log | Device too slow for SmolVLM2 | Disable VLM or switch to 256M variant |

---

## 15.2 Log Analysis Commands

```bash
# Count alerts by rule
sqlite3 logs/events.db \
  "SELECT json_extract(data, '$.alert.rule_id'), COUNT(*)
   FROM events
   WHERE json_extract(data, '$.event_type') = 'ALERT_FIRED'
   GROUP BY 1;"

# Average FPS over last hour
sqlite3 logs/events.db \
  "SELECT AVG(json_extract(data, '$.pipeline_metrics.fps'))
   FROM events
   WHERE timestamp > datetime('now', '-1 hour');"

# Find all low-confidence detections for active learning
sqlite3 logs/events.db \
  "SELECT * FROM active_learning
   WHERE confidence < 0.35
   ORDER BY timestamp DESC LIMIT 20;"

# Check detection distribution by class
sqlite3 logs/events.db \
  "SELECT json_extract(value, '$.class_name'), COUNT(*)
   FROM events, json_each(json_extract(data, '$.detections'))
   GROUP BY 1 ORDER BY 2 DESC;"

# Find all CRITICAL alerts in last 24 hours
sqlite3 logs/events.db \
  "SELECT timestamp, json_extract(data, '$.alert.rule_id'), json_extract(data, '$.alert.message')
   FROM events
   WHERE json_extract(data, '$.alert.severity') = 'CRITICAL'
     AND timestamp > datetime('now', '-24 hours');"
```

---

## 15.3 Disaster Recovery Procedures

### Scenario: Model Regression After Retraining

```
1. Identify: Performance dashboard shows mAP50 dropped below threshold
2. Contain: Stop deployment of new model
3. Recover:
   a. git log --oneline models/  # Find last known good commit
   b. dvc checkout models/yolo11n/weights/best.pt  # Restore weights
   c. git checkout <good-commit> -- configs/  # Restore configs
4. Verify: Run evaluation script against restored model
5. Root cause: Compare training configs and dataset versions
```

### Scenario: Dataset Corruption

```
1. Identify: QA pipeline reports critical errors
2. Contain: Do not start training with corrupted data
3. Recover:
   a. dvc diff  # Identify changed files
   b. dvc checkout data/  # Restore from DVC remote
4. Verify: Re-run QA pipeline
5. Root cause: Check annotation workflow for source of corruption
```

### Scenario: Device Failure During Field Test

```
1. Identify: Camera feed lost or pipeline crash
2. Contain: TTS speaks "System error, please wait"
3. Recover:
   a. Watchdog thread restarts crashed component (max 3 retries)
   b. If camera lost, wait 10s and retry
   c. If unrecoverable, graceful shutdown with log flush
4. Verify: Check logs/events.db for crash details
5. Escalate: If repeated, report for engineering investigation
```

### Scenario: Scenario YAML Corruption

`configs/risk_rules.yaml` was retired in Phase 6 (ADR-P6-03). The authored
knowledge now lives as one file per scenario in `configs/scenarios/`, and the
compiled artifact `data/scenario_engine/build/scenarios.compiled.json` is
**generated** — never hand-edit it, and never `git checkout` it as a recovery
step, because the next compile will overwrite whatever you restored.

```
1. Identify: "Cannot load scenarios" / ScenarioRuntimeError on start-up, or
   "Scenarios hot-reloaded" absent after an edit. The engine names the file and
   the field.
2. Contain: A failed reload leaves the previously loaded set active — a bad edit
   never disarms a running pipeline. A failed *start-up* refuses to run at all,
   which is deliberate: a silent engine is indistinguishable from a working one
   that has seen no hazards.
3. Recover:
   a. git diff configs/scenarios/          # See what changed
   b. git checkout HEAD configs/scenarios/ # Restore the AUTHORED masters
   c. python scripts/scenarios/30_compile_scenarios.py   # Regenerate
   d. python scripts/scenarios/31_validate_scenarios.py  # Is it still wise?
   e. rule_engine.reload_rules()           # Hot-reload into the running pipeline
4. Verify: 30_compile_scenarios.py --check exits 0 (1 = validation failure,
   2 = the committed artifact has drifted from the sources).
```

### Scenario: The pipeline starts but never alerts

Three causes, in the order worth checking:

```
1. Every scenario is still `status: draft`. This is the expected state until a
   clinical reviewer sets reviewed_by/reviewed_on. The engine refuses to start
   rather than running silently — check the start-up error, which counts the
   drafts.
2. No `room` was passed to build_pipeline(). Room-scoped scenarios (SC-KIT-*,
   SC-BTH-*, SC-COR-*) are gated on it, so a camera with no room configured is
   inert for most of the set.
3. The hazard has not dwelled long enough. SC-KIT-001 needs 16 minutes,
   SC-KIT-002 11, SC-MOB-001 6. Call rule_engine.last_decisions() — it reports
   per-scenario why each did or did not fire ("dwelling", "cooling_down",
   "wrong_room", "quiet_hours_suppress", "max_per_day").
```

---

Previous: [release_checklists.md](./release_checklists.md)

Next: [future_modules.md](./future_modules.md)

Related: [../02_technical_architecture_specification/error_handling.md](../02_technical_architecture_specification/error_handling.md)
