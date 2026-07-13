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
| No alerts generated | Rule cooldowns active or config issue | Check cooldown timers; verify `risk_rules.yaml` |
| TTS silent (no audio) | Speaker muted or Piper crash | Check device volume; verify TTS health check |
| "Camera source unavailable" | Camera permission or hardware | Check OS camera permissions; try different camera index |
| High false positive rate | Low confidence threshold | Increase `conf_threshold` for affected class |
| High false negative rate | High confidence threshold or poor training data | Decrease threshold; collect more training data |
| SQLite "database is locked" | Concurrent write from multiple threads | Verify single logger instance; check threading |
| Out of memory error | VLM model too large for device | Disable VLM (`vlm_enabled: false`); use smaller variant |
| `yaml.scanner.ScannerError` on config load | YAML syntax error | Validate YAML with `yamllint`; check indentation |
| `piper: command not found` | Piper TTS not installed | Install Piper: `pip install piper-tts` or download binary |
| Alert repeating too frequently | Cooldown too short for class | Increase `cooldown_seconds` in `risk_rules.yaml` |
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

### Scenario: Rule Engine YAML Corruption

```
1. Identify: "YAML parse error" in logs; no alerts being generated
2. Contain: Pipeline continues using last valid rule set (in-memory)
3. Recover:
   a. git diff configs/risk_rules.yaml  # See what changed
   b. git checkout HEAD configs/risk_rules.yaml  # Restore
   c. rule_engine.reload_rules()  # Trigger hot-reload
4. Verify: Check that rules fire correctly on next hazard detection
```

---

Previous: [release_checklists.md](./release_checklists.md)

Next: [future_modules.md](./future_modules.md)

Related: [../02_technical_architecture_specification/error_handling.md](../02_technical_architecture_specification/error_handling.md)
