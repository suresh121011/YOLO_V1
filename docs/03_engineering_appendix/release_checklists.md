# Release & Pre-Deployment Checklists

## Purpose

Pre-release quality gates, pre-deployment checklist, and dataset release checklist.

## Dependencies

Reads:
- training_scripts.md
- qa_pipeline.md

Used By:
- troubleshooting.md

Related:
- ../02_technical_architecture_specification/architecture_decisions.md
- ../01_executive_implementation_plan/validation_strategy.md

---

## 13.1 Pre-Release Checklist

```markdown
## Release v{X.Y.Z} Checklist

### Code Quality
- [ ] All unit tests passing (`python -m pytest tests/unit/ -v`)
- [ ] All integration tests passing (`python -m pytest tests/integration/ -v`)
- [ ] Code formatted with `black` (0 violations)
- [ ] Type checking with `mypy --strict` (0 errors)
- [ ] Security scan with `bandit` (0 high-severity findings)
- [ ] Dependency audit with `pip-audit` (0 known vulnerabilities)

### Model Quality
- [ ] QA pipeline passes with 0 critical errors
- [ ] mAP50 >= 0.70 on validation set
- [ ] Safety class recall >= 0.80
- [ ] False positive rate < 10% per class on test videos
- [ ] Model hash recorded in release notes

### Performance
- [ ] FPS >= 15 on target device
- [ ] End-to-end latency < 2,000ms
- [ ] RAM usage < 2GB peak
- [ ] CPU usage < 60% steady state
- [ ] 30-minute thermal stability test passed

### Documentation
- [ ] README updated with version number
- [ ] CHANGELOG updated
- [ ] API docs reflect any interface changes
- [ ] Deployment guide updated if config changed

### Deployment
- [ ] Model exported to target format (ONNX/TFLite)
- [ ] Config files updated for target device
- [ ] Feature flags set appropriately
- [ ] Git tag created (v{X.Y.Z})
- [ ] DVC tag created if dataset changed
```

---

## 13.2 Pre-Deployment Checklist

```markdown
## Deployment to {target_device} Checklist

### Environment
- [ ] Python version matches requirements (>=3.10)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Piper TTS binary available and in PATH
- [ ] Camera accessible (`v4l2-ctl --list-devices` or OS camera check)

### Model & Config
- [ ] Model weights in expected path
- [ ] Model hash verified on load (check startup logs)
- [ ] Config YAML validated (no schema errors on startup)
- [ ] Feature flags reviewed and set correctly for this device

### System State
- [ ] SQLite database directory writable (`logs/`)
- [ ] Alert queue empty at startup
- [ ] Speaker output tested (TTS health check passes)
- [ ] FPS benchmark run on this specific device

### Testing
- [ ] Run 5-minute smoke test (no crashes)
- [ ] Trigger each of 6 rules manually to verify speech output
- [ ] Verify SQLite events.db records events
- [ ] Check active learning log is populated (if enabled)
```

---

## 13.3 Dataset Release Checklist

```markdown
## Dataset v{X.Y.Z} Release Checklist

### Quality Gates
- [ ] All 23 classes have >= 200 annotated instances
- [ ] Safety-critical classes have >= 500 instances
- [ ] QA pipeline: 0 critical errors, 0 warnings
- [ ] No train/val leakage detected
- [ ] No duplicate images (perceptual hash check)
- [ ] All bounding boxes valid (coordinates in [0,1], class IDs in [0,22])

### Annotation Review
- [ ] ML Lead spot-checked 50 images per new class
- [ ] Inter-annotator agreement score >= 0.85 for new batches
- [ ] Class definition document reviewed for any ambiguous cases

### Versioning
- [ ] DATASET_CHANGELOG.md updated
- [ ] DVC commit created and pushed to remote
- [ ] Git tag created (dataset-v{X.Y.Z})
- [ ] Train/val split regenerated if major version bump
```

---

Previous: [annotation_guide.md](./annotation_guide.md)

Next: [troubleshooting.md](./troubleshooting.md)

Related: [../02_technical_architecture_specification/architecture_decisions.md](../02_technical_architecture_specification/architecture_decisions.md)
