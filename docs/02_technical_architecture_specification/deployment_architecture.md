# Engineering Standards

## Purpose

Code standards, Git conventions, dependency policy, and model versioning convention.

## Dependencies

Reads:
- api_contracts.md

Used By:
- architecture_decisions.md

Related:
- ../01_executive_implementation_plan/engineering_standards.md

---

## Code Standards

| Standard | Tool | Enforcement |
|:---------|:-----|:------------|
| Style | `black` (88-char line length) | Pre-commit hook |
| Import ordering | `isort` | Pre-commit hook |
| Type hints | All public functions | `mypy --strict` in CI |
| Docstrings | All public classes and methods | Google-style docstrings |
| Test coverage | `pytest-cov` | Minimum 70% coverage |
| Security scanning | `bandit` | CI gate |

## Git Conventions

```
feat(detector): add class-specific confidence thresholds
fix(rule_engine): correct cooldown timer reset on rule reload
refactor(orchestrator): extract frame skip logic into strategy class
docs(tts): add fallback behavior documentation
test(event_memory): add window boundary condition tests
```

## Dependency Policy

- All dependencies pinned with exact versions in `requirements.txt`
- `requirements-dev.txt` for development-only tools
- No transitive dependency pinning (lock file approach)
- Security audit with `pip-audit` before each release
- CUDA/hardware dependencies isolated in `requirements-gpu.txt`

## Model Versioning Convention

```
yolo11n-v{major}.{minor}.{patch}
  major: taxonomy change (new/removed class)
  minor: retraining with new data (≥100 new images)
  patch: hyperparameter-only retrain
```

---

Previous: [api_contracts.md](./api_contracts.md)

Next: [architecture_decisions.md](./architecture_decisions.md)

Related: [../01_executive_implementation_plan/engineering_standards.md](../01_executive_implementation_plan/engineering_standards.md)
