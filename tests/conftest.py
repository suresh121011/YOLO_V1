"""
tests/ — Elderly Assistant System Test Suite
============================================

Test pyramid:
    tests/unit/         — Fast unit tests (no I/O, mocked dependencies)
    tests/integration/  — Component pair tests
    tests/system/       — End-to-end scenario tests (require full pipeline)
    tests/performance/  — Latency budget validation tests

Run all tests:
    pytest tests/ -v

Run by layer:
    pytest tests/unit/ -v -m unit
    pytest tests/integration/ -v -m integration
    pytest tests/system/ -v -m system
    pytest tests/performance/ -v -m performance

Run with coverage:
    pytest tests/ --cov=src --cov-report=term-missing

Markers (defined in pyproject.toml):
    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.system
    @pytest.mark.performance
    @pytest.mark.slow
"""
