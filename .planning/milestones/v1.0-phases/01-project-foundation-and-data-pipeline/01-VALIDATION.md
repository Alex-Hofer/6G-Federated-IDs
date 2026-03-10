---
phase: 1
slug: project-foundation-and-data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0.0 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | INFR-01, INFR-02 | unit | `python -m pytest tests/test_config.py -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0 | INFR-02 | unit | `python -m pytest tests/test_config.py::test_seed -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | DATA-01 | unit | `python -m pytest tests/test_loader.py -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | DATA-02 | unit | `python -m pytest tests/test_preprocess.py::test_feature_selection -x` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | DATA-03 | unit | `python -m pytest tests/test_preprocess.py::test_scaler_no_leakage -x` | ❌ W0 | ⬜ pending |
| 01-02-04 | 02 | 1 | DATA-04 | unit | `python -m pytest tests/test_preprocess.py::test_class_weights -x` | ❌ W0 | ⬜ pending |
| 01-02-05 | 02 | 1 | DATA-05 | unit | `python -m pytest tests/test_partition.py -x` | ❌ W0 | ⬜ pending |
| 01-02-06 | 02 | 1 | INFR-02 | unit | `python -m pytest tests/test_preprocess.py::test_reproducibility -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures: synthetic DataFrame mimicking CICIDS2017 structure (~100 rows), test config dict
- [ ] `tests/test_loader.py` — stubs for DATA-01: CSV loading, column stripping, Inf/NaN removal, label mapping
- [ ] `tests/test_preprocess.py` — stubs for DATA-02, DATA-03, DATA-04, INFR-02: feature selection, scaler fitting, class weights, reproducibility
- [ ] `tests/test_partition.py` — stubs for DATA-05: stratified IID partitioning, class ratio verification
- [ ] `tests/test_config.py` — stubs for INFR-01: YAML loading, env var interpolation, validation errors
- [ ] Framework install: `pip install -e ".[dev]"` — pytest included in dev extras

*Testing approach: Tests use synthetic DataFrames (no actual CICIDS2017 CSVs required). conftest.py provides fixtures mimicking CICIDS2017 column names, data types, and deliberate Inf/NaN values.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pip install -e .` succeeds | INFR-02 | Environment-dependent | Run `pip install -e ".[dev]"` in a clean venv and verify no errors |
| CICIDS2017 CSV download instructions | DATA-01 | External dependency | Verify README contains UNB download URL and expected filenames |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
