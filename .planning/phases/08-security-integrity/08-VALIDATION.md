---
phase: 8
slug: security-integrity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `python -m pytest tests/test_security.py -x` |
| **Full suite command** | `python -m pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_security.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_inf_raises -x` | ❌ W0 | ⬜ pending |
| 8-01-02 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_nan_raises -x` | ❌ W0 | ⬜ pending |
| 8-01-03 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_bad_labels_raises -x` | ❌ W0 | ⬜ pending |
| 8-01-04 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_preprocess_validation -x` | ❌ W0 | ⬜ pending |
| 8-01-05 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_partition_ratio_raises -x` | ❌ W0 | ⬜ pending |
| 8-02-01 | 02 | 2 | SEC-02 | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_empty_results -x` | ❌ W0 | ⬜ pending |
| 8-02-02 | 02 | 2 | SEC-02 | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_zero_example_clients -x` | ❌ W0 | ⬜ pending |
| 8-02-03 | 02 | 2 | SEC-02 | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_nan_params -x` | ❌ W0 | ⬜ pending |
| 8-02-04 | 02 | 2 | SEC-02 | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_all_clients_invalid -x` | ❌ W0 | ⬜ pending |
| 8-02-05 | 02 | 2 | SEC-03 | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_save_json -x` | ❌ W0 | ⬜ pending |
| 8-02-06 | 02 | 2 | SEC-03 | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_load_json -x` | ❌ W0 | ⬜ pending |
| 8-02-07 | 02 | 2 | SEC-03 | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_pkl_migration_error -x` | ❌ W0 | ⬜ pending |
| 8-03-01 | 03 | 3 | SEC-06 | unit | `python -m pytest tests/test_security.py::TestConfigValidation::test_invalid_log_level -x` | ❌ W0 | ⬜ pending |
| 8-03-02 | 03 | 3 | SEC-06 | unit | `python -m pytest tests/test_security.py::TestConfigValidation::test_invalid_config_extension -x` | ❌ W0 | ⬜ pending |
| 8-03-03 | 03 | 3 | SEC-07 | unit | `python -m pytest tests/test_security.py::TestConfigSafety::test_no_config_mutation -x` | ❌ W0 | ⬜ pending |
| 8-03-04 | 03 | 3 | SEC-09 | integration | `python -m pytest tests/test_security.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_security.py` — new file covering SEC-01 through SEC-09 validation tests
- [ ] `src/federated_ids/exceptions.py` — new module defining `DataValidationError`
- [ ] Update `tests/test_preprocess.py` line 204 — change `scaler.pkl` to `scaler.json` artifact check

*Existing infrastructure covers pytest framework and conftest.py fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pyproject.toml` deps match actual usage | SEC-04, SEC-05 | Requires inspecting actual imports vs declared deps | Run `grep -r "import flwr\|import tqdm\|import joblib" src/` and verify zero results; check pyproject.toml no longer lists them |
| fl/__init__.py docstring accuracy | SEC-08 | Semantic correctness of documentation | Read docstring and verify it correctly states no Flower dependency |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
