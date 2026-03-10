---
phase: 3
slug: federated-learning-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/test_fl.py -x -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_fl.py -x -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | FLRN-01 | unit | `pytest tests/test_fl.py::test_client_get_set_parameters -x` | No -- Wave 0 | pending |
| 03-01-02 | 01 | 0 | FLRN-01 | unit | `pytest tests/test_fl.py::test_client_fit -x` | No -- Wave 0 | pending |
| 03-01-03 | 01 | 0 | FLRN-02 | unit | `pytest tests/test_fl.py::test_fedavg_aggregate -x` | No -- Wave 0 | pending |
| 03-01-04 | 01 | 0 | FLRN-02 | unit | `pytest tests/test_fl.py::test_fedavg_equal_weights -x` | No -- Wave 0 | pending |
| 03-01-05 | 01 | 0 | FLRN-03 | unit | `pytest tests/test_fl.py::test_config_driven_rounds_clients -x` | No -- Wave 0 | pending |
| 03-01-06 | 01 | 0 | FLRN-03 | unit | `pytest tests/test_fl.py::test_cli_overrides -x` | No -- Wave 0 | pending |
| 03-01-07 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_fl.py::test_round_metrics_keys -x` | No -- Wave 0 | pending |
| 03-01-08 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_fl.py::test_metrics_json_output -x` | No -- Wave 0 | pending |
| 03-01-09 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_fl.py::test_convergence_check -x` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_fl.py` -- stubs for FLRN-01, FLRN-02, FLRN-03, EVAL-01
- [ ] `tests/conftest.py` -- add FL-specific fixtures (synthetic partitions, mock clients)
- [ ] No framework install needed -- pytest already configured

*Existing infrastructure covers framework requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Global model convergence over rounds | EVAL-01 | Convergence depends on data/hyperparameters; unit test uses synthetic data | Run full FL loop with real partitions, visually confirm F1 trend |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
