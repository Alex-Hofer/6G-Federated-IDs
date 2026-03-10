---
phase: 7
slug: verify-phase4-and-integration-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q` + `python scripts/verify_phase4.py`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | BUG-FIX | unit | `python -m pytest tests/test_train.py::TestWeightedLossConfig -x` | No — W0 | pending |
| 07-01-02 | 01 | 1 | EVAL-02 | smoke | `python scripts/verify_phase4.py` | No — W0 | pending |
| 07-01-03 | 01 | 1 | EVAL-03 | smoke | `python scripts/verify_phase4.py` | No — W0 | pending |
| 07-01-04 | 01 | 1 | EVAL-04 | smoke | `python scripts/verify_phase4.py` | No — W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `scripts/verify_phase4.py` — verification script covering EVAL-02, EVAL-03, EVAL-04
- [ ] `tests/test_train.py::TestWeightedLossConfig` — regression test for weighted_loss fix (append to existing file)
- [ ] `.planning/phases/07-.../07-VERIFICATION.md` — verification report

*All test infrastructure (pytest, pyproject.toml config) already exists from prior phases.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
