---
phase: 6
slug: verify-phase1-requirements
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v && python scripts/verify_phase1.py` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v && python scripts/verify_phase1.py`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | DATA-02 | unit | `python -m pytest tests/test_preprocess.py::TestSelectFeaturesReExport -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | DATA-01 | verification script | `python scripts/verify_phase1.py` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | DATA-02 | verification script | `python scripts/verify_phase1.py` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | DATA-03 | verification script | `python scripts/verify_phase1.py` | ❌ W0 | ⬜ pending |
| 06-01-05 | 01 | 1 | DATA-05 | verification script | `python scripts/verify_phase1.py` | ❌ W0 | ⬜ pending |
| 06-01-06 | 01 | 1 | INFR-01 | verification script | `python scripts/verify_phase1.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/verify_phase1.py` — standalone verification script covering DATA-01, DATA-02, DATA-03, DATA-05, INFR-01
- [ ] `tests/test_preprocess.py::TestSelectFeaturesReExport` — regression test for select_features import from package
- [ ] `src/federated_ids/data/__init__.py` — fix: add select_features re-export (code change, not test gap)

*All Wave 0 items are prerequisites for verification.*

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
