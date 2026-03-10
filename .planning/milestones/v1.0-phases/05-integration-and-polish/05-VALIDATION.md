---
phase: 5
slug: integration-and-polish
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `.venv/Scripts/python -m pytest tests/test_integration.py -x -v` |
| **Full suite command** | `.venv/Scripts/python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/Scripts/python -m pytest tests/test_integration.py -x -v`
- **After every plan wave:** Run `.venv/Scripts/python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | MODL-03, INFR-02 | integration | `.venv/Scripts/python -m pytest tests/test_integration.py::TestFullPipeline::test_pipeline_produces_all_outputs -x` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 2 | INFR-02 | manual | README review | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_integration.py` — end-to-end pipeline integration test (covers MODL-03, INFR-02)
- [ ] Integration test fixture generating synthetic CICIDS2017 CSV with ~500 rows

*Existing test infrastructure (88 tests, conftest.py with _CICIDS_COLUMNS) covers all unit-level concerns. Only the new integration test file is needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README documents full reproduction steps | INFR-02 | Content quality requires human review | Read README.md, verify setup/download/config/run instructions are complete and accurate |
| Output screenshots embedded in README | INFR-02 | Visual output requires real data run | Run pipeline with real data, verify docs/*.png are generated and README embeds them correctly |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
