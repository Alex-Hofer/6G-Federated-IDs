---
phase: 4
slug: evaluation-and-visualization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `.venv/Scripts/python.exe -m pytest tests/test_eval.py -x -q` |
| **Full suite command** | `.venv/Scripts/python.exe -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/Scripts/python.exe -m pytest tests/test_eval.py -x -q`
- **After every plan wave:** Run `.venv/Scripts/python.exe -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | EVAL-02 | unit | `pytest tests/test_eval.py::TestEvaluateDetailed -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | EVAL-02 | unit | `pytest tests/test_eval.py::TestConfusionMatrixPlot -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | EVAL-02 | unit | `pytest tests/test_eval.py::TestClassificationReport -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | EVAL-03 | unit | `pytest tests/test_eval.py::TestConvergencePlot -x` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 1 | ALL | unit | `pytest tests/test_eval.py::TestClientComparison -x` | ❌ W0 | ⬜ pending |
| 04-01-06 | 01 | 1 | ALL | unit | `pytest tests/test_eval.py::TestEvalCLI -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | EVAL-04 | unit | `pytest tests/test_eval.py::TestTensorBoardLogging -x` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 2 | EVAL-04 | unit | `pytest tests/test_eval.py::TestTensorBoardFallback -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eval.py` — stubs for EVAL-02, EVAL-03, EVAL-04
- [ ] `pip install tensorboard` — not yet in environment
- [ ] Update `pyproject.toml` — add tensorboard dependency

*Existing infrastructure covers pytest framework and conftest.py fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Plot visual quality | EVAL-02, EVAL-03 | Subjective visual assessment | Open PNG files and verify labels, colors, layout match spec |
| TensorBoard UI renders | EVAL-04 | Requires TensorBoard UI | Run `tensorboard --logdir outputs/tensorboard/` and verify scalars appear |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
