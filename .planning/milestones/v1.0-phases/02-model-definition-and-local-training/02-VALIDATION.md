---
phase: 2
slug: model-definition-and-local-training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0.0 (already in dev dependencies) |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` (testpaths = ["tests"]) |
| **Quick run command** | `python -m pytest tests/test_model.py tests/test_train.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_model.py tests/test_train.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 0 | MODL-01 | unit | `python -m pytest tests/test_model.py::TestMLP -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 0 | MODL-02 | unit | `python -m pytest tests/test_train.py::TestTrainOneEpoch -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 0 | MODL-03 | unit | `python -m pytest tests/test_train.py::TestCheckpointing -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 0 | DATA-04 | unit | `python -m pytest tests/test_train.py::TestWeightedLoss -x` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 1 | MODL-01 | unit | `python -m pytest tests/test_model.py::TestMLP::test_forward_shape -x` | ❌ W0 | ⬜ pending |
| 02-01-06 | 01 | 1 | MODL-02 | unit | `python -m pytest tests/test_train.py::TestEvaluate -x` | ❌ W0 | ⬜ pending |
| 02-01-07 | 01 | 2 | MODL-03 | unit | `python -m pytest tests/test_train.py::TestCheckpointing::test_load_checkpoint -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_model.py` — stubs for MODL-01 (MLP instantiation, forward pass, parameter count)
- [ ] `tests/test_train.py` — stubs for MODL-02, MODL-03, DATA-04 (training loop, evaluation, checkpointing, weighted loss)
- [ ] `tests/conftest.py` — add synthetic data fixtures for model/training tests (small tensors, mock class_weights.json)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| F1 > 0.80 on DDoS in 5 epochs | MODL-02 | Requires real CICIDS2017 data; too slow for CI | Run `python -m federated_ids.model.train` on full dataset, verify console output |
| Attack-class recall > 0.70 | DATA-04 | Requires real data + class weights | Same as above, check recall metric in output |
| Summary table copy-pasteable for thesis | MODL-02 | Visual/formatting check | Inspect console output format after training |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
