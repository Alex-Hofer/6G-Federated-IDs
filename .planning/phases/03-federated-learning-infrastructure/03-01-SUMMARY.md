---
phase: 03-federated-learning-infrastructure
plan: 01
subsystem: fl
tags: [federated-learning, fedavg, numpy, pytorch, tdd]

# Dependency graph
requires:
  - phase: 02-model-definition-and-local-training
    provides: "MLP model, train_one_epoch, evaluate functions"
provides:
  - "FederatedClient class with get/set_parameters and fit"
  - "fedavg_aggregate weighted averaging function"
  - "server_evaluate global model evaluation"
  - "FL-specific test fixtures (fl_train_loaders, fl_test_loader, fl_criterion)"
affects: [03-02-PLAN, 04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [numpy-parameter-transport, in-place-copy-for-optimizer-safety, tdd-red-green]

key-files:
  created:
    - src/federated_ids/fl/client.py
    - src/federated_ids/fl/server.py
    - tests/test_fl.py
  modified:
    - src/federated_ids/fl/__init__.py
    - tests/conftest.py

key-decisions:
  - "In-place parameter copy via copy_() instead of load_state_dict to preserve optimizer tensor references"
  - "get_parameters returns numpy copies (not views) to prevent mutation bugs across FL rounds"

patterns-established:
  - "NumPy parameter transport: get_parameters returns independent copies, set_parameters uses in-place copy_()"
  - "FL client mirrors Flower NumPyClient interface for future migration compatibility"

requirements-completed: [FLRN-01, FLRN-02]

# Metrics
duration: 6min
completed: 2026-03-09
---

# Phase 3 Plan 1: FL Client and FedAvg Summary

**FederatedClient with NumPy parameter transport and FedAvg weighted aggregation, tested via TDD with 5 unit tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-09T19:23:14Z
- **Completed:** 2026-03-09T19:29:20Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- FederatedClient class with get/set_parameters roundtrip and fit() local training
- FedAvg aggregation producing mathematically correct weighted averages
- Server-side evaluation loading global params and returning all 5 metric keys
- 5 new FL unit tests passing, 68 total tests green (no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Test scaffolding and FL-specific conftest fixtures** - `64a4020` (test)
2. **Task 2: FederatedClient class and FedAvg aggregation** - `26c6f56` (feat)

_TDD flow: Task 1 = RED (failing tests), Task 2 = GREEN (implementation passing all tests)_

## Files Created/Modified
- `src/federated_ids/fl/client.py` - FederatedClient class mirroring Flower NumPyClient interface
- `src/federated_ids/fl/server.py` - fedavg_aggregate weighted averaging and server_evaluate
- `src/federated_ids/fl/__init__.py` - Public API exports for FL subpackage
- `tests/test_fl.py` - 5 unit tests covering params, fit, aggregation, evaluation
- `tests/conftest.py` - FL fixtures: fl_train_loaders, fl_test_loader, fl_criterion

## Decisions Made
- **In-place copy for set_parameters:** Used `copy_()` on state_dict tensors instead of `load_state_dict` to preserve optimizer tensor references. `load_state_dict` can replace tensor objects, causing the optimizer to silently stop producing gradient updates.
- **Independent copies in get_parameters:** Returns `.detach().numpy().copy()` instead of `.numpy()` views. On CPU, `.numpy()` returns a memory-shared view that gets mutated when training modifies parameters, causing snapshot comparisons to silently return equality.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed get_parameters returning views instead of copies**
- **Found during:** Task 2 (FederatedClient implementation)
- **Issue:** `.cpu().numpy()` on CPU tensors returns a memory-shared view. When `initial_params = get_parameters()` was compared to post-training params, both pointed to the same memory, so the comparison always showed no change.
- **Fix:** Changed to `.cpu().detach().numpy().copy()` to return independent copies.
- **Files modified:** `src/federated_ids/fl/client.py`
- **Verification:** test_client_fit now correctly detects parameter changes after training.
- **Committed in:** `26c6f56` (Task 2 commit)

**2. [Rule 1 - Bug] Used in-place copy instead of load_state_dict for set_parameters**
- **Found during:** Task 2 (FederatedClient implementation)
- **Issue:** `load_state_dict` with new `torch.tensor()` objects breaks optimizer references. The optimizer holds pointers to the original parameter tensors, but `load_state_dict` replaces them.
- **Fix:** Used `state_dict[key].copy_(torch.tensor(new_val))` which modifies values in-place while preserving tensor identity.
- **Files modified:** `src/federated_ids/fl/client.py`
- **Verification:** test_client_fit confirms parameters change after training (optimizer correctly updates the model).
- **Committed in:** `26c6f56` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes essential for correctness. The numpy-view and optimizer-reference bugs are well-known PyTorch/FL pitfalls. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- FL building blocks (client, server, aggregation) ready for Plan 02 orchestration loop
- All interfaces match the contracts specified in PLAN.md context
- FederatedClient mirrors Flower NumPyClient for future migration path

## Self-Check: PASSED

- All 5 created/modified files exist on disk
- Both task commits (64a4020, 26c6f56) found in git log
- Line counts: client.py=136 (min 60), server.py=111 (min 40), test_fl.py=170 (min 80)
- All 68 tests pass (5 new FL + 63 existing)

---
*Phase: 03-federated-learning-infrastructure*
*Completed: 2026-03-09*
