---
phase: 02-model-definition-and-local-training
plan: 02
subsystem: model-training
tags: [pytorch, training-loop, evaluation, checkpointing, sklearn-metrics, cli]

requires:
  - phase: 02-model-definition-and-local-training
    plan: 01
    provides: MLP nn.Module class
  - phase: 01-project-foundation-and-data-pipeline
    provides: data pipeline with cached tensors, class_weights.json, partition_iid, config, seed, device
provides:
  - train_one_epoch() and evaluate() functions for FL client reuse in Phase 3
  - standalone_train() entry point for single-client model validation
  - Best-model checkpointing on F1 improvement
  - Console script federated-ids-train
affects: [03 federated learning (Flower clients reuse train_one_epoch/evaluate), 04 evaluation]

tech-stack:
  added: []
  patterns: [class-weighted CrossEntropyLoss from JSON, stratified train/val split, best-F1 checkpointing]

key-files:
  created:
    - src/federated_ids/model/train.py
    - src/federated_ids/model/__main__.py
    - tests/test_train.py
  modified:
    - config/default.yaml
    - pyproject.toml
    - tests/conftest.py

key-decisions:
  - "Class-weighted loss loaded from class_weights.json with explicit device placement"
  - "Stratified train/val split within client partition for per-epoch validation"
  - "Best-model checkpoint saved only on F1 improvement (not loss)"
  - "Auto-run data pipeline if cached tensors missing"
  - "standalone_epochs and val_split as optional config keys (not in _REQUIRED_NESTED)"

patterns-established:
  - "TDD for training code: test functions in isolation with synthetic data before integration"
  - "Dual invocation pattern: CLI argparse when config_path=None, direct call for programmatic use"

requirements-completed: [MODL-02, MODL-03, DATA-04]

duration: 4min
completed: 2026-03-09
---

# Phase 2 Plan 2: Local Training Loop Summary

**Training loop with class-weighted loss, F1-based checkpointing, sklearn evaluation metrics, and standalone CLI entry point for single-client model validation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-09T16:11:31Z
- **Completed:** 2026-03-09T16:15:48Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `train_one_epoch()` with batch-weighted loss accumulation for FL client reuse
- `evaluate()` returning binary accuracy, F1, precision, recall via sklearn
- `_print_summary_table()` for thesis-appendix-ready epoch metrics table
- `standalone_train()` with auto-pipeline, stratified val split, F1 checkpointing, global test evaluation
- `__main__.py` entry point for `python -m federated_ids.model.train`
- Console script `federated-ids-train` registered in pyproject.toml
- Config keys `standalone_epochs` and `val_split` added as optional training parameters
- 11 new unit tests (19 total with existing model tests) all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Config keys and console script** - `3ddd755` (chore)
2. **Task 2 (RED): Failing tests for training** - `9fc6da8` (test)
3. **Task 2 (GREEN): Implement train.py and __main__.py** - `7fac909` (feat)
4. **Task 2 (FIX): Stabilize checkpoint test** - `493ee19` (fix)

_TDD task had separate RED and GREEN commits._

## Files Created/Modified

- `src/federated_ids/model/train.py` - Training loop, evaluation, checkpointing, standalone entry point (270 lines)
- `src/federated_ids/model/__main__.py` - Module entry point for standalone training
- `tests/test_train.py` - 11 unit tests for training, evaluation, checkpointing, weighted loss
- `tests/conftest.py` - New fixtures: synthetic_train_data, synthetic_class_weights_file, sample_model
- `config/default.yaml` - Added standalone_epochs (5) and val_split (0.2) training keys
- `pyproject.toml` - Added federated-ids-train console script entry

## Decisions Made

- Class-weighted loss loaded from class_weights.json with explicit `.to(device)` placement to prevent device mismatch
- Stratified train/val split within client partition ensures validation class ratio matches training
- Best-model checkpoint saved only on F1 improvement (not loss) since F1 better represents DDoS detection quality on imbalanced data
- Auto-run data pipeline if cached tensors missing, enabling zero-config standalone training
- standalone_epochs and val_split as optional config keys accessed via `.get()` with defaults, avoiding breaking Phase 3 configs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed flaky checkpoint test**
- **Found during:** Task 2 verification
- **Issue:** `test_checkpoint_saved_on_f1_improvement` failed because 1 epoch with default LR on random data produced F1=0 (model predicted all-benign)
- **Fix:** Used class-weighted loss, higher LR (0.01), and 10 training epochs so model reliably learns minority class
- **Files modified:** tests/test_train.py
- **Commit:** 493ee19

## Issues Encountered

None beyond the test fix above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `train_one_epoch()` and `evaluate()` ready for Flower client integration (Phase 3)
- Both functions accept model, loader, criterion, optimizer, device -- matching FL client pattern
- Checkpoint save/load pattern established for FL round-level checkpointing
- All Phase 2 requirements complete (MODL-01, MODL-02, MODL-03, DATA-04 consumed)

## Self-Check: PASSED

- All 3 created files exist on disk
- All 3 modified files verified
- All 4 task commits verified in git log (3ddd755, 9fc6da8, 7fac909, 493ee19)
- All 19 unit tests pass (11 new + 8 existing)
- Imports verified: train_one_epoch, evaluate, standalone_train

---
*Phase: 02-model-definition-and-local-training*
*Completed: 2026-03-09*
