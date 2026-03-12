---
phase: 08-security-integrity
plan: 01
subsystem: data
tags: [validation, security, json-serialization, exceptions, assert-removal]

# Dependency graph
requires: []
provides:
  - DataValidationError custom exception class (ValueError subclass)
  - Assert-free validation in loader.py, preprocess.py, partition.py
  - JSON scaler serialization with versioned envelope format
  - _load_scaler_json with array length and finite value validation
affects: [08-02, 08-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [if/raise over assert for validation, JSON envelope for model artifacts]

key-files:
  created:
    - src/federated_ids/exceptions.py
  modified:
    - src/federated_ids/data/loader.py
    - src/federated_ids/data/preprocess.py
    - src/federated_ids/data/partition.py
    - tests/test_preprocess.py

key-decisions:
  - "Use if/raise DataValidationError instead of assert for all validation gates -- immune to python -O"
  - "JSON envelope format {version, type, params} for scaler serialization -- no pickle/joblib"
  - "No auto-conversion from pkl to json -- clean break with migration error message"
  - "DataValidationError extends ValueError for backward compatibility with existing except clauses"

patterns-established:
  - "Validation pattern: if condition: raise DataValidationError(f'message with {actual} vs {expected}')"
  - "Artifact serialization: versioned JSON envelope with validation on load"

requirements-completed: [SEC-01, SEC-03, SEC-05]

# Metrics
duration: 5min
completed: 2026-03-12
---

# Phase 8 Plan 1: Assert Replacement & Scaler Migration Summary

**Replace all 10 assert-based validation gates with DataValidationError and migrate scaler from pickle to versioned JSON envelope**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-12T10:22:18Z
- **Completed:** 2026-03-12T10:28:17Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created DataValidationError(ValueError) exception class in new exceptions.py module
- Replaced all 10 assert statements across 3 files with if/raise patterns including actual vs expected values
- Migrated scaler serialization from joblib/pickle to JSON with versioned envelope format
- Added _load_scaler_json with array length validation, finite value checks, and pkl migration guard
- Removed joblib dependency entirely from the codebase

## Task Commits

Each task was committed atomically:

1. **Task 1: Create exception module and replace all 10 asserts** - `10524d9` (feat)
2. **Task 2: Migrate scaler serialization from pickle to JSON and remove joblib** - `357aa83` (feat)

## Files Created/Modified
- `src/federated_ids/exceptions.py` - New module with DataValidationError(ValueError) class
- `src/federated_ids/data/loader.py` - 3 asserts replaced with if/raise DataValidationError
- `src/federated_ids/data/preprocess.py` - 6 asserts replaced, joblib removed, _save/_load_scaler_json added, scaler.pkl -> scaler.json
- `src/federated_ids/data/partition.py` - 1 assert replaced with if/raise DataValidationError
- `tests/test_preprocess.py` - Updated artifact check from scaler.pkl to scaler.json

## Decisions Made
- DataValidationError extends ValueError for backward compatibility with existing except clauses
- JSON envelope format uses version field ("1.0") for future-proofing
- No auto-conversion from .pkl to .json: clean break with a descriptive migration error
- _load_scaler_json sets n_samples_seen_ = np.int64(1) as sentinel since original value is not preserved

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated test expectation from scaler.pkl to scaler.json**
- **Found during:** Task 2 (scaler migration)
- **Issue:** test_artifacts_saved asserted existence of scaler.pkl, which no longer exists after migration
- **Fix:** Changed assertion to check for scaler.json instead
- **Files modified:** tests/test_preprocess.py
- **Verification:** All 35 tests pass
- **Committed in:** 357aa83 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Test fix was necessary consequence of the scaler filename change. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DataValidationError is available for import by all future plans in this phase
- Plans 08-02 and 08-03 can build on the validation pattern established here
- All 35 existing tests pass, confirming no behavioral regression

---
*Phase: 08-security-integrity*
*Completed: 2026-03-12*
