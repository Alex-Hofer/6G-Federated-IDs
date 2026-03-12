---
phase: 05-integration-and-polish
plan: 01
subsystem: infra
tags: [pipeline, cli, integration-test, orchestration]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    provides: "preprocess.main() entry point and _cache_exists() for tensor caching"
  - phase: 03-federated-learning-infrastructure
    provides: "run_federated_training() FL orchestration loop"
  - phase: 04-evaluation-and-visualization
    provides: "eval.__main__.main() evaluation pipeline with plots and reports"
provides:
  - "run_pipeline() end-to-end orchestrator chaining preprocess -> FL training -> evaluation"
  - "federated-ids-run-all CLI console script"
  - "Integration test validating full pipeline on synthetic CICIDS2017 data"
affects: [05-02-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: ["thin orchestrator calling existing entry points with explicit config_path", "lazy imports for stage modules"]

key-files:
  created:
    - src/federated_ids/pipeline.py
    - tests/test_integration.py
  modified:
    - pyproject.toml

key-decisions:
  - "Lazy imports for stage modules to keep pipeline module import lightweight"
  - "No try/except around stages -- fail-fast by letting exceptions propagate"
  - "Registered pytest slow marker in pyproject.toml to suppress unknown mark warning"

patterns-established:
  - "Pipeline orchestrator pattern: load config once, pass config_path explicitly to all stages"
  - "Integration test pattern: synthetic CSV with injected inf/nan, minimal config for fast execution"

requirements-completed: [MODL-03, INFR-02]

# Metrics
duration: 3min
completed: 2026-03-10
---

# Phase 5 Plan 1: End-to-End Pipeline Summary

**Pipeline runner chaining preprocess, FL training, and evaluation into single `federated-ids-run-all` command, validated by integration test on synthetic CICIDS2017 data**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-10T09:00:03Z
- **Completed:** 2026-03-10T09:02:31Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Pipeline orchestrator module chains all three stages sequentially with cache-aware preprocessing skip
- Console script `federated-ids-run-all` registered in pyproject.toml for zero-config experiment execution
- Integration test validates full pipeline produces all output artifacts (checkpoint, metrics JSON, plots, classification report)
- Full test suite (89 tests) passes with no regressions in under 5 seconds

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pipeline runner module and register console script** - `f671262` (feat)
2. **Task 2: Create integration test with synthetic CICIDS2017 data** - `f29ce7d` (test)

## Files Created/Modified
- `src/federated_ids/pipeline.py` - End-to-end pipeline orchestrator with run_pipeline(), _print_pipeline_summary(), and main()
- `tests/test_integration.py` - Integration test with synthetic 500-row CICIDS2017 CSV and full output validation
- `pyproject.toml` - Added federated-ids-run-all console script and pytest slow marker registration

## Decisions Made
- Lazy imports for stage modules (preprocess, FL server, eval) to keep pipeline module import lightweight, consistent with fl/server.py pattern
- No try/except around stages -- fail-fast by letting exceptions propagate directly, per user decision in plan
- Registered `slow` pytest marker in pyproject.toml to eliminate unknown mark warning without adding pytest-timeout dependency

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Registered pytest slow marker in pyproject.toml**
- **Found during:** Task 2 (Integration test)
- **Issue:** `@pytest.mark.slow` produced an unknown mark warning without marker registration
- **Fix:** Added `markers` list to `[tool.pytest.ini_options]` in pyproject.toml
- **Files modified:** pyproject.toml
- **Verification:** Full test suite runs with zero warnings about unknown marks
- **Committed in:** f29ce7d (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Minor config addition for pytest hygiene. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline runner is fully operational and tested
- Ready for 05-02 (documentation, polish, and thesis-readiness tasks)
- All 89 tests pass, all Phase 1-4 modules remain stable

---
*Phase: 05-integration-and-polish*
*Completed: 2026-03-10*
