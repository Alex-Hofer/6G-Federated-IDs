---
phase: 03-federated-learning-infrastructure
plan: 02
subsystem: fl
tags: [fedavg, orchestration, cli, convergence, checkpointing, metrics-json]

# Dependency graph
requires:
  - phase: 03-federated-learning-infrastructure (Plan 01)
    provides: FederatedClient, fedavg_aggregate, server_evaluate
provides:
  - run_federated_training orchestration loop
  - check_convergence early-vs-late F1 comparison
  - save_fl_metrics JSON persistence with embedded config
  - print_config_banner ASCII summary
  - federated-ids-train-fl CLI entry point
  - fl/__main__.py with --config, --num-clients, --num-rounds
affects: [04-evaluation-framework]

# Tech tracking
tech-stack:
  added: []
  patterns: [orchestration-loop, fresh-model-per-client-per-round, fresh-optimizer-per-round, convergence-check, fl-summary-table]

key-files:
  created:
    - src/federated_ids/fl/__main__.py
  modified:
    - src/federated_ids/fl/server.py
    - src/federated_ids/fl/__init__.py
    - pyproject.toml
    - tests/test_fl.py

key-decisions:
  - "Fresh MLP and Adam optimizer created per client per round to avoid shared state bugs (Pitfall 6 & 2)"
  - "Convergence check uses mean F1 of first n vs last n rounds with adaptive n for short histories"
  - "Metrics JSON embeds full config snapshot for thesis reproducibility"
  - "CLI overrides (--num-clients, --num-rounds) mutate config dict before passing to run_federated_training"

patterns-established:
  - "FL orchestration: fresh model + optimizer per client per round, FedAvg aggregate, server_evaluate on shared eval model"
  - "FL summary table: same fixed-width format as standalone training but with Round column header"
  - "FL CLI: dual-invocation pattern (argparse for CLI, kwargs for programmatic) matching standalone_train"

requirements-completed: [FLRN-03, EVAL-01]

# Metrics
duration: 5min
completed: 2026-03-09
---

# Phase 3 Plan 2: FL Training Orchestration Summary

**Complete federated training pipeline with FedAvg orchestration loop, CLI entry point, convergence verification, and metrics/checkpoint persistence**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-09T19:32:47Z
- **Completed:** 2026-03-09T19:37:33Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Full FL orchestration loop with configurable rounds/clients, per-round metrics logging, best-model checkpointing on F1
- Convergence check comparing early vs late F1 means with adaptive window
- Metrics JSON persistence with embedded config for thesis reproducibility
- CLI entry point (python -m federated_ids.fl and federated-ids-train-fl) with config overrides
- 12 FL tests (7 new), 75 total tests passing with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Orchestration loop, config banner, metrics persistence, convergence check, and checkpointing** - `04558af` (feat, TDD)
2. **Task 2: CLI entry point, console script, and config update** - `22e95bc` (feat)

## Files Created/Modified
- `src/federated_ids/fl/server.py` - Extended with run_federated_training, check_convergence, save_fl_metrics, print_config_banner, _print_fl_summary_table
- `src/federated_ids/fl/__main__.py` - CLI entry point with --config, --num-clients, --num-rounds
- `src/federated_ids/fl/__init__.py` - Exports updated with 4 new public functions
- `pyproject.toml` - Added federated-ids-train-fl console script
- `tests/test_fl.py` - 7 new tests: convergence pass/fail, metrics JSON, round keys, config banner, config-driven rounds, CLI overrides

## Decisions Made
- Fresh MLP and Adam optimizer created per client per round to prevent shared model state and stale optimizer state (research Pitfalls 6 & 2)
- Convergence check uses adaptive n: if history shorter than 2*n rounds, n = max(1, len//2)
- Metrics JSON embeds config snapshot (num_clients, num_rounds, hidden_layers, seed, etc.) for full reproducibility
- CLI overrides mutate the loaded config dict in place before passing to orchestration, matching the standalone_train dual-invocation pattern
- Suppress federated_ids.model.train logger at WARNING level during FL to reduce per-client noise

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- FL training pipeline complete: can be invoked via CLI or programmatically
- Phase 3 fully complete (both plans done)
- Ready for Phase 4 (Evaluation Framework) which will consume FL metrics and model checkpoints

---
*Phase: 03-federated-learning-infrastructure*
*Completed: 2026-03-09*
