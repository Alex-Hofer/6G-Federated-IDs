---
status: testing
phase: 03-federated-learning-infrastructure
source: 03-01-SUMMARY.md, 03-02-SUMMARY.md
started: 2026-03-10T10:00:00Z
updated: 2026-03-10T10:00:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: All Tests Pass
expected: |
  Run `python -m pytest` from the project root. All 75 tests pass (including 12 FL tests). No failures, no errors.
awaiting: user response

## Tests

### 1. All Tests Pass
expected: Run `python -m pytest` from the project root. All 75 tests pass (including 12 FL tests). No failures, no errors.
result: [pending]

### 2. FL CLI Entry Point Works
expected: Run `python -m federated_ids.fl --help` (or `federated-ids-train-fl --help`). Shows usage info with --config, --num-clients, --num-rounds options.
result: [pending]

### 3. FL Training Runs End-to-End
expected: Run `python -m federated_ids.fl` with default config. Training executes multiple rounds, printing a per-round summary table with columns for Round, Loss, Accuracy, Precision, Recall, F1. Completes without errors.
result: [pending]

### 4. Best Model Checkpoint Saved
expected: After a training run completes, a best-model checkpoint file (.pt or similar) exists in the output directory. The checkpoint corresponds to the round with the highest F1 score.
result: [pending]

### 5. Metrics JSON Created with Config
expected: After training, a metrics JSON file is saved. Opening it shows per-round metrics AND an embedded config section containing num_clients, num_rounds, hidden_layers, seed, etc.
result: [pending]

### 6. Convergence Report Displayed
expected: At the end of training, the output includes a convergence check result (converged or not converged), comparing early vs late round F1 scores.
result: [pending]

### 7. CLI Config Overrides
expected: Run with `--num-clients 2 --num-rounds 3`. Training uses exactly 2 clients and runs for exactly 3 rounds (visible in the output table having 3 round rows).
result: [pending]

## Summary

total: 7
passed: 0
issues: 0
pending: 7
skipped: 0

## Gaps

[none yet]
