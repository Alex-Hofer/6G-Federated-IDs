# Phase 3: Federated Learning Infrastructure - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire model and data into Flower client-server protocol with FedAvg aggregation across multiple clients. Clients train locally on their data partitions, send only model weights to the server, and receive aggregated global weights back each round. Per-round metrics are logged to console. No evaluation plots, no TensorBoard, no end-to-end orchestration in this phase.

</domain>

<decisions>
## Implementation Decisions

### Flower API & Simulation Mode
- Use Flower's `start_simulation` engine — single-process, no socket/port conflicts
- Virtual clients avoid memory overhead of separate processes
- Client logic (NumPyClient subclass) must be modular and mirror what would be deployed on a real 6G edge node — clean separation so it could be swapped to real client-server later
- All clients share the same device (CPU or single GPU) — Flower simulation handles client sequencing internally

### Aggregation Strategy
- Standard FedAvg weighted by dataset size (num_examples) — Flower's default behavior
- Simulates 6G edge environment where different nodes capture varying amounts of network traffic
- No custom aggregation function needed — use FedAvg out of the box
- `fraction_fit: 1.0` — all clients participate every round (configurable via config)

### Entry Point & CLI
- Dual entry point: `python -m federated_ids.fl` AND console_scripts entry point (e.g., `federated-ids-train-fl`)
- Consistent with Phase 1 (data pipeline) and Phase 2 (standalone training) patterns
- CLI arguments `--config`, `--num-clients`, `--num-rounds` override config.yaml values
- CLI overrides essential for thesis workflow: quick debugging runs without modifying baseline config
- Auto-run data pipeline if processed data doesn't exist — single command from raw data to federated training

### Start Banner
- Print ASCII config summary banner before training starts: clients, rounds, local epochs, model architecture, device, strategy
- Acts as sanity check for thesis experiments and provides easy-to-copy metadata for research logs

### Per-Round Metrics Output (EVAL-01)
- Server-side evaluation: after each round, evaluate global model on held-out test set
- One-line per round format: `Round  1/20 — loss: 0.412, acc: 0.88, F1: 0.72, prec: 0.81, rec: 0.65`
- Summary table after training completes — all rounds at a glance, thesis-appendix ready
- Global metrics only — suppress individual client training logs for clean console output
- Reflects privacy-preserving nature: focus on global model convergence, not per-client internals

### Global Model Checkpoint
- Save best global model based on highest F1 across all rounds (not always final round)
- Save to `outputs/checkpoints/global_model.pt` — state_dict only, consistent with Phase 2
- Log checkpoint events: `Saved global model (round 15, F1=0.94)`
- Phase 4 evaluation loads this checkpoint for comparison against Phase 2 local baseline

### Convergence Tracking & Verification
- Save all per-round metrics to JSON: `outputs/metrics/fl_metrics.json`
- JSON includes embedded config (num_clients, num_rounds, local_epochs, strategy, model, device) for full reproducibility
- Automated convergence check after training: compare avg F1 of first 3 rounds vs last 3 rounds, print pass/fail
- This is the formal success criterion: federated training must demonstrably improve F1 over rounds

### Claude's Discretion
- Exact NumPyClient implementation (get_parameters, set_parameters, fit, evaluate methods)
- How to structure server-side evaluate function within FedAvg strategy
- Internal parameter serialization (ndarray <-> state_dict conversion)
- Test structure and test cases for FL infrastructure
- Exact summary table formatting library or approach
- How to suppress Flower's default logging to keep console clean

</decisions>

<specifics>
## Specific Ideas

- Config summary banner format should match the preview: clean ASCII box with aligned key-value pairs
- Per-round metric format must exactly match Phase 2's per-epoch format for consistency across thesis
- Metrics JSON is the primary data source for Phase 4 convergence plots (F1 vs Rounds)
- Best-model checkpointing mirrors Phase 2's approach but at the round level instead of epoch level
- The FL entry point should feel identical to standalone training from a user perspective — same patterns, same auto-pipeline behavior

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `federated_ids/model/train.py`: `train_one_epoch()` and `evaluate()` — FL clients wrap these directly
- `federated_ids/model/model.py`: MLP model definition — instantiated per-client in `client_fn`
- `federated_ids/data/partition.py`: IID partitioning — provides per-client DataLoaders
- `federated_ids/config.py`: Config loading with validation — use for all FL config access
- `federated_ids/seed.py`: Global seed utility — call before simulation starts
- `federated_ids/device.py`: Auto-detect CUDA/MPS/CPU — shared device for all clients

### Established Patterns
- Logging module used throughout (not print statements)
- Config accessed via `load_config('config/default.yaml')` returning nested dict
- Module entry points via `__main__.py` files
- Console scripts registered in pyproject.toml
- Google-style docstrings on all modules and functions
- Artifacts saved to configured output directories

### Integration Points
- `fl/` subpackage exists but is empty — ready for `client.py`, `server.py`, `__main__.py`
- Config already has `federation` section with `num_clients`, `num_rounds`, `fraction_fit`
- `data/processed/` artifacts (tensors, class_weights.json, scaler) consumed by FL pipeline
- Global test set (`X_test.pt`, `y_test.pt`) used for server-side evaluation
- `outputs/checkpoints/` directory used by Phase 2 — FL adds `global_model.pt` alongside `best_model.pt`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-federated-learning-infrastructure*
*Context gathered: 2026-03-09*
