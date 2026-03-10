# Phase 2: Model Definition and Local Training - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Define an MLP model and validate it with local training on a single client's data partition, confirming the model architecture and training loop produce reasonable DDoS detection metrics before federation. No federated learning, no multi-client coordination, no evaluation plots in this phase.

</domain>

<decisions>
## Implementation Decisions

### Training Metrics & Output
- Full metric suite per epoch: loss, accuracy, precision, recall, F1
- Use Python logging module (consistent with Phase 1 patterns)
- Format: `Epoch 3/5 — loss: 0.142, acc: 0.95, F1: 0.83, prec: 0.87, rec: 0.79`
- Print a summary table after training completes (all epochs at a glance) — easy to copy into thesis
- Per-epoch metrics computed on validation data (separate evaluation pass), not training batch metrics

### Checkpoint Strategy
- Save model weights only (state_dict) — no optimizer state, no metadata embedded in .pt file
- Save only when validation F1 improves — single `best_model.pt` file, overwritten on improvement
- Save to `outputs/checkpoints/best_model.pt` (under configured output_dir)
- Log checkpoint events to console: `Saved best model (epoch 3, F1=0.84)` — metadata in logs, not in file

### Standalone Training Mode
- Entry point: both `python -m federated_ids.model.train` and console script (pyproject.toml)
- Add `training.standalone_epochs` config key (default 5), separate from `training.local_epochs` (1) for FL
- Auto-run data pipeline if processed data doesn't exist — single command from raw data to trained model
- CLI argument `--client-id` (default 0) to select which client partition to train on

### Validation During Training
- Client-local validation split: split the selected client's partition into train/val
- Add `training.val_split` config key (default 0.2) — configurable train/val ratio within client data
- Best-model checkpoint based on client-local val F1 (methodologically sound — no test set leakage)
- After training completes, run one final evaluation on the global test set and log results separately — shows true generalization

### Claude's Discretion
- Optimizer choice (Adam expected based on config LR, but Claude can decide)
- Exact summary table formatting
- Internal training loop structure (batch iteration, gradient clipping, etc.)
- Test framework and test structure for model/training code
- How to load class weights from Phase 1 artifacts into the loss function

</decisions>

<specifics>
## Specific Ideas

- Summary table after training should be thesis-appendix ready (copy-pasteable)
- Success criteria require F1 > 0.80 within 5 epochs and attack-class recall > 0.70 — standalone_epochs default of 5 matches this
- Final global test set evaluation provides the "true" generalization metric for the thesis
- Training should reuse Phase 1's class_weights.json for the weighted cross-entropy loss

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `federated_ids/config.py`: Config loading with env var interpolation and validation — use for all config access
- `federated_ids/seed.py`: Global seed utility (numpy, random, torch, torch.cuda) — call before training
- `federated_ids/device.py`: Auto-detect CUDA/MPS/CPU — use for model and tensor placement
- `federated_ids/data/preprocess.py`: Full pipeline producing X_train.pt, y_train.pt, X_test.pt, y_test.pt, class_weights.json
- `federated_ids/data/partition.py`: IID partitioning into client DataLoaders with stratified splits

### Established Patterns
- Logging module used throughout Phase 1 (not print statements)
- Config accessed via `load_config('config/default.yaml')` returning nested dict
- Artifacts saved to configured directories (data.processed_dir, output_dir)
- Google-style docstrings on all modules and functions
- Module entry points via `__main__.py` files

### Integration Points
- Model defined here is consumed by FL clients in Phase 3 (Flower NumPyClient wraps it)
- Training loop's `train_one_epoch()` function will be called by FL clients in Phase 3
- Checkpoint saved here is loaded for evaluation in Phase 4
- Config keys added here (standalone_epochs, val_split) extend the existing config/default.yaml

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-model-definition-and-local-training*
*Context gathered: 2026-03-09*
