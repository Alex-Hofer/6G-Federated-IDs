# Roadmap: 6G Federated IDS

## Overview

This roadmap delivers a privacy-preserving Intrusion Detection System using Federated Learning across simulated 6G edge nodes. The build follows a strict dependency chain: data pipeline first (where most pitfalls live), then model definition and local training validation, then Flower-based federated learning infrastructure, then evaluation and visualization, and finally integration and polish. Each phase delivers a verifiable capability that the next phase depends on. By the end, the system detects DDoS attacks across a federated network of edge nodes without any client sharing its raw network traffic data.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Project Foundation and Data Pipeline** - Scaffold the project and build a validated CICIDS2017 preprocessing pipeline that produces clean, normalized, partitioned DataLoaders
- [ ] **Phase 2: Model Definition and Local Training** - Define the MLP model and validate local training on a single partition with correct loss and metrics
- [ ] **Phase 3: Federated Learning Infrastructure** - Wire model and data into Flower client-server protocol with FedAvg aggregation across multiple clients
- [ ] **Phase 4: Evaluation and Visualization** - Evaluate the global federated model on held-out data and produce publication-quality plots
- [ ] **Phase 5: Integration and Polish** - Tie all components into a single runnable pipeline with orchestration, end-to-end validation, and documentation

## Phase Details

### Phase 1: Project Foundation and Data Pipeline
**Goal**: Users can run the data pipeline and get clean, normalized, partitioned PyTorch DataLoaders ready for ML training, with all project infrastructure in place
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, INFR-01, INFR-02
**Success Criteria** (what must be TRUE):
  1. Running the data pipeline on CICIDS2017 CSVs produces DataLoaders with zero Inf/NaN values and no identifier columns (IP, port, flow ID, timestamp)
  2. Features are reduced from 78+ raw columns to 20-40 informative ones, normalized with a StandardScaler fitted only on training data
  3. Data is partitioned IID across a configurable number of clients (2-5) with stratified splits that preserve class ratios, verified by printing class distributions per partition
  4. A YAML configuration file controls all hyperparameters (learning rate, epochs, batch size, FL rounds, number of clients) and fixed random seeds ensure reproducible outputs
  5. The project has a working pyproject.toml with pinned dependencies, and `pip install -e .` succeeds
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffold, config system, seed utility, device detection
- [x] 01-02-PLAN.md — Data loading, cleaning, feature selection, normalization, class weights
- [x] 01-03-PLAN.md — IID partitioning, DataLoaders, pipeline entry point, README

### Phase 2: Model Definition and Local Training
**Goal**: Users can train an MLP model on a single client's data partition and see it achieve reasonable DDoS detection metrics, confirming the model architecture and training loop work before federation
**Depends on**: Phase 1
**Requirements**: MODL-01, MODL-02, MODL-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. An MLP model (3 hidden layers, ReLU, dropout, binary output) is defined in a single model.py file and can be instantiated with configurable layer sizes
  2. Running local training on one client's data partition produces per-epoch metrics (loss, accuracy, F1) printed to console, with F1 above 0.80 on DDoS detection within 5 epochs
  3. Class-weighted cross-entropy loss is applied so that the model does not simply predict "benign" for everything (attack-class recall is above 0.70)
  4. The best model checkpoint is saved based on F1-score and can be reloaded for evaluation
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — MLP model definition (nn.Module) with configurable architecture and unit tests
- [x] 02-02-PLAN.md — Training loop, evaluation, checkpointing, standalone entry point, and tests

### Phase 3: Federated Learning Infrastructure
**Goal**: Users can run a pure-Python FedAvg federated training loop with multiple virtual clients, per-round metrics logged to console, and global model convergence demonstrated
**Depends on**: Phase 1, Phase 2
**Requirements**: FLRN-01, FLRN-02, FLRN-03, EVAL-01
**Success Criteria** (what must be TRUE):
  1. A Flower server starts and accepts connections from 2-5 client instances running simultaneously on a single machine
  2. Clients train locally on their data partitions, send only model weights (no raw data) to the server, and receive aggregated global weights back each round
  3. FedAvg aggregation completes for a configurable number of rounds (default 10-20), with per-round accuracy, precision, recall, and F1 logged to console
  4. The global model improves over rounds (F1 in later rounds is higher than F1 in early rounds, demonstrating convergence)
**Plans:** 2 plans

Plans:
- [ ] 03-01-PLAN.md — FederatedClient class, FedAvg aggregation, server-side evaluation, and unit tests
- [ ] 03-02-PLAN.md — Orchestration loop, CLI entry point, metrics persistence, checkpointing, convergence check

### Phase 4: Evaluation and Visualization
**Goal**: Users can evaluate the final federated model on held-out test data and generate publication-quality plots that demonstrate the system works
**Depends on**: Phase 3
**Requirements**: EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. A confusion matrix and full classification report (precision, recall, F1 per class) are generated by evaluating the final global model on a held-out test set that no client trained on
  2. Convergence plots (loss and F1 over FL rounds) are saved as PNG files showing training progression
  3. Per-client performance comparison is visualized, showing that all clients contribute meaningfully to the federated model
  4. TensorBoard logging captures training metrics for real-time monitoring during FL rounds
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Integration and Polish
**Goal**: Users can run the entire pipeline end-to-end with a single command and understand how to set up, configure, and reproduce the experiment
**Depends on**: Phase 3, Phase 4
**Requirements**: MODL-03, INFR-02
**Success Criteria** (what must be TRUE):
  1. A shell script or entry point launches the server and N clients, runs federated training, evaluates the global model, and saves all outputs (metrics, plots, checkpoints) to an organized directory structure
  2. Running the full pipeline from a clean environment (fresh install, data download, training, evaluation) completes without errors and produces all expected outputs
  3. A README documents setup instructions, data download steps, configuration options, and usage examples sufficient for someone unfamiliar with the project to reproduce results
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Foundation and Data Pipeline | 3/3 | Complete | 2026-03-09 |
| 2. Model Definition and Local Training | 2/2 | Complete | 2026-03-09 |
| 3. Federated Learning Infrastructure | 0/2 | Not started | - |
| 4. Evaluation and Visualization | 0/? | Not started | - |
| 5. Integration and Polish | 0/? | Not started | - |
