# Requirements: 6G Federated IDS

**Defined:** 2026-03-09
**Core Value:** Detect DDoS attacks across a federated network of edge nodes without any client ever sharing its raw network traffic data.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: Load CICIDS2017 CSV files and clean data (handle inf/NaN values, whitespace column names, constant columns)
- [x] **DATA-02**: Select and engineer features (reduce 78+ raw features to 20-40 informative ones)
- [x] **DATA-03**: Normalize features with StandardScaler fitted on training data only (no data leakage)
- [x] **DATA-04**: Handle class imbalance via weighted cross-entropy loss for DDoS minority class
- [x] **DATA-05**: Partition data IID across 2-5 clients with stratified splits maintaining class ratios

### Model

- [x] **MODL-01**: Define MLP model in PyTorch (3-layer feed-forward, ReLU, dropout, binary classification)
- [ ] **MODL-02**: Implement local PyTorch training loop with configurable hyperparameters
- [ ] **MODL-03**: Implement model checkpointing to save the best-performing global model based on F1-score during training

### Federated Learning

- [ ] **FLRN-01**: Implement Flower NumPyClient wrapping local PyTorch training
- [ ] **FLRN-02**: Implement Flower server with FedAvg aggregation strategy
- [ ] **FLRN-03**: Support configurable number of FL rounds and participating clients

### Evaluation & Visualization

- [ ] **EVAL-01**: Log per-round metrics (accuracy, precision, recall, F1) to console
- [ ] **EVAL-02**: Generate confusion matrix and classification report on held-out test set
- [ ] **EVAL-03**: Save convergence plots (loss and accuracy over FL rounds) as PNG
- [ ] **EVAL-04**: Log training metrics to TensorBoard for real-time monitoring

### Infrastructure

- [x] **INFR-01**: Configuration file (YAML/JSON) for all hyperparameters (LR, epochs, batch size, FL rounds, num clients)
- [x] **INFR-02**: Reproducibility via fixed seeds, pyproject.toml, and documented hyperparameters

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Privacy & Security

- **PRIV-01**: Differential privacy via DP-SGD (Opacus) with configurable epsilon budget
- **PRIV-02**: Secure aggregation for encrypted model updates
- **PRIV-03**: Robust aggregation (FedMedian, Krum, Trimmed Mean) against poisoned updates
- **PRIV-04**: Model poisoning attack simulation (label-flipping, gradient-scaling)

### Advanced FL

- **ADVF-01**: Non-IID data partitioning (Dirichlet-based splits with configurable alpha)
- **ADVF-02**: FedProx aggregation strategy for non-IID robustness
- **ADVF-03**: Multi-class attack detection (DDoS subtypes + additional CICIDS2017 categories)
- **ADVF-04**: Client selection strategies (partial participation with fraction_fit < 1.0)
- **ADVF-05**: Asynchronous federated learning for heterogeneous client speeds

### Evaluation Enhancements

- **EVLX-01**: Statistical significance via multi-seed experiments (mean + std)
- **EVLX-02**: Communication efficiency tracking (bytes transmitted per round)
- **EVLX-03**: MLflow experiment tracking integration

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real network traffic capture | Requires network infrastructure, introduces privacy/legal issues, data is unlabeled |
| CNN/LSTM/Autoencoder architectures | Simple MLP validates the FL concept without model complexity obscuring results |
| Web dashboard / GUI | Console + TensorBoard + saved plots sufficient for proof of concept |
| Docker containerization | Single-machine simulation sufficient for v1 |
| Multiple datasets simultaneously | Each dataset requires separate preprocessing; CICIDS2017 only for v1 |
| Automated hyperparameter tuning | Manual tuning with documented search sufficient; Optuna/Ray Tune adds complexity |
| Blockchain-based verification | Academic novelty without practical value for proof of concept |
| Real 6G hardware integration | Simulation only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| MODL-01 | Phase 2 | Complete |
| MODL-02 | Phase 2 | Pending |
| MODL-03 | Phase 2 | Pending |
| FLRN-01 | Phase 3 | Pending |
| FLRN-02 | Phase 3 | Pending |
| FLRN-03 | Phase 3 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| INFR-01 | Phase 1 | Complete |
| INFR-02 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-09 after 01-01 execution*
