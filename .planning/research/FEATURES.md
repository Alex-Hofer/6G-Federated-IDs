# Feature Research

**Domain:** Federated Learning Intrusion Detection System for 6G Edge Networks
**Researched:** 2026-03-09
**Confidence:** MEDIUM (based on training data knowledge of FL-IDS literature, Flower framework, CICIDS2017 dataset; no live source verification available)

## Feature Landscape

### Table Stakes (Users Expect These)

These are the features without which the system cannot function as a credible FL-IDS proof of concept. Missing any of these makes the project feel incomplete or unvalidatable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **CICIDS2017 data loading and cleaning** | Standard benchmark dataset for IDS research; any FL-IDS paper uses a recognized dataset | Low | Must handle infinity/NaN values, drop constant columns, parse timestamps. CICIDS2017 has known data quality issues (inf values in flow bytes/s, flow packets/s). |
| **Feature selection and engineering** | Raw CICIDS2017 has 78+ features, many redundant or noisy; models need curated input | Medium | Remove highly correlated features (>0.95 threshold). Standard practice: keep 20-40 most informative features via mutual information or variance thresholding. |
| **Data normalization / standardization** | Neural networks require normalized inputs; FL requires consistent preprocessing across clients | Low | StandardScaler or MinMaxScaler. Must fit scaler on training data only (no data leakage). In FL context: either use global statistics or per-client normalization with documented approach. |
| **Binary classification (benign vs DDoS)** | Core IDS function -- detect attacks | Low | v1 scope per PROJECT.md. Simpler than multi-class, validates the FL pipeline cleanly. |
| **IID data partitioning across clients** | Simulates federated data distribution; required to test FL pipeline correctness | Low | Random stratified split maintaining class ratios across 2-5 clients. Must preserve train/test split boundaries. |
| **MLP model definition (PyTorch)** | The local model each client trains; MLP is the stated architecture | Low | Input layer matching feature count, 2-3 hidden layers with ReLU, output layer with sigmoid (binary) or softmax. Dropout for regularization. |
| **Local client training loop** | Each client must train on its own data partition | Medium | Standard PyTorch training: forward pass, loss computation (BCE or CrossEntropy), backpropagation, optimizer step. Must handle class imbalance (DDoS is minority class in CICIDS2017). |
| **Flower client implementation** | Integrates local training with FL framework; handles get_parameters, fit, evaluate | Medium | Implement flwr.client.NumPyClient. Must correctly serialize/deserialize PyTorch state_dict to/from NumPy arrays. |
| **Flower server with FedAvg** | Central aggregation server orchestrating training rounds | Low | flwr.server.start_server with FedAvg strategy. Configure min_fit_clients, min_evaluate_clients, num_rounds. |
| **Per-round federated metrics** | Must show learning progress across FL rounds | Medium | Accuracy, precision, recall, F1-score per round. Flower supports server-side metric aggregation via evaluate_metrics_aggregation_fn. |
| **Final model evaluation** | Validate the global model's detection capability | Medium | Evaluate aggregated global model on held-out test set. Report accuracy, precision, recall, F1, and confusion matrix. |
| **Confusion matrix visualization** | Standard IDS evaluation artifact; shows TP/FP/TN/FN breakdown | Low | Matplotlib/seaborn heatmap. Critical for understanding false positive rate (key IDS metric). |
| **Training convergence plots** | Show that federated training actually converges | Low | Loss and accuracy over FL rounds. Matplotlib line plots. Essential to demonstrate FL is working. |
| **Reproducibility controls** | Research credibility requires reproducible results | Low | Fixed random seeds (Python, NumPy, PyTorch, CUDA), documented hyperparameters, requirements.txt/pyproject.toml with pinned versions. |
| **Class imbalance handling** | CICIDS2017 DDoS flows are minority class (~3-8% depending on day files used); naive training yields degenerate models | Medium | Weighted loss function (class weights inversely proportional to frequency) or oversampling (SMOTE). Weighted loss is simpler and preferred in FL since SMOTE creates synthetic samples. |

### Differentiators (Competitive Advantage / Research Value)

These elevate the project from "basic tutorial" to "credible research contribution." Not expected in every FL-IDS system but significantly increase value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Non-IID data partitioning** | Real 6G edge nodes see different traffic patterns; IID is unrealistic. Non-IID tests robustness of FL approach. | Medium | Dirichlet distribution (alpha parameter controls heterogeneity) or label-skew partitioning. PROJECT.md defers this to after v1, which is correct -- validate IID first. |
| **FedProx aggregation strategy** | Handles statistical heterogeneity (non-IID data) better than FedAvg by adding proximal term to local loss | Low | Flower has built-in FedProx strategy. Only requires setting proximal_mu parameter. Makes sense when non-IID is introduced. |
| **Robust aggregation (FedMedian, Krum, Trimmed Mean)** | Defends against poisoned model updates from compromised edge nodes -- critical for security-focused IDS | Medium | Flower provides FedMedian and FedTrimmedAvg. Krum may need custom implementation. PROJECT.md defers this to v2. |
| **Differential privacy (DP-SGD)** | Formal privacy guarantee beyond "raw data stays local"; protects against model inversion and membership inference | High | Flower has experimental DP support. Alternatively use Opacus (PyTorch DP library). Adds noise to gradients, requires privacy budget (epsilon) tracking. Significant accuracy tradeoff. |
| **Multi-class attack detection** | Detect DDoS subtypes or additional attack categories (brute force, botnet, infiltration, web attacks) | Medium | CICIDS2017 supports 14 attack types. Requires multi-class output layer, per-class metrics, more complex evaluation. Significantly broadens detection capability. |
| **Per-client performance dashboard** | Visualize how each client performs individually vs the global model; detect struggling clients | Medium | Per-client accuracy/F1 tracked across rounds. Reveals if specific data partitions are problematic. Matplotlib subplots or simple web dashboard. |
| **Model poisoning attack simulation** | Inject malicious model updates to test system resilience; validates need for robust aggregation | High | Implement label-flipping or gradient-scaling attacks on subset of clients. Measure impact on global model. Pairs with robust aggregation as the defense. |
| **Communication efficiency metrics** | Track bytes transmitted per round; critical for bandwidth-constrained 6G edge | Low | Log model parameter size, number of rounds to convergence. Compare compression approaches if implemented. |
| **Model compression / quantization** | Reduce model size for edge deployment and lower communication overhead | High | Pruning, quantization (INT8), knowledge distillation. PyTorch supports post-training quantization. Relevant for 6G edge resource constraints. |
| **Secure aggregation protocol** | Cryptographic guarantee that server cannot inspect individual model updates | High | Flower has experimental SecAgg support. Complex to implement correctly. PROJECT.md correctly defers this. |
| **Automated hyperparameter reporting** | Log all hyperparameters (learning rate, batch size, epochs, FL rounds, clients) in structured format | Low | JSON/YAML config file consumed by both server and client. Makes experiments reproducible and comparable. |
| **Experiment tracking integration** | MLflow, Weights & Biases, or TensorBoard integration for tracking experiments across runs | Medium | Flower has built-in logging support. MLflow is lightweight and self-hosted. Adds significant value for research iteration. |
| **Statistical significance testing** | Run multiple seeds, report mean +/- std for all metrics | Low | Run 3-5 seeds, compute confidence intervals. Transforms single-run demo into credible research result. |
| **Client selection strategies** | Not all clients participate every round; simulate realistic partial participation | Low | Flower supports fraction_fit and fraction_evaluate parameters. Random client selection is default. More sophisticated: select based on data quality or loss. |
| **Gradient compression** | Reduce communication cost by compressing gradient updates (top-k sparsification, quantization) | High | Custom Flower strategy needed. Significant implementation effort but directly relevant to 6G bandwidth constraints. |
| **Asynchronous FL support** | Clients update at different rates (realistic for heterogeneous 6G edge devices) | High | Flower primarily supports synchronous FL. Async requires custom server logic or alternative frameworks. Adds realism but significant complexity. |

### Anti-Features (Commonly Requested, Often Problematic)

Things that seem like good ideas but should be deliberately excluded from v1 (and possibly v2) because they add complexity without proportional value, or are actively harmful to the project's goals.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Real network traffic capture** | "Use real data for realism" | Requires network infrastructure, introduces privacy/legal issues, data is unlabeled and noisy. Derails focus from FL pipeline to data engineering. | Use CICIDS2017 benchmark. It is real captured traffic, just pre-labeled. Sufficient for proof of concept. |
| **Deep learning architectures (CNN/LSTM/Transformer)** | "Better detection accuracy" | Adds model complexity that obscures whether FL works. Harder to debug federated training. Longer training times on single machine. | MLP first. If FL pipeline works with MLP, swapping in deeper models is straightforward later. The FL infrastructure is architecture-agnostic. |
| **Web-based dashboard UI** | "Better visualization and monitoring" | Significant frontend engineering effort (Flask/FastAPI + React/Streamlit). Distracts from core ML/FL work. | Matplotlib plots saved to disk + console logging. For v2, consider Streamlit (minimal effort) or Flower's built-in monitoring. |
| **Docker/Kubernetes deployment** | "Production-ready deployment" | Massive infrastructure complexity. Single-machine simulation is the stated scope. Container orchestration is orthogonal to FL-IDS research. | Python virtual environment + clear README. Docker can wrap the final product trivially once it works. |
| **Real-time inference pipeline** | "Deploy as actual IDS" | Requires streaming data pipeline, low-latency serving, network tap integration. Completely different system from training pipeline. | Batch evaluation on test set demonstrates detection capability. Real-time serving is a deployment concern, not a research concern. |
| **Custom FL framework** | "Flower is too limiting" | Massive implementation effort to rebuild client-server communication, serialization, aggregation. Flower handles this well. | Use Flower. It supports custom strategies, custom serialization, and pluggable components. Extend, don't replace. |
| **GAN-based data augmentation** | "Generate synthetic attack traffic" | Complex to train, generated data quality is hard to validate, adds significant scope. Doesn't help validate FL pipeline. | Use existing CICIDS2017 attack samples. Apply simpler augmentation (SMOTE) if class imbalance is severe. |
| **Blockchain-based model verification** | "Ensure model integrity in federated setting" | Enormous complexity, minimal practical benefit for proof of concept, performance overhead. Popular in papers but rarely implemented usefully. | Trust model is single-machine simulation. If integrity is needed later, hash model updates with standard cryptography. |
| **AutoML / Neural Architecture Search** | "Find optimal model automatically" | Computationally expensive, adds framework dependencies (Optuna/Ray Tune), distracts from FL focus. | Manual hyperparameter tuning with documented search. Grid search over 2-3 key parameters is sufficient. |
| **Support for multiple datasets simultaneously** | "Show generalizability" | Each dataset requires separate preprocessing pipeline, different feature spaces, different class definitions. Multiplies work without deepening FL understanding. | Use CICIDS2017 only for v1. Design preprocessing as a module so swapping datasets later is possible, but don't build multi-dataset support upfront. |

## Feature Dependencies

```
Data Loading & Cleaning
  |
  v
Feature Selection & Engineering
  |
  v
Data Normalization
  |
  +---> IID Data Partitioning ---> Non-IID Partitioning (v2)
  |
  v
Class Imbalance Handling
  |
  v
MLP Model Definition
  |
  +---> Local Client Training Loop
  |       |
  |       v
  |     Flower Client Implementation
  |       |
  |       v
  |     Flower Server (FedAvg) ---------> Robust Aggregation (v2)
  |       |                                   |
  |       |                                   v
  |       |                             Model Poisoning Simulation (v2)
  |       |
  |       +---> FedProx (after non-IID)
  |       |
  |       v
  |     Per-Round Metrics
  |       |
  |       v
  |     Training Convergence Plots
  |       |
  |       v
  |     Per-Client Performance Dashboard (v1.x)
  |
  v
Final Model Evaluation
  |
  +---> Confusion Matrix
  |
  +---> Statistical Significance (multiple seeds) (v1.x)

Reproducibility Controls (independent -- implement from day 1)

Automated Hyperparameter Reporting (independent -- implement from day 1)

Communication Efficiency Metrics (independent -- add anytime after FL runs)

Differential Privacy (requires: local training loop + Opacus integration) (v2)

Secure Aggregation (requires: Flower server) (v2+)
```

### Key Dependency Insights

1. **Data pipeline must come first.** Everything downstream depends on clean, normalized, partitioned data. Get this right or nothing else works.
2. **Flower client/server are the integration point.** Local training and data pipeline feed into it; metrics and evaluation flow out of it.
3. **Non-IID unlocks FedProx.** There is no reason to use FedProx with IID data. Implement non-IID partitioning before switching aggregation strategies.
4. **Robust aggregation and poisoning simulation are paired.** Building attack simulation without defense (or vice versa) demonstrates a problem without a solution.
5. **Differential privacy is independent of aggregation.** DP operates at the client level (gradient noise), while robust aggregation operates at the server level. They can be developed independently.

## MVP Definition

### Launch With (v1)

The minimum system that demonstrates "federated learning for DDoS detection works and preserves privacy." Aligned with PROJECT.md active requirements.

1. **CICIDS2017 data pipeline** -- load, clean (handle inf/NaN), select features, normalize, handle class imbalance with weighted loss
2. **IID data partitioning** -- stratified random split across 2-5 simulated clients
3. **MLP model** -- 3-layer feed-forward network with dropout, binary classification (benign vs DDoS)
4. **Flower client** -- NumPyClient implementation wrapping PyTorch training loop
5. **Flower server** -- FedAvg strategy, configurable number of rounds and clients
6. **Per-round metrics** -- accuracy, precision, recall, F1 logged to console each round
7. **Final evaluation** -- confusion matrix + classification report on held-out test set
8. **Convergence plots** -- loss and accuracy curves over FL rounds, saved as PNG
9. **Per-client performance plots** -- per-client metrics across rounds, saved as PNG
10. **Reproducibility** -- fixed seeds, requirements.txt, documented hyperparameters
11. **Configuration file** -- YAML/JSON config for all hyperparameters (learning rate, epochs, batch size, FL rounds, number of clients)

**v1 success criteria:** Global model achieves F1 > 0.90 on DDoS detection after federated training converges within 10-20 rounds. All metrics visible in console output and saved plots.

### Add After Validation (v1.x)

Features that enhance the system once v1 proves the FL pipeline works. Each can be added independently.

1. **Non-IID data partitioning** -- Dirichlet-based splits with configurable heterogeneity (alpha parameter). Tests robustness of FL approach.
2. **FedProx strategy** -- Drop-in replacement via Flower strategy parameter. Only meaningful with non-IID data.
3. **Multi-class attack detection** -- Expand to detect DDoS subtypes or additional CICIDS2017 attack categories. Requires output layer change and per-class evaluation.
4. **Statistical significance** -- Run experiments across 3-5 random seeds, report mean and standard deviation for all metrics.
5. **Communication efficiency tracking** -- Log model size in bytes, total bytes transmitted per round, rounds to convergence.
6. **Experiment tracking** -- MLflow integration for comparing runs across hyperparameter configurations.
7. **Per-client performance dashboard** -- Detailed per-client metrics visualization showing individual learning curves and data distribution statistics.
8. **Client selection strategies** -- Partial participation (fraction_fit < 1.0) to simulate realistic scenarios where not all edge nodes are available.

### Future Consideration (v2+)

Features that represent significant scope expansion. Each is a research contribution in itself.

1. **Robust aggregation** (FedMedian, Krum, Trimmed Mean) -- Defense against Byzantine/poisoned clients
2. **Model poisoning attack simulation** -- Label-flipping and gradient-scaling attacks to test robustness
3. **Differential privacy** (DP-SGD via Opacus) -- Formal privacy guarantees with epsilon tracking
4. **Secure aggregation** -- Cryptographic protection of model updates during transmission
5. **Model compression/quantization** -- Edge deployment optimization for resource-constrained 6G devices
6. **Gradient compression** -- Top-k sparsification or quantized gradients for bandwidth efficiency
7. **Asynchronous federated learning** -- Handle heterogeneous client speeds
8. **Additional datasets** -- NSL-KDD, UNSW-NB15, or custom 5G/6G traffic captures for generalizability

## Feature Prioritization Matrix

| Feature | Impact | Effort | Risk | Priority | Phase |
|---------|--------|--------|------|----------|-------|
| Data pipeline (load, clean, features) | Critical | Low | Low | P0 | v1 |
| Data normalization | Critical | Low | Low | P0 | v1 |
| Class imbalance handling | High | Low | Low | P0 | v1 |
| IID partitioning | Critical | Low | Low | P0 | v1 |
| MLP model definition | Critical | Low | Low | P0 | v1 |
| Local training loop | Critical | Medium | Medium | P0 | v1 |
| Flower client | Critical | Medium | Medium | P0 | v1 |
| Flower server + FedAvg | Critical | Low | Low | P0 | v1 |
| Per-round metrics | High | Medium | Low | P0 | v1 |
| Final evaluation + confusion matrix | High | Low | Low | P0 | v1 |
| Convergence plots | High | Low | Low | P0 | v1 |
| Per-client performance plots | Medium | Low | Low | P0 | v1 |
| Reproducibility controls | High | Low | Low | P0 | v1 |
| Config file for hyperparameters | Medium | Low | Low | P0 | v1 |
| Non-IID partitioning | High | Medium | Medium | P1 | v1.x |
| FedProx | Medium | Low | Low | P1 | v1.x |
| Multi-class detection | Medium | Medium | Low | P1 | v1.x |
| Statistical significance (multi-seed) | Medium | Low | Low | P1 | v1.x |
| Communication metrics | Low | Low | Low | P1 | v1.x |
| Experiment tracking (MLflow) | Medium | Medium | Low | P2 | v1.x |
| Robust aggregation | High | Medium | Medium | P2 | v2 |
| Poisoning simulation | High | High | Medium | P2 | v2 |
| Differential privacy | Medium | High | High | P3 | v2 |
| Secure aggregation | Low | High | High | P3 | v2+ |
| Model compression | Medium | High | Medium | P3 | v2+ |

## Key Implementation Notes

### CICIDS2017 Preprocessing Specifics

CICIDS2017 has well-documented data quality issues that must be handled:

1. **Infinity values**: `Flow Bytes/s` and `Flow Packets/s` columns contain `inf` values. Replace with column max or drop rows.
2. **NaN values**: Several features have NaN entries. Drop rows or impute with median.
3. **Constant columns**: Some features have zero variance. Drop them before normalization.
4. **Timestamp column**: `Timestamp` must be dropped (not a useful feature, causes data leakage if day-files are mixed).
5. **Label encoding**: Map `BENIGN` to 0, all DDoS variants to 1 (binary) or individual class labels (multi-class).
6. **Feature count**: After cleaning, expect approximately 40-50 usable features. Further reduction via correlation analysis or feature importance brings this to 20-30.
7. **Day file selection**: DDoS attacks are in the Friday data file (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`). Other day files contain other attack types.

### Flower Integration Specifics

1. **NumPyClient interface**: Implement `get_parameters()`, `fit()`, `evaluate()`. The `fit()` method receives global model parameters and returns updated local parameters plus the number of training samples.
2. **Parameter conversion**: PyTorch state_dict values must be converted to/from NumPy arrays: `[val.cpu().numpy() for val in model.state_dict().values()]`.
3. **Metric aggregation**: Use `evaluate_metrics_aggregation_fn` in strategy config to aggregate per-client metrics (weighted average by sample count).
4. **Server config**: `ServerConfig(num_rounds=N)` controls training rounds. Strategies accept `min_fit_clients`, `min_evaluate_clients`, `min_available_clients`.

### Class Imbalance Strategy

For v1, use **weighted cross-entropy loss** rather than resampling:
- Compute class weights from training data: `weight = total_samples / (num_classes * class_count)`
- Pass to `torch.nn.CrossEntropyLoss(weight=class_weights)`
- This works naturally in FL (each client computes local weights from its partition)
- SMOTE is problematic in FL because synthetic samples may not represent the client's true data distribution

## Confidence Notes

| Area | Confidence | Rationale |
|------|------------|-----------|
| CICIDS2017 preprocessing | HIGH | Well-documented dataset with extensive literature on known issues and standard preprocessing steps |
| MLP architecture for IDS | HIGH | Well-established baseline in IDS literature; PROJECT.md correctly identifies it as sufficient for proof of concept |
| Flower framework capabilities | MEDIUM | Based on training data up to early 2025; Flower evolves rapidly and API may have changed. Verify current API against Flower docs before implementation. |
| FedAvg for IID baseline | HIGH | Standard, well-understood; correct choice for v1 |
| Class imbalance in CICIDS2017 | HIGH | Extensively documented; DDoS is minority class requiring weighted loss |
| Differential privacy via Opacus | MEDIUM | Opacus exists and integrates with PyTorch, but Flower+Opacus integration may require custom work. Verify compatibility. |
| Robust aggregation in Flower | MEDIUM | FedMedian and FedTrimmedAvg were available in Flower as of early 2025. Krum may need custom strategy. Verify current strategy list. |
| Secure aggregation in Flower | LOW | Was experimental/in-development as of early 2025. Status may have changed significantly. |
| Non-IID Dirichlet partitioning | HIGH | Standard technique, well-documented in FL literature, straightforward implementation |

---
*Feature research for: Federated Learning IDS for 6G Edge Networks*
*Researched: 2026-03-09*
*Sources: Training data knowledge of FL-IDS literature, CICIDS2017 dataset documentation, Flower framework (up to early 2025). No live web verification was possible during this research session. Flower API details and version-specific features should be verified against current documentation before implementation.*
