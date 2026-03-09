# Project Research Summary

**Project:** 6G Federated IDS
**Domain:** Privacy-Preserving Federated Learning for Network Intrusion Detection
**Researched:** 2026-03-09
**Confidence:** MEDIUM

## Executive Summary

This project is a federated learning intrusion detection system (FL-IDS) that trains MLP classifiers across simulated 6G edge nodes to detect DDoS attacks, using the CICIDS2017 benchmark dataset. The expert-recommended approach is well-established in academic literature: use Flower as the FL framework (it handles all client-server gRPC communication and FedAvg aggregation), PyTorch for the MLP model, and Pandas/Scikit-learn for the notoriously tricky CICIDS2017 preprocessing pipeline. The technology choices are mature and interoperate cleanly. Python 3.11, Flower 1.x (classic API), and CPU-only PyTorch are the right choices for a proof-of-concept of this scale.

The recommended build approach is strictly dependency-ordered: data pipeline first, then model definition, then FL infrastructure, then evaluation and visualization. This ordering is not arbitrary -- seven of ten identified critical pitfalls originate in the data preprocessing phase, and every downstream component depends on clean, correctly partitioned data. The CICIDS2017 dataset has well-documented but easily overlooked data quality issues (infinity values, NaN entries, inconsistent column names, severe class imbalance) that will silently corrupt the entire FL pipeline if not addressed upfront with explicit validation assertions.

The key risks are (1) data quality issues in CICIDS2017 producing silent model corruption or misleading metrics, (2) class imbalance causing inflated accuracy that masks poor attack detection, and (3) Flower API version drift between tutorial code and the installed version. All three are mitigable with disciplined preprocessing, proper metric selection (F1/recall over accuracy), and version pinning. The project scope as defined in PROJECT.md is well-calibrated -- it correctly defers robust aggregation, differential privacy, non-IID splits, and complex architectures to future versions, keeping v1 focused on proving that the federated pipeline works.

## Key Findings

### Recommended Stack

The stack centers on Python 3.11 with Flower (flwr) for federated learning orchestration, PyTorch for model training, and Pandas/Scikit-learn for data preprocessing. All components have verified compatibility. CPU-only PyTorch is sufficient for MLP training on tabular data. Development tooling is modern and minimal: pyproject.toml for packaging, ruff for linting/formatting, pytest for testing, and YAML for configuration. See `.planning/research/STACK.md` for full details.

**Core technologies:**
- **Flower (flwr >=1.12):** FL framework -- handles gRPC communication, FedAvg aggregation, and federated evaluation. Use the stable 1.x "classic" API (NumPyClient, start_server), not the newer ServerApp/ClientApp API which has less tutorial coverage.
- **PyTorch (torch >=2.4):** Neural network training -- industry standard, CPU-only install sufficient. Provides nn.Module for MLP, DataLoader for batching, and standard training loop primitives.
- **Pandas (>=2.2) + Scikit-learn (>=1.5):** Data pipeline -- Pandas for CSV loading and CICIDS2017 cleaning, Scikit-learn for StandardScaler normalization, train/test splitting, and all evaluation metrics (F1, precision, recall, confusion matrix).
- **Matplotlib (>=3.9) + Seaborn (>=0.13):** Visualization -- convergence plots and confusion matrix heatmaps, saved as static PNGs.
- **PyYAML (>=6.0):** Configuration -- centralized hyperparameter management in a single YAML file rather than scattered constants.

### Expected Features

The feature landscape is well-defined with clear v1/v1.x/v2 boundaries. See `.planning/research/FEATURES.md` for the full prioritization matrix and dependency graph.

**Must have (table stakes for v1):**
- CICIDS2017 data loading, cleaning (Inf/NaN handling), and feature engineering
- Data normalization with global StandardScaler statistics
- Class imbalance handling via weighted cross-entropy loss
- IID stratified data partitioning across 2-5 simulated clients
- MLP model definition (3 hidden layers, ReLU, dropout, binary output)
- Local PyTorch training loop with BCE/CrossEntropy loss
- Flower NumPyClient implementation (get_parameters, fit, evaluate)
- Flower server with FedAvg strategy and configurable rounds
- Per-round federated metrics (accuracy, precision, recall, F1)
- Final global model evaluation with confusion matrix and classification report
- Training convergence plots and per-client performance comparison plots
- Reproducibility controls (fixed seeds for Python, NumPy, PyTorch)
- YAML configuration file for all hyperparameters

**Should have (v1.x differentiators):**
- Non-IID data partitioning (Dirichlet distribution)
- FedProx aggregation strategy (pairs with non-IID)
- Multi-class attack detection (expand beyond DDoS)
- Statistical significance testing (multi-seed runs)
- Communication efficiency metrics

**Defer (v2+):**
- Robust aggregation (FedMedian, Krum) and poisoning simulation
- Differential privacy (DP-SGD via Opacus)
- Secure aggregation, model compression, gradient compression
- Asynchronous FL, additional datasets

### Architecture Approach

The architecture follows a clean component separation pattern with 7 modules in a flat `src/` directory, each mapping to exactly one responsibility. The system has four distinct phases: data preparation (runs once), client initialization (per-client), federated training (repeats for R rounds), and post-training evaluation (runs once). Raw data never crosses the wire -- only model parameter tensors (50-200 KB per round for a small MLP) are transmitted via Flower's gRPC layer. See `.planning/research/ARCHITECTURE.md` for complete data flow diagrams.

**Major components:**
1. **Data Pipeline** (`data_utils.py`) -- CSV loading, Inf/NaN cleaning, feature selection, global normalization, IID partitioning into client shards, PyTorch DataLoader creation
2. **Model Definition** (`model.py`) -- MLP architecture as nn.Module subclass; shared by client, server, and evaluation (single source of truth to prevent parameter shape mismatch)
3. **FL Client** (`client.py`) -- Flower NumPyClient subclass; receives global weights, trains locally, returns updated weights and metrics
4. **FL Server** (`server.py`) -- FedAvg strategy configuration, round management, metric aggregation via weighted averaging
5. **Evaluation** (`evaluate.py`) -- Final global model assessment on held-out test set; produces accuracy, precision, recall, F1, confusion matrix
6. **Visualization** (`visualize.py`) -- Convergence curves, confusion matrix heatmaps, per-client comparisons; saves to `outputs/plots/`
7. **Configuration** (`config.py`) -- Single source of truth for all hyperparameters, feature column lists, file paths, and random seeds

### Critical Pitfalls

The top 5 pitfalls, synthesized from 10 identified in `.planning/research/PITFALLS.md`:

1. **CICIDS2017 Inf/NaN values crash training silently** -- Replace infinities with NaN, then drop or impute NaN rows immediately after CSV loading, before any other processing. Assert zero Inf/NaN values remain. This is the single most common first-time failure with this dataset.
2. **Class imbalance inflates accuracy to meaningless levels** -- A naive model predicting "benign" for everything scores 80%+ accuracy. Use weighted cross-entropy loss (weights inversely proportional to class frequency) and report F1/recall for the attack class as primary metrics, not accuracy.
3. **Data leakage through identifier columns and scaler fitting** -- Remove Source IP, Destination IP, Source Port, Destination Port, Flow ID, and Timestamp before training. Fit StandardScaler on training data only, never on the full dataset. Compute global scaler statistics before client partitioning to ensure consistent feature spaces.
4. **Flower parameter shape mismatch between client and server** -- Define the model in a single `model.py` file imported everywhere. Use `strict=True` in `load_state_dict`. Never duplicate the model class definition across files. Log parameter shapes in the first round to verify alignment.
5. **Evaluating the global model on client training data instead of held-out data** -- Hold out a global test set before client partitioning. Each client also maintains a separate local test set for per-round evaluation. Final evaluation uses the global held-out set that no client ever trained on.

## Implications for Roadmap

Based on research, the project naturally decomposes into 5 phases following a strict dependency chain. The data pipeline must be validated before FL infrastructure is built, and FL infrastructure must work before evaluation and visualization can be meaningful.

### Phase 1: Project Setup and Data Pipeline
**Rationale:** Seven of ten critical pitfalls originate in data preprocessing. Everything downstream depends on clean, correctly partitioned data. This phase also pins tool versions to prevent API incompatibilities (Pitfall 9).
**Delivers:** A fully validated CICIDS2017 preprocessing pipeline that produces clean, normalized, partitioned PyTorch DataLoaders ready for FL training. Also delivers project scaffolding (pyproject.toml, config.py, .gitignore, directory structure).
**Addresses:** Data loading/cleaning, feature engineering, normalization, class imbalance handling, IID partitioning, reproducibility controls, configuration file.
**Avoids:** Pitfalls 1 (Inf/NaN), 2 (class imbalance), 3 (data leakage), 6 (column name inconsistency), 8 (missing DDoS files), 9 (Flower version pinning), 10 (scaling inconsistency).

### Phase 2: Model Definition and Local Training
**Rationale:** The MLP model is the most-depended-upon module (imported by client, server, and evaluation). It must be correct and stable before FL integration. Local training loop validation in isolation catches bugs before they are obscured by the federated layer.
**Delivers:** A working MLP model with a validated local training loop that achieves reasonable metrics on a single client's data partition. Confirms the model architecture, loss function, optimizer, and class weighting work correctly.
**Addresses:** MLP model definition, local client training loop, class imbalance via weighted loss.
**Avoids:** Pitfall 4 (parameter shape mismatch -- establish the single model.py pattern here).

### Phase 3: Federated Learning Infrastructure
**Rationale:** With data pipeline and model validated independently, this phase integrates them through Flower's client-server protocol. This is the core technical challenge and the project's reason for existing. Building it on validated components isolates FL-specific issues from data/model bugs.
**Delivers:** A working Flower server with FedAvg and 2-5 connected clients that complete multiple training rounds. Per-round metrics logged to console. Global model weights correctly aggregated and redistributed.
**Addresses:** Flower client implementation (NumPyClient), Flower server with FedAvg, per-round federated metrics, multi-client orchestration.
**Avoids:** Pitfall 4 (parameter mismatch -- single model.py), Pitfall 7 (non-convergence -- start with conservative hyperparameters: E=1, lr=0.001, 10-20 rounds).

### Phase 4: Evaluation and Visualization
**Rationale:** Meaningful evaluation requires a working federated training pipeline. This phase adds the artifacts that demonstrate the system works: confusion matrix, convergence plots, per-client comparisons, classification report. Without Phase 3 complete, there is nothing to evaluate.
**Delivers:** Final global model evaluation on held-out test set with full metric suite. Publication-quality plots saved as PNGs. Per-client performance comparison. Complete console output showing training progression and final results.
**Addresses:** Final model evaluation, confusion matrix visualization, training convergence plots, per-client performance plots.
**Avoids:** Pitfall 2 (report F1/recall, not accuracy), Pitfall 5 (evaluate on held-out data, not training data).

### Phase 5: Integration and Polish
**Rationale:** With all components working, this phase ties everything together into a runnable system with clear entry points, documentation, and end-to-end orchestration.
**Delivers:** Shell script to launch server + N clients. End-to-end test confirming the full pipeline runs from data loading through evaluation and plot generation. README with setup instructions, data download guide, and usage examples.
**Addresses:** Orchestration script, end-to-end integration testing, documentation, experiment reproducibility.
**Avoids:** Technical debt patterns identified in PITFALLS.md (hardcoded paths, missing seeds, undocumented setup).

### Phase Ordering Rationale

- **Data first (Phase 1) is non-negotiable.** The architecture research confirms this and the pitfalls research proves it -- 7/10 critical pitfalls live in the data layer. FL training on dirty data produces silent garbage.
- **Model before FL (Phase 2 before 3)** because `model.py` is imported by client, server, and evaluation. Establishing it as a stable, tested module before building the FL layer prevents the most insidious pitfall (parameter shape mismatch). Local training validation also provides a centralized-training baseline to compare against federated results.
- **FL before evaluation (Phase 3 before 4)** because evaluation artifacts are meaningless without a trained federated model. Building visualization code before having real data to plot wastes effort on format assumptions.
- **Integration last (Phase 5)** because it depends on all components existing. Orchestration scripts cannot be tested without a working pipeline.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Data Pipeline):** The CICIDS2017 dataset has many documented quirks, but the exact file structure and column names should be verified against the actual downloaded data. The recommended preprocessing steps are well-established but dataset version variations exist across download mirrors.
- **Phase 3 (FL Infrastructure):** Flower's API has been evolving rapidly. The recommended 1.x classic API (NumPyClient, start_server) is stable but the exact import paths and function signatures should be verified against the pinned Flower version's documentation before writing FL code. Research the current Flower version to confirm 1.x API is still supported.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Model + Local Training):** Standard PyTorch MLP definition and training loop. Extremely well-documented, no surprises expected.
- **Phase 4 (Evaluation + Visualization):** Standard Scikit-learn metrics and Matplotlib/Seaborn plotting. Completely routine.
- **Phase 5 (Integration):** Shell scripting and documentation. No research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Core technologies (PyTorch, Pandas, Scikit-learn) are HIGH confidence. Flower version specifics are MEDIUM -- the framework evolves rapidly and exact version numbers should be verified against PyPI before project start. |
| Features | MEDIUM-HIGH | Feature landscape is well-defined by PROJECT.md constraints and FL-IDS literature. The v1/v1.x/v2 scoping is sound. Class imbalance handling and evaluation metric choices are HIGH confidence from extensive IDS literature. |
| Architecture | MEDIUM-HIGH | Component separation, data flow, and FL training patterns are well-established in Flower tutorials and FL-IDS papers. The flat `src/` structure is appropriate for project scope. Flower 1.x vs 2.x API boundary is MEDIUM confidence. |
| Pitfalls | HIGH | CICIDS2017 data quality issues and FL evaluation methodology pitfalls are among the most-discussed topics in network security ML literature. These pitfalls are dataset-inherent and framework-inherent, not speculative. |

**Overall confidence:** MEDIUM

The core domain knowledge (FL-IDS with CICIDS2017) is well-researched with HIGH confidence. The overall rating is MEDIUM because the Flower framework API specifics could not be verified against current documentation and may have shifted since the training data cutoff. This is a bounded risk -- if the API has changed, the adaptation is mechanical (update import paths and function calls), not architectural.

### Gaps to Address

- **Flower version verification:** Confirm current stable Flower version on PyPI. Verify that the 1.x classic API (NumPyClient, start_server, start_numpy_client) is still supported. If Flower has fully migrated to 2.x ClientApp/ServerApp, adjust the FL implementation plan accordingly. This is the single largest uncertainty.
- **CICIDS2017 download source:** Identify the current official download URL. Multiple mirrors exist with slightly different file sets. Verify that the Friday DDoS CSV is present and has the expected column structure before building the pipeline.
- **PyTorch + Flower compatibility:** After pinning exact versions, run a minimal "hello world" Flower + PyTorch integration test (2 clients, 1 round, random data) to confirm the stack works together before building the full pipeline.
- **Per-client scaler vs global scaler decision:** The research recommends global scaler statistics computed before partitioning. This is the right approach for IID simulation but should be validated empirically -- verify that per-client feature distributions are sufficiently similar after IID splitting.

## Sources

### Primary (HIGH confidence)
- CICIDS2017 dataset documentation (University of New Brunswick, unb.ca/cic/datasets/ids-2017.html) -- dataset format, known issues, class distribution, feature descriptions
- PyTorch documentation (pytorch.org/docs/stable) -- nn.Module, DataLoader, training loop patterns, state_dict serialization
- Scikit-learn documentation (scikit-learn.org/stable) -- StandardScaler, train_test_split, classification_report, confusion_matrix
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017) -- FedAvg algorithm, client drift analysis
- Published FL-IDS academic literature (2020-2024) -- CICIDS2017 preprocessing patterns, class imbalance strategies, FL evaluation methodology

### Secondary (MEDIUM confidence)
- Flower framework documentation (flower.ai/docs) -- API patterns, FedAvg configuration, simulation mode. MEDIUM because the framework evolves rapidly; API details based on training data through early 2025.
- Li et al., "Federated Learning: Challenges, Methods, and Future Directions" (2020) -- FL convergence issues, data heterogeneity survey
- Flower GitHub community discussions -- API migration patterns, NumPyClient usage, parameter serialization

### Tertiary (LOW confidence)
- Flower 2.x / next-gen API details (ClientApp, ServerApp, SuperLink) -- status and stability uncertain, may have changed significantly since training data cutoff
- Opacus (differential privacy for PyTorch) + Flower integration -- known to exist but integration maturity is unverified
- Flower SecAgg (secure aggregation) -- was experimental as of early 2025, current status unknown

---
*Research completed: 2026-03-09*
*Ready for roadmap: yes*
