# 6G Federated IDS

## What This Is

A privacy-preserving Intrusion Detection System (IDS) for 6G edge networks using Federated Learning. Multiple simulated edge clients train local MLP models on CICIDS2017 network traffic data to detect DDoS attacks, sharing only model weights via FedAvg aggregation. Raw data never leaves the edge device — privacy by design. Shipped as a single-command Python pipeline with publication-quality evaluation outputs.

## Core Value

Detect DDoS attacks across a federated network of edge nodes without any client ever sharing its raw network traffic data.

## Requirements

### Validated

- Central Flower server orchestrates federated training rounds across multiple clients — v1.0
- Each client trains a local MLP model on its partition of CICIDS2017 data — v1.0
- Only model weights are transmitted between clients and server (no raw data) — v1.0
- FedAvg aggregation combines client model updates into a global model — v1.0
- System detects DDoS attacks with measurable accuracy (precision, recall, F1) — v1.0
- CICIDS2017 dataset is preprocessed and split IID across 2-5 simulated clients — v1.0
- Console output shows per-round metrics (accuracy, precision, recall, F1) — v1.0
- Saved plots visualize training convergence and per-client performance — v1.0
- Confusion matrix generated for final model evaluation — v1.0
- Multiple client instances can run simultaneously on a single machine — v1.0

### Active

(None — next milestone requirements TBD via `/gsd:new-milestone`)

### Out of Scope

- Data poisoning defense (robust aggregation) — defer to v2, focus on IDS pipeline first
- Differential privacy (DP-SGD) — FL-only privacy sufficient for proof of concept
- Secure aggregation — added complexity not needed for v1 trust model
- Non-IID data splits — IID first to validate pipeline, non-IID adds convergence challenges
- Docker containerization — single-machine simulation sufficient for v1
- Additional attack types (brute force, botnet, infiltration) — DDoS-only keeps scope tight
- CNN/LSTM/Autoencoder architectures — simple MLP validates the concept
- Real 6G hardware/network integration — simulation only

## Context

**Current state:** v1.0 shipped (2026-03-10). 36 Python files, 6,643 LOC. 104 commits. All 17 v1 requirements verified.

- **Dataset:** CICIDS2017 — well-labeled benchmark with DDoS flows, widely used in IDS research
- **Framework:** Flower (flwr) for FL client-server communication and FedAvg aggregation
- **Architecture:** Modular package `federated_ids` with layered imports: config/seed/device -> data -> model -> fl -> eval -> pipeline. 5 CLI entry points.
- **ML pipeline:** Pandas/Scikit-learn for preprocessing, PyTorch for model/training, tensor caching for fast re-runs
- **Evaluation:** Confusion matrix, classification report, convergence plots (PNG), per-client comparison, TensorBoard logging
- **Tech debt:** 10 non-critical items (see milestones/v1.0-MILESTONE-AUDIT.md)

## Constraints

- **Tech stack**: Python, PyTorch, Flower, Pandas, NumPy, Scikit-learn — user preference
- **Scale**: 2-5 simulated clients on a single machine — proof of concept scope
- **Dataset**: CICIDS2017 with IID partitioning — standard benchmark for reproducibility
- **Aggregation**: FedAvg — Flower's default, well-understood baseline
- **Model**: Simple MLP (feed-forward) — fast to train, easy to federate

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CICIDS2017 over NSL-KDD | More modern, richer attack types, better labeled flows | Good — 78+ features, clean binary DDoS labels |
| Simple MLP over deeper models | Validates FL pipeline without model complexity obscuring results | Good — fast training, clear FL convergence signal |
| FedAvg over FedProx | Simpler baseline, sufficient for IID data distribution | Good — stable convergence across clients |
| IID split over non-IID | Validates pipeline correctness before tackling convergence challenges | Good — clean baseline for v2 non-IID comparison |
| FL-only privacy (no DP/SA) | Proof of concept — raw data stays local, which is the core privacy guarantee | Good — sufficient for thesis v1 |
| DDoS-only detection for v1 | Keeps classification scope focused, can expand attack types in v2 | Good — binary classification keeps evaluation clear |
| 5-phase linear dependency chain | Data -> Model -> FL -> Eval -> Integration, based on research pitfall analysis | Good — natural build order, minimal rework |
| Tensor caching (.pt files) | Skip expensive CSV loading/preprocessing on subsequent runs | Good — dramatically faster iteration |
| F1-based checkpointing (not loss) | Better DDoS detection quality metric for imbalanced classes | Good — saves best detector, not lowest loss |
| TDD for bug fixes (Phases 6-7) | Write failing test first, then fix, ensures regression coverage | Good — caught weighted_loss bug cleanly |

---
*Last updated: 2026-03-10 after v1.0 milestone*
