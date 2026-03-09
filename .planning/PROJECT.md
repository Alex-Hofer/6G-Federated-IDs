# 6G Federated IDS

## What This Is

A privacy-preserving Intrusion Detection System (IDS) for 6G edge networks using Federated Learning. Edge devices (clients) train a local neural network on their own network traffic data to detect DDoS attacks, then share only model weights with a central server for aggregation. Raw network data never leaves the edge device — privacy by design.

## Core Value

Detect DDoS attacks across a federated network of edge nodes without any client ever sharing its raw network traffic data.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Central Flower server orchestrates federated training rounds across multiple clients
- [ ] Each client trains a local MLP model on its partition of CICIDS2017 data
- [ ] Only model weights are transmitted between clients and server (no raw data)
- [ ] FedAvg aggregation combines client model updates into a global model
- [ ] System detects DDoS attacks with measurable accuracy (precision, recall, F1)
- [ ] CICIDS2017 dataset is preprocessed and split IID across 2-5 simulated clients
- [ ] Console output shows per-round metrics (accuracy, precision, recall, F1)
- [ ] Saved plots visualize training convergence and per-client performance
- [ ] Confusion matrix generated for final model evaluation
- [ ] Multiple client instances can run simultaneously on a single machine

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

- **Dataset:** CICIDS2017 — well-labeled benchmark with DDoS flows, widely used in IDS research
- **Framework:** Flower (flwr) is the leading open-source FL framework for Python/PyTorch, handles client-server communication and aggregation strategies out of the box
- **Architecture:** Two scripts — `server.py` (Flower server with FedAvg strategy) and `client.py` (Flower client with local PyTorch MLP training). Multiple client instances simulate edge nodes
- **ML pipeline:** Pandas/Scikit-learn for data preprocessing and feature engineering, PyTorch for model definition and training, NumPy for numerical operations
- **Evaluation:** Per-round federated metrics + final global model evaluation with confusion matrix and convergence plots

## Constraints

- **Tech stack**: Python, PyTorch, Flower, Pandas, NumPy, Scikit-learn — user preference
- **Scale**: 2-5 simulated clients on a single machine — proof of concept scope
- **Dataset**: CICIDS2017 with IID partitioning — standard benchmark for reproducibility
- **Aggregation**: FedAvg — Flower's default, well-understood baseline
- **Model**: Simple MLP (feed-forward) — fast to train, easy to federate

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CICIDS2017 over NSL-KDD | More modern, richer attack types, better labeled flows | — Pending |
| Simple MLP over deeper models | Validates FL pipeline without model complexity obscuring results | — Pending |
| FedAvg over FedProx | Simpler baseline, sufficient for IID data distribution | — Pending |
| IID split over non-IID | Validates pipeline correctness before tackling convergence challenges | — Pending |
| FL-only privacy (no DP/SA) | Proof of concept — raw data stays local, which is the core privacy guarantee | — Pending |
| DDoS-only detection for v1 | Keeps classification scope focused, can expand attack types in v2 | — Pending |

---
*Last updated: 2026-03-09 after initialization*
