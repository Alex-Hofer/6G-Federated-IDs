# 6G Federated IDS

## What This Is

A privacy-preserving Intrusion Detection System (IDS) for 6G edge networks using Federated Learning. Edge devices train local MLP classifiers on their network traffic data (CIC-IDS2017) and share only model weights with a central aggregation server — raw data never leaves the edge. This is a research/thesis project demonstrating that federated learning can achieve competitive detection accuracy while preserving data privacy.

## Core Value

Detect network anomalies (DDoS attacks) with high accuracy while ensuring raw network traffic data never leaves the edge device (Privacy by Design).

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. -->

- [ ] Central FL server using Flower that orchestrates training rounds and aggregates model weights via FedAvg
- [ ] Client script that trains a local MLP classifier on CIC-IDS2017 data and sends weights to the server
- [ ] Support for 3-5 simultaneous simulated edge nodes (client instances)
- [ ] Data preprocessing pipeline: load CIC-IDS2017, clean, normalize, and split into binary labels (Normal vs Attack)
- [ ] IID data partitioning across clients (each client gets a balanced random subset)
- [ ] Binary classification: Normal traffic vs Attack traffic
- [ ] Evaluation metrics: Accuracy, F1-Score, Precision, Recall per round
- [ ] Track and report number of communication rounds to convergence
- [ ] Generate matplotlib plots: accuracy/loss over FL rounds, confusion matrix
- [ ] Reproducible experiments: seeded randomness, configurable hyperparameters

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Data Poisoning defense — complex FL-security topic, deferred to v2
- Non-IID data distribution — adds convergence complexity, deferred to v2
- Multi-class attack classification — binary is cleaner for initial thesis contribution
- FedProx or other aggregation strategies — FedAvg baseline first
- Centralized vs federated comparison — deferred (could be v2 experiment)
- Web dashboard or GUI — CLI + plots sufficient for thesis
- Real network traffic capture — using CIC-IDS2017 benchmark dataset
- Deployment on actual edge hardware — simulation only

## Context

- **Dataset:** CIC-IDS2017 — widely used IDS benchmark with 80+ network flow features and labeled attack types including DDoS. Well-documented in literature, good for reproducibility and comparison with existing work.
- **6G framing:** The 6G edge network context is conceptual — the architecture demonstrates how federated IDS would work in a 6G MEC (Multi-access Edge Computing) environment. The simulation abstracts away the physical network layer.
- **Thesis context:** Results need to be reproducible with clear metrics. Plots and logged metrics are the primary deliverables alongside the codebase.
- **Flower framework:** Handles FL orchestration (client-server communication, round management). Simplifies implementation so focus can be on the IDS model and data pipeline.

## Constraints

- **Tech stack**: Python, PyTorch, Flower (flwr), Pandas, NumPy, Scikit-learn, Matplotlib — established and thesis-appropriate
- **Architecture**: Central server script + client script (multi-instance) — standard Flower pattern
- **Dataset**: CIC-IDS2017 must be downloaded separately (large CSV files, not bundled in repo)
- **Reproducibility**: All experiments must be seeded and configurable for thesis requirements

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Binary classification over multi-class | Cleaner baseline for initial thesis contribution | — Pending |
| FedAvg over FedProx | Standard baseline, well-understood, sufficient for IID setting | — Pending |
| IID data split over non-IID | Simpler convergence, establishes baseline before harder scenarios | — Pending |
| MLP over LSTM/CNN | Tabular features (CIC-IDS2017 flow stats) suit feed-forward nets; faster training | — Pending |
| Flower over custom FL | Production-grade FL framework, reduces boilerplate, well-documented | — Pending |

---
*Last updated: 2026-03-09 after initialization*
