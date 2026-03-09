# Architecture Research

**Domain:** Federated Learning Intrusion Detection System for 6G Edge Networks
**Researched:** 2026-03-09
**Confidence:** MEDIUM-HIGH (based on training knowledge of Flower, PyTorch FL-IDS patterns, and CICIDS2017 pipeline; no live doc verification possible)

## Standard Architecture

### System Overview

```
+-------------------------------------------------------------------+
|                        ORCHESTRATION LAYER                        |
|                                                                    |
|   +--------------------+    Flower gRPC     +-------------------+ |
|   |   FL SERVER         |<=================>|   FL CLIENT 1     | |
|   |   (server.py)       |    (weights only)  |   (client.py)     | |
|   |                     |                    |                   | |
|   |  - FedAvg Strategy  |<======+===========>|   FL CLIENT 2     | |
|   |  - Round mgmt       |      |            |   (client.py)     | |
|   |  - Global model     |      |            |                   | |
|   |  - Metric aggreg.   |      +==========>|   FL CLIENT N     | |
|   +--------------------+                    |   (client.py)     | |
|            |                                 +-------------------+ |
|            v                                         |             |
|   +--------------------+                    +-------------------+ |
|   |   EVALUATION        |                    |   LOCAL TRAINING   | |
|   |   (evaluate.py)     |                    |                   | |
|   |                     |                    |  - Data loading   | |
|   |  - Global metrics   |                    |  - Preprocessing  | |
|   |  - Confusion matrix |                    |  - MLP forward    | |
|   |  - Per-class stats  |                    |  - Backprop       | |
|   +--------------------+                    |  - Local eval     | |
|            |                                 +-------------------+ |
|            v                                         ^             |
|   +--------------------+                    +-------------------+ |
|   |   VISUALIZATION     |                    |   DATA PIPELINE    | |
|   |   (visualize.py)    |                    |   (data_utils.py)  | |
|   |                     |                    |                   | |
|   |  - Loss curves      |                    |  - CSV loading    | |
|   |  - Accuracy plots   |                    |  - Feature eng.   | |
|   |  - Confusion matrix |                    |  - Normalization  | |
|   |  - Per-client comp. |                    |  - IID splitting  | |
|   +--------------------+                    |  - DataLoaders    | |
|                                              +-------------------+ |
|                                                      ^             |
|                                              +-------------------+ |
|                                              |   RAW DATA         | |
|                                              |   CICIDS2017 CSV   | |
|                                              +-------------------+ |
+-------------------------------------------------------------------+
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation | Depends On |
|-----------|----------------|------------------------|------------|
| **Data Pipeline** (`data_utils.py`) | Load CICIDS2017 CSVs, clean, engineer features, normalize, partition IID across clients, produce PyTorch DataLoaders | Pandas for loading/cleaning, Scikit-learn for scaling/encoding, PyTorch Dataset/DataLoader for batching | Raw CICIDS2017 CSV files |
| **Model Definition** (`model.py`) | Define the MLP architecture, forward pass, loss function selection | PyTorch `nn.Module` subclass with linear layers, ReLU, dropout, sigmoid/softmax output | None (pure definition) |
| **FL Client** (`client.py`) | Implement Flower client interface: receive global weights, train locally, return updated weights and metrics | `flwr.client.NumPyClient` subclass with `get_parameters`, `set_parameters`, `fit`, `evaluate` methods | Data Pipeline, Model Definition |
| **FL Server** (`server.py`) | Configure aggregation strategy, manage training rounds, collect and aggregate metrics | `flwr.server.start_server` or `flwr.server.ServerApp` with `FedAvg` strategy configuration | Model Definition (for initial params) |
| **Evaluation** (`evaluate.py`) | Compute final global model metrics: accuracy, precision, recall, F1, confusion matrix | Scikit-learn metrics on held-out test set using the final aggregated model | Model Definition, Data Pipeline |
| **Visualization** (`visualize.py`) | Generate training convergence plots, confusion matrix heatmaps, per-client comparisons | Matplotlib/Seaborn for plots, saved to `outputs/` directory | Evaluation results, training history |
| **Configuration** (`config.py` or constants) | Central hyperparameter management: rounds, learning rate, batch size, client count, feature columns | Python dataclass or simple constants module | None |

## Recommended Project Structure

```
6g-federated-ids/
├── .planning/                  # Project planning (existing)
│   ├── PROJECT.md
│   ├── config.json
│   └── research/
│       └── ARCHITECTURE.md     # This file
│
├── data/                       # Data directory (gitignored)
│   └── cicids2017/             # Raw CSV files from CICIDS2017
│       ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│       └── ...                 # Other day CSVs if needed
│
├── src/                        # All source code
│   ├── __init__.py
│   ├── server.py               # Flower server entry point
│   ├── client.py               # Flower client entry point
│   ├── model.py                # MLP model definition
│   ├── data_utils.py           # Data loading, preprocessing, partitioning
│   ├── evaluate.py             # Global model evaluation
│   ├── visualize.py            # Plot generation
│   └── config.py               # Hyperparameters and constants
│
├── outputs/                    # Generated artifacts (gitignored)
│   ├── plots/                  # Saved figures
│   │   ├── convergence.png
│   │   ├── confusion_matrix.png
│   │   └── per_client.png
│   ├── metrics/                # Saved metric logs
│   │   └── history.json
│   └── models/                 # Saved model checkpoints
│       └── global_model.pth
│
├── scripts/                    # Convenience scripts
│   └── run_simulation.sh       # Launch server + N clients
│
├── tests/                      # Unit and integration tests
│   ├── test_data_utils.py
│   ├── test_model.py
│   └── test_client.py
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # data/, outputs/, __pycache__, etc.
└── README.md                   # Project overview
```

### Why This Structure

- **`src/` flat module**: For a proof-of-concept with 6-7 files, a flat source directory is cleaner than nested packages. Each file maps to exactly one component from the architecture diagram. No hunting for code.
- **`data/` separate and gitignored**: CICIDS2017 CSVs are 1-2 GB. They must never enter version control. The data pipeline reads from this known path.
- **`outputs/` separate and gitignored**: All generated artifacts (plots, models, metric logs) go here. Clean separation between source and output.
- **`model.py` separate from `client.py`**: The model definition is used by the server (for initial parameter shapes), the client (for training), and evaluation (for inference). Sharing it via import prevents duplication and drift.
- **`config.py` as single source of truth**: Every magic number (learning rate, batch size, number of rounds, feature columns) lives in one place. No scattered constants.

## Architectural Patterns

### Pattern 1: NumPyClient Interface (Flower 1.x)

The Flower `NumPyClient` is the correct abstraction for this project. It handles serialization of PyTorch tensors to/from NumPy arrays automatically.

**What:** Subclass `flwr.client.NumPyClient` and implement four methods.
**Why:** Avoids manual protobuf serialization. Flower handles gRPC transport of NumPy arrays. The client just works with familiar NumPy/PyTorch types.

```python
class IDSClient(flwr.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        """Return model weights as list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model weights from list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train locally, return updated weights + num_examples."""
        self.set_parameters(parameters)
        train_loss = train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": train_loss}

    def evaluate(self, parameters, config):
        """Evaluate on local test set, return loss + num_examples + metrics."""
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        }
```

**Confidence:** HIGH -- this is the canonical Flower pattern, stable across 1.x releases.

### Pattern 2: Server-Side Strategy Configuration

**What:** Configure `FedAvg` with metric aggregation callbacks and optional initial parameters.
**Why:** The server needs to aggregate per-client metrics (not just weights) to track convergence. Setting initial parameters avoids the "first client defines the model" race condition.

```python
import flwr as fl
from flwr.server.strategy import FedAvg
from model import IDS_MLP

def weighted_average(metrics):
    """Aggregate client metrics weighted by number of examples."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Get initial parameters from a fresh model
model = IDS_MLP(input_dim=NUM_FEATURES)
initial_params = [val.cpu().numpy() for _, val in model.state_dict().items()]

strategy = FedAvg(
    fraction_fit=1.0,            # Use all available clients each round
    fraction_evaluate=1.0,       # Evaluate on all clients
    min_fit_clients=2,           # Minimum clients for training
    min_evaluate_clients=2,      # Minimum clients for evaluation
    min_available_clients=2,     # Wait for at least 2 clients
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

**Confidence:** HIGH -- standard FedAvg configuration pattern.

### Pattern 3: Centralized Data Preprocessing with Client-Specific Partitioning

**What:** Preprocess the full dataset once, then partition into N client shards.
**Why:** Ensures consistent feature engineering and normalization across all clients. In production FL, each client would have its own data -- in simulation, we split a single dataset to emulate this.

```python
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load CICIDS2017, clean, engineer features, return clean DataFrame."""
    # 1. Load CSV
    # 2. Drop NaN/Inf rows
    # 3. Map labels to binary (Benign=0, DDoS=1)
    # 4. Select relevant numeric features
    # 5. Return clean DataFrame

def partition_data(df: pd.DataFrame, num_clients: int, seed: int = 42):
    """IID partition: shuffle and split into N equal parts."""
    # Returns list of N DataFrames

def create_dataloaders(partition: pd.DataFrame, test_split: float = 0.2):
    """Scale features, split train/test, return PyTorch DataLoaders."""
    # 1. Train/test split (stratified)
    # 2. Fit StandardScaler on train, transform both
    # 3. Create TensorDatasets
    # 4. Return (trainloader, testloader)
```

**Confidence:** HIGH -- this is the standard simulation pattern in FL-IDS literature.

### Pattern 4: Metric History Logging

**What:** Capture per-round, per-client metrics and persist them for visualization.
**Why:** FL training is expensive. You do not want to re-run experiments just because you forgot to save metrics. The Flower `History` object captures server-side aggregated metrics automatically, but per-client granularity requires explicit logging.

```python
# In the client's fit() and evaluate() methods, return metrics dict.
# On the server, use evaluate_metrics_aggregation_fn to capture them.
# After training completes, save the History object:

history = fl.server.start_server(...)  # Returns History
# history.metrics_distributed["accuracy"] = [(round, value), ...]
# Save to JSON for visualization
```

**Confidence:** HIGH -- `start_server` returns `History` object in Flower 1.x.

## Data Flow

### FL Training Flow

```
PHASE 1: DATA PREPARATION (runs once, before FL starts)
=========================================================
  CICIDS2017 CSV files
       |
       v
  [load_and_preprocess()]
  - Read CSV with Pandas
  - Drop columns: 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp'
  - Replace inf with NaN, drop NaN rows
  - Binary label encoding: 'BENIGN' -> 0, 'DDoS' -> 1
  - Select ~78 numeric feature columns
       |
       v
  Clean DataFrame (N rows x F features + 1 label)
       |
       v
  [partition_data(num_clients=K)]
  - Shuffle rows (seeded)
  - Split into K equal-size partitions
       |
       v
  K DataFrames, one per simulated client

PHASE 2: CLIENT INITIALIZATION (each client independently)
===========================================================
  Client i's DataFrame partition
       |
       v
  [create_dataloaders()]
  - Stratified train/test split (80/20)
  - Fit StandardScaler on train features only (prevent data leakage)
  - Transform train and test features
  - Wrap in TensorDataset -> DataLoader (batch_size=64)
       |
       v
  (trainloader_i, testloader_i) + local MLP model instance

PHASE 3: FEDERATED TRAINING (repeats for R rounds)
====================================================
  Round r begins
       |
       v
  Server sends global model weights to selected clients
       |                          (gRPC, serialized NumPy arrays)
       v
  Each client i:
    1. set_parameters(global_weights)     <- overwrite local model
    2. Train for E local epochs on trainloader_i
       - Forward pass through MLP
       - BCEWithLogitsLoss (binary) or CrossEntropyLoss
       - Adam optimizer, lr=0.001
       - Backpropagation
    3. Return: (updated_weights, num_train_examples, {"loss": train_loss})
       |
       v
  Server receives all client updates
       |
       v
  [FedAvg Aggregation]
  - Weighted average of client weights (weighted by num_examples)
  - new_global_weight[layer] = sum(n_i * w_i[layer]) / sum(n_i)
       |
       v
  Server optionally evaluates:
  - Sends new global weights to clients for evaluate()
  - Each client runs inference on testloader_i
  - Returns: (loss, num_test_examples, {accuracy, precision, recall, f1})
  - Server aggregates metrics via weighted_average()
       |
       v
  Log round metrics to History
  Repeat for next round

PHASE 4: POST-TRAINING (runs once, after FL completes)
=======================================================
  Final global model weights (from last round)
       |
       v
  [evaluate.py]
  - Load fresh model, set final weights
  - Run inference on full held-out test set
  - Compute: accuracy, precision, recall, F1, confusion matrix
  - Print classification report
       |
       v
  Metrics + confusion matrix data
       |
       v
  [visualize.py]
  - Plot convergence curves (loss/accuracy vs. round)
  - Plot confusion matrix heatmap
  - Plot per-client metric comparison (bar chart)
  - Save all to outputs/plots/
```

### Evaluation Flow

```
Final Global Weights ──> Load into fresh MLP
                              |
                              v
                    Full Test Set (unseen during FL)
                              |
                              v
                    Model inference (no grad)
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             Per-sample           Per-sample
             predictions          true labels
                    |                   |
                    +--------+----------+
                             |
                             v
                    sklearn.metrics
                    - accuracy_score
                    - precision_score
                    - recall_score
                    - f1_score
                    - confusion_matrix
                    - classification_report
                             |
                             v
                    Console output + saved JSON
                             |
                             v
                    visualize.py reads JSON/data
                    - matplotlib confusion matrix
                    - convergence line plots
                    - per-client bar charts
                             |
                             v
                    outputs/plots/*.png
```

## Communication Protocol

Flower uses gRPC for client-server communication. The key data exchanged:

| Direction | Data | Format | Size |
|-----------|------|--------|------|
| Server -> Client | Global model weights | List of NumPy arrays (one per layer) | ~50-200 KB for a small MLP |
| Client -> Server | Updated local weights | List of NumPy arrays | Same as above |
| Client -> Server | Training metrics | Python dict (loss, num_examples) | Negligible |
| Client -> Server | Evaluation metrics | Python dict (accuracy, precision, recall, f1) | Negligible |
| Server -> Client | Config dict | Python dict (local_epochs, lr, etc.) | Negligible |

**Key point:** Raw training data NEVER crosses the wire. Only model parameter tensors (weights and biases) are transmitted. This is the core privacy guarantee of FL.

## Build Order (Dependency Graph)

Components must be built in this order based on dependencies:

```
Phase 1: FOUNDATION (no dependencies)
  ├── config.py          # Constants and hyperparameters
  └── model.py           # MLP architecture (nn.Module)

Phase 2: DATA (depends on config)
  └── data_utils.py      # Load, preprocess, partition, DataLoaders
      Tests: Can load CSV, clean data, produce correct shapes

Phase 3: FL CLIENT (depends on model + data)
  └── client.py          # NumPyClient with fit/evaluate
      Tests: Can instantiate, get/set params, run one fit step

Phase 4: FL SERVER (depends on model + config)
  └── server.py          # FedAvg strategy, start_server
      Integration test: Server + 2 clients complete 1 round

Phase 5: EVALUATION (depends on model + data)
  └── evaluate.py        # Post-training metrics and classification report

Phase 6: VISUALIZATION (depends on evaluation outputs)
  └── visualize.py       # Plots from metric history

Phase 7: INTEGRATION
  └── run_simulation.sh  # End-to-end orchestration script
```

**Critical dependency insight:** `model.py` is the most depended-upon module. It must be correct and stable early because the client, server (for initial params), and evaluation all import it. Changes to the model shape ripple everywhere.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Fitting Scaler on Full Dataset Before Splitting

**What:** Calling `StandardScaler.fit()` on the entire dataset (train + test) before partitioning or splitting.
**Why bad:** Introduces data leakage. Test set statistics influence training feature scaling. Evaluation metrics become unreliable -- your reported F1 is higher than real-world performance.
**Instead:** Fit the scaler on each client's training split only, then transform both train and test with that scaler. Each client gets its own scaler instance.

### Anti-Pattern 2: Hardcoding Feature Columns by Index

**What:** Selecting features with `df.iloc[:, 0:78]` instead of by column name.
**Why bad:** CICIDS2017 CSV column ordering varies between download sources and day-files. An index-based approach silently uses wrong features when the CSV layout shifts.
**Instead:** Define an explicit `FEATURE_COLUMNS` list in `config.py`. Select by name: `df[FEATURE_COLUMNS]`. Fail loudly if a column is missing.

### Anti-Pattern 3: Monolithic client.py That Does Everything

**What:** Putting data loading, preprocessing, model definition, training loops, and Flower client logic all in one file.
**Why bad:** Impossible to test components independently. Cannot reuse the model for evaluation. Cannot change data preprocessing without touching FL logic.
**Instead:** Separate concerns into `data_utils.py`, `model.py`, and `client.py`. Each module has a single responsibility and can be tested in isolation.

### Anti-Pattern 4: Not Setting Random Seeds

**What:** Forgetting to seed `torch`, `numpy`, `random`, and `sklearn` splits.
**Why bad:** FL experiments are inherently variable (multiple clients, async). Without fixed seeds, you cannot reproduce results or meaningfully compare experiments.
**Instead:** Set seeds in `config.py` and apply them at the top of every entry point:
```python
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

### Anti-Pattern 5: Using start_server + start_client in Production-Like Setup When Simulation Suffices

**What:** Running `server.py` and multiple `client.py` processes manually for a single-machine proof of concept.
**Why this is nuanced:** Flower offers `flwr.simulation.start_simulation()` which runs server and clients in a single process using virtual clients. This is simpler for development. However, the PROJECT.md explicitly specifies `server.py` + `client.py` as separate scripts, which better demonstrates the real FL architecture with actual gRPC communication.
**Recommendation:** Use the separate `server.py` + `client.py` approach as specified in the project requirements. It is closer to how a real 6G edge deployment would work and makes the architecture more transparent for academic/demonstration purposes. Use a shell script to orchestrate launching the processes.

### Anti-Pattern 6: Ignoring Class Imbalance in CICIDS2017

**What:** Training with standard loss without accounting for the severe class imbalance in CICIDS2017 (DDoS flows are a small fraction of total traffic in most day-files).
**Why bad:** Model learns to predict "Benign" for everything and achieves high accuracy but zero recall on DDoS.
**Instead:** Use one or more of: (a) class weights in the loss function, (b) stratified sampling in DataLoaders, (c) oversampling minority class, (d) use the Friday afternoon DDoS-specific CSV which has better balance. Report F1/precision/recall, not just accuracy.

## Flower API Considerations

### Flower 1.x vs 2.x API

Flower underwent a significant API redesign between 1.x and the newer API direction. Key differences:

**Flower 1.x (stable, well-documented, recommended for this project):**
- `flwr.client.NumPyClient` -- subclass and implement methods
- `flwr.client.start_numpy_client(server_address, client)` -- connect client to server
- `flwr.server.start_server(server_address, config, strategy)` -- start server
- Returns `History` object with metrics

**Flower 2.x / Next-gen API (emerging, uses `ClientApp`/`ServerApp`):**
- `flwr.client.ClientApp` with decorators
- `flwr.server.ServerApp` with decorators
- More declarative, designed for `flwr run` CLI
- Still stabilizing; less community example code available

**Recommendation:** Use the **Flower 1.x API** (`NumPyClient`, `start_server`, `start_numpy_client`). It is mature, extensively documented, and has the most example code available. The project should pin `flwr>=1.5,<2.0` (or similar) to avoid breaking changes.

**Confidence:** MEDIUM -- The Flower 2.x API evolution is actively ongoing. The boundary between 1.x and 2.x was shifting as of my last training data. Verify current stable version before pinning.

## Scalability Considerations

| Concern | 2-3 Clients (v1) | 10+ Clients (v2) | 100+ Clients (future) |
|---------|-------------------|-------------------|------------------------|
| Process management | Shell script | Shell script + process manager | Flower simulation API or container orchestration |
| Memory | Each client loads own partition (~200 MB each) | Partition sizes shrink, still fits in RAM | Need streaming data loading |
| Communication | gRPC on localhost, negligible latency | Same machine, still fine | Need actual network, compression |
| Aggregation time | Instant with FedAvg | Sub-second | May need async aggregation |
| Data heterogeneity | IID, no issues | Consider non-IID | Need FedProx or similar |

For this v1 proof of concept with 2-5 clients on a single machine, scalability is not a concern. The architecture supports later scaling because the component boundaries (separate processes, gRPC communication) mirror a real distributed deployment.

## Sources

- Flower framework documentation and tutorials (flower.ai/docs) -- HIGH confidence for 1.x API patterns
- CICIDS2017 dataset documentation (University of New Brunswick) -- HIGH confidence for data schema
- Published FL-IDS literature using CICIDS2017 + Flower/PyTorch -- MEDIUM confidence (synthesized from multiple training examples)
- PyTorch documentation for nn.Module, DataLoader, optimizer patterns -- HIGH confidence

**Note:** All architecture recommendations are based on training knowledge through May 2025. The Flower framework version and API stability should be verified against current documentation before implementation begins.

---
*Architecture research for: Federated Learning IDS for 6G Edge Networks*
*Researched: 2026-03-09*
