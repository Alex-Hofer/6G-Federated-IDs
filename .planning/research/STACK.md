# Stack Research

**Domain:** Privacy-Preserving Federated Learning IDS for 6G Edge Networks
**Researched:** 2026-03-09
**Confidence:** MEDIUM (versions based on training data through early 2025; could not verify absolute latest releases via PyPI due to tool restrictions -- verify with `pip install --upgrade` before project start)

## Recommended Stack

### Python Runtime

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.x | Runtime | Best balance of performance and library compatibility. 3.11 brought 10-60% speedups over 3.10. Flower, PyTorch, and scikit-learn all have mature 3.11 support. Avoid 3.12+ initially -- some ML libraries lag on 3.12/3.13 C extension compatibility. **Confidence: HIGH** |

### Core ML/FL Framework

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| flwr (Flower) | >=1.12.0 | Federated Learning framework | The leading open-source FL framework. Handles client-server gRPC communication, FedAvg aggregation strategy, and federated evaluation out of the box. Active development, strong PyTorch integration, excellent documentation. Use `flwr[simulation]` extra for single-machine multi-client simulation. **Confidence: MEDIUM** (1.12 was latest known; may be higher now) |
| torch (PyTorch) | >=2.4.0 | Neural network training | Industry standard for research ML. `torch.compile()` available since 2.0 for optional speedups. Mature, well-documented, huge community. CPU-only install sufficient for MLP on CICIDS2017 scale. **Confidence: MEDIUM** (2.4/2.5 were latest known; verify) |
| torchvision | Match torch version | PyTorch utilities | Install matching version for compatibility, even if not directly used -- prevents dependency conflicts. **Confidence: HIGH** |

### Data Preprocessing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pandas | >=2.2.0 | CSV loading, data manipulation | CICIDS2017 ships as CSV files. Pandas is the standard for tabular data loading, filtering, column operations. v2.x has significant performance improvements via Arrow backend. **Confidence: MEDIUM** (2.2 was latest known) |
| numpy | >=1.26.0 | Numerical operations | Underpins all ML computation. Required by PyTorch, pandas, scikit-learn. Pin to >=1.26 for NumPy 2.0 transition compatibility. **Confidence: MEDIUM** |
| scikit-learn | >=1.5.0 | Preprocessing and evaluation | Use for: `StandardScaler` (feature normalization), `train_test_split`, `LabelEncoder`, `classification_report`, `confusion_matrix`, `precision_recall_fscore_support`. The standard toolkit for ML preprocessing and metrics. **Confidence: MEDIUM** (1.5 was latest known) |

### Visualization

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| matplotlib | >=3.9.0 | Core plotting library | Training convergence curves, per-round metric plots. Stable, universally supported, integrates with all ML libraries. **Confidence: MEDIUM** |
| seaborn | >=0.13.0 | Statistical visualizations | Confusion matrix heatmaps (`seaborn.heatmap`), distribution plots for feature analysis. Built on matplotlib but produces publication-quality plots with less code. **Confidence: MEDIUM** |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| pip | latest | Package management | Use `pip install -e .` with pyproject.toml for development install. Prefer pip over conda for this project -- simpler dependency resolution for PyTorch+Flower. |
| pyproject.toml | N/A | Project configuration | Modern Python packaging standard (PEP 621). Define dependencies, optional extras, tool configs in one file. Replaces setup.py, setup.cfg, requirements.txt. |
| venv | stdlib | Virtual environment | Use Python's built-in `python -m venv .venv`. No need for conda/poetry for a project this size. |
| ruff | >=0.5.0 | Linting and formatting | Replaces flake8 + black + isort. Extremely fast (Rust-based), single tool. Use `ruff check` and `ruff format`. |
| pytest | >=8.0.0 | Testing | Test preprocessing pipeline, model architecture, federated round logic. Essential for validating data pipeline correctness. |
| tqdm | >=4.66.0 | Progress bars | Training loop progress, data loading progress. Small but important for UX during long preprocessing/training. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyYAML | >=6.0 | Configuration files | Store hyperparameters (learning rate, epochs, num_clients, batch_size) in YAML config rather than hardcoded. Cleaner than argparse for experiment management. |
| logging (stdlib) | N/A | Structured logging | Use Python's built-in logging module instead of print(). Set per-module log levels. Critical for debugging federated training across multiple client processes. |

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows (Git Bash / MSYS2):
source .venv/Scripts/activate
# On Linux/macOS:
# source .venv/bin/activate

# Core FL + ML stack
pip install "flwr[simulation]>=1.12.0" "torch>=2.4.0" "torchvision>=0.19.0"

# Data preprocessing + evaluation
pip install "pandas>=2.2.0" "numpy>=1.26.0" "scikit-learn>=1.5.0"

# Visualization
pip install "matplotlib>=3.9.0" "seaborn>=0.13.0"

# Development utilities
pip install "tqdm>=4.66.0" "PyYAML>=6.0"

# Dev tools (not runtime dependencies)
pip install "ruff>=0.5.0" "pytest>=8.0.0"
```

**Important: PyTorch CPU-only variant.** For this project (MLP on tabular data), GPU is unnecessary. If you want a smaller install:
```bash
# CPU-only PyTorch (significantly smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## pyproject.toml Template

```toml
[project]
name = "federated-ids-6g"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "flwr[simulation]>=1.12.0",
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "tqdm>=4.66.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.5.0",
    "pytest>=8.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **Flower (flwr)** | PySyft | PySyft if you need differential privacy or secure multi-party computation baked in. Flower is simpler, faster to prototype, better PyTorch integration, and has more active community. PySyft adds complexity not needed for v1. |
| **Flower (flwr)** | FedML | FedML if you need MLOps-style experiment tracking or pre-built FL benchmarks. Flower is more lightweight and flexible for custom pipelines. FedML has heavier dependencies. |
| **Flower (flwr)** | TFF (TensorFlow Federated) | Only if already in TensorFlow ecosystem. TFF is tightly coupled to TensorFlow, poor PyTorch support. Flower is framework-agnostic. |
| **PyTorch** | TensorFlow/Keras | Only if team has deep TF experience. PyTorch dominates research, better debugging (eager execution default), simpler custom training loops. |
| **pandas** | Polars | Polars if CICIDS2017 loading is a bottleneck (>10s). Polars is 5-10x faster for large CSV reads. However, pandas is more widely documented for ML pipelines, and CICIDS2017 (~2.8M rows) loads fine in pandas on modern hardware. Not worth the additional learning curve for v1. |
| **matplotlib+seaborn** | Plotly | Plotly if you want interactive dashboards. For this project (saved static plots for evaluation), matplotlib+seaborn is simpler and more standard in ML research. |
| **ruff** | flake8+black+isort | No reason to use the trio anymore. Ruff does everything they do, 10-100x faster, single config. |
| **venv** | conda | Conda if you need GPU CUDA management or complex C library dependencies. For CPU-only PyTorch + pure Python FL, venv is simpler. |
| **venv** | poetry | Poetry adds dependency resolution overhead not needed here. pip + pyproject.toml is sufficient. |
| **PyYAML** | Hydra / OmegaConf | Hydra if you need experiment sweeps, multi-run configs, or complex config composition. Overkill for v1 with a single config file. Consider for v2 if you add hyperparameter tuning. |
| **StandardScaler** | MinMaxScaler | MinMaxScaler if features must stay in [0,1] range. StandardScaler (zero mean, unit variance) is the standard choice for neural network inputs and handles outliers better -- important for CICIDS2017 which has extreme flow-level features. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **TensorFlow Federated** | Locked to TensorFlow ecosystem, poor PyTorch interop, less active community for custom FL pipelines | Flower -- framework-agnostic, simpler API, better docs |
| **Jupyter Notebooks for production code** | Notebooks create hidden state bugs, can't be properly version-controlled, make federated client/server scripts impossible to run as separate processes | Regular .py scripts. Use notebooks only for one-off EDA or result visualization, never for the training pipeline. |
| **requirements.txt** | No metadata, no optional dependencies, no tool configuration, legacy approach | pyproject.toml -- modern standard, one file for everything |
| **print() for logging** | No log levels, no timestamps, impossible to filter by module in multi-client scenarios | Python `logging` module with per-module loggers |
| **Manual feature selection** | Tempting to hand-pick features from CICIDS2017, but error-prone and hard to reproduce | Use all numeric features after dropping identifiers (Source IP, Dest IP, Timestamp, Flow ID). Let the MLP learn relevance. Drop columns with >50% NaN or zero variance. |
| **torch.nn.DataParallel** | For multi-GPU parallelism, irrelevant for this project (CPU MLP training) | Single-device training. If ever needed, use `torch.nn.parallel.DistributedDataParallel` instead (DataParallel is effectively deprecated). |
| **Custom FL communication protocol** | Building your own gRPC/socket client-server wastes weeks and introduces bugs | Flower handles all communication, serialization, and aggregation |
| **One-hot encoding for attack labels** | Unnecessary complexity for binary classification (DDoS vs Benign) | `LabelEncoder` or simple mapping: `{"BENIGN": 0, "DDoS": 1}` |
| **conda** | Adds environment complexity, slower dependency resolution, potential conflicts with pip-installed Flower | venv + pip. Flower is pip-native. |
| **Global random seeds only** | Setting `random.seed()` alone misses NumPy and PyTorch RNGs | Set all three: `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)` |

## Version Compatibility Matrix

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| flwr >=1.12 | torch >=2.0 | Flower's PyTorch integration works with any PyTorch 2.x. Flower uses NumPy arrays for weight serialization (`ndarrays_to_parameters`). |
| torch >=2.4 | Python 3.11 | Verified compatible. Python 3.12 support may have edge cases with some torch extensions. |
| torch >=2.4 | numpy >=1.26 | PyTorch 2.4+ supports NumPy 2.0, but pin >=1.26 to stay in safe compatibility range. |
| pandas >=2.2 | numpy >=1.26 | Pandas 2.2 works with both NumPy 1.x and 2.x. |
| scikit-learn >=1.5 | numpy >=1.26, pandas >=2.0 | Full interop. sklearn accepts pandas DataFrames directly. |
| matplotlib >=3.9 | numpy >=1.26, pandas >=2.0 | Plots pandas Series/DataFrames natively. |
| seaborn >=0.13 | matplotlib >=3.4, pandas >=1.5 | Always install matching matplotlib. |
| flwr[simulation] | ray (auto-installed) | Flower's simulation mode uses Ray for multi-client orchestration on single machine. Ray is installed automatically as a dependency. |

## CICIDS2017-Specific Stack Notes

The CICIDS2017 dataset has specific preprocessing requirements that influence tool choices:

1. **File format:** 8 CSV files, one per day of capture. Use `pandas.concat()` to merge. Total ~2.8M rows, ~80 columns.
2. **Column naming:** Column names have leading/trailing whitespace (e.g., " Destination Port"). Use `df.columns = df.columns.str.strip()` immediately after loading.
3. **Infinity values:** Some flow features contain `inf` values (e.g., Flow Bytes/s). Use `numpy.isinf()` detection and replace with `numpy.nan`, then impute or drop.
4. **NaN handling:** Several columns have NaN values. Use `pandas.DataFrame.dropna()` or `sklearn.impute.SimpleImputer`.
5. **Class imbalance:** Benign traffic vastly outnumbers DDoS. Use `sklearn.utils.class_weight.compute_class_weight()` for weighted loss, or undersample benign class.
6. **Feature scaling:** Flow-level features span vastly different ranges (0 to millions). `StandardScaler` is mandatory before feeding to MLP.
7. **Label mapping:** Labels are strings ("BENIGN", "DDoS"). Map to integers for PyTorch: 0=Benign, 1=DDoS.

## Flower API Usage Pattern

For this project, use Flower's **low-level API** (not the new `ServerApp`/`ClientApp` high-level API introduced in 1.8+). The low-level API is:

- More explicit and easier to understand for a learning project
- Better documented in tutorials and research papers
- Gives direct control over training loops

Key Flower components to use:
- `flwr.client.NumPyClient` -- subclass this for the client
- `flwr.server.start_server()` -- launch the FL server
- `flwr.client.start_client()` -- connect a client to the server
- `flwr.server.strategy.FedAvg` -- aggregation strategy
- `flwr.common.ndarrays_to_parameters` / `parameters_to_ndarrays` -- weight serialization

**Note on Flower API evolution:** Flower has been evolving its API surface rapidly (SuperLink, SuperNode, Fleet API). For a v1 proof of concept, stick with the stable `start_server`/`start_client` pattern. The newer `ServerApp`/`ClientApp` pattern is valid but has less community tutorial coverage and adds abstraction that may obscure the learning goals.

## Sources

- Flower documentation (flower.ai/docs) -- API patterns, FedAvg configuration, simulation mode
- PyTorch documentation (pytorch.org/docs/stable) -- torch.nn, DataLoader, training loop patterns
- scikit-learn documentation (scikit-learn.org/stable) -- preprocessing, metrics API
- CICIDS2017 dataset documentation (unb.ca/cic/datasets/ids-2017.html) -- dataset format, known issues
- Version numbers based on training data through early 2025; marked MEDIUM confidence where exact latest version could not be verified via PyPI

**Version verification note:** Before starting development, run `pip install --upgrade flwr torch scikit-learn pandas numpy matplotlib seaborn` to get absolute latest versions, then pin exact versions in pyproject.toml with `pip freeze`. The minimum versions listed above are known-good; latest will be equal or higher.

---
*Stack research for: Federated Learning IDS for 6G Edge Networks*
*Researched: 2026-03-09*
