# Phase 1: Project Foundation and Data Pipeline - Research

**Researched:** 2026-03-09
**Domain:** Python project scaffolding, CICIDS2017 data preprocessing, PyTorch DataLoader creation, IID federated partitioning
**Confidence:** HIGH

## Summary

Phase 1 establishes the entire project foundation and builds a validated CICIDS2017 preprocessing pipeline. This phase carries the highest pitfall density in the entire project -- seven of ten identified critical pitfalls originate in the data layer. The research confirms that every decision in this phase (cleaning order, scaler fitting, partition strategy, feature selection, label mapping) has downstream consequences for model training and federated learning correctness. Getting this wrong produces silent data corruption that surfaces as mysteriously bad metrics in later phases.

The technical approach is well-established: load the Friday-WorkingHours-Afternoon-DDos CSV (~225K rows, ~97K benign / ~128K DDoS), immediately clean Inf/NaN values and strip column whitespace, drop identifier columns, apply a hybrid feature selection (domain shortlist + statistical filters), fit a global StandardScaler on training data only before partitioning, then split IID across clients with stratified splits preserving class ratios. The stack is mature Python ML tooling (pandas, scikit-learn, PyTorch, PyYAML) with no exotic dependencies.

The Flower framework (flwr) has evolved significantly since the early domain research. The current stable version is 1.26.1, and while `start_server`/`start_client` and `NumPyClient` still work with backward compatibility, they are deprecated in favor of `ClientApp`/`ServerApp`. For Phase 1, we only need to pin the version -- actual Flower usage comes in Phase 3. PyTorch is at version 2.10.0 (Jan 2026).

**Primary recommendation:** Build the preprocessing pipeline as three distinct modules (loader, preprocess, partition) with explicit validation assertions between each stage. Use a global train/test split BEFORE client partitioning, fit the StandardScaler on the global training set, and save all artifacts (scaler, feature list, class weights) to data/processed/ for reuse.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Manual download -- user places CSVs in data/raw/ manually, README documents the UNB source URL and expected files
- Expect original UNB filenames (e.g., 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
- Load only DDoS-relevant CSV files (Friday-WorkingHours), not all 8 CSVs
- Data layout: data/raw/ for original CSVs, data/processed/ for pipeline output, both gitignored
- Hybrid feature selection: start with domain-informed shortlist of DDoS-relevant features, then apply statistical filters (drop zero-variance, near-constant, highly correlated >0.95)
- Target 20-30 informative features from the 78+ raw columns
- Detailed logging: print/save a report of dropped features with reasons and final feature list
- Binary label: map all DDoS subtypes (DDoS, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest) to 1, BENIGN to 0
- Global hold-out test set (20%) created BEFORE client partitioning
- Rows with Inf/NaN values are deleted (not imputed)
- StandardScaler fitted ONLY on training set, then applied to test set and all client partitions (no data leakage)
- All numerical features converted to float32 during preprocessing
- Subpackage layout: src/federated_ids/ with subpackages data/, model/, fl/, eval/
- Phase 1 focuses on data/ subpackage: loader.py, preprocess.py, partition.py
- pip + pyproject.toml with pinned dependencies, pip install -e . for dev
- Basic unit tests in tests/
- outputs/ directory for plots, checkpoints, logs
- Auto-detect CUDA/MPS/CPU, fall back to CPU
- Global seed utility fixing numpy, random, torch, torch.cuda -- configurable via config.yaml
- Python script entry point: python -m federated_ids.data.preprocess + console_scripts entry point
- Override config with --config path/to/config.yaml flag
- Pipeline saves intermediate artifacts: fitted StandardScaler (.pkl), selected feature list (.json), class distribution stats to data/processed/
- Pipeline both saves processed tensors to data/processed/ AND returns DataLoaders in memory
- Flat YAML config with top-level sections: data, model, training, federation
- Environment variable interpolation supported (e.g., ${DATA_DIR:-./data}) for portability
- Config validation on load: check all required keys, fail fast with clear error messages
- Heavily commented config.yaml -- each parameter has a one-line explanation (thesis-appendix ready)
- All directory paths configurable via YAML, CWD as default base
- Google-style docstrings for all modules and functions

### Claude's Discretion
- Exact statistical thresholds for feature filtering (variance, correlation cutoffs)
- Specific domain features to include in the initial shortlist
- Internal pipeline logging format (print vs logging module)
- Test framework choice (pytest vs unittest)
- Exact console_scripts entry point name

### Deferred Ideas (OUT OF SCOPE)
- None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Load CICIDS2017 CSV files and clean data (handle inf/NaN values, whitespace column names, constant columns) | CICIDS2017 CSV structure verified: 79 columns + Label, leading whitespace on column names, Inf values in Flow Bytes/s and Flow Packets/s. Friday DDoS file has ~225K rows. Cleaning pattern: strip columns, replace Inf with NaN, dropna, drop constant columns. |
| DATA-02 | Select and engineer features (reduce 78+ raw features to 20-40 informative ones) | Hybrid approach confirmed viable: domain shortlist of DDoS-relevant flow features + statistical filters (zero-variance, correlation >0.95). Target 20-30 features. Full CICIDS2017 column list researched. |
| DATA-03 | Normalize features with StandardScaler fitted on training data only (no data leakage) | scikit-learn StandardScaler is the standard tool. MUST fit on global training set before client partitioning to ensure consistent feature spaces across all clients. Save fitted scaler as .pkl artifact. |
| DATA-04 | Handle class imbalance via weighted cross-entropy loss for DDoS minority class | Friday DDoS file: ~97K BENIGN (43%) vs ~128K DDoS (57%). Note: DDoS is actually the MAJORITY class in this specific file. Class weights should still be computed for correctness and portability to other data splits. Use sklearn.utils.class_weight.compute_class_weight(). |
| DATA-05 | Partition data IID across 2-5 clients with stratified splits maintaining class ratios | Use sklearn StratifiedShuffleSplit or StratifiedKFold for IID partitioning. Apply AFTER global train/test split. Each client partition preserves the global class ratio. Verify with per-partition class distribution logging. |
| INFR-01 | Configuration file (YAML) for all hyperparameters | PyYAML with custom loader for ${VAR:-default} env var interpolation. Flat structure with sections: data, model, training, federation. Config validation on load. |
| INFR-02 | Reproducibility via fixed seeds, pyproject.toml, and documented hyperparameters | Seed utility setting random.seed, numpy.random.seed, torch.manual_seed, torch.cuda.manual_seed_all. pyproject.toml with pinned dependencies. |
</phase_requirements>

## Standard Stack

### Core (Phase 1 Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11.x | Runtime | Best balance of performance and ML library compatibility. 3.11 has 10-60% speedups over 3.10. Avoid 3.12+ for potential C extension issues. **HIGH confidence** |
| pandas | >=2.2.0 | CSV loading, data cleaning, feature analysis | Standard for tabular data. CICIDS2017 ships as CSV. v2.x has Arrow backend for performance. **HIGH confidence** |
| numpy | >=1.26.0 | Numerical operations, Inf/NaN detection | Underpins all ML computation. Pin >=1.26 for NumPy 2.0 transition compatibility. **HIGH confidence** |
| scikit-learn | >=1.5.0 | StandardScaler, train_test_split, class_weight, stratified splitting | The standard toolkit for ML preprocessing. Provides everything needed for normalization, splitting, and class weight computation. **HIGH confidence** |
| torch | >=2.4.0 | TensorDataset, DataLoader, tensor operations, seed utilities | Current stable is 2.10.0 (Jan 2026). Any 2.x works. CPU-only sufficient for data pipeline. **HIGH confidence** |
| PyYAML | >=6.0 | Configuration file loading | Lightweight YAML parser. Needs custom constructor for env var interpolation. **HIGH confidence** |
| joblib | (bundled with sklearn) | Saving fitted StandardScaler as .pkl | sklearn's persistence tool. Already installed as sklearn dependency. **HIGH confidence** |

### Development Tools

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0.0 | Unit testing | Test data pipeline correctness: no NaN/Inf, correct feature count, class ratios, scaler fitting. **Recommendation: use pytest over unittest** -- better assertions, fixtures, parametrize. |
| ruff | >=0.5.0 | Linting and formatting | Replaces flake8+black+isort. Single tool, Rust-based, fast. |
| tqdm | >=4.66.0 | Progress bars | Data loading progress during CSV reads. |

### Pinned for Future Phases (install now, use later)

| Library | Version | Purpose | Why Pin Now |
|---------|---------|---------|-------------|
| flwr | >=1.13.0 | Federated Learning framework | Pin now to avoid version drift. NumPyClient + start_server still work with backward compatibility in 1.26.x but are deprecated since 1.7/1.13. Phase 3 will decide classic vs new API. **MEDIUM confidence on exact API surface** |
| matplotlib | >=3.9.0 | Plotting (Phase 4) | Install now for potential EDA during development. |
| seaborn | >=0.13.0 | Statistical visualization (Phase 4) | Install alongside matplotlib. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyYAML + custom env interpolation | OmegaConf / Hydra | OmegaConf has built-in ${env:VAR,default} interpolation but adds a dependency. PyYAML with ~20 lines of custom constructor is sufficient for a single config file. OmegaConf is better if config complexity grows. |
| PyYAML + custom env interpolation | pyyaml-env-tag | Small library that adds !ENV tag for env vars. Adds a dependency for minimal benefit over custom constructor. |
| pandas CSV loading | Polars | Polars is 5-10x faster but CICIDS2017 (~225K rows for Friday DDoS) loads in <2s with pandas. Not worth the learning curve. |
| pytest | unittest | unittest is stdlib but pytest has better assertion messages, fixtures via conftest.py, and parametrize for testing multiple partitions. pytest is the de facto standard. |
| joblib .pkl for scaler | pickle directly | joblib is optimized for numpy arrays (which StandardScaler uses internally). Already bundled with sklearn. |

**Installation:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash

# Install all dependencies via editable install
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure (Phase 1 Scope)

```
6g-federated-ids/
├── src/
│   └── federated_ids/
│       ├── __init__.py          # Package init, version
│       ├── config.py            # YAML loading, env var interpolation, validation
│       ├── seed.py              # Global seed utility (random, numpy, torch, cuda)
│       ├── device.py            # Auto-detect CUDA/MPS/CPU
│       └── data/
│           ├── __init__.py
│           ├── loader.py        # CSV loading, column cleaning, label mapping
│           ├── preprocess.py    # Feature selection, normalization, pipeline orchestration
│           └── partition.py     # IID stratified partitioning, DataLoader creation
├── tests/
│   ├── conftest.py              # Shared fixtures (small test DataFrame, config)
│   ├── test_loader.py           # CSV loading, cleaning assertions
│   ├── test_preprocess.py       # Feature selection, scaler fitting, no NaN/Inf
│   ├── test_partition.py        # Stratified splits, class ratio preservation
│   └── test_config.py          # Config loading, validation, env var interpolation
├── config/
│   └── default.yaml             # Heavily commented default config
├── data/
│   ├── raw/                     # User places CSVs here (gitignored)
│   └── processed/               # Pipeline output (gitignored)
├── outputs/                     # Plots, checkpoints, logs (gitignored)
├── pyproject.toml               # Project metadata, dependencies, tool config
├── .gitignore                   # data/, outputs/, .venv/, __pycache__/
└── README.md                    # Setup, data download instructions
```

### Pattern 1: Three-Stage Pipeline with Validation Gates

**What:** Split preprocessing into three distinct stages (load -> preprocess -> partition), each with explicit validation assertions at the boundary.

**When to use:** Always for this pipeline. Each stage has different failure modes and should be independently testable.

```python
# Stage 1: loader.py -- Load and clean raw CSV
def load_cicids2017(data_dir: str, files: list[str]) -> pd.DataFrame:
    """Load CICIDS2017 CSVs, strip column whitespace, map labels."""
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Clean Inf/NaN
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)

    # Map labels: BENIGN -> 0, all DDoS variants -> 1
    ddos_labels = ['DDoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest']
    combined['Label'] = combined['Label'].str.strip()
    combined['Label'] = combined['Label'].apply(lambda x: 1 if x in ddos_labels else 0)

    # VALIDATION GATE
    assert not combined.isin([np.inf, -np.inf]).any().any(), "Inf values remain"
    assert not combined.isna().any().any(), "NaN values remain"
    assert set(combined['Label'].unique()) == {0, 1}, "Labels not binary"

    return combined

# Stage 2: preprocess.py -- Feature selection, normalization
def preprocess(df: pd.DataFrame, config: dict) -> tuple:
    """Select features, split train/test, fit scaler, return processed data."""
    # Drop identifier columns
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                 'Destination Port', 'Timestamp', 'Protocol', 'Label']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df['Label'].values.astype(np.int64)

    # Global train/test split BEFORE partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config['seed']
    )

    # Fit StandardScaler on training data ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # VALIDATION GATE
    assert not np.isnan(X_train).any(), "NaN after scaling"
    assert not np.isinf(X_train).any(), "Inf after scaling"

    return X_train, X_test, y_train, y_test, scaler, feature_cols

# Stage 3: partition.py -- IID partitioning and DataLoader creation
def partition_iid(X: np.ndarray, y: np.ndarray, num_clients: int,
                  seed: int) -> list[tuple]:
    """Stratified IID partition into num_clients shards."""
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    partitions = []
    for _, idx in skf.split(X, y):
        partitions.append((X[idx], y[idx]))

    # VALIDATION GATE: verify class ratios per partition
    global_ratio = np.mean(y)
    for i, (_, y_part) in enumerate(partitions):
        part_ratio = np.mean(y_part)
        assert abs(part_ratio - global_ratio) < 0.05, \
            f"Client {i} class ratio {part_ratio:.3f} deviates from global {global_ratio:.3f}"

    return partitions
```

### Pattern 2: YAML Config with Environment Variable Interpolation

**What:** Custom PyYAML constructor that resolves `${VAR:-default}` patterns using os.environ.

**When to use:** For the config loading system. Enables portability across machines/clusters.

```python
import re
import os
import yaml

_ENV_PATTERN = re.compile(r'\$\{(\w+)(?::-(.*?))?\}')

def _env_constructor(loader, node):
    """Resolve ${VAR:-default} patterns in YAML values."""
    value = loader.construct_scalar(node)
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ''
        return os.environ.get(var_name, default)
    return _ENV_PATTERN.sub(replacer, value)

def _create_loader():
    """Create a YAML loader with env var interpolation."""
    loader = yaml.SafeLoader
    # Apply pattern to all string values
    loader.add_implicit_resolver('!env', _ENV_PATTERN, None)
    loader.add_constructor('!env', _env_constructor)
    return loader

def load_config(path: str) -> dict:
    """Load and validate YAML config with env var interpolation."""
    with open(path) as f:
        config = yaml.load(f, Loader=_create_loader())
    _validate_config(config)
    return config
```

### Pattern 3: Global Seed Utility

**What:** Single function that seeds all RNG sources for reproducibility.

**When to use:** Called once at the start of any entry point.

```python
import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Pattern 4: Device Auto-Detection

**What:** Detect the best available compute device.

```python
import torch

def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Anti-Patterns to Avoid

- **Fitting scaler on full dataset before splitting:** Introduces data leakage. Test set statistics influence training feature scaling. Evaluation metrics become unreliable.
- **Hardcoding feature columns by index:** CICIDS2017 column ordering varies between download sources. Select by column name, fail loudly if missing.
- **Independent per-client scalers:** Each client would map the same raw value to different scaled values. Use a single global scaler fitted on the global training set.
- **Loading all 8 CSVs when only Friday DDoS is needed:** Wastes memory and introduces irrelevant attack types. Load only what CONTEXT.md specifies.
- **Using float64 throughout:** 2x memory for no accuracy benefit with an MLP. Convert to float32 immediately after preprocessing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature normalization | Custom mean/std computation | `sklearn.preprocessing.StandardScaler` | Handles edge cases (zero variance columns), provides fit/transform separation, serializable with joblib |
| Train/test split with stratification | Manual index shuffling | `sklearn.model_selection.train_test_split(stratify=y)` | Guarantees class ratio preservation, handles edge cases with small classes |
| Stratified K-fold partitioning | Manual partition loop | `sklearn.model_selection.StratifiedKFold` | Ensures each fold has the same class distribution as the full dataset |
| Class weight computation | Manual counting + formula | `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` | Handles multi-class correctly, returns weights suitable for torch.nn.CrossEntropyLoss |
| YAML parsing | Custom parser | `PyYAML` with custom env var constructor | Robust, handles all YAML edge cases, well-tested |
| Tensor dataset creation | Manual batching | `torch.utils.data.TensorDataset` + `DataLoader` | Handles batching, shuffling, multiprocessing data loading |
| Object serialization | Custom save/load | `joblib.dump` / `joblib.load` for sklearn objects | Optimized for numpy arrays, handles compression |

**Key insight:** The entire preprocessing pipeline is a composition of well-tested sklearn and PyTorch utilities. The only custom code needed is the CICIDS2017-specific cleaning logic (column stripping, label mapping, identifier column removal) and the pipeline orchestration.

## Common Pitfalls

### Pitfall 1: CICIDS2017 Infinity Values Crash Training Silently
**What goes wrong:** `Flow Bytes/s` and `Flow Packets/s` contain `inf` values from CICFlowMeter dividing by zero for zero-duration flows. These propagate through StandardScaler to produce NaN, which makes PyTorch loss become NaN.
**Why it happens:** CICFlowMeter tool bug. Pandas reads `inf` as `float('inf')` which NumPy and PyTorch propagate silently.
**How to avoid:** Immediately after loading: `df.replace([np.inf, -np.inf], np.nan, inplace=True)` then `df.dropna()`. Assert zero Inf/NaN remain before proceeding.
**Warning signs:** Loss becomes NaN on first forward pass, model predicts all-same values.

### Pitfall 2: Column Name Whitespace in CICIDS2017 CSVs
**What goes wrong:** Column names have leading/trailing whitespace (e.g., ` Destination Port` vs `Destination Port`). KeyError when selecting features by name.
**Why it happens:** CICFlowMeter inconsistent output formatting across CSV generation runs.
**How to avoid:** First operation after pd.read_csv: `df.columns = df.columns.str.strip()`. Also strip Label values: `df['Label'] = df['Label'].str.strip()`.
**Warning signs:** KeyError on column access, unexpected column count after merge.

### Pitfall 3: Data Leakage Through Scaler Fitting
**What goes wrong:** StandardScaler fitted on full dataset (train + test) before splitting. Test metrics are inflated by 5-20%.
**Why it happens:** Test set statistics influence training feature scaling.
**How to avoid:** Split FIRST, then fit scaler on training set only, then transform both train and test. In this project: global train/test split -> fit scaler on train -> transform everything -> partition across clients.
**Warning signs:** Test performance suspiciously close to train performance.

### Pitfall 4: Identifier Columns Create Data Leakage
**What goes wrong:** Leaving Source IP, Destination IP, Source Port, Destination Port, Flow ID, or Timestamp in the feature set. Model memorizes IP patterns instead of learning attack signatures.
**Why it happens:** These columns are numeric and correlate with attack labels (attacks come from specific IPs/ports).
**How to avoid:** Explicitly drop all identifier columns before feature selection. The CONTEXT.md specifies this.
**Warning signs:** Feature importance analysis shows IPs/ports as top features, model degrades on different day's data.

### Pitfall 5: Class Ratio Imbalance After IID Partitioning
**What goes wrong:** Random partitioning without stratification can give one client mostly benign and another mostly DDoS, especially with small client counts (2-3).
**Why it happens:** With 2-5 clients, random sampling variance is significant.
**How to avoid:** Use `StratifiedKFold` or `StratifiedShuffleSplit` which guarantees class ratio preservation in each partition. Verify by logging per-partition class distributions.
**Warning signs:** One client shows much higher/lower accuracy than others.

### Pitfall 6: Friday DDoS File Class Distribution Surprise
**What goes wrong:** Assuming DDoS is the minority class. In the Friday-WorkingHours-Afternoon-DDos file specifically, DDoS is actually the MAJORITY (~128K DDoS vs ~97K BENIGN, roughly 57% vs 43%).
**Why it happens:** The overall CICIDS2017 dataset is heavily benign-biased, but the Friday DDoS file is specifically filtered for the DDoS attack period.
**How to avoid:** Compute class weights from the actual data, not from assumptions. Log class distribution immediately after loading. The class weights will still be useful for ensuring balanced learning.
**Warning signs:** Class weights seem inverted from expectations.

### Pitfall 7: PyYAML Env Var Constructor Modifies Global Loader
**What goes wrong:** Adding an implicit resolver or constructor to `yaml.SafeLoader` modifies the class globally, affecting all YAML loading in the process (including third-party libraries).
**Why it happens:** PyYAML's `add_implicit_resolver` and `add_constructor` are class methods that modify the loader class itself.
**How to avoid:** Create a custom loader class that inherits from SafeLoader, add constructors to the subclass only:
```python
class EnvYamlLoader(yaml.SafeLoader):
    pass
EnvYamlLoader.add_implicit_resolver('!env', _ENV_PATTERN, None)
EnvYamlLoader.add_constructor('!env', _env_constructor)
```
**Warning signs:** Other YAML loading in the project behaves unexpectedly.

## Code Examples

### CICIDS2017 Column List (Expected After Stripping Whitespace)

The CICIDS2017 MachineLearningCSV files contain ~79 feature columns plus a Label column. Key columns (verified from dataset documentation and published analyses):

**Identifier columns to DROP:**
- `Flow ID` -- unique flow identifier
- `Source IP` -- source IP address
- `Source Port` -- source port number
- `Destination IP` -- destination IP address
- `Destination Port` -- destination port number
- `Timestamp` -- flow timestamp
- `Protocol` -- network protocol (keep or drop based on discretion)

**Domain-Informed DDoS Feature Shortlist (recommendation for initial selection):**

Flow-level features (most relevant for DDoS detection):
- `Flow Duration` -- duration of the flow
- `Total Fwd Packets`, `Total Backward Packets` -- packet counts
- `Total Length of Fwd Packets`, `Total Length of Bwd Packets` -- byte counts
- `Flow Bytes/s`, `Flow Packets/s` -- rate features (contain Inf, clean first)
- `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`, `Flow IAT Min` -- inter-arrival times
- `Fwd IAT Total`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Max`, `Fwd IAT Min`
- `Bwd IAT Total`, `Bwd IAT Mean`, `Bwd IAT Std`, `Bwd IAT Max`, `Bwd IAT Min`

Packet-level features:
- `Fwd Packet Length Max/Min/Mean/Std` -- forward packet sizes
- `Bwd Packet Length Max/Min/Mean/Std` -- backward packet sizes
- `Min Packet Length`, `Max Packet Length`, `Packet Length Mean`, `Packet Length Std`, `Packet Length Variance`
- `Average Packet Size`

Flag features (important for DDoS patterns):
- `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`
- `Fwd PSH Flags`, `Bwd PSH Flags`
- `Down/Up Ratio`

Rate features:
- `Fwd Packets/s`, `Bwd Packets/s`

Header features:
- `Fwd Header Length`, `Bwd Header Length`

**Statistical filtering thresholds (recommendation):**
- Zero-variance threshold: drop features where `variance < 1e-10`
- Near-constant threshold: drop features where >99% of values are the same
- Correlation threshold: drop one of each pair with Pearson correlation > 0.95 (keep the one with higher variance)

### CICIDS2017 File Information

**Files to load (per CONTEXT.md decision -- Friday DDoS only):**
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (~77 MB, ~225K rows)
  - BENIGN: ~97,718 rows (43.3%)
  - DDoS: ~128,027 rows (56.7%)

**Other Friday files (NOT loaded per CONTEXT.md, but be aware):**
- `Friday-WorkingHours-Morning.pcap_ISCX.csv` (~58 MB, contains Bot attacks)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (~77 MB, contains PortScan attacks)

**All 8 CICIDS2017 MachineLearningCSV files (for reference):**
1. `Monday-WorkingHours.pcap_ISCX.csv` (benign only)
2. `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP/SSH brute force)
3. `Wednesday-workingHours.pcap_ISCX.csv` (DoS/Heartbleed -- note lowercase 'w')
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
5. `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Bot)
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS -- PRIMARY)

### DDoS Label Mapping

Per CONTEXT.md, these labels should all map to 1 (attack):
```python
DDOS_LABELS = {'DDoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'}
BENIGN_LABEL = 'BENIGN'

# Note: The Friday DDoS file primarily contains 'DDoS' and 'BENIGN' labels.
# The DoS variants (Hulk, GoldenEye, etc.) appear in other day files
# (Wednesday). Including them in the mapping is defensive coding for
# potential future expansion to load additional files.
```

### pyproject.toml Template

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "federated-ids-6g"
version = "0.1.0"
description = "Privacy-preserving Federated Learning IDS for 6G Edge Networks"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.4.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "flwr>=1.13.0",
    "tqdm>=4.66.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.5.0",
]

[project.scripts]
federated-ids-preprocess = "federated_ids.data.preprocess:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Default Config YAML Structure

```yaml
# ============================================================
# Federated IDS Configuration
# ============================================================
# This file controls all hyperparameters for the federated
# learning intrusion detection system. Each parameter includes
# a brief explanation suitable for thesis documentation.
#
# Environment variables: Use ${VAR_NAME:-default} syntax.
# Example: ${DATA_DIR:-./data} uses $DATA_DIR if set, else ./data
# ============================================================

# --- Data Pipeline ---
data:
  # Path to directory containing raw CICIDS2017 CSV files
  raw_dir: "${DATA_DIR:-./data}/raw"
  # Path to directory for preprocessed pipeline output
  processed_dir: "${DATA_DIR:-./data}/processed"
  # CSV files to load (Friday DDoS for binary classification)
  files:
    - "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
  # Fraction of data held out for global test set (before client partitioning)
  test_size: 0.2
  # Target number of features after selection (informational, actual count
  # depends on statistical filtering)
  target_features: 30
  # Correlation threshold: drop one feature from pairs with |r| above this
  correlation_threshold: 0.95
  # Variance threshold: drop features with variance below this value
  variance_threshold: 1.0e-10

# --- Model Architecture ---
model:
  # Hidden layer sizes for the MLP (3 layers as per requirements)
  hidden_layers: [128, 64, 32]
  # Dropout probability for regularization
  dropout: 0.3
  # Number of output classes (binary: benign vs DDoS)
  num_classes: 2

# --- Training ---
training:
  # Learning rate for Adam optimizer
  learning_rate: 0.001
  # Number of local training epochs per FL round
  local_epochs: 1
  # Mini-batch size for DataLoader
  batch_size: 64
  # Use class-weighted cross-entropy loss to handle imbalance
  weighted_loss: true

# --- Federation ---
federation:
  # Number of simulated federated clients
  num_clients: 3
  # Number of federated learning rounds
  num_rounds: 20
  # Fraction of clients participating in each round
  fraction_fit: 1.0

# --- Infrastructure ---
seed: 42
# Directory for output artifacts (plots, checkpoints, metrics)
output_dir: "${OUTPUT_DIR:-./outputs}"
# Logging level: DEBUG, INFO, WARNING, ERROR
log_level: "INFO"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `flwr.client.start_numpy_client()` | `flwr.client.start_client(client.to_client())` | Flower 1.7 (deprecated) | Must use `to_client()` adapter; `start_numpy_client` still works but emits deprecation warning |
| `flwr.server.start_server()` / `start_client()` | `ClientApp` / `ServerApp` + `flwr run` | Flower 1.13 (deprecated) | Classic API still works with backward compat in 1.26.x. Phase 3 must decide which API to use. |
| PyTorch <2.0 training loops | PyTorch 2.x with optional `torch.compile()` | PyTorch 2.0 (2023) | For MLP on tabular data, `torch.compile()` is unnecessary. Standard training loop is fine. |
| `requirements.txt` | `pyproject.toml` (PEP 621) | 2022+ | Modern standard. One file for metadata, dependencies, and tool config. |
| `setup.py` / `setup.cfg` | `pyproject.toml` with setuptools backend | 2022+ | Simpler, declarative, no executable setup script. |

**Deprecated/outdated:**
- `flwr.client.start_numpy_client()`: deprecated since Flower 1.7. Use `start_client()` with `to_client()`.
- `flwr.server.start_server()` + `flwr.client.start_client()`: deprecated since Flower 1.13 in favor of `ServerApp`/`ClientApp`. Still functional in 1.26.x for backward compatibility.
- `requirements.txt` for project dependencies: replaced by pyproject.toml.

## Open Questions

1. **Flower API decision for Phase 3**
   - What we know: Flower 1.26.1 is current. `NumPyClient` + `start_server`/`start_client` still work but are deprecated since 1.7/1.13. The new `ClientApp`/`ServerApp` API is the recommended approach.
   - What's unclear: Whether the project should use the classic (more tutorial coverage, simpler for learning) or new API (future-proof, recommended by Flower team).
   - Recommendation: Pin `flwr>=1.13.0` in Phase 1. Defer the API choice to Phase 3 research. Phase 1 only needs to install the dependency, not use it.

2. **Wednesday DoS file inclusion**
   - What we know: CONTEXT.md says "Load only DDoS-relevant CSV files (Friday-WorkingHours)". The Wednesday file contains DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest which are mapped to label 1.
   - What's unclear: Whether the user might want to include Wednesday DoS data for more attack variety.
   - Recommendation: Follow CONTEXT.md strictly -- Friday DDoS file only. The label mapping includes DoS variants defensively, but we only load the Friday file.

3. **Exact feature count after statistical filtering**
   - What we know: Target is 20-30 features. Starting from ~79 columns minus ~7 identifier columns = ~72 numeric features. After zero-variance + correlation filtering, expect 25-40 survivors.
   - What's unclear: Exact count depends on the data. Cannot predict without running the pipeline.
   - Recommendation: Log the full feature selection report (which features dropped, why). Make the thresholds configurable in YAML. Target is informational, not a hard constraint.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | CSV loading produces DataFrame with stripped columns, no Inf/NaN, binary labels | unit | `python -m pytest tests/test_loader.py -x` | Wave 0 |
| DATA-02 | Feature selection reduces columns to 20-40, zero-variance and correlated features removed | unit | `python -m pytest tests/test_preprocess.py::test_feature_selection -x` | Wave 0 |
| DATA-03 | StandardScaler fitted on train only, test/partitions transformed consistently | unit | `python -m pytest tests/test_preprocess.py::test_scaler_no_leakage -x` | Wave 0 |
| DATA-04 | Class weights computed from actual data distribution, saved to processed/ | unit | `python -m pytest tests/test_preprocess.py::test_class_weights -x` | Wave 0 |
| DATA-05 | IID partitioning preserves class ratios across 2-5 clients, verified by distribution check | unit | `python -m pytest tests/test_partition.py -x` | Wave 0 |
| INFR-01 | YAML config loads with env var interpolation, validates required keys, fails on missing | unit | `python -m pytest tests/test_config.py -x` | Wave 0 |
| INFR-02 | Setting seed produces identical outputs on repeated runs | unit | `python -m pytest tests/test_preprocess.py::test_reproducibility -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -x -q` (quick, fail-fast)
- **Per wave merge:** `python -m pytest tests/ -v` (full suite, verbose)
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps

- [ ] `tests/conftest.py` -- shared fixtures: small synthetic DataFrame mimicking CICIDS2017 structure (~100 rows), test config dict
- [ ] `tests/test_loader.py` -- covers DATA-01: CSV loading, column stripping, Inf/NaN removal, label mapping
- [ ] `tests/test_preprocess.py` -- covers DATA-02, DATA-03, DATA-04, INFR-02: feature selection, scaler fitting, class weights, reproducibility
- [ ] `tests/test_partition.py` -- covers DATA-05: stratified IID partitioning, class ratio verification, configurable client count
- [ ] `tests/test_config.py` -- covers INFR-01: YAML loading, env var interpolation, validation errors
- [ ] Framework install: `pip install -e ".[dev]"` -- pytest included in dev extras

**Testing approach for data pipeline:** Tests should NOT require the actual CICIDS2017 CSV files. Create synthetic test DataFrames in conftest.py that mimic the CICIDS2017 structure (same column names, similar data types, include deliberate Inf/NaN values for testing cleanup). This allows CI/CD and testing without downloading 77+ MB of data.

## Sources

### Primary (HIGH confidence)
- [CICIDS2017 dataset documentation (UNB)](https://www.unb.ca/cic/datasets/ids-2017.html) -- dataset format, file list, attack schedule
- [PyTorch documentation](https://pytorch.org/docs/stable) -- DataLoader, TensorDataset, seed utilities
- [scikit-learn documentation](https://scikit-learn.org/stable) -- StandardScaler, train_test_split, StratifiedKFold, compute_class_weight
- [pandas documentation](https://pandas.pydata.org/docs/) -- CSV loading, column operations
- CICIDS2017 preprocessing patterns from published FL-IDS literature

### Secondary (MEDIUM confidence)
- [Flower PyPI](https://pypi.org/project/flwr/) -- version 1.26.1 confirmed current stable
- [Flower NumPyClient docs](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html) -- NumPyClient still documented but start_numpy_client deprecated since 1.7
- [Flower upgrade to 1.13 guide](https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html) -- start_server/start_client deprecated in favor of ServerApp/ClientApp
- [PyTorch releases](https://github.com/pytorch/pytorch/releases) -- PyTorch 2.10.0 released Jan 2026
- [pyaml_env / pyyaml-env-tag](https://github.com/mkaranasou/pyaml_env) -- env var interpolation patterns for PyYAML

### Tertiary (LOW confidence)
- Friday DDoS file row counts (~97K benign, ~128K DDoS) -- from search results citing published analyses, should be verified against actual downloaded data
- Exact feature list after statistical filtering -- cannot predict without running on actual data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are mature, well-documented, widely used in ML/IDS research
- Architecture: HIGH -- three-stage pipeline with validation gates is a proven pattern, project structure follows user decisions precisely
- CICIDS2017 specifics: HIGH -- dataset issues are extensively documented in literature (Inf/NaN, whitespace, class distribution)
- Pitfalls: HIGH -- seven of ten identified pitfalls apply to this phase, all are well-documented in IDS literature
- Flower version: MEDIUM -- confirmed 1.26.1 is current and backward compat works, but exact API deprecation timeline in future versions is uncertain
- Feature selection outcome: LOW -- exact features surviving statistical filters depends on the actual data

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (30 days -- stack is stable, CICIDS2017 is static)
