# 6G Federated IDS

Privacy-preserving Intrusion Detection System using Federated Learning for 6G edge networks. A federated MLP classifier detects DDoS attacks on the CICIDS2017 dataset without any client sharing raw network traffic data. Each edge node trains locally on its partition, and only model weight updates are aggregated via the Flower framework.

## Setup

### Prerequisites

- Python 3.11 or 3.12

### Installation

```bash
git clone https://github.com/<your-org>/6g-federated-ids.git
cd 6g-federated-ids
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows cmd
# source .venv/Scripts/activate  # Windows Git Bash
pip install -e ".[dev]"
```

## Data Download

This project uses the **CICIDS2017** dataset from the University of New Brunswick Canadian Institute for Cybersecurity.

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Navigate to the **MachineLearningCSV** download section
3. Download: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
4. Place the file in: `data/raw/`

Expected file properties:
- Size: approximately 77 MB
- Rows: approximately 225,000
- Contains: BENIGN traffic and DDoS attack variants (DDoS, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest)

```
data/
  raw/
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## Usage

### Run the preprocessing pipeline

The pipeline loads raw CSV data, selects features, normalizes, splits train/test, partitions across federated clients, and creates PyTorch DataLoaders.

```bash
python -m federated_ids.data.preprocess
```

With a custom config file:

```bash
python -m federated_ids.data.preprocess --config path/to/config.yaml
```

Or using the installed console script:

```bash
federated-ids-preprocess
```

### Pipeline output

After running, processed artifacts are saved to `data/processed/`:

| File | Description |
|------|-------------|
| `scaler.pkl` | Fitted StandardScaler (joblib) |
| `features.json` | Selected feature names |
| `class_weights.json` | Computed class weights for balanced loss |
| `class_distribution.json` | Per-split class counts |
| `X_train.pt`, `X_test.pt` | Cached feature tensors |
| `y_train.pt`, `y_test.pt` | Cached label tensors |

On subsequent runs, the pipeline detects cached tensors and skips the expensive loading and preprocessing steps.

## Configuration

Configuration is defined in `config/default.yaml`. Key parameters:

| Parameter | Section | Default | Description |
|-----------|---------|---------|-------------|
| `seed` | root | `42` | Global random seed for reproducibility |
| `num_clients` | `federation` | `3` | Number of federated client partitions |
| `test_size` | `data` | `0.2` | Fraction held out as global test set |
| `batch_size` | `training` | `64` | Mini-batch size for DataLoaders |
| `correlation_threshold` | `data` | `0.95` | Drop one feature from pairs with \|r\| above this |
| `target_features` | `data` | `30` | Informational target for feature count after selection |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Base directory for raw and processed data |
| `OUTPUT_DIR` | `./outputs` | Directory for training outputs and plots |

## Testing

Tests use synthetic data fixtures and do not require downloading the CICIDS2017 dataset.

```bash
python -m pytest tests/ -v
```

Run a specific test module:

```bash
python -m pytest tests/test_partition.py -v
```

## Project Structure

```
src/federated_ids/
    __init__.py
    config.py          # YAML config loading with env var interpolation
    seed.py            # Global seed for reproducibility
    device.py          # Auto-detect CUDA/MPS/CPU device
    data/
        __init__.py    # Convenience imports
        __main__.py    # python -m federated_ids.data entry point
        loader.py      # CICIDS2017 CSV loading and cleaning
        preprocess.py  # Feature selection, normalization, train/test split
        partition.py   # IID stratified partitioning and DataLoader creation
    model/             # MLP model definition (Phase 2)
    fl/                # Flower federated learning (Phase 3)
    eval/              # Evaluation metrics and plotting (Phase 4)
config/
    default.yaml       # Default configuration
tests/
    conftest.py        # Shared test fixtures
    test_config.py     # Config loading tests
    test_loader.py     # Data loader tests
    test_preprocess.py # Preprocessing tests
    test_partition.py  # Partitioning and DataLoader tests
```

## License

This project is developed as part of a thesis on federated learning for network intrusion detection in 6G environments.
