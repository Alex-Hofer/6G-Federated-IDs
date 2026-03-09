"""Shared test fixtures for the federated IDS test suite.

Provides reusable fixtures for configuration testing, including
a minimal valid config dict and temporary config files. Also provides
synthetic CICIDS2017 DataFrames for data pipeline testing and
synthetic model/training fixtures for training loop tests.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from federated_ids.model.model import MLP


@pytest.fixture
def sample_config_dict():
    """Return a minimal valid configuration dictionary.

    Matches the structure of config/default.yaml with all required
    sections and keys present, using simple default values suitable
    for testing.
    """
    return {
        "data": {
            "raw_dir": "./data/raw",
            "processed_dir": "./data/processed",
            "files": ["test-file.csv"],
            "test_size": 0.2,
            "target_features": 30,
            "correlation_threshold": 0.95,
            "variance_threshold": 1e-10,
        },
        "model": {
            "hidden_layers": [128, 64, 32],
            "dropout": 0.3,
            "num_classes": 2,
        },
        "training": {
            "learning_rate": 0.001,
            "local_epochs": 1,
            "batch_size": 64,
            "weighted_loss": True,
        },
        "federation": {
            "num_clients": 3,
            "num_rounds": 20,
            "fraction_fit": 1.0,
        },
        "seed": 42,
        "output_dir": "./outputs",
        "log_level": "INFO",
    }


@pytest.fixture
def tmp_config_file(sample_config_dict):
    """Write a valid config dict to a temporary YAML file.

    Returns:
        Path to the temporary YAML file. The file is automatically
        cleaned up after the test completes.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(sample_config_dict, f)
        path = f.name

    yield path

    os.unlink(path)


# ---------------------------------------------------------------------------
# CICIDS2017 synthetic data fixtures
# ---------------------------------------------------------------------------

# All 78 feature columns expected in CICIDS2017 MachineLearningCSV files,
# plus the Label column.  Leading whitespace on some names deliberately
# mimics the real CSV quirk (Pitfall 2).
_CICIDS_COLUMNS = [
    " Destination Port",  # leading whitespace
    " Flow Duration",  # leading whitespace
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    "FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
]


@pytest.fixture
def sample_cicids_df():
    """Create a synthetic DataFrame mimicking CICIDS2017 structure.

    Returns a DataFrame with ~100 rows containing:
    - Leading whitespace on some column names (real CSV quirk)
    - np.inf values in ``Flow Bytes/s`` and ``Flow Packets/s`` columns
    - NaN values scattered in other columns
    - Label column with values including leading whitespace
    - Mix of benign (~40%) and DDoS (~60%) rows
    """
    rng = np.random.RandomState(42)
    n_rows = 100

    data = {}
    for col in _CICIDS_COLUMNS:
        data[col] = rng.rand(n_rows).astype(np.float64) * 1000

    # Inject np.inf in Flow Bytes/s (indices 5, 15, 25)
    data["Flow Bytes/s"][5] = np.inf
    data["Flow Bytes/s"][15] = -np.inf
    data["Flow Bytes/s"][25] = np.inf

    # Inject np.inf in Flow Packets/s (index 35)
    data[" Flow Packets/s"][35] = np.inf

    # Inject NaN in other columns (indices 45, 55)
    data[" Flow Duration"][45] = np.nan
    data[" Total Fwd Packets"][55] = np.nan

    # Build labels: ~40 benign, ~60 DDoS (with whitespace on some labels)
    labels = []
    for i in range(n_rows):
        if i < 40:
            # Benign with occasional leading whitespace
            labels.append(" BENIGN" if i % 5 == 0 else "BENIGN")
        elif i < 70:
            labels.append("DDoS")
        elif i < 85:
            labels.append("DoS Hulk")
        else:
            labels.append("DoS GoldenEye")

    data[" Label"] = labels  # leading whitespace on column name

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_csv_file(tmp_path, sample_cicids_df):
    """Write the synthetic CICIDS2017 DataFrame to a CSV file.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
        sample_cicids_df: Synthetic DataFrame from the ``sample_cicids_df`` fixture.

    Returns:
        The ``tmp_path`` directory containing the CSV file
        named ``test_data.csv``.
    """
    csv_path = tmp_path / "test_data.csv"
    sample_cicids_df.to_csv(csv_path, index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Model / training synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_train_data():
    """Create synthetic training data for model/training tests.

    Returns:
        Tuple of (X_train, y_train) as numpy arrays with ~200 samples,
        10 features, binary labels with ~30% class 1 (imbalanced).
    """
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 10

    X = rng.randn(n_samples, n_features).astype(np.float32)
    # ~30% class 1 (imbalanced)
    y = (rng.rand(n_samples) < 0.3).astype(np.int64)

    return X, y


@pytest.fixture
def synthetic_class_weights_file(tmp_path):
    """Write a temporary class_weights.json file with imbalanced weights.

    Returns:
        Path to the temporary JSON file.
    """
    weights = {"0": 0.6, "1": 1.8}
    weights_path = tmp_path / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f)

    return str(weights_path)


@pytest.fixture
def sample_model():
    """Create a small MLP model for fast test execution.

    Returns:
        MLP instance with input_dim=10, hidden_layers=[32, 16],
        num_classes=2, dropout=0.1.
    """
    return MLP(input_dim=10, hidden_layers=[32, 16], num_classes=2, dropout=0.1)
