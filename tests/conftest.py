"""Shared test fixtures for the federated IDS test suite.

Provides reusable fixtures for configuration testing, including
a minimal valid config dict and temporary config files.
"""

import os
import tempfile

import pytest
import yaml


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
