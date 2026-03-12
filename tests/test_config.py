"""Tests for configuration loading, validation, and environment variable interpolation.

Covers:
- Loading the default config file and verifying all sections
- Environment variable interpolation with ${VAR:-default} syntax
- Validation errors for missing required keys
- FileNotFoundError for nonexistent config paths
- Seed reproducibility across torch and numpy
- Device detection returning a valid torch.device
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

from federated_ids.config import load_config
from federated_ids.device import get_device
from federated_ids.seed import set_global_seed


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_default_config(self):
        """Loading config/default.yaml returns a dict with all required sections."""
        config = load_config("config/default.yaml")
        assert isinstance(config, dict)
        for section in ["data", "model", "training", "federation", "seed"]:
            assert section in config, f"Missing section: {section}"

    def test_load_from_tmp_file(self, tmp_config_file):
        """Loading a valid temp config file succeeds."""
        config = load_config(tmp_config_file)
        assert config["seed"] == 42
        assert config["data"]["test_size"] == 0.2

    def test_nonexistent_file(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("nonexistent/path/config.yaml")


class TestEnvVarInterpolation:
    """Tests for ${VAR:-default} environment variable interpolation."""

    def test_env_var_interpolation(self, sample_config_dict):
        """Environment variable values are resolved in YAML strings."""
        # Set an env var and create a config referencing it
        os.environ["TEST_FID_DATA_DIR"] = "/custom/data/path"
        try:
            config_dict = sample_config_dict.copy()
            config_dict["data"] = sample_config_dict["data"].copy()
            config_dict["data"]["raw_dir"] = "${TEST_FID_DATA_DIR:-./data}/raw"

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config_dict, f)
                path = f.name

            config = load_config(path)
            assert config["data"]["raw_dir"] == "/custom/data/path/raw"
        finally:
            del os.environ["TEST_FID_DATA_DIR"]
            os.unlink(path)

    def test_env_var_default_fallback(self, sample_config_dict):
        """When env var is not set, the default value after :- is used."""
        # Ensure the variable is NOT set
        os.environ.pop("TEST_FID_NONEXISTENT_VAR", None)

        config_dict = sample_config_dict.copy()
        config_dict["data"] = sample_config_dict["data"].copy()
        config_dict["data"]["raw_dir"] = "${TEST_FID_NONEXISTENT_VAR:-./fallback}/raw"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            path = f.name

        try:
            config = load_config(path)
            assert config["data"]["raw_dir"] == "./fallback/raw"
        finally:
            os.unlink(path)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_missing_key_validation(self):
        """Config missing a required section raises ValueError with descriptive message."""
        # Config without "data" section
        incomplete_config = {
            "model": {"hidden_layers": [64], "dropout": 0.3, "num_classes": 2},
            "training": {"learning_rate": 0.001, "local_epochs": 1, "batch_size": 64},
            "federation": {"num_clients": 3, "num_rounds": 10, "fraction_fit": 1.0},
            "seed": 42,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(incomplete_config, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required section.*data"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_missing_nested_key_validation(self):
        """Config with missing nested key raises ValueError listing the missing key."""
        incomplete_config = {
            "data": {
                "raw_dir": "./data/raw",
                # Missing: processed_dir, files, test_size
            },
            "model": {"hidden_layers": [64], "dropout": 0.3, "num_classes": 2},
            "training": {"learning_rate": 0.001, "local_epochs": 1, "batch_size": 64},
            "federation": {"num_clients": 3, "num_rounds": 10, "fraction_fit": 1.0},
            "seed": 42,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(incomplete_config, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="data.processed_dir"):
                load_config(path)
        finally:
            os.unlink(path)


class TestSeedReproducibility:
    """Tests for the global seed utility."""

    def test_seed_torch_reproducibility(self):
        """set_global_seed(42) produces identical torch.rand outputs on repeated calls."""
        set_global_seed(42)
        first = torch.rand(5)

        set_global_seed(42)
        second = torch.rand(5)

        assert torch.equal(first, second), (
            f"Torch outputs differ after setting same seed:\n"
            f"  first:  {first}\n"
            f"  second: {second}"
        )

    def test_seed_numpy_reproducibility(self):
        """set_global_seed(42) produces identical np.random.rand outputs on repeated calls."""
        set_global_seed(42)
        first = np.random.rand(5)

        set_global_seed(42)
        second = np.random.rand(5)

        np.testing.assert_array_equal(first, second)


class TestDeviceDetection:
    """Tests for device auto-detection."""

    def test_device_returns_valid(self):
        """get_device() returns a torch.device with type cpu, cuda, or mps."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps"), (
            f"Unexpected device type: {device.type}"
        )
