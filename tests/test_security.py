"""Tests for security validation gates and input integrity checks.

Verifies that all validation gates added in Phase 8 (Plans 01 and 02)
correctly reject invalid inputs with appropriate error messages.
Covers: assert replacements (loader, preprocess, partition), FedAvg
aggregation edge cases, scaler JSON serialization, config validation,
and config safety (no mutation).

All tests use synthetic data -- no real CICIDS2017 CSV needed.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from federated_ids.exceptions import DataValidationError


class TestAssertReplacements:
    """Validation gates that replaced assert statements raise DataValidationError."""

    def test_loader_inf_raises(self):
        """DataFrame with Inf values after cleaning triggers DataValidationError."""
        from federated_ids.data.loader import load_cicids2017

        # We need a CSV file that, after cleaning, still has Inf values.
        # The loader replaces Inf with NaN and drops NaN rows, so normally
        # Inf is cleaned. To test the gate, we need a situation where Inf
        # remains after cleaning. Since the loader replaces all Inf with NaN
        # then drops NaN, the Inf gate fires only if something re-introduces
        # Inf after the replace+drop step. In normal flow this is impossible,
        # so we test the validation logic directly on the numeric check path.
        #
        # Instead, we test by constructing a DataFrame that has Inf only in
        # non-numeric columns (which wouldn't be caught by replace), or by
        # testing the validation function directly. Since the loader performs
        # the validation inline, we'll create a CSV where Inf is in a string
        # that gets parsed as Inf.
        #
        # Actually, the simplest approach: the loader checks numeric columns
        # for Inf AFTER replacing Inf->NaN and dropping NaN. If the data is
        # clean, this won't fire. The gate is a safety net. We can verify
        # the gate exists and would fire by testing the condition directly.
        #
        # For a proper integration test, we'll monkeypatch DataFrame.replace
        # to be a no-op so Inf survives.
        import unittest.mock as mock

        rng = np.random.RandomState(42)
        n = 20
        data = {
            "Flow ID": ["id"] * n,
            "Source IP": ["1.2.3.4"] * n,
            "Destination IP": ["5.6.7.8"] * n,
            "Source Port": [80] * n,
            "Destination Port": [443] * n,
            "Timestamp": ["2021-01-01"] * n,
            "Feature1": rng.rand(n) * 100,
            "Label": ["BENIGN"] * 10 + ["DDoS"] * 10,
        }
        df = pd.DataFrame(data)
        # Inject Inf that will survive if replace is skipped
        df.loc[0, "Feature1"] = np.inf
        df.loc[1, "Feature1"] = -np.inf

        # Write to temp CSV, load with monkeypatched replace to skip Inf cleanup
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            df.to_csv(csv_path, index=False)

            original_replace = pd.DataFrame.replace

            def noop_replace(self, *args, **kwargs):
                """Skip replace to let Inf survive."""
                return self

            with mock.patch.object(
                pd.DataFrame, "replace", noop_replace
            ):
                # Also need to patch dropna to not drop the Inf rows
                original_dropna = pd.DataFrame.dropna

                def noop_dropna(self, *args, **kwargs):
                    """Skip dropna to let Inf rows survive."""
                    return None  # inplace=True returns None

                with mock.patch.object(
                    pd.DataFrame, "dropna", noop_dropna
                ):
                    with pytest.raises(DataValidationError, match="Inf"):
                        load_cicids2017(tmpdir, ["test.csv"])

    def test_loader_nan_raises(self):
        """DataFrame with NaN values after cleaning triggers DataValidationError."""
        from federated_ids.data.loader import load_cicids2017
        import unittest.mock as mock

        rng = np.random.RandomState(42)
        n = 20
        data = {
            "Flow ID": ["id"] * n,
            "Source IP": ["1.2.3.4"] * n,
            "Destination IP": ["5.6.7.8"] * n,
            "Source Port": [80] * n,
            "Destination Port": [443] * n,
            "Timestamp": ["2021-01-01"] * n,
            "Feature1": rng.rand(n) * 100,
            "Label": ["BENIGN"] * 10 + ["DDoS"] * 10,
        }
        df = pd.DataFrame(data)
        df.loc[0, "Feature1"] = np.nan

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            df.to_csv(csv_path, index=False)

            # Patch dropna to be a no-op so NaN survives
            def noop_dropna(self, *args, **kwargs):
                return None  # inplace=True returns None

            with mock.patch.object(pd.DataFrame, "dropna", noop_dropna):
                with pytest.raises(DataValidationError, match="NaN"):
                    load_cicids2017(tmpdir, ["test.csv"])

    def test_loader_bad_labels_raises(self):
        """DataFrame with labels {0, 1, 2} triggers DataValidationError."""
        from federated_ids.data.loader import load_cicids2017
        import unittest.mock as mock

        rng = np.random.RandomState(42)
        n = 20
        data = {
            "Flow ID": ["id"] * n,
            "Source IP": ["1.2.3.4"] * n,
            "Destination IP": ["5.6.7.8"] * n,
            "Source Port": [80] * n,
            "Destination Port": [443] * n,
            "Timestamp": ["2021-01-01"] * n,
            "Feature1": rng.rand(n) * 100,
            "Label": ["BENIGN"] * 10 + ["DDoS"] * 5 + ["Unknown"] * 5,
        }
        df = pd.DataFrame(data)

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            df.to_csv(csv_path, index=False)

            # "Unknown" maps to 0 (not in DDOS_LABELS), so labels will be {0, 1}
            # which is valid. To get labels {0, 1, 2} we need to patch the
            # label mapping. Instead, we'll directly test the validation gate
            # by checking that the function validates unique_labels == {0, 1}.
            # Since load_cicids2017 maps everything to 0 or 1, we can't get
            # {0, 1, 2} without patching. Let's patch the apply to inject a 2.
            original_apply = pd.Series.apply

            def patched_apply(self, func, *args, **kwargs):
                result = original_apply(self, func, *args, **kwargs)
                if hasattr(result, 'iloc') and len(result) > 0:
                    # Check if this is the label mapping (result contains 0s and 1s)
                    if set(result.unique()).issubset({0, 1}):
                        result.iloc[-1] = 2
                return result

            with mock.patch.object(pd.Series, "apply", patched_apply):
                with pytest.raises(DataValidationError, match=r"\{.*2.*\}"):
                    load_cicids2017(tmpdir, ["test.csv"])

    def test_preprocess_nan_raises(self, sample_cicids_df, sample_config_dict, tmp_path):
        """X_train with NaN after scaling triggers DataValidationError."""
        from federated_ids.data.preprocess import preprocess
        import unittest.mock as mock

        # Create a clean df for preprocessing
        df = sample_cicids_df.copy()
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        ddos = {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
        df["Label"] = df["Label"].apply(lambda x: 1 if x in ddos else 0)

        sample_config_dict["data"]["processed_dir"] = str(tmp_path)

        # Patch StandardScaler.fit_transform to inject NaN
        original_fit_transform = StandardScaler.fit_transform

        def inject_nan_fit_transform(self, X, y=None, **kwargs):
            result = original_fit_transform(self, X, y, **kwargs)
            result[0, 0] = np.nan
            return result

        with mock.patch.object(
            StandardScaler, "fit_transform", inject_nan_fit_transform
        ):
            with pytest.raises(DataValidationError, match="NaN"):
                preprocess(df, sample_config_dict)

    def test_preprocess_inf_raises(self, sample_cicids_df, sample_config_dict, tmp_path):
        """X_train with Inf after scaling triggers DataValidationError."""
        from federated_ids.data.preprocess import preprocess
        import unittest.mock as mock

        df = sample_cicids_df.copy()
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        ddos = {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
        df["Label"] = df["Label"].apply(lambda x: 1 if x in ddos else 0)

        sample_config_dict["data"]["processed_dir"] = str(tmp_path)

        original_fit_transform = StandardScaler.fit_transform

        def inject_inf_fit_transform(self, X, y=None, **kwargs):
            result = original_fit_transform(self, X, y, **kwargs)
            result[0, 0] = np.inf
            return result

        with mock.patch.object(
            StandardScaler, "fit_transform", inject_inf_fit_transform
        ):
            with pytest.raises(DataValidationError, match="Inf"):
                preprocess(df, sample_config_dict)

    def test_preprocess_dtype_raises(self, sample_cicids_df, sample_config_dict, tmp_path):
        """X_train with float64 dtype after scaling triggers DataValidationError."""
        from federated_ids.data.preprocess import preprocess
        import unittest.mock as mock

        df = sample_cicids_df.copy()
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        ddos = {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
        df["Label"] = df["Label"].apply(lambda x: 1 if x in ddos else 0)

        sample_config_dict["data"]["processed_dir"] = str(tmp_path)

        # The preprocess function does:
        #   X_train = scaler.fit_transform(X_train).astype(np.float32)
        # We need the .astype(np.float32) chain to produce float64.
        # Patch fit_transform to return an array whose astype is a no-op.
        original_fit_transform = StandardScaler.fit_transform

        class Float64Array(np.ndarray):
            """ndarray subclass that ignores astype(float32) to stay float64."""

            def astype(self, dtype, *args, **kwargs):
                if dtype == np.float32:
                    return self  # stay float64
                return super().astype(dtype, *args, **kwargs)

        def return_float64(self, X, y=None, **kwargs):
            result = original_fit_transform(self, X, y, **kwargs)
            f64 = result.astype(np.float64).view(Float64Array)
            return f64

        with mock.patch.object(
            StandardScaler, "fit_transform", return_float64
        ):
            with pytest.raises(DataValidationError, match="float64"):
                preprocess(df, sample_config_dict)

    def test_partition_ratio_raises(self):
        """Partition with >5% class ratio deviation triggers DataValidationError."""
        from federated_ids.data.partition import partition_iid

        # Create data with very skewed partition possibility
        # With 2 clients and perfectly separable classes, stratified k-fold
        # should preserve ratio. To trigger the >5% deviation, we need to
        # mock StratifiedKFold to produce a bad split.
        import unittest.mock as mock
        from sklearn.model_selection import StratifiedKFold

        n = 100
        X = np.random.RandomState(42).randn(n, 5).astype(np.float32)
        # 50% class 1 globally
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)

        # Mock StratifiedKFold to produce an imbalanced split
        original_split = StratifiedKFold.split

        def bad_split(self, X, y=None, groups=None):
            """Produce intentionally imbalanced splits."""
            # Client 0: mostly class 0, Client 1: mostly class 1
            indices_0 = list(range(0, 40)) + list(range(50, 55))  # 40 zeros, 5 ones = 11% class1
            indices_1 = list(range(40, 50)) + list(range(55, 100))  # 10 zeros, 45 ones
            yield (np.array(indices_1), np.array(indices_0))
            yield (np.array(indices_0), np.array(indices_1))

        with mock.patch.object(StratifiedKFold, "split", bad_split):
            with pytest.raises(DataValidationError, match="deviates"):
                partition_iid(X, y, num_clients=2, seed=42)


class TestFedAvgValidation:
    """FedAvg aggregation rejects invalid inputs and handles edge cases."""

    def test_empty_results(self):
        """Empty results list triggers DataValidationError."""
        from federated_ids.fl.server import fedavg_aggregate

        with pytest.raises(DataValidationError, match="empty results"):
            fedavg_aggregate([])

    def test_zero_example_clients_skipped(self):
        """Mix of valid and zero-example clients aggregates using only valid."""
        from federated_ids.fl.server import fedavg_aggregate

        params_valid = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        params_zero = [np.array([10.0, 20.0]), np.array([30.0, 40.0])]

        results = [
            (params_valid, 10),  # valid
            (params_zero, 0),    # zero examples -- should be skipped
        ]
        aggregated = fedavg_aggregate(results)

        # Only the valid client contributes, so result equals its params
        np.testing.assert_array_almost_equal(aggregated[0], params_valid[0])
        np.testing.assert_array_almost_equal(aggregated[1], params_valid[1])

    def test_all_zero_example_raises(self):
        """All clients with zero examples triggers DataValidationError."""
        from federated_ids.fl.server import fedavg_aggregate

        results = [
            ([np.array([1.0])], 0),
            ([np.array([2.0])], 0),
        ]
        with pytest.raises(DataValidationError, match="All clients filtered"):
            fedavg_aggregate(results)

    def test_nan_params_skipped(self):
        """Mix of valid and NaN-param clients aggregates using only valid."""
        from federated_ids.fl.server import fedavg_aggregate

        params_valid = [np.array([1.0, 2.0])]
        params_nan = [np.array([float("nan"), 2.0])]

        results = [
            (params_valid, 10),  # valid
            (params_nan, 10),    # NaN params -- should be skipped
        ]
        aggregated = fedavg_aggregate(results)

        np.testing.assert_array_almost_equal(aggregated[0], params_valid[0])

    def test_all_nan_raises(self):
        """All clients with NaN params triggers DataValidationError."""
        from federated_ids.fl.server import fedavg_aggregate

        results = [
            ([np.array([float("nan")])], 10),
            ([np.array([float("nan")])], 5),
        ]
        with pytest.raises(DataValidationError, match="All clients filtered"):
            fedavg_aggregate(results)

    def test_valid_aggregation(self):
        """Normal inputs produce correct weighted average (regression test)."""
        from federated_ids.fl.server import fedavg_aggregate

        # Client 0: params [1.0, 2.0], 10 examples
        # Client 1: params [3.0, 4.0], 30 examples
        # Weighted avg: (1*10 + 3*30)/40 = 100/40 = 2.5
        #               (2*10 + 4*30)/40 = 140/40 = 3.5
        results = [
            ([np.array([1.0, 2.0])], 10),
            ([np.array([3.0, 4.0])], 30),
        ]
        aggregated = fedavg_aggregate(results)

        np.testing.assert_array_almost_equal(
            aggregated[0], np.array([2.5, 3.5])
        )


class TestScalerJson:
    """Scaler JSON save/load round-trip and validation."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save StandardScaler to JSON, load back, verify attributes match."""
        from federated_ids.data.preprocess import _save_scaler_json, _load_scaler_json

        scaler = StandardScaler()
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        scaler.fit(X)

        path = str(tmp_path / "scaler.json")
        _save_scaler_json(scaler, path)
        loaded = _load_scaler_json(path)

        np.testing.assert_array_almost_equal(loaded.mean_, scaler.mean_)
        np.testing.assert_array_almost_equal(loaded.scale_, scaler.scale_)
        np.testing.assert_array_almost_equal(loaded.var_, scaler.var_)
        assert loaded.n_features_in_ == scaler.n_features_in_

    def test_json_envelope_format(self, tmp_path):
        """Saved JSON has version, type, params keys."""
        from federated_ids.data.preprocess import _save_scaler_json

        scaler = StandardScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler.fit(X)

        path = str(tmp_path / "scaler.json")
        _save_scaler_json(scaler, path)

        with open(path) as f:
            envelope = json.load(f)

        assert "version" in envelope, "Missing 'version' key"
        assert "type" in envelope, "Missing 'type' key"
        assert "params" in envelope, "Missing 'params' key"
        assert envelope["type"] == "StandardScaler"
        assert envelope["version"] == "1.0"

    def test_load_mismatched_lengths(self, tmp_path):
        """JSON with array length != n_features_in_ triggers DataValidationError."""
        from federated_ids.data.preprocess import _load_scaler_json

        envelope = {
            "version": "1.0",
            "type": "StandardScaler",
            "params": {
                "mean_": [1.0, 2.0, 3.0],
                "scale_": [1.0, 2.0],  # length 2 != n_features_in_ 3
                "var_": [1.0, 2.0, 3.0],
                "n_features_in_": 3,
            },
        }
        path = str(tmp_path / "scaler.json")
        with open(path, "w") as f:
            json.dump(envelope, f)

        with pytest.raises(DataValidationError, match="length"):
            _load_scaler_json(path)

    def test_load_non_finite_values(self, tmp_path):
        """JSON with NaN/Inf in arrays triggers DataValidationError."""
        from federated_ids.data.preprocess import _load_scaler_json

        envelope = {
            "version": "1.0",
            "type": "StandardScaler",
            "params": {
                "mean_": [1.0, float("nan"), 3.0],
                "scale_": [1.0, 2.0, 3.0],
                "var_": [1.0, 2.0, 3.0],
                "n_features_in_": 3,
            },
        }
        path = str(tmp_path / "scaler.json")
        with open(path, "w") as f:
            json.dump(envelope, f)

        with pytest.raises(DataValidationError, match="non-finite"):
            _load_scaler_json(path)

    def test_pkl_migration_error(self, tmp_path):
        """Only .pkl exists triggers DataValidationError with regenerate instruction."""
        from federated_ids.data.preprocess import _load_scaler_json

        # Create a dummy .pkl file but no .json
        pkl_path = str(tmp_path / "scaler.pkl")
        with open(pkl_path, "wb") as f:
            f.write(b"fake pickle data")

        json_path = str(tmp_path / "scaler.json")

        with pytest.raises(DataValidationError, match="regenerate"):
            _load_scaler_json(json_path)


class TestConfigValidation:
    """Config validation rejects invalid log levels and config paths."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level):
        """All 5 valid log levels are accepted without error."""
        from federated_ids.config import validate_log_level

        result = validate_log_level(level)
        assert result == level

    @pytest.mark.parametrize("level", ["VERBOSE", "info", ""])
    def test_invalid_log_level(self, level):
        """Invalid log levels raise ValueError."""
        from federated_ids.config import validate_log_level

        with pytest.raises(ValueError, match="Invalid log_level"):
            validate_log_level(level)

    @pytest.mark.parametrize("path", ["config.yaml", "path/to/config.yml"])
    def test_valid_config_paths(self, path):
        """Valid YAML config paths are accepted without error."""
        from federated_ids.config import validate_config_path

        result = validate_config_path(path)
        assert result == path

    @pytest.mark.parametrize("path", ["config.json", "config.txt"])
    def test_invalid_config_extension(self, path):
        """Invalid config extensions raise ValueError."""
        from federated_ids.config import validate_config_path

        with pytest.raises(ValueError, match="must have .yaml or .yml"):
            validate_config_path(path)


class TestConfigSafety:
    """Config dict must not be mutated during FL operations."""

    def test_no_config_mutation(self, tmp_path):
        """Config dict is not mutated by save_fl_metrics (no _device key added)."""
        from federated_ids.fl.server import save_fl_metrics

        config = {
            "federation": {"num_clients": 3, "num_rounds": 5, "fraction_fit": 1.0},
            "training": {"learning_rate": 0.001, "local_epochs": 1, "batch_size": 64},
            "model": {"hidden_layers": [32, 16]},
            "seed": 42,
        }

        # Deep copy to compare before/after
        import copy
        config_before = copy.deepcopy(config)

        output_path = str(tmp_path / "metrics" / "fl_metrics.json")
        history = [{"loss": 0.5, "accuracy": 0.9, "f1": 0.85, "precision": 0.88, "recall": 0.82}]

        save_fl_metrics(history, config, output_path, device="cpu")

        # Verify no mutation
        assert "_device" not in config, "Config should not have _device key added"
        assert config == config_before, "Config dict was mutated during save_fl_metrics"
