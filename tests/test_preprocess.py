"""Tests for feature selection, normalization, and preprocessing.

Verifies that the preprocessing pipeline correctly:
- Removes identifier columns
- Drops zero-variance and highly correlated features
- Generates a feature drop report
- Splits train/test with stratification
- Fits StandardScaler on training data only (no leakage)
- Outputs float32 arrays
- Computes class weights
- Saves artifacts to processed directory
- Produces reproducible results with the same seed

All tests use synthetic data fixtures -- no real CICIDS2017 CSV needed.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from federated_ids.data.loader import IDENTIFIER_COLS, load_cicids2017
from federated_ids.data.preprocess import preprocess, select_features


def _make_clean_df(sample_cicids_df):
    """Create a cleaned DataFrame suitable for preprocessing tests.

    Simulates what load_cicids2017 would produce: stripped column names,
    no Inf/NaN, binary labels.
    """
    df = sample_cicids_df.copy()
    df.columns = df.columns.str.strip()
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Map labels to binary
    ddos_labels = {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
    df["Label"] = df["Label"].apply(lambda x: 1 if x in ddos_labels else 0)

    return df


@pytest.fixture
def clean_df(sample_cicids_df):
    """Cleaned DataFrame ready for preprocessing."""
    return _make_clean_df(sample_cicids_df)


@pytest.fixture
def config_dict(sample_config_dict):
    """Config dict with all required keys for preprocessing."""
    return sample_config_dict


class TestIdentifierColumnRemoval:
    """Identifier columns are removed from feature set."""

    def test_identifier_columns_removed(self, clean_df, config_dict):
        """Identifier columns (Flow ID, Source IP, etc.) are not in selected features."""
        filtered_df, report = select_features(clean_df, config_dict)
        remaining_cols = set(filtered_df.columns) - {"Label"}
        for id_col in IDENTIFIER_COLS:
            assert id_col not in remaining_cols, (
                f"Identifier column '{id_col}' still present"
            )


class TestZeroVariance:
    """Zero-variance features are dropped."""

    def test_zero_variance_dropped(self, clean_df, config_dict):
        """A constant column (all same value) is dropped during feature selection."""
        # Add a constant column
        clean_df["ConstantFeature"] = 42.0
        filtered_df, report = select_features(clean_df, config_dict)
        assert "ConstantFeature" not in filtered_df.columns, (
            "Constant feature should be dropped"
        )


class TestHighCorrelation:
    """Highly correlated feature pairs are reduced."""

    def test_high_correlation_dropped(self, clean_df, config_dict):
        """Two perfectly correlated columns have one member dropped."""
        # Add two perfectly correlated columns
        clean_df["CorrFeatureA"] = np.arange(len(clean_df), dtype=np.float64)
        clean_df["CorrFeatureB"] = clean_df["CorrFeatureA"] * 2.0 + 1.0
        filtered_df, report = select_features(clean_df, config_dict)
        # At most one of the pair should remain
        remaining = set(filtered_df.columns)
        both_present = "CorrFeatureA" in remaining and "CorrFeatureB" in remaining
        assert not both_present, "Both correlated features should not remain"


class TestFeatureDropReport:
    """Feature drop report is generated."""

    def test_feature_drop_report_structure(self, clean_df, config_dict):
        """Report dict has entries with 'reason' keys."""
        clean_df["ConstantFeature"] = 42.0
        _, report = select_features(clean_df, config_dict)
        assert isinstance(report, dict), "Report should be a dict"
        # At least the constant feature should appear in the report
        assert len(report) > 0, "Report should have at least one entry"
        # Each entry should have a reason
        for feature_name, info in report.items():
            assert "reason" in info, (
                f"Report entry for '{feature_name}' missing 'reason' key"
            )

    def test_feature_count_reasonable(self, clean_df, config_dict):
        """Feature count after selection is between 15 and 50."""
        filtered_df, _ = select_features(clean_df, config_dict)
        n_features = len([c for c in filtered_df.columns if c != "Label"])
        assert 15 <= n_features <= 50, (
            f"Expected 15-50 features, got {n_features}"
        )


class TestTrainTestSplit:
    """Train/test split properties."""

    def test_train_test_split_ratio(self, clean_df, config_dict):
        """Train/test split is approximately 80/20."""
        result = preprocess(clean_df, config_dict)
        total = len(result["y_train"]) + len(result["y_test"])
        test_ratio = len(result["y_test"]) / total
        assert 0.15 <= test_ratio <= 0.25, (
            f"Test ratio {test_ratio:.2f} not near 0.20"
        )

    def test_stratified_split(self, clean_df, config_dict):
        """Train and test class ratios match global ratio within 5% tolerance."""
        result = preprocess(clean_df, config_dict)
        train_ratio = np.mean(result["y_train"])
        test_ratio = np.mean(result["y_test"])
        assert abs(train_ratio - test_ratio) < 0.05, (
            f"Class ratio mismatch: train={train_ratio:.3f}, test={test_ratio:.3f}"
        )


class TestScalerNoLeakage:
    """StandardScaler is fitted on training data only."""

    def test_scaler_no_leakage(self, clean_df, config_dict):
        """StandardScaler fitted on train only -- test data mean is NOT zero.

        If the scaler were fitted on the full dataset (train + test), the
        test set feature means would be close to zero. By fitting on train
        only, the test set means should deviate from zero, proving no leakage.
        """
        result = preprocess(clean_df, config_dict)
        test_means = np.mean(result["X_test"], axis=0)
        # Not ALL test feature means should be zero (some may be close by chance)
        assert not np.allclose(
            test_means, 0.0, atol=0.05
        ), "Test set means are all ~0, suggesting scaler was fit on full data"


class TestOutputDtype:
    """Output arrays are float32."""

    def test_float32_output(self, clean_df, config_dict):
        """All output feature arrays are float32 dtype."""
        result = preprocess(clean_df, config_dict)
        assert result["X_train"].dtype == np.float32, (
            f"X_train dtype is {result['X_train'].dtype}, expected float32"
        )
        assert result["X_test"].dtype == np.float32, (
            f"X_test dtype is {result['X_test'].dtype}, expected float32"
        )


class TestClassWeights:
    """Class weights computation."""

    def test_class_weights_computed(self, clean_df, config_dict):
        """Class weights dict has keys 0 and 1 with positive values."""
        result = preprocess(clean_df, config_dict)
        weights = result["class_weights"]
        assert isinstance(weights, dict), "Class weights should be a dict"
        assert 0 in weights and 1 in weights, (
            f"Class weights keys: {list(weights.keys())}, expected [0, 1]"
        )
        assert weights[0] > 0, "Weight for class 0 should be positive"
        assert weights[1] > 0, "Weight for class 1 should be positive"


class TestArtifactsSaved:
    """Artifacts are saved to processed directory."""

    def test_artifacts_saved(self, clean_df, config_dict, tmp_path):
        """Scaler, features, and class weights are saved to processed dir."""
        config_dict["data"]["processed_dir"] = str(tmp_path)
        result = preprocess(clean_df, config_dict)

        assert os.path.isfile(
            os.path.join(str(tmp_path), "scaler.json")
        ), "scaler.json not found"
        assert os.path.isfile(
            os.path.join(str(tmp_path), "features.json")
        ), "features.json not found"
        assert os.path.isfile(
            os.path.join(str(tmp_path), "class_weights.json")
        ), "class_weights.json not found"

    def test_features_json_content(self, clean_df, config_dict, tmp_path):
        """features.json contains a list of feature name strings."""
        config_dict["data"]["processed_dir"] = str(tmp_path)
        result = preprocess(clean_df, config_dict)

        with open(os.path.join(str(tmp_path), "features.json")) as f:
            features = json.load(f)
        assert isinstance(features, list), "features.json should contain a list"
        assert len(features) > 0, "features.json should not be empty"
        assert all(isinstance(f, str) for f in features), (
            "All feature names should be strings"
        )

    def test_class_weights_json_content(self, clean_df, config_dict, tmp_path):
        """class_weights.json contains weights for classes 0 and 1."""
        config_dict["data"]["processed_dir"] = str(tmp_path)
        result = preprocess(clean_df, config_dict)

        with open(os.path.join(str(tmp_path), "class_weights.json")) as f:
            weights = json.load(f)
        assert "0" in weights or 0 in weights, "Missing weight for class 0"
        assert "1" in weights or 1 in weights, "Missing weight for class 1"


class TestReproducibility:
    """Reproducibility with same seed."""

    def test_reproducibility(self, clean_df, config_dict, tmp_path):
        """Two runs with seed=42 produce identical train/test splits and scaled values."""
        config_dict["seed"] = 42

        # Run 1
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config1 = config_dict.copy()
        config1["data"] = config_dict["data"].copy()
        config1["data"]["processed_dir"] = str(dir1)
        result1 = preprocess(clean_df.copy(), config1)

        # Run 2
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config2 = config_dict.copy()
        config2["data"] = config_dict["data"].copy()
        config2["data"]["processed_dir"] = str(dir2)
        result2 = preprocess(clean_df.copy(), config2)

        np.testing.assert_array_equal(
            result1["X_train"], result2["X_train"],
            err_msg="X_train differs between identical runs"
        )
        np.testing.assert_array_equal(
            result1["X_test"], result2["X_test"],
            err_msg="X_test differs between identical runs"
        )
        np.testing.assert_array_equal(
            result1["y_train"], result2["y_train"],
            err_msg="y_train differs between identical runs"
        )
        np.testing.assert_array_equal(
            result1["y_test"], result2["y_test"],
            err_msg="y_test differs between identical runs"
        )


class TestSelectFeaturesReExport:
    """Regression test: select_features must be importable from federated_ids.data."""

    def test_import_from_package(self):
        """select_features is importable from federated_ids.data and is callable."""
        from federated_ids.data import select_features

        assert callable(select_features), (
            "select_features should be callable"
        )
