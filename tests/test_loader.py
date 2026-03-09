"""Tests for CICIDS2017 data loading and cleaning.

Verifies that the loader correctly handles CICIDS2017 CSV quirks:
column whitespace, Inf/NaN values, and label mapping to binary
{0, 1} classification.

All tests use synthetic data fixtures -- no real CICIDS2017 CSV needed.
"""

import numpy as np
import pandas as pd
import pytest

from federated_ids.data.loader import load_cicids2017


class TestColumnCleaning:
    """Column name whitespace stripping."""

    def test_column_names_are_stripped(self, sample_csv_file):
        """Loading a CSV with leading whitespace column names produces stripped names."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        # No column should have leading or trailing whitespace
        for col in df.columns:
            assert col == col.strip(), f"Column '{col}' has whitespace"

    def test_label_column_exists(self, sample_csv_file):
        """After loading, a 'Label' column (stripped) exists."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        assert "Label" in df.columns


class TestInfNaNRemoval:
    """Inf and NaN value cleaning."""

    def test_inf_values_removed(self, sample_csv_file):
        """Loading a CSV with np.inf values in Flow Bytes/s drops those rows."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        assert not df.isin([np.inf, -np.inf]).any().any(), "Inf values remain"

    def test_nan_values_removed(self, sample_csv_file):
        """Loading a CSV with NaN values drops those rows."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        assert not df.isna().any().any(), "NaN values remain"

    def test_no_inf_or_nan_in_any_column(self, sample_csv_file):
        """After loading, no Inf or NaN values remain in any column."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        # Check numeric columns for Inf
        numeric_df = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric_df.values).any(), "Inf in numeric columns"
        assert not np.isnan(numeric_df.values).any(), "NaN in numeric columns"


class TestLabelMapping:
    """Label mapping to binary {0, 1}."""

    def test_benign_maps_to_zero(self, sample_csv_file):
        """Label mapping converts 'BENIGN' (including with whitespace) to 0."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        # Original data had BENIGN labels -- they should now be 0
        assert 0 in df["Label"].values

    def test_ddos_variants_map_to_one(self, sample_csv_file):
        """DDoS variants ('DDoS', 'DoS Hulk', 'DoS GoldenEye', etc.) map to 1."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        assert 1 in df["Label"].values

    def test_only_binary_labels(self, sample_csv_file):
        """Only binary labels {0, 1} exist after mapping."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        unique_labels = set(df["Label"].unique())
        assert unique_labels == {0, 1}, f"Expected {{0, 1}}, got {unique_labels}"


class TestReturnType:
    """Return type validation."""

    def test_returns_dataframe(self, sample_csv_file):
        """Function returns a pandas DataFrame."""
        result = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        assert isinstance(result, pd.DataFrame)


class TestErrorHandling:
    """Error handling for invalid inputs."""

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Loading with a nonexistent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_cicids2017(str(tmp_path), ["nonexistent.csv"])

    def test_rows_are_dropped(self, sample_csv_file, sample_cicids_df):
        """Some rows are dropped due to Inf/NaN values."""
        df = load_cicids2017(str(sample_csv_file), ["test_data.csv"])
        # Original had 100 rows; some had Inf/NaN so result should be smaller
        assert len(df) < len(sample_cicids_df)
        assert len(df) > 0, "All rows were dropped"
