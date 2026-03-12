"""Tests for IID partitioning and DataLoader creation.

Verifies that partition_iid produces the correct number of client partitions
with preserved class ratios, and that create_dataloaders wraps them in
PyTorch DataLoaders with correct batch sizes and dtypes.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from federated_ids.data.partition import create_dataloaders, partition_iid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Create synthetic preprocessed data mimicking preprocess() output.

    Returns X_train, y_train, X_test, y_test with a class ratio of
    approximately 40% benign (0) and 60% DDoS (1), matching CICIDS2017
    binary classification proportions.
    """
    rng = np.random.RandomState(42)
    n_train = 300
    n_test = 75
    n_features = 20

    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)

    # ~40% class 0 (benign), ~60% class 1 (DDoS)
    y_train = np.array([0] * 120 + [1] * 180, dtype=np.int64)
    rng.shuffle(y_train)
    y_test = np.array([0] * 30 + [1] * 45, dtype=np.int64)
    rng.shuffle(y_test)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# partition_iid tests
# ---------------------------------------------------------------------------


class TestPartitionIID:
    """Tests for the partition_iid function."""

    def test_partition_count_3_clients(self, synthetic_data):
        """partition_iid with num_clients=3 produces exactly 3 partitions."""
        X_train, y_train, _, _ = synthetic_data
        partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)
        assert len(partitions) == 3

    def test_partition_count_2_clients(self, synthetic_data):
        """partition_iid with num_clients=2 produces exactly 2 partitions."""
        X_train, y_train, _, _ = synthetic_data
        partitions = partition_iid(X_train, y_train, num_clients=2, seed=42)
        assert len(partitions) == 2

    def test_class_ratio_preserved(self, synthetic_data):
        """Each partition's class ratio is within 5% of the global ratio."""
        X_train, y_train, _, _ = synthetic_data
        num_clients = 3
        partitions = partition_iid(X_train, y_train, num_clients=num_clients, seed=42)

        # Global class 1 ratio
        global_ratio = np.mean(y_train == 1)

        for i, (X_part, y_part) in enumerate(partitions):
            part_ratio = np.mean(y_part == 1)
            diff = abs(part_ratio - global_ratio)
            assert diff <= 0.05, (
                f"Partition {i} class ratio {part_ratio:.3f} deviates "
                f"{diff:.3f} from global {global_ratio:.3f} (max 0.05)"
            )

    def test_no_data_loss(self, synthetic_data):
        """Total samples across all partitions equals total input samples."""
        X_train, y_train, _, _ = synthetic_data
        num_clients = 3
        partitions = partition_iid(X_train, y_train, num_clients=num_clients, seed=42)

        total_samples = sum(len(y_part) for _, y_part in partitions)
        assert total_samples == len(y_train), (
            f"Data loss: {total_samples} partition samples vs {len(y_train)} total"
        )

    def test_partition_reproducibility(self, synthetic_data):
        """With same seed, partition indices are identical on repeated runs."""
        X_train, y_train, _, _ = synthetic_data

        partitions_a = partition_iid(X_train, y_train, num_clients=3, seed=42)
        partitions_b = partition_iid(X_train, y_train, num_clients=3, seed=42)

        for i, ((Xa, ya), (Xb, yb)) in enumerate(zip(partitions_a, partitions_b)):
            np.testing.assert_array_equal(
                ya, yb, err_msg=f"Partition {i} labels differ between runs"
            )
            np.testing.assert_array_equal(
                Xa, Xb, err_msg=f"Partition {i} features differ between runs"
            )


# ---------------------------------------------------------------------------
# create_dataloaders tests
# ---------------------------------------------------------------------------


class TestCreateDataLoaders:
    """Tests for the create_dataloaders function."""

    def test_dataloader_count(self, synthetic_data):
        """create_dataloaders returns correct number of client DataLoaders."""
        X_train, y_train, X_test, y_test = synthetic_data
        partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)

        client_loaders, test_loader = create_dataloaders(
            partitions, X_test, y_test, batch_size=32
        )
        assert len(client_loaders) == 3

    def test_dataloader_batching(self, synthetic_data):
        """DataLoader yields correct batch size (last batch may be smaller)."""
        X_train, y_train, X_test, y_test = synthetic_data
        batch_size = 32
        partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)

        client_loaders, _ = create_dataloaders(
            partitions, X_test, y_test, batch_size=batch_size
        )

        for loader in client_loaders:
            batches = list(loader)
            # All batches except last should be exactly batch_size
            for batch_X, batch_y in batches[:-1]:
                assert batch_X.shape[0] == batch_size
            # Last batch should be <= batch_size
            last_X, last_y = batches[-1]
            assert last_X.shape[0] <= batch_size

    def test_dataloader_dtypes(self, synthetic_data):
        """X is float32, y is int64 in DataLoader batches."""
        X_train, y_train, X_test, y_test = synthetic_data
        partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)

        client_loaders, test_loader = create_dataloaders(
            partitions, X_test, y_test, batch_size=32
        )

        # Check client loaders
        for loader in client_loaders:
            batch_X, batch_y = next(iter(loader))
            assert batch_X.dtype == torch.float32, f"Expected float32, got {batch_X.dtype}"
            assert batch_y.dtype == torch.int64, f"Expected int64, got {batch_y.dtype}"

        # Check test loader
        test_X, test_y = next(iter(test_loader))
        assert test_X.dtype == torch.float32
        assert test_y.dtype == torch.int64

    def test_test_dataloader_separate(self, synthetic_data):
        """Global test DataLoader is separate and not partitioned."""
        X_train, y_train, X_test, y_test = synthetic_data
        partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)

        client_loaders, test_loader = create_dataloaders(
            partitions, X_test, y_test, batch_size=32
        )

        # test_loader should be a single DataLoader, not a list
        assert isinstance(test_loader, DataLoader)

        # Collect all test samples
        test_samples = 0
        for batch_X, batch_y in test_loader:
            test_samples += batch_X.shape[0]

        # Test loader should contain all test data
        assert test_samples == len(y_test), (
            f"Test loader has {test_samples} samples, expected {len(y_test)}"
        )
