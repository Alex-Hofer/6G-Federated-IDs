"""IID stratified partitioning and DataLoader creation for federated learning.

Implements Stage 3 of the three-stage data pipeline (load -> preprocess -> partition).
Splits preprocessed training data across a configurable number of clients using
stratified partitioning to preserve class ratios, then wraps each partition
in a PyTorch DataLoader for training.

Example:
    >>> from federated_ids.data.partition import partition_iid, create_dataloaders
    >>> partitions = partition_iid(X_train, y_train, num_clients=3, seed=42)
    >>> client_loaders, test_loader = create_dataloaders(
    ...     partitions, X_test, y_test, batch_size=64
    ... )
"""

import logging

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition training data into IID client splits with preserved class ratios.

    Uses ``StratifiedKFold`` to create non-overlapping partitions where each
    client receives an approximately equal share of the data with class
    distributions matching the global ratio within 5%.

    With K folds, the K test portions are non-overlapping and collectively
    cover all data, so each fold's test indices become one client's partition.

    Args:
        X: Training features array of shape ``(n_samples, n_features)``.
        y: Training labels array of shape ``(n_samples,)``.
        num_clients: Number of federated clients (partitions) to create.
        seed: Random seed for reproducible partitioning.

    Returns:
        List of ``(X_partition, y_partition)`` tuples, one per client.
        Each partition contains a non-overlapping subset of the training data.

    Raises:
        AssertionError: If any partition's class ratio deviates more than 5%
            from the global ratio.
    """
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)

    global_ratio = np.mean(y == 1)
    logger.info(
        "Partitioning %d samples across %d clients (global class-1 ratio: %.3f)",
        len(y),
        num_clients,
        global_ratio,
    )

    partitions = []
    for fold_idx, (_, test_indices) in enumerate(skf.split(X, y)):
        X_part = X[test_indices]
        y_part = y[test_indices]

        # Log per-partition statistics
        unique, counts = np.unique(y_part, return_counts=True)
        total = len(y_part)
        dist = {int(u): f"{c} ({c / total * 100:.1f}%)" for u, c in zip(unique, counts)}
        logger.info("  Client %d: %d samples, distribution: %s", fold_idx, total, dist)

        # Validation gate: class ratio within 5% of global
        part_ratio = np.mean(y_part == 1)
        deviation = abs(part_ratio - global_ratio)
        assert deviation <= 0.05, (
            f"VALIDATION FAILED: Client {fold_idx} class-1 ratio {part_ratio:.3f} "
            f"deviates {deviation:.3f} from global {global_ratio:.3f} (max 0.05)"
        )

        partitions.append((X_part, y_part))

    logger.info("IID partitioning complete: %d clients", len(partitions))
    return partitions


def create_dataloaders(
    partitions: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[list[DataLoader], DataLoader]:
    """Wrap partitioned data and test data in PyTorch DataLoaders.

    Converts NumPy arrays to typed PyTorch tensors (float32 features, int64
    labels) and creates shuffled DataLoaders for client training and a
    non-shuffled DataLoader for global test evaluation.

    Args:
        partitions: List of ``(X_partition, y_partition)`` tuples from
            ``partition_iid``.
        X_test: Test features array of shape ``(n_test, n_features)``.
        y_test: Test labels array of shape ``(n_test,)``.
        batch_size: Mini-batch size for all DataLoaders.

    Returns:
        A tuple of:
        - List of client DataLoaders (one per partition, shuffled)
        - Global test DataLoader (not shuffled)
    """
    client_loaders = []
    for i, (X_part, y_part) in enumerate(partitions):
        X_tensor = torch.tensor(X_part, dtype=torch.float32)
        y_tensor = torch.tensor(y_part, dtype=torch.int64)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
        logger.info(
            "  Client %d DataLoader: %d samples, %d batches (batch_size=%d)",
            i,
            len(y_part),
            len(loader),
            batch_size,
        )

    # Global test DataLoader (not shuffled for consistent evaluation)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(
        "  Test DataLoader: %d samples, %d batches (batch_size=%d)",
        len(y_test),
        len(test_loader),
        batch_size,
    )

    return client_loaders, test_loader
