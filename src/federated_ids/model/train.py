"""Local training loop with evaluation, checkpointing, and standalone entry point.

Provides reusable ``train_one_epoch()`` and ``evaluate()`` functions that will
be called by Flower clients during federated learning (Phase 3), plus a
``standalone_train()`` entry point for validating the MLP model on a single
client's data partition before federation.

Standalone training flow:
1. Load config, set seed, detect device
2. Auto-run data pipeline if cached tensors are missing
3. Partition training data, select one client
4. Split client data into train/val (stratified)
5. Train for ``standalone_epochs`` with per-epoch validation metrics
6. Save best checkpoint on F1 improvement
7. Print summary table and evaluate on global test set

Example:
    >>> from federated_ids.model.train import train_one_epoch, evaluate
    >>> loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    >>> metrics = evaluate(model, val_loader, criterion, device)
    >>> print(f"F1: {metrics['f1']:.2f}")
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from federated_ids.config import load_config
from federated_ids.data.partition import partition_iid
from federated_ids.data.preprocess import _cache_exists
from federated_ids.device import get_device
from federated_ids.model.model import MLP
from federated_ids.seed import set_global_seed

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch on the provided data.

    Args:
        model: Neural network model to train.
        train_loader: DataLoader yielding ``(features, labels)`` batches.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimizer instance (e.g. Adam).
        device: Compute device for tensors.

    Returns:
        Average training loss over the epoch (total loss / num samples).
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(train_loader.dataset)


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model on a dataset and compute classification metrics.

    Args:
        model: Trained neural network model.
        data_loader: DataLoader yielding ``(features, labels)`` batches.
        criterion: Loss function for computing evaluation loss.
        device: Compute device for tensors.

    Returns:
        Dict with keys: ``loss``, ``accuracy``, ``f1``, ``precision``,
        ``recall``.  All classification metrics use binary averaging.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / len(data_loader.dataset)

    return {
        "loss": avg_loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
    }


def _print_summary_table(history: list[dict]) -> None:
    """Log a thesis-appendix-ready summary table of all training epochs.

    Produces a fixed-width table with columns: Epoch, Loss, Acc, F1, Prec, Rec.

    Args:
        history: List of metric dicts from ``evaluate()``, one per epoch.
    """
    logger.info("--- Training Summary ---")
    header = f"{'Epoch':>5}  {'Loss':>7}  {'Acc':>5}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}"
    logger.info(header)
    logger.info("-" * len(header))

    for i, metrics in enumerate(history, start=1):
        logger.info(
            "%5d  %7.4f  %5.3f  %5.3f  %5.3f  %5.3f",
            i,
            metrics["loss"],
            metrics["accuracy"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

    logger.info("-" * len(header))
    logger.info("--- End Training Summary ---")


def standalone_train(
    config_path: str | None = None,
    client_id: int | None = None,
) -> None:
    """Run standalone local training on a single client partition.

    Validates the MLP model on one client's data before federation.
    Supports both CLI invocation (``federated-ids-train``) and
    programmatic use from other modules.

    Args:
        config_path: Path to YAML config file. If ``None``, parsed from
            CLI ``--config`` argument (default ``config/default.yaml``).
        client_id: Client partition index. If ``None``, parsed from
            CLI ``--client-id`` argument (default ``0``).
    """
    # --- Argument parsing ---
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Run standalone local training on a single client partition."
        )
        parser.add_argument(
            "--config",
            default="config/default.yaml",
            help="Path to YAML configuration file (default: config/default.yaml)",
        )
        parser.add_argument(
            "--client-id",
            type=int,
            default=0,
            help="Client partition index to train on (default: 0)",
        )
        args = parser.parse_args()
        config_path = args.config
        client_id = args.client_id

    if client_id is None:
        client_id = 0

    # --- Setup ---
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_global_seed(seed)
    device = get_device()

    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]

    processed_dir = data_config.get("processed_dir", "./data/processed")
    num_clients = config["federation"]["num_clients"]
    batch_size = training_config["batch_size"]
    standalone_epochs = training_config.get("standalone_epochs", 5)
    val_split = training_config.get("val_split", 0.2)
    output_dir = config.get("output_dir", "./outputs")

    # --- Data pipeline (auto-run if cached data missing) ---
    if not _cache_exists(processed_dir):
        logger.info(
            "Cached data not found in %s. Running data pipeline...",
            processed_dir,
        )
        from federated_ids.data.preprocess import main as run_pipeline

        run_pipeline(config_path)

    # Load cached tensors
    logger.info("Loading cached tensors from %s", processed_dir)
    X_train = (
        torch.load(os.path.join(processed_dir, "X_train.pt"), weights_only=True)
        .numpy()
    )
    y_train = (
        torch.load(os.path.join(processed_dir, "y_train.pt"), weights_only=True)
        .numpy()
    )
    X_test = (
        torch.load(os.path.join(processed_dir, "X_test.pt"), weights_only=True)
        .numpy()
    )
    y_test = (
        torch.load(os.path.join(processed_dir, "y_test.pt"), weights_only=True)
        .numpy()
    )

    # --- Partition and select client ---
    partitions = partition_iid(X_train, y_train, num_clients=num_clients, seed=seed)
    X_client, y_client = partitions[client_id]
    logger.info(
        "Selected client %d: %d samples", client_id, len(y_client)
    )

    # --- Stratified train/val split ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_client,
        y_client,
        test_size=val_split,
        stratify=y_client,
        random_state=seed,
    )
    logger.info(
        "Client %d split: train=%d, val=%d (val_split=%.2f)",
        client_id,
        len(y_tr),
        len(y_val),
        val_split,
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.int64),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # --- Model ---
    input_dim = X_train.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_layers=model_config["hidden_layers"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config["learning_rate"]
    )

    # --- Class-weighted loss ---
    weights_path = os.path.join(processed_dir, "class_weights.json")
    with open(weights_path) as f:
        raw_weights = json.load(f)

    num_classes = model_config["num_classes"]
    weight_tensor = torch.tensor(
        [raw_weights[str(i)] for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    logger.info("Using class-weighted loss: %s (device=%s)", raw_weights, device)

    # --- Training loop ---
    best_f1 = 0.0
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    history: list[dict] = []

    for epoch in range(1, standalone_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d -- loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
            epoch,
            standalone_epochs,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        # Checkpoint on F1 improvement
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "Saved best model (epoch %d, F1=%.2f)", epoch, best_f1
            )

        history.append(val_metrics)

    # --- Summary table ---
    _print_summary_table(history)

    # --- Final global test set evaluation ---
    logger.info("--- Global Test Set Evaluation ---")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Test -- loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
    )
    logger.info("--- End Global Test Set Evaluation ---")
