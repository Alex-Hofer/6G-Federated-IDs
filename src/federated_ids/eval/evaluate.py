"""Detailed model evaluation and per-client comparison for federated IDS.

Provides ``evaluate_detailed()`` which mirrors the inference logic of
``model.train.evaluate()`` but additionally returns the raw ``y_true``
and ``y_pred`` arrays needed for confusion matrices and classification
reports.  Also provides ``evaluate_per_client()`` which trains each
client locally for a fair total-epoch comparison against the global
federated model.

The existing ``evaluate()`` in ``model/train.py`` is deliberately
NOT modified -- this keeps Phase 3 code stable.

Example::

    metrics, y_true, y_pred = evaluate_detailed(model, test_loader, criterion, device)
    print(f"F1: {metrics['f1']:.3f}, samples: {len(y_true)}")
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


def evaluate_detailed(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a model and return metrics together with raw predictions.

    Mirrors the inference logic of ``federated_ids.model.train.evaluate()``
    but returns ``(metrics_dict, y_true, y_pred)`` instead of just the
    metrics dict.  This enables downstream analysis such as confusion
    matrices and classification reports without a second forward pass.

    Args:
        model: Trained neural network model.
        data_loader: DataLoader yielding ``(features, labels)`` batches.
        criterion: Loss function for computing evaluation loss.
        device: Compute device for tensors.

    Returns:
        Tuple of ``(metrics_dict, y_true, y_pred)`` where:
        - ``metrics_dict`` has keys: ``loss``, ``accuracy``, ``f1``,
          ``precision``, ``recall`` (binary averaging).
        - ``y_true`` is a 1-D NumPy array of ground-truth labels.
        - ``y_pred`` is a 1-D NumPy array of predicted labels.
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

    metrics = {
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

    return metrics, y_true, y_pred


def evaluate_per_client(
    config: dict,
    config_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Train each client locally and compare against the global model.

    For a fair comparison (per research Pitfall 3), each local-only model
    is trained for ``num_rounds * local_epochs`` total epochs -- the same
    total compute budget as a federated client receives across all rounds.

    Args:
        config: Full configuration dictionary.
        config_path: Optional path to the YAML config file, used if the
            data pipeline needs to be auto-run.

    Returns:
        Tuple of ``(client_metrics_list, global_metrics)`` where:
        - ``client_metrics_list`` is a list of metric dicts (one per client).
        - ``global_metrics`` is the metric dict for the global checkpoint.
    """
    from federated_ids.data.partition import create_dataloaders, partition_iid
    from federated_ids.data.preprocess import _cache_exists
    from federated_ids.device import get_device
    from federated_ids.model.model import MLP
    from federated_ids.model.train import train_one_epoch
    from federated_ids.seed import set_global_seed

    # --- Extract config ---
    fed_config = config["federation"]
    trn_config = config["training"]
    mdl_config = config["model"]
    data_config = config.get("data", {})

    num_clients = fed_config["num_clients"]
    num_rounds = fed_config["num_rounds"]
    local_epochs = trn_config["local_epochs"]
    total_epochs = num_rounds * local_epochs
    batch_size = trn_config["batch_size"]
    learning_rate = trn_config["learning_rate"]
    hidden_layers = mdl_config["hidden_layers"]
    num_classes = mdl_config.get("num_classes", 2)
    dropout = mdl_config.get("dropout", 0.3)

    seed = config.get("seed", 42)
    output_dir = config.get("output_dir", "./outputs")
    processed_dir = data_config.get("processed_dir", "./data/processed")

    set_global_seed(seed)
    device = get_device()

    # --- Data pipeline auto-run ---
    if not _cache_exists(processed_dir):
        logger.info(
            "Cached data not found in %s. Running data pipeline...",
            processed_dir,
        )
        from federated_ids.data.preprocess import main as run_pipeline

        run_pipeline(config_path)

    # --- Load cached tensors ---
    X_train = torch.load(
        os.path.join(processed_dir, "X_train.pt"), weights_only=True
    ).numpy()
    y_train = torch.load(
        os.path.join(processed_dir, "y_train.pt"), weights_only=True
    ).numpy()
    X_test = torch.load(
        os.path.join(processed_dir, "X_test.pt"), weights_only=True
    ).numpy()
    y_test = torch.load(
        os.path.join(processed_dir, "y_test.pt"), weights_only=True
    ).numpy()

    # --- Partition data ---
    partitions = partition_iid(X_train, y_train, num_clients=num_clients, seed=seed)
    client_loaders, test_loader = create_dataloaders(
        partitions, X_test, y_test, batch_size=batch_size
    )

    # --- Class-weighted loss ---
    weighted_loss = trn_config.get("weighted_loss", False)
    if weighted_loss:
        weights_path = os.path.join(processed_dir, "class_weights.json")
        with open(weights_path) as f:
            raw_weights = json.load(f)
        weight_tensor = torch.tensor(
            [raw_weights[str(i)] for i in range(num_classes)],
            dtype=torch.float32,
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    input_dim = X_train.shape[1]

    # --- Train each client locally ---
    client_metrics_list: list[dict] = []

    for client_id in range(num_clients):
        logger.info(
            "Training local-only model for client %d (%d epochs)...",
            client_id,
            total_epochs,
        )

        model = MLP(input_dim, hidden_layers, num_classes, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for _epoch in range(total_epochs):
            train_one_epoch(model, client_loaders[client_id], criterion, optimizer, device)

        # Evaluate on test set
        metrics, _, _ = evaluate_detailed(model, test_loader, criterion, device)
        logger.info(
            "Client %d local-only -- F1: %.3f, Prec: %.3f, Rec: %.3f",
            client_id,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )
        client_metrics_list.append(metrics)

    # --- Evaluate global model ---
    checkpoint_path = os.path.join(output_dir, "checkpoints", "global_model.pt")
    state_dict = torch.load(checkpoint_path, weights_only=True)

    global_model = MLP(input_dim, hidden_layers, num_classes, dropout).to(device)
    global_model.load_state_dict(state_dict, strict=True)

    global_metrics, _, _ = evaluate_detailed(global_model, test_loader, criterion, device)
    logger.info(
        "Global model -- F1: %.3f, Prec: %.3f, Rec: %.3f",
        global_metrics["f1"],
        global_metrics["precision"],
        global_metrics["recall"],
    )

    return client_metrics_list, global_metrics
