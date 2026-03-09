"""Server-side FedAvg aggregation and global model evaluation.

Implements the Federated Averaging (FedAvg) algorithm for aggregating
client model updates, and a server-side evaluation function that loads
aggregated parameters into a model and computes metrics on a held-out
test set.

These functions are decoupled from any framework (no Flower dependency)
and operate on plain NumPy arrays, making them easy to test and to
integrate with the pure-Python orchestration loop (Plan 02).

Example::

    results = [(client_params_1, n_1), (client_params_2, n_2)]
    global_params = fedavg_aggregate(results)
    metrics = server_evaluate(global_params, model, test_loader, criterion, device)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch

from federated_ids.model.train import evaluate

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Compute the Federated Averaging weighted aggregation of client parameters.

    For each model layer, computes the weighted average across all clients:
    ``aggregated[i] = sum(params_i * n_i / total_n)`` where ``n_i`` is the
    number of training examples on client ``i``.

    Args:
        results: List of ``(parameters, num_examples)`` tuples, one per
            client.  ``parameters`` is a list of NumPy arrays (one per
            model layer) and ``num_examples`` is the number of training
            samples that client used.

    Returns:
        List of NumPy arrays containing the weighted-average parameters.
    """
    total_examples = sum(n for _, n in results)
    num_layers = len(results[0][0])

    # Initialize accumulator with zeros matching parameter shapes
    aggregated = [np.zeros_like(results[0][0][i]) for i in range(num_layers)]

    for params, num_examples in results:
        weight = num_examples / total_examples
        for i in range(num_layers):
            aggregated[i] += params[i] * weight

    logger.info(
        "FedAvg aggregation: %d clients, %d total examples",
        len(results),
        total_examples,
    )

    return aggregated


def server_evaluate(
    global_params: list[np.ndarray],
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the global model on a test dataset.

    Loads ``global_params`` into the model and delegates to
    ``federated_ids.model.train.evaluate`` for metric computation.

    Args:
        global_params: List of NumPy arrays representing the global
            (aggregated) model weights.
        model: PyTorch model matching the parameter structure.
        test_loader: DataLoader yielding ``(features, labels)`` batches
            for evaluation.
        criterion: Loss function for computing evaluation loss.
        device: Compute device for tensors.

    Returns:
        Dict with keys: ``loss``, ``accuracy``, ``f1``, ``precision``,
        ``recall``.
    """
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {k: torch.tensor(v).to(device) for k, v in zip(keys, global_params)}
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    metrics = evaluate(model, test_loader, criterion, device)

    logger.info(
        "Server evaluation -- loss: %.4f, acc: %.3f, F1: %.3f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["f1"],
    )

    return metrics
