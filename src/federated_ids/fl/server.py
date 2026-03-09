"""Server-side FedAvg aggregation, global evaluation, and federated training orchestration.

Implements the Federated Averaging (FedAvg) algorithm for aggregating
client model updates, a server-side evaluation function, and the complete
federated training orchestration loop with metrics logging, checkpointing,
and convergence verification.

These functions are decoupled from any framework (no Flower dependency)
and operate on plain NumPy arrays, making them easy to test and to
integrate with the pure-Python orchestration loop.

Example::

    results = [(client_params_1, n_1), (client_params_2, n_2)]
    global_params = fedavg_aggregate(results)
    metrics = server_evaluate(global_params, model, test_loader, criterion, device)

    # Full orchestration:
    config = load_config("config/default.yaml")
    history = run_federated_training(config)
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from statistics import mean

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


def print_config_banner(config: dict, device: str, num_features: int) -> None:
    """Print an ASCII box with aligned key-value pairs summarising the FL config.

    Args:
        config: Full configuration dictionary.
        device: String representation of the compute device (e.g. ``"cpu"``).
        num_features: Number of input features in the dataset.
    """
    fed = config.get("federation", {})
    trn = config.get("training", {})
    mdl = config.get("model", {})
    seed = config.get("seed", 42)

    lines = [
        ("Clients", str(fed.get("num_clients", "?"))),
        ("Rounds", str(fed.get("num_rounds", "?"))),
        ("Local epochs", str(trn.get("local_epochs", "?"))),
        ("Batch size", str(trn.get("batch_size", "?"))),
        ("Learning rate", str(trn.get("learning_rate", "?"))),
        ("Hidden layers", str(mdl.get("hidden_layers", "?"))),
        ("Dropout", str(mdl.get("dropout", "?"))),
        ("Features", str(num_features)),
        ("Device", str(device)),
        ("Strategy", f"FedAvg (fraction_fit={fed.get('fraction_fit', 1.0)})"),
        ("Seed", str(seed)),
    ]

    max_key = max(len(k) for k, _ in lines)
    width = max_key + 4 + max(len(v) for _, v in lines)
    border = "+" + "-" * (width + 2) + "+"

    logger.info(border)
    logger.info("| %-*s |", width, "Federated Training Configuration")
    logger.info(border)
    for key, val in lines:
        logger.info("| %-*s  %s%s |", max_key, key, val, " " * (width - max_key - 2 - len(val)))
    logger.info(border)


def check_convergence(history: list[dict], n: int = 3) -> bool:
    """Check whether federated training converged by comparing early vs late F1.

    If the mean F1 of the last ``n`` rounds exceeds the mean F1 of the first
    ``n`` rounds, training is considered to have converged.

    Args:
        history: List of per-round metric dicts, each containing an ``"f1"`` key.
        n: Number of rounds to compare from each end. Adjusted downward
            if ``history`` is shorter than ``2 * n``.

    Returns:
        ``True`` if late F1 exceeds early F1 (convergence), ``False`` otherwise.
    """
    if len(history) < 2 * n:
        n = max(1, len(history) // 2)

    early_f1 = mean([h["f1"] for h in history[:n]])
    late_f1 = mean([h["f1"] for h in history[-n:]])

    passed = late_f1 > early_f1
    status = "PASS" if passed else "FAIL"
    logger.info(
        "Convergence check: early F1=%.4f, late F1=%.4f -> %s",
        early_f1,
        late_f1,
        status,
    )
    return passed


def save_fl_metrics(
    history: list[dict],
    config: dict,
    output_path: str,
) -> None:
    """Persist per-round FL metrics and embedded config to a JSON file.

    Args:
        history: List of per-round metric dicts.
        config: Full configuration dictionary (relevant keys are extracted).
        output_path: Path to the output JSON file.
    """
    fed = config.get("federation", {})
    trn = config.get("training", {})
    mdl = config.get("model", {})

    payload = {
        "config": {
            "num_clients": fed.get("num_clients"),
            "num_rounds": fed.get("num_rounds"),
            "local_epochs": trn.get("local_epochs"),
            "batch_size": trn.get("batch_size"),
            "learning_rate": trn.get("learning_rate"),
            "hidden_layers": mdl.get("hidden_layers"),
            "strategy": "FedAvg",
            "fraction_fit": fed.get("fraction_fit"),
            "device": str(config.get("_device", "cpu")),
            "seed": config.get("seed", 42),
        },
        "rounds": [
            {"round": i + 1, **metrics} for i, metrics in enumerate(history)
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved FL metrics to %s", output_path)


def _print_fl_summary_table(history: list[dict]) -> None:
    """Log a thesis-appendix-ready summary table of all FL rounds.

    Args:
        history: List of per-round metric dicts from ``server_evaluate``.
    """
    logger.info("--- FL Training Summary ---")
    header = f"{'Round':>5}  {'Loss':>7}  {'Acc':>5}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}"
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
    logger.info("--- End FL Training Summary ---")


def run_federated_training(
    config: dict,
    config_path: str | None = None,
) -> list[dict]:
    """Run the complete federated training orchestration loop.

    Orchestrates data loading, client creation, FedAvg aggregation,
    server evaluation, checkpointing, and metrics persistence across
    a configurable number of rounds and clients.

    Args:
        config: Full configuration dictionary.
        config_path: Optional path to the YAML config file, used if the
            data pipeline needs to be auto-run.

    Returns:
        List of per-round metric dicts (one dict per round).
    """
    from federated_ids.data.partition import create_dataloaders, partition_iid
    from federated_ids.data.preprocess import _cache_exists
    from federated_ids.device import get_device
    from federated_ids.fl.client import FederatedClient
    from federated_ids.model.model import MLP
    from federated_ids.seed import set_global_seed

    # --- Extract config sections ---
    fed_config = config["federation"]
    trn_config = config["training"]
    mdl_config = config["model"]
    data_config = config.get("data", {})

    num_clients = fed_config["num_clients"]
    num_rounds = fed_config["num_rounds"]
    local_epochs = trn_config["local_epochs"]
    batch_size = trn_config["batch_size"]
    learning_rate = trn_config["learning_rate"]
    hidden_layers = mdl_config["hidden_layers"]
    num_classes = mdl_config.get("num_classes", 2)
    dropout = mdl_config.get("dropout", 0.3)

    seed = config.get("seed", 42)
    output_dir = config.get("output_dir", "./outputs")
    processed_dir = data_config.get("processed_dir", "./data/processed")

    # --- Setup ---
    set_global_seed(seed)
    device = get_device()

    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Suppress client-level training noise
    logging.getLogger("federated_ids.model.train").setLevel(logging.WARNING)

    # --- Data pipeline auto-run ---
    if not _cache_exists(processed_dir):
        logger.info(
            "Cached data not found in %s. Running data pipeline...",
            processed_dir,
        )
        from federated_ids.data.preprocess import main as run_pipeline

        run_pipeline(config_path)

    # --- Load cached tensors ---
    logger.info("Loading cached tensors from %s", processed_dir)
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

    # --- Initial model and parameters ---
    input_dim = X_train.shape[1]
    init_model = MLP(input_dim, hidden_layers, num_classes, dropout).to(device)

    # Print config banner
    print_config_banner(config, device=str(device), num_features=input_dim)

    # Get initial global parameters (standalone helper, same as FederatedClient)
    global_params = [
        val.cpu().detach().numpy().copy()
        for val in init_model.state_dict().values()
    ]

    # Shared evaluation model (re-used each round)
    eval_model = MLP(input_dim, hidden_layers, num_classes, dropout).to(device)

    # --- FL Loop ---
    history: list[dict] = []
    best_f1 = 0.0
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "global_model.pt")

    for round_num in range(1, num_rounds + 1):
        round_results = []

        for client_id in range(num_clients):
            # Fresh model per client per round (Pitfall 6: avoid shared state)
            client_model = MLP(input_dim, hidden_layers, num_classes, dropout).to(device)
            # Fresh optimizer per client per round (Pitfall 2: reset optimizer)
            client_optimizer = torch.optim.Adam(
                client_model.parameters(), lr=learning_rate
            )
            client = FederatedClient(
                client_model, client_loaders[client_id], criterion, client_optimizer, device
            )
            params, n_examples, _ = client.fit(
                global_params, {"local_epochs": local_epochs}
            )
            round_results.append((params, n_examples))

        # Aggregate
        global_params = fedavg_aggregate(round_results)

        # Evaluate
        metrics = server_evaluate(
            global_params, eval_model, test_loader, criterion, device
        )

        # One-line per-round log
        logger.info(
            "Round %2d/%d -- loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
            round_num,
            num_rounds,
            metrics["loss"],
            metrics["accuracy"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

        # Checkpoint on F1 improvement
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            # Load global params into eval model and save state_dict
            keys = list(eval_model.state_dict().keys())
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(keys, global_params)}
            )
            torch.save(state_dict, checkpoint_path)
            logger.info(
                "Saved global model (round %d, F1=%.2f)", round_num, best_f1
            )

        history.append(metrics)

    # --- Post-training ---
    _print_fl_summary_table(history)
    check_convergence(history)

    # Store device in config for metrics persistence
    config["_device"] = str(device)
    metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
    save_fl_metrics(history, config, metrics_path)
    logger.info("FL metrics saved to %s", metrics_path)

    return history
