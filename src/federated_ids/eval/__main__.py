"""Entry point for evaluation: ``python -m federated_ids.eval``.

Orchestrates the full evaluation pipeline: loads the global model
checkpoint, evaluates on the test set, generates confusion matrix,
classification report, convergence plot, and per-client comparison
chart.

Usage::

    python -m federated_ids.eval --config config/default.yaml
    federated-ids-evaluate --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from federated_ids.config import load_config
from federated_ids.data.preprocess import _cache_exists
from federated_ids.device import get_device
from federated_ids.eval.evaluate import evaluate_detailed, evaluate_per_client
from federated_ids.eval.plots import (
    plot_client_comparison,
    plot_confusion_matrix,
    plot_convergence,
    save_classification_report,
)
from federated_ids.model.model import MLP
from federated_ids.seed import set_global_seed

logger = logging.getLogger(__name__)


def main(config_path: str | None = None) -> None:
    """Run the full evaluation pipeline.

    Orchestrates model evaluation, confusion matrix, classification
    report, convergence plot, and per-client comparison in a single
    command.

    Args:
        config_path: Path to YAML config file. If ``None``, parsed from
            CLI ``--config`` argument (default ``config/default.yaml``).
    """
    # --- Argument parsing ---
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Evaluate the global federated model and generate plots."
        )
        parser.add_argument(
            "--config",
            default="config/default.yaml",
            help="Path to YAML configuration file (default: config/default.yaml)",
        )
        args = parser.parse_args()
        config_path = args.config

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
    model_config = config["model"]
    training_config = config["training"]

    processed_dir = data_config.get("processed_dir", "./data/processed")
    output_dir = config.get("output_dir", "./outputs")
    batch_size = training_config["batch_size"]

    # --- Data pipeline auto-run ---
    if not _cache_exists(processed_dir):
        logger.info(
            "Cached data not found in %s. Running data pipeline...",
            processed_dir,
        )
        from federated_ids.data.preprocess import main as run_pipeline

        run_pipeline(config_path)

    # --- Load cached test tensors ---
    logger.info("Loading test tensors from %s", processed_dir)
    X_test = torch.load(
        os.path.join(processed_dir, "X_test.pt"), weights_only=True
    )
    y_test = torch.load(
        os.path.join(processed_dir, "y_test.pt"), weights_only=True
    )

    # --- Load global model checkpoint ---
    checkpoint_path = os.path.join(output_dir, "checkpoints", "global_model.pt")
    logger.info("Loading global model from %s", checkpoint_path)

    input_dim = X_test.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_layers=model_config["hidden_layers"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # --- Create test DataLoader ---
    from torch.utils.data import DataLoader, TensorDataset

    test_dataset = TensorDataset(
        X_test.to(torch.float32),
        y_test.to(torch.int64),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Class-weighted criterion ---
    num_classes = model_config["num_classes"]
    weighted_loss = training_config.get("weighted_loss", False)
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

    # --- Evaluate global model ---
    logger.info("Evaluating global model on test set...")
    metrics, y_true, y_pred = evaluate_detailed(model, test_loader, criterion, device)
    logger.info(
        "Global model -- loss: %.4f, acc: %.3f, F1: %.3f, prec: %.3f, rec: %.3f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    # --- Generate plots and reports ---
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion matrix
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    logger.info("Saved confusion matrix: %s", cm_path)

    # Classification report
    report_path = os.path.join(plots_dir, "classification_report.txt")
    report = save_classification_report(y_true, y_pred, report_path)
    logger.info("Classification Report:\n%s", report)
    logger.info("Saved classification report: %s", report_path)

    # Convergence plot
    metrics_json_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
    convergence_path = os.path.join(plots_dir, "convergence.png")
    if os.path.isfile(metrics_json_path):
        plot_convergence(metrics_json_path, convergence_path)
        logger.info("Saved convergence plot: %s", convergence_path)
    else:
        logger.warning(
            "FL metrics file not found at %s -- skipping convergence plot",
            metrics_json_path,
        )

    # Per-client comparison
    logger.info("Training local-only models for per-client comparison...")
    client_metrics, global_metrics = evaluate_per_client(config, config_path)
    comparison_path = os.path.join(plots_dir, "client_comparison.png")
    plot_client_comparison(client_metrics, global_metrics, comparison_path)
    logger.info("Saved client comparison plot: %s", comparison_path)

    # --- Summary ---
    logger.info("--- Evaluation Complete ---")
    logger.info("Artifacts saved to %s:", plots_dir)
    logger.info("  - confusion_matrix.png")
    logger.info("  - classification_report.txt")
    if os.path.isfile(convergence_path):
        logger.info("  - convergence.png")
    logger.info("  - client_comparison.png")
    logger.info("--- End Evaluation ---")


if __name__ == "__main__":
    main()
