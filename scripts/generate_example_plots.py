#!/usr/bin/env python3
"""Generate example plots for README embedding using synthetic data.

Creates realistic-looking confusion matrix and convergence plots by
calling the project's own plotting functions with synthetic data that
mimics a decent federated learning model.

Usage::

    python scripts/generate_example_plots.py

Outputs:
    docs/confusion_matrix.png
    docs/convergence.png
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from federated_ids.eval.plots import plot_confusion_matrix, plot_convergence


def main() -> None:
    """Generate example plots with synthetic data."""
    rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # 1. Confusion matrix with realistic predictions
    #    ~600 BENIGN (label 0), ~400 DDoS (label 1)
    #    ~95% accuracy on BENIGN, ~90% accuracy on DDoS
    # ------------------------------------------------------------------
    n_benign = 600
    n_ddos = 400

    y_true_benign = np.zeros(n_benign, dtype=int)
    y_pred_benign = np.where(
        rng.rand(n_benign) < 0.95, 0, 1
    )  # 95% correct

    y_true_ddos = np.ones(n_ddos, dtype=int)
    y_pred_ddos = np.where(
        rng.rand(n_ddos) < 0.90, 1, 0
    )  # 90% correct

    y_true = np.concatenate([y_true_benign, y_true_ddos])
    y_pred = np.concatenate([y_pred_benign, y_pred_ddos])

    cm_path = os.path.join("docs", "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Generated: {cm_path}")

    # ------------------------------------------------------------------
    # 2. Convergence plot with 10 rounds showing clear convergence
    #    Loss:  0.60 -> 0.15 (decreasing)
    #    F1:    0.70 -> 0.93 (increasing)
    # ------------------------------------------------------------------
    n_rounds = 10
    loss_start, loss_end = 0.60, 0.15
    f1_start, f1_end = 0.70, 0.93

    rounds_data = []
    for r in range(1, n_rounds + 1):
        progress = (r - 1) / (n_rounds - 1)
        # Exponential-ish decay for loss, logarithmic-ish growth for F1
        loss = loss_start + (loss_end - loss_start) * (1 - np.exp(-3 * progress)) / (1 - np.exp(-3))
        f1 = f1_start + (f1_end - f1_start) * (1 - np.exp(-3 * progress)) / (1 - np.exp(-3))
        # Add small noise for realism
        loss += rng.normal(0, 0.01)
        f1 += rng.normal(0, 0.005)
        rounds_data.append({
            "round": r,
            "loss": round(float(loss), 4),
            "f1": round(float(f1), 4),
        })

    metrics = {"rounds": rounds_data}

    # Write temporary metrics JSON, generate plot, clean up
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="fl_metrics_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(metrics, f, indent=2)

        conv_path = os.path.join("docs", "convergence.png")
        plot_convergence(tmp_path, conv_path)
        print(f"Generated: {conv_path}")
    finally:
        os.unlink(tmp_path)

    print("Done. Both example plots are ready for README embedding.")


if __name__ == "__main__":
    main()
