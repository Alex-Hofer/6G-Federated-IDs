"""Publication-quality plots for federated IDS evaluation.

Generates confusion matrix, convergence, and per-client comparison
plots using seaborn with the ``whitegrid`` theme and ``muted`` palette.
All plots are saved at 150 DPI with tight bounding boxes and all
figures are closed after saving to prevent state pollution.

Matplotlib uses the ``Agg`` backend (headless rendering) so that plots
can be generated on servers without a display.

Example::

    plot_confusion_matrix(y_true, y_pred, "outputs/plots/confusion_matrix.png")
    plot_convergence("outputs/metrics/fl_metrics.json", "outputs/plots/convergence.png")
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless rendering -- before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

# Global theme applied once at module load
sns.set_theme(style="whitegrid", palette="muted")

_FIGSIZE = (10, 6)
_DPI = 150
_CLASS_NAMES = ["BENIGN", "DDoS"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """Save a confusion matrix heatmap with percentage and count annotations.

    Each cell displays ``"XX.X%\\n(count)"`` where percentages are computed
    relative to the total number of samples (NOT row-normalised) per
    research Pitfall 2.

    Args:
        y_true: Ground-truth labels (1-D array).
        y_pred: Predicted labels (1-D array).
        output_path: File path for the saved PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    cm_pct = cm / total * 100

    # Build annotation array: "XX.X%\n(count)"
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})"

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.heatmap(
        cm_pct,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=_CLASS_NAMES,
        yticklabels=_CLASS_NAMES,
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    metrics_path: str,
    output_path: str,
) -> None:
    """Save a two-subplot convergence plot (loss + F1 over FL rounds).

    Reads ``fl_metrics.json`` produced by ``fl.server.save_fl_metrics``
    and creates side-by-side line plots with markers.

    Args:
        metrics_path: Path to the ``fl_metrics.json`` file.
        output_path: File path for the saved PNG.
    """
    with open(metrics_path) as f:
        data = json.load(f)

    rounds_data = data["rounds"]
    rounds = [r["round"] for r in rounds_data]
    losses = [r["loss"] for r in rounds_data]
    f1s = [r["f1"] for r in rounds_data]

    palette = sns.color_palette("muted", n_colors=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE)

    ax1.plot(rounds, losses, marker="o", color=palette[0], linewidth=2)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Rounds")

    ax2.plot(rounds, f1s, marker="s", color=palette[1], linewidth=2)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score over Rounds")

    fig.suptitle("Federated Learning Convergence", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_client_comparison(
    client_metrics: list[dict],
    global_metrics: dict,
    output_path: str,
) -> None:
    """Save a grouped bar chart comparing per-client and global metrics.

    Plots F1, Precision, and Recall as three groups on the x-axis with
    one bar per client plus one bar for the global federated model.

    Args:
        client_metrics: List of metric dicts, one per client.
        global_metrics: Metric dict for the global model.
        output_path: File path for the saved PNG.
    """
    metric_names = ["f1", "precision", "recall"]
    metric_labels = ["F1", "Precision", "Recall"]
    n_clients = len(client_metrics)
    n_groups = len(metric_names)
    n_bars = n_clients + 1  # clients + global

    palette = sns.color_palette("muted", n_colors=n_bars)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for i in range(n_clients):
        values = [client_metrics[i][m] for m in metric_names]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=f"Client {i}", color=palette[i])

    # Global model bar
    global_values = [global_metrics[m] for m in metric_names]
    offset = (n_clients - n_bars / 2 + 0.5) * bar_width
    ax.bar(
        x + offset,
        global_values,
        bar_width,
        label="Global (FedAvg)",
        color=palette[n_clients],
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Client vs Global Model Performance")
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> str:
    """Save a sklearn classification report as text and return the string.

    Uses ``target_names=["BENIGN", "DDoS"]`` for readable class labels.

    Args:
        y_true: Ground-truth labels (1-D array).
        y_pred: Predicted labels (1-D array).
        output_path: File path for the saved text report.

    Returns:
        The classification report as a formatted string.
    """
    report = classification_report(
        y_true, y_pred, target_names=_CLASS_NAMES, zero_division=0
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report
