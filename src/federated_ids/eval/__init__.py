"""Evaluation and visualization utilities.

This subpackage provides model evaluation, metrics computation,
and publication-quality plot generation.
"""

from federated_ids.eval.evaluate import evaluate_detailed, evaluate_per_client
from federated_ids.eval.plots import (
    plot_client_comparison,
    plot_confusion_matrix,
    plot_convergence,
    save_classification_report,
)

__all__ = [
    "evaluate_detailed",
    "evaluate_per_client",
    "plot_client_comparison",
    "plot_confusion_matrix",
    "plot_convergence",
    "save_classification_report",
]
