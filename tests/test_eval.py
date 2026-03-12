"""Unit tests for the evaluation and visualization module.

Tests cover:
- evaluate_detailed: returns metrics dict + raw y_true/y_pred arrays
- plot_confusion_matrix: saves annotated confusion matrix PNG
- plot_convergence: saves two-subplot convergence PNG from fl_metrics.json
- plot_client_comparison: saves grouped bar chart PNG
- save_classification_report: saves text file and returns report string
"""

import json
import os

import numpy as np
import pytest
import torch

from federated_ids.eval.evaluate import evaluate_detailed, evaluate_per_client
from federated_ids.eval.plots import (
    plot_client_comparison,
    plot_confusion_matrix,
    plot_convergence,
    save_classification_report,
)


class TestEvaluateDetailed:
    """Test evaluate_detailed returns (metrics_dict, y_true, y_pred)."""

    def test_returns_tuple_of_three(self, sample_model, fl_test_loader, fl_criterion):
        device = torch.device("cpu")
        result = evaluate_detailed(sample_model, fl_test_loader, fl_criterion, device)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_metrics_dict_has_required_keys(self, sample_model, fl_test_loader, fl_criterion):
        device = torch.device("cpu")
        metrics, _, _ = evaluate_detailed(sample_model, fl_test_loader, fl_criterion, device)
        assert isinstance(metrics, dict)
        for key in ("loss", "accuracy", "f1", "precision", "recall"):
            assert key in metrics, f"Missing key: {key}"

    def test_y_true_y_pred_are_numpy_arrays(self, sample_model, fl_test_loader, fl_criterion):
        device = torch.device("cpu")
        _, y_true, y_pred = evaluate_detailed(sample_model, fl_test_loader, fl_criterion, device)
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)

    def test_y_true_y_pred_correct_length(self, sample_model, fl_test_loader, fl_criterion):
        device = torch.device("cpu")
        _, y_true, y_pred = evaluate_detailed(sample_model, fl_test_loader, fl_criterion, device)
        # fl_test_loader has 50 samples
        assert len(y_true) == 50
        assert len(y_pred) == 50


class TestConfusionMatrixPlot:
    """Test plot_confusion_matrix saves a valid PNG."""

    def test_saves_png_file(self, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        output_path = str(tmp_path / "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, output_path)
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0


class TestConvergencePlot:
    """Test plot_convergence reads fl_metrics.json and saves PNG."""

    def test_saves_png_from_metrics_json(self, tmp_path):
        # Write synthetic fl_metrics.json
        metrics_data = {
            "config": {"num_clients": 3, "num_rounds": 5},
            "rounds": [
                {"round": i + 1, "loss": 1.0 - i * 0.1, "f1": 0.5 + i * 0.08,
                 "accuracy": 0.6 + i * 0.05, "precision": 0.55 + i * 0.07,
                 "recall": 0.5 + i * 0.09}
                for i in range(5)
            ],
        }
        metrics_path = str(tmp_path / "fl_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)

        output_path = str(tmp_path / "convergence.png")
        plot_convergence(metrics_path, output_path)
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0


class TestClientComparison:
    """Test plot_client_comparison saves a grouped bar chart PNG."""

    def test_saves_png_file(self, tmp_path):
        client_metrics = [
            {"f1": 0.85, "precision": 0.80, "recall": 0.90},
            {"f1": 0.82, "precision": 0.78, "recall": 0.87},
            {"f1": 0.88, "precision": 0.84, "recall": 0.92},
        ]
        global_metrics = {"f1": 0.90, "precision": 0.86, "recall": 0.94}
        output_path = str(tmp_path / "client_comparison.png")
        plot_client_comparison(client_metrics, global_metrics, output_path)
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0


class TestClassificationReport:
    """Test save_classification_report saves text and returns string."""

    def test_saves_text_file(self, tmp_path):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        output_path = str(tmp_path / "classification_report.txt")
        report = save_classification_report(y_true, y_pred, output_path)
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0

    def test_returns_string_with_class_names(self, tmp_path):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        output_path = str(tmp_path / "classification_report.txt")
        report = save_classification_report(y_true, y_pred, output_path)
        assert isinstance(report, str)
        assert "BENIGN" in report
        assert "DDoS" in report


class TestEvalCLI:
    """Test that the CLI entry point is importable and callable."""

    def test_main_is_importable(self):
        from federated_ids.eval.__main__ import main

        assert callable(main)
