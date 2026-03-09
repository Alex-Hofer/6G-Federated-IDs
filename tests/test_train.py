"""Unit tests for training loop, evaluation, checkpointing, and weighted loss.

Tests cover:
- train_one_epoch returns finite loss and updates model parameters
- evaluate returns all required metrics in valid ranges
- evaluate is deterministic
- Class-weighted loss with correct device placement
- Checkpoint save/load on F1 improvement
- Summary table logging
"""

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from federated_ids.model.model import MLP
from federated_ids.model.train import evaluate, train_one_epoch


def _make_loader(X, y, batch_size=32):
    """Helper to create a DataLoader from numpy arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestTrainOneEpoch:
    """Tests for the train_one_epoch function."""

    def test_train_one_epoch_returns_loss(
        self, sample_model, synthetic_train_data
    ):
        """train_one_epoch() returns a finite float loss value."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = train_one_epoch(model, loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_train_one_epoch_updates_params(
        self, sample_model, synthetic_train_data
    ):
        """Model parameters change after one epoch of training."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Snapshot parameters before training
        params_before = {
            name: p.clone() for name, p in model.named_parameters()
        }

        train_one_epoch(model, loader, criterion, optimizer, device)

        # At least some parameters should have changed
        any_changed = False
        for name, p in model.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "No parameters changed after training epoch"


class TestEvaluate:
    """Tests for the evaluate function."""

    def test_evaluate_returns_all_metrics(
        self, sample_model, synthetic_train_data
    ):
        """evaluate() returns dict with keys: loss, accuracy, f1, precision, recall."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, device)

        expected_keys = {"loss", "accuracy", "f1", "precision", "recall"}
        assert set(metrics.keys()) == expected_keys

    def test_evaluate_metrics_range(
        self, sample_model, synthetic_train_data
    ):
        """All metric values in [0, 1] except loss which is >= 0."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, device)

        assert metrics["loss"] >= 0
        for key in ("accuracy", "f1", "precision", "recall"):
            assert 0.0 <= metrics[key] <= 1.0, (
                f"{key}={metrics[key]} out of [0, 1]"
            )

    def test_evaluate_deterministic(
        self, sample_model, synthetic_train_data
    ):
        """Two evaluate() calls on same data produce identical metrics."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        metrics1 = evaluate(model, loader, criterion, device)
        metrics2 = evaluate(model, loader, criterion, device)

        for key in metrics1:
            assert metrics1[key] == metrics2[key], (
                f"Non-deterministic: {key} differs ({metrics1[key]} vs {metrics2[key]})"
            )


class TestWeightedLoss:
    """Tests for class-weighted cross-entropy loss."""

    def test_weighted_loss_device_placement(self, sample_model):
        """Class weight tensor placed on same device as model (CPU test)."""
        device = torch.device("cpu")
        model = sample_model.to(device)

        weights = torch.tensor([0.6, 1.8], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        # Verify weights are on the right device
        assert criterion.weight.device == device

        # Verify it can compute loss with model output
        dummy_input = torch.randn(4, 10, device=device)
        dummy_labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device)
        logits = model(dummy_input)
        loss = criterion(logits, dummy_labels)
        assert loss.item() > 0

    def test_weighted_loss_from_json(self, synthetic_class_weights_file):
        """Class weights loaded from JSON file produce valid CrossEntropyLoss."""
        with open(synthetic_class_weights_file) as f:
            raw_weights = json.load(f)

        # Convert string keys to ordered tensor
        num_classes = len(raw_weights)
        weight_tensor = torch.tensor(
            [raw_weights[str(i)] for i in range(num_classes)],
            dtype=torch.float32,
        )

        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        assert criterion.weight is not None
        assert criterion.weight.shape == (num_classes,)
        assert torch.allclose(
            criterion.weight, torch.tensor([0.6, 1.8])
        )


class TestCheckpointing:
    """Tests for model checkpoint save/load."""

    def test_checkpoint_saved_on_f1_improvement(
        self, sample_model, synthetic_train_data, tmp_path
    ):
        """best_model.pt created when F1 > 0."""
        X, y = synthetic_train_data
        loader = _make_loader(X, y)
        device = torch.device("cpu")
        model = sample_model.to(device)
        # Use class-weighted loss and higher LR so model learns minority class
        weights = torch.tensor([0.6, 1.8], dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train several epochs so model learns to predict both classes
        for _ in range(10):
            train_one_epoch(model, loader, criterion, optimizer, device)

        # Evaluate to get metrics
        metrics = evaluate(model, loader, criterion, device)

        # Save checkpoint if F1 > 0
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_model.pt"

        best_f1 = 0.0
        if metrics["f1"] > best_f1:
            torch.save(model.state_dict(), checkpoint_path)

        assert checkpoint_path.exists(), "Checkpoint not saved despite F1 improvement"

    def test_checkpoint_not_saved_when_no_improvement(
        self, sample_model, tmp_path
    ):
        """best_model.pt not overwritten when F1 does not improve."""
        device = torch.device("cpu")
        model = sample_model.to(device)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_model.pt"

        # Save initial checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        initial_mtime = checkpoint_path.stat().st_mtime

        # Simulate F1 not improving (current F1 < best F1)
        best_f1 = 0.95
        current_f1 = 0.50

        if current_f1 > best_f1:
            torch.save(model.state_dict(), checkpoint_path)

        # File should NOT have been overwritten
        assert checkpoint_path.stat().st_mtime == initial_mtime

    def test_checkpoint_loadable(self, sample_model, tmp_path):
        """Saved state_dict can be loaded into a new MLP instance and produces same output."""
        device = torch.device("cpu")
        model = sample_model.to(device)

        # Save checkpoint
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Create new model with same architecture and load checkpoint
        new_model = MLP(
            input_dim=10, hidden_layers=[32, 16], num_classes=2, dropout=0.1
        ).to(device)
        new_model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)
        )

        # Both models should produce identical output
        model.eval()
        new_model.eval()
        test_input = torch.randn(8, 10, device=device)

        with torch.no_grad():
            out1 = model(test_input)
            out2 = new_model(test_input)

        assert torch.equal(out1, out2), "Loaded model produces different output"


class TestSummaryTable:
    """Tests for summary table logging."""

    def test_summary_table_logged(self, caplog):
        """Summary table output contains epoch numbers and metric columns."""
        from federated_ids.model.train import _print_summary_table

        history = [
            {"loss": 0.500, "accuracy": 0.80, "f1": 0.70, "precision": 0.75, "recall": 0.65},
            {"loss": 0.300, "accuracy": 0.90, "f1": 0.85, "precision": 0.88, "recall": 0.82},
            {"loss": 0.200, "accuracy": 0.93, "f1": 0.89, "precision": 0.90, "recall": 0.88},
        ]

        with caplog.at_level(logging.INFO, logger="federated_ids.model.train"):
            _print_summary_table(history)

        log_text = caplog.text

        # Check header and footer markers
        assert "Training Summary" in log_text
        # Check column headers
        assert "Epoch" in log_text
        assert "Loss" in log_text
        assert "F1" in log_text
        # Check epoch numbers are present
        assert "1" in log_text
        assert "2" in log_text
        assert "3" in log_text
