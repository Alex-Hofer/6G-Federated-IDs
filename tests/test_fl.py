"""Unit tests for federated learning client and server components.

Tests FederatedClient parameter handling, local training (fit), FedAvg
weighted aggregation, server-side evaluation, orchestration loop,
convergence checking, metrics persistence, config banner, and
TensorBoard metric logging integration.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from federated_ids.fl.client import FederatedClient
from federated_ids.fl.server import (
    check_convergence,
    fedavg_aggregate,
    print_config_banner,
    run_federated_training,
    save_fl_metrics,
    server_evaluate,
)


class TestFederatedClientParameters:
    """Test FederatedClient.get_parameters() and set_parameters() roundtrip."""

    def test_client_get_set_parameters(
        self, sample_model, fl_train_loaders, fl_criterion
    ):
        """get_parameters returns list of numpy arrays matching model
        state_dict; set_parameters restores them so model output is identical.
        """
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        client = FederatedClient(
            model=sample_model,
            train_loader=fl_train_loaders[0],
            criterion=fl_criterion,
            optimizer=optimizer,
            device=device,
        )

        # Get original parameters
        original_params = client.get_parameters()
        assert isinstance(original_params, list)
        assert all(isinstance(p, np.ndarray) for p in original_params)
        assert len(original_params) == len(list(sample_model.state_dict().keys()))

        # Perturb parameters
        perturbed = [p + 0.5 for p in original_params]
        client.set_parameters(perturbed)

        # Get parameters again -- should match perturbed values
        restored = client.get_parameters()
        for p_set, p_got in zip(perturbed, restored):
            np.testing.assert_array_almost_equal(p_set, p_got, decimal=6)

        # Verify model output is deterministic with set parameters
        sample_input = torch.randn(4, 10)
        sample_model.eval()
        out1 = sample_model(sample_input).detach().numpy()

        # Re-set same parameters
        client.set_parameters(restored)
        sample_model.eval()
        out2 = sample_model(sample_input).detach().numpy()

        np.testing.assert_array_almost_equal(out1, out2, decimal=6)


class TestFederatedClientFit:
    """Test FederatedClient.fit() local training."""

    def test_client_fit(self, sample_model, fl_train_loaders, fl_criterion):
        """fit() returns (parameters, num_examples, metrics) where
        num_examples matches dataset size and parameters differ from input.
        """
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        client = FederatedClient(
            model=sample_model,
            train_loader=fl_train_loaders[0],
            criterion=fl_criterion,
            optimizer=optimizer,
            device=device,
        )

        # Capture initial parameters
        initial_params = client.get_parameters()

        # Run fit
        updated_params, num_examples, metrics = client.fit(
            parameters=initial_params, config={"local_epochs": 2}
        )

        # Check return types
        assert isinstance(updated_params, list)
        assert all(isinstance(p, np.ndarray) for p in updated_params)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)

        # num_examples must match the dataset size
        assert num_examples == len(fl_train_loaders[0].dataset)

        # Parameters must have changed (training happened)
        any_changed = any(
            not np.array_equal(p_init, p_upd)
            for p_init, p_upd in zip(initial_params, updated_params)
        )
        assert any_changed, "Parameters should change after training"


class TestFedAvgAggregation:
    """Test FedAvg weighted aggregation logic."""

    def test_fedavg_aggregate(self):
        """Given 2 clients with different weights (70/30 split),
        aggregated params equal the expected weighted average.
        """
        # Client 1: 70 examples, params all 1.0
        params_1 = [np.ones((3, 2)), np.ones((3,))]
        n_1 = 70

        # Client 2: 30 examples, params all 3.0
        params_2 = [np.full((3, 2), 3.0), np.full((3,), 3.0)]
        n_2 = 30

        results = [(params_1, n_1), (params_2, n_2)]
        aggregated = fedavg_aggregate(results)

        # Expected: (1.0 * 70 + 3.0 * 30) / 100 = 1.6
        expected_val = (1.0 * 70 + 3.0 * 30) / 100.0
        for layer in aggregated:
            np.testing.assert_allclose(layer, expected_val, rtol=1e-6)

    def test_fedavg_equal_weights(self):
        """Given 3 clients with equal num_examples,
        aggregated params equal simple arithmetic mean.
        """
        params_a = [np.array([1.0, 2.0, 3.0])]
        params_b = [np.array([4.0, 5.0, 6.0])]
        params_c = [np.array([7.0, 8.0, 9.0])]

        results = [(params_a, 100), (params_b, 100), (params_c, 100)]
        aggregated = fedavg_aggregate(results)

        expected = np.array([4.0, 5.0, 6.0])  # simple mean
        np.testing.assert_allclose(aggregated[0], expected, rtol=1e-6)


class TestServerEvaluate:
    """Test server-side evaluation."""

    def test_server_evaluate(self, sample_model, fl_test_loader, fl_criterion):
        """server_evaluate() returns dict with all 5 required metric keys."""
        device = torch.device("cpu")

        # Extract current model parameters as numpy arrays
        global_params = [
            val.cpu().numpy() for val in sample_model.state_dict().values()
        ]

        metrics = server_evaluate(
            global_params=global_params,
            model=sample_model,
            test_loader=fl_test_loader,
            criterion=fl_criterion,
            device=device,
        )

        assert isinstance(metrics, dict)
        required_keys = {"loss", "accuracy", "f1", "precision", "recall"}
        assert required_keys == set(metrics.keys()), (
            f"Missing keys: {required_keys - set(metrics.keys())}"
        )

        # All metric values should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"


class TestConvergenceCheck:
    """Test check_convergence with crafted history lists."""

    def test_convergence_check_pass(self):
        """check_convergence returns True when later rounds have higher F1."""
        history = [
            {"f1": 0.50, "loss": 0.5, "accuracy": 0.6, "precision": 0.5, "recall": 0.5},
            {"f1": 0.55, "loss": 0.4, "accuracy": 0.65, "precision": 0.55, "recall": 0.55},
            {"f1": 0.60, "loss": 0.35, "accuracy": 0.7, "precision": 0.6, "recall": 0.6},
            {"f1": 0.70, "loss": 0.3, "accuracy": 0.75, "precision": 0.7, "recall": 0.7},
            {"f1": 0.80, "loss": 0.25, "accuracy": 0.8, "precision": 0.8, "recall": 0.8},
            {"f1": 0.85, "loss": 0.2, "accuracy": 0.85, "precision": 0.85, "recall": 0.85},
        ]
        assert check_convergence(history) is True

    def test_convergence_check_fail(self):
        """check_convergence returns False when later rounds have lower F1."""
        history = [
            {"f1": 0.80, "loss": 0.2, "accuracy": 0.8, "precision": 0.8, "recall": 0.8},
            {"f1": 0.75, "loss": 0.25, "accuracy": 0.75, "precision": 0.75, "recall": 0.75},
            {"f1": 0.70, "loss": 0.3, "accuracy": 0.7, "precision": 0.7, "recall": 0.7},
            {"f1": 0.60, "loss": 0.35, "accuracy": 0.65, "precision": 0.6, "recall": 0.6},
            {"f1": 0.55, "loss": 0.4, "accuracy": 0.6, "precision": 0.55, "recall": 0.55},
            {"f1": 0.50, "loss": 0.5, "accuracy": 0.55, "precision": 0.5, "recall": 0.5},
        ]
        assert check_convergence(history) is False


class TestMetricsJsonOutput:
    """Test save_fl_metrics JSON file creation."""

    def test_metrics_json_output(self):
        """save_fl_metrics creates valid JSON with 'config' and 'rounds' keys."""
        history = [
            {"f1": 0.70, "loss": 0.3, "accuracy": 0.75, "precision": 0.7, "recall": 0.7},
            {"f1": 0.80, "loss": 0.2, "accuracy": 0.85, "precision": 0.8, "recall": 0.8},
        ]
        config = {
            "federation": {"num_clients": 3, "num_rounds": 2, "fraction_fit": 1.0},
            "training": {"learning_rate": 0.001, "local_epochs": 1, "batch_size": 64},
            "model": {"hidden_layers": [128, 64, 32]},
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metrics", "fl_metrics.json")
            save_fl_metrics(history, config, output_path)

            assert os.path.isfile(output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert "config" in data
            assert "rounds" in data
            assert len(data["rounds"]) == 2
            assert data["rounds"][0]["round"] == 1
            assert data["rounds"][1]["round"] == 2


class TestRoundMetricsKeys:
    """Test that each round dict has all 5 required metric keys."""

    def test_round_metrics_keys(
        self, sample_model, fl_train_loaders, fl_test_loader, fl_criterion
    ):
        """Each round dict in history contains loss, accuracy, f1, precision, recall."""
        device = torch.device("cpu")

        # Run a minimal 2-round FL loop inline
        from federated_ids.fl.client import FederatedClient
        from federated_ids.fl.server import fedavg_aggregate, server_evaluate
        from federated_ids.model.model import MLP

        # Get initial parameters
        global_params = [
            val.cpu().detach().numpy().copy()
            for val in sample_model.state_dict().values()
        ]

        history = []
        num_rounds = 2
        num_clients = min(2, len(fl_train_loaders))

        for _round in range(num_rounds):
            round_results = []
            for cid in range(num_clients):
                model = MLP(input_dim=10, hidden_layers=[32, 16], num_classes=2, dropout=0.1)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                client = FederatedClient(
                    model, fl_train_loaders[cid], fl_criterion, optimizer, device
                )
                params, n_examples, _ = client.fit(
                    global_params, {"local_epochs": 1}
                )
                round_results.append((params, n_examples))

            global_params = fedavg_aggregate(round_results)

            eval_model = MLP(input_dim=10, hidden_layers=[32, 16], num_classes=2, dropout=0.1)
            eval_model.to(device)
            metrics = server_evaluate(
                global_params, eval_model, fl_test_loader, fl_criterion, device
            )
            history.append(metrics)

        required_keys = {"loss", "accuracy", "f1", "precision", "recall"}
        for i, metrics in enumerate(history):
            assert required_keys == set(metrics.keys()), (
                f"Round {i + 1} missing keys: {required_keys - set(metrics.keys())}"
            )


class TestConfigBanner:
    """Test print_config_banner does not raise."""

    def test_config_banner(self):
        """print_config_banner with sample config does not raise an exception."""
        config = {
            "federation": {"num_clients": 3, "num_rounds": 20, "fraction_fit": 1.0},
            "training": {"learning_rate": 0.001, "local_epochs": 1, "batch_size": 64},
            "model": {"hidden_layers": [128, 64, 32], "dropout": 0.3},
            "seed": 42,
        }
        # Should not raise
        print_config_banner(config, device="cpu", num_features=25)


class TestConfigDrivenRoundsClients:
    """Test run_federated_training with config-driven rounds and clients."""

    def test_config_driven_rounds_clients(self, tmp_path):
        """run_federated_training with 2 rounds and 2 clients produces
        exactly 2 metric dicts in history.
        """
        # Create synthetic cached tensors
        rng = np.random.RandomState(42)
        n_train, n_test, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        y_train = (rng.rand(n_train) < 0.3).astype(np.int64)
        X_test = rng.randn(n_test, n_features).astype(np.float32)
        y_test = (rng.rand(n_test) < 0.3).astype(np.int64)

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(torch.tensor(X_train), os.path.join(processed_dir, "X_train.pt"))
        torch.save(torch.tensor(y_train), os.path.join(processed_dir, "y_train.pt"))
        torch.save(torch.tensor(X_test), os.path.join(processed_dir, "X_test.pt"))
        torch.save(torch.tensor(y_test), os.path.join(processed_dir, "y_test.pt"))

        # Create class weights file
        weights = {"0": 0.6, "1": 1.8}
        with open(os.path.join(processed_dir, "class_weights.json"), "w") as f:
            json.dump(weights, f)

        output_dir = str(tmp_path / "outputs")

        config = {
            "data": {"processed_dir": processed_dir},
            "model": {
                "hidden_layers": [32, 16],
                "dropout": 0.1,
                "num_classes": 2,
            },
            "training": {
                "learning_rate": 0.001,
                "local_epochs": 1,
                "batch_size": 32,
                "weighted_loss": True,
            },
            "federation": {
                "num_clients": 2,
                "num_rounds": 2,
                "fraction_fit": 1.0,
            },
            "seed": 42,
            "output_dir": output_dir,
            "log_level": "WARNING",
        }

        history = run_federated_training(config)

        assert len(history) == 2, f"Expected 2 rounds, got {len(history)}"


class TestCLIOverrides:
    """Test CLI entry point with programmatic overrides."""

    def test_cli_overrides(self):
        """main() with num_clients=5 and num_rounds=3 overrides config values."""
        from unittest.mock import patch

        from federated_ids.fl.__main__ import main

        with patch(
            "federated_ids.fl.__main__.run_federated_training"
        ) as mock_run:
            mock_run.return_value = []
            main(
                config_path="config/default.yaml",
                num_clients=5,
                num_rounds=3,
            )

            mock_run.assert_called_once()
            call_config = mock_run.call_args[0][0]
            assert call_config["federation"]["num_clients"] == 5
            assert call_config["federation"]["num_rounds"] == 3


class TestTensorBoardLogging:
    """Test TensorBoard SummaryWriter integration in FL training loop."""

    def test_tensorboard_scalars_logged_per_round(self, tmp_path):
        """run_federated_training logs 5 scalars per round via SummaryWriter.

        Uses 2 rounds, expects 10 total add_scalar calls (5 per round)
        with correct tag names, and close() called once.
        """
        # Create synthetic cached tensors
        rng = np.random.RandomState(42)
        n_train, n_test, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        y_train = (rng.rand(n_train) < 0.3).astype(np.int64)
        X_test = rng.randn(n_test, n_features).astype(np.float32)
        y_test = (rng.rand(n_test) < 0.3).astype(np.int64)

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(torch.tensor(X_train), os.path.join(processed_dir, "X_train.pt"))
        torch.save(torch.tensor(y_train), os.path.join(processed_dir, "y_train.pt"))
        torch.save(torch.tensor(X_test), os.path.join(processed_dir, "X_test.pt"))
        torch.save(torch.tensor(y_test), os.path.join(processed_dir, "y_test.pt"))

        output_dir = str(tmp_path / "outputs")
        num_rounds = 2

        config = {
            "data": {"processed_dir": processed_dir},
            "model": {
                "hidden_layers": [32, 16],
                "dropout": 0.1,
                "num_classes": 2,
            },
            "training": {
                "learning_rate": 0.001,
                "local_epochs": 1,
                "batch_size": 32,
                "weighted_loss": False,
            },
            "federation": {
                "num_clients": 2,
                "num_rounds": num_rounds,
                "fraction_fit": 1.0,
            },
            "seed": 42,
            "output_dir": output_dir,
            "log_level": "WARNING",
        }

        mock_writer = MagicMock()

        with patch("federated_ids.fl.server._HAS_TENSORBOARD", True), \
             patch("federated_ids.fl.server.SummaryWriter", return_value=mock_writer):
            history = run_federated_training(config)

        # 5 scalars per round * 2 rounds = 10 calls
        assert mock_writer.add_scalar.call_count == 5 * num_rounds

        # Verify correct tag names were used
        expected_tags = {
            "Global/loss", "Global/accuracy", "Global/f1",
            "Global/precision", "Global/recall",
        }
        actual_tags = {call.args[0] for call in mock_writer.add_scalar.call_args_list}
        assert actual_tags == expected_tags, (
            f"Expected tags {expected_tags}, got {actual_tags}"
        )

        # Verify close() was called exactly once
        mock_writer.close.assert_called_once()

        # Verify correct round numbers were used as step
        round_nums = {call.args[2] for call in mock_writer.add_scalar.call_args_list}
        assert round_nums == {1, 2}

    def test_tensorboard_log_dir_created(self, tmp_path):
        """SummaryWriter is initialized with log_dir under output_dir/tensorboard."""
        rng = np.random.RandomState(42)
        n_train, n_test, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        y_train = (rng.rand(n_train) < 0.3).astype(np.int64)
        X_test = rng.randn(n_test, n_features).astype(np.float32)
        y_test = (rng.rand(n_test) < 0.3).astype(np.int64)

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(torch.tensor(X_train), os.path.join(processed_dir, "X_train.pt"))
        torch.save(torch.tensor(y_train), os.path.join(processed_dir, "y_train.pt"))
        torch.save(torch.tensor(X_test), os.path.join(processed_dir, "X_test.pt"))
        torch.save(torch.tensor(y_test), os.path.join(processed_dir, "y_test.pt"))

        output_dir = str(tmp_path / "outputs")

        config = {
            "data": {"processed_dir": processed_dir},
            "model": {"hidden_layers": [32, 16], "dropout": 0.1, "num_classes": 2},
            "training": {
                "learning_rate": 0.001, "local_epochs": 1,
                "batch_size": 32, "weighted_loss": False,
            },
            "federation": {"num_clients": 2, "num_rounds": 1, "fraction_fit": 1.0},
            "seed": 42,
            "output_dir": output_dir,
            "log_level": "WARNING",
        }

        mock_writer = MagicMock()
        mock_sw_class = MagicMock(return_value=mock_writer)

        with patch("federated_ids.fl.server._HAS_TENSORBOARD", True), \
             patch("federated_ids.fl.server.SummaryWriter", mock_sw_class):
            run_federated_training(config)

        # SummaryWriter should have been called with correct log_dir
        expected_tb_dir = os.path.join(output_dir, "tensorboard")
        mock_sw_class.assert_called_once_with(log_dir=expected_tb_dir)


class TestTensorBoardFallback:
    """Test graceful degradation when TensorBoard is not installed."""

    def test_no_tensorboard_no_error(self, tmp_path):
        """run_federated_training completes without error when _HAS_TENSORBOARD is False."""
        rng = np.random.RandomState(42)
        n_train, n_test, n_features = 200, 50, 10
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        y_train = (rng.rand(n_train) < 0.3).astype(np.int64)
        X_test = rng.randn(n_test, n_features).astype(np.float32)
        y_test = (rng.rand(n_test) < 0.3).astype(np.int64)

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(torch.tensor(X_train), os.path.join(processed_dir, "X_train.pt"))
        torch.save(torch.tensor(y_train), os.path.join(processed_dir, "y_train.pt"))
        torch.save(torch.tensor(X_test), os.path.join(processed_dir, "X_test.pt"))
        torch.save(torch.tensor(y_test), os.path.join(processed_dir, "y_test.pt"))

        output_dir = str(tmp_path / "outputs")

        config = {
            "data": {"processed_dir": processed_dir},
            "model": {"hidden_layers": [32, 16], "dropout": 0.1, "num_classes": 2},
            "training": {
                "learning_rate": 0.001, "local_epochs": 1,
                "batch_size": 32, "weighted_loss": False,
            },
            "federation": {"num_clients": 2, "num_rounds": 2, "fraction_fit": 1.0},
            "seed": 42,
            "output_dir": output_dir,
            "log_level": "WARNING",
        }

        with patch("federated_ids.fl.server._HAS_TENSORBOARD", False):
            history = run_federated_training(config)

        # Should complete normally with 2 rounds of metrics
        assert len(history) == 2
        for metrics in history:
            assert "loss" in metrics
            assert "f1" in metrics
