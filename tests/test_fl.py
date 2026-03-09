"""Unit tests for federated learning client and server components.

Tests FederatedClient parameter handling, local training (fit), FedAvg
weighted aggregation, and server-side evaluation.  These tests run in
isolation (no Flower dependency) using synthetic data from conftest.py.
"""

import numpy as np
import torch

from federated_ids.fl.client import FederatedClient
from federated_ids.fl.server import fedavg_aggregate, server_evaluate


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
