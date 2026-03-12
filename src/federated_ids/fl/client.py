"""Federated learning client with NumPy-based parameter transport.

Implements ``FederatedClient``, a pure-Python federated client that mirrors
the Flower ``NumPyClient`` interface.  This design allows the project to
run federated training without Flower (using the orchestration loop in
Plan 02) while remaining compatible for future Flower migration.

Parameter transport uses NumPy arrays -- the same format Flower's
``NumPyClient`` expects -- so switching to Flower later requires only
subclassing ``flwr.client.NumPyClient`` and delegating to this class.

Example::

    client = FederatedClient(model, train_loader, criterion, optimizer, device)
    params = client.get_parameters()
    updated, n, metrics = client.fit(params, {"local_epochs": 3})
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch

from federated_ids.model.train import train_one_epoch

logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client handling local training and parameter exchange.

    Mirrors the Flower ``NumPyClient`` interface (``get_parameters``,
    ``set_parameters``, ``fit``) so that migration to Flower requires
    minimal changes.

    Args:
        model: PyTorch model to train locally.
        train_loader: DataLoader yielding ``(features, labels)`` batches
            for this client's data partition.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        optimizer: Optimizer instance (e.g. ``Adam``).
        device: Compute device for tensors (``cpu`` or ``cuda``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def get_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays.

        Returns independent copies (not views) so that callers can safely
        store a snapshot without it being mutated by subsequent training.

        Returns:
            List of NumPy arrays, one per entry in ``model.state_dict()``,
            ordered consistently with ``state_dict().keys()``.
        """
        return [
            val.cpu().detach().numpy().copy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters from a list of NumPy arrays into the model.

        Copies values **in-place** into existing model parameter and buffer
        tensors using ``data.copy_()``.  This preserves the tensor objects
        that the optimizer references, avoiding the common bug where
        ``load_state_dict`` replaces tensors and the optimizer silently
        stops producing gradient updates.

        Args:
            parameters: List of NumPy arrays matching the model's
                ``state_dict`` structure (same order and shapes).
        """
        keys = list(self.model.state_dict().keys())
        state_dict = self.model.state_dict()
        for key, new_val in zip(keys, parameters):
            state_dict[key].copy_(torch.tensor(new_val))

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Run local training and return updated parameters.

        Sets the model weights from ``parameters``, trains for
        ``config["local_epochs"]`` epochs on this client's data, and
        returns the updated weights along with the number of training
        examples.

        Args:
            parameters: Global model parameters to start from.
            config: Training configuration dict. Recognised keys:

                - ``local_epochs`` (int): Number of local training epochs.
                  Defaults to 1 if not provided.

        Returns:
            Tuple of ``(updated_parameters, num_examples, metrics)`` where:

            - ``updated_parameters``: List of NumPy arrays with trained weights.
            - ``num_examples``: Number of training samples used.
            - ``metrics``: Empty dict (reserved for future per-client metrics).
        """
        self.set_parameters(parameters)
        local_epochs = config.get("local_epochs", 1)

        self.model.to(self.device)
        for epoch in range(local_epochs):
            loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )
            logger.debug("Local epoch %d/%d -- loss: %.4f", epoch + 1, local_epochs, loss)

        num_examples = len(self.train_loader.dataset)
        return self.get_parameters(), num_examples, {}
