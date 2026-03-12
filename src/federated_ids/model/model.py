"""MLP neural network for binary DDoS detection.

Defines a configurable Multi-Layer Perceptron (MLP) used for binary
classification of network traffic as BENIGN or DDoS.  The architecture
is parameterised so that hidden layer sizes, dropout rate, and the
number of output classes can be set from the project configuration
(config/default.yaml).

The forward pass produces **raw logits** -- no softmax is applied.
This is intentional: ``torch.nn.CrossEntropyLoss`` applies
``log_softmax`` internally, so adding softmax in the model would
create a double-softmax bug that prevents learning.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary DDoS classification.

    Builds a feed-forward network dynamically from a list of hidden
    layer sizes.  Each hidden layer is followed by ReLU activation and
    dropout.  The final output layer maps to ``num_classes`` logits
    without any activation (suitable for ``CrossEntropyLoss``).

    Args:
        input_dim: Number of input features (columns after feature
            selection in the preprocessing pipeline).
        hidden_layers: Sizes of hidden layers, e.g. ``[128, 64, 32]``.
        num_classes: Number of output classes (2 for binary DDoS
            detection).
        dropout: Dropout probability applied after each hidden layer.

    Example::

        model = MLP(input_dim=25, hidden_layers=[128, 64, 32],
                     num_classes=2, dropout=0.3)
        logits = model(torch.randn(16, 25))  # shape (16, 2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification layer -- NO activation (raw logits)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Compute raw logits for each sample in the batch.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tensor of shape ``(batch_size, num_classes)`` containing
            raw (un-normalised) logits.
        """
        return self.network(x)
