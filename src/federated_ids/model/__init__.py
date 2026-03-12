"""MLP model definition for DDoS detection.

This subpackage defines the neural network architecture
used for binary classification of network traffic.

Public API
----------
MLP : torch.nn.Module
    Multi-Layer Perceptron for binary DDoS classification.
    Configurable hidden layers, dropout, and output classes.
"""

from federated_ids.model.model import MLP

__all__ = ["MLP"]
