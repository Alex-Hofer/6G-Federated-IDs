"""Unit tests for the MLP model used in binary DDoS classification.

Tests cover instantiation with various configurations, forward pass
output shapes, raw logit output (no softmax), dropout layer presence,
parameter count sanity, and input dimension enforcement.
"""

import pytest
import torch

from federated_ids.model.model import MLP


@pytest.fixture
def default_model():
    """Create an MLP with default CICIDS2017 configuration.

    Uses the same hyperparameters as config/default.yaml:
    input_dim=25 (typical after feature selection), hidden_layers=[128,64,32],
    num_classes=2 (BENIGN vs DDoS), dropout=0.3.
    """
    return MLP(input_dim=25, hidden_layers=[128, 64, 32], num_classes=2, dropout=0.3)


@pytest.fixture
def sample_batch():
    """Create a synthetic input batch of 16 samples with 25 features."""
    return torch.randn(16, 25)


class TestMLPInstantiation:
    """Tests for MLP constructor and configuration."""

    def test_mlp_instantiation(self, default_model):
        """MLP can be created with default config without error."""
        assert isinstance(default_model, MLP)

    def test_different_hidden_layers(self):
        """MLP works with a different hidden layer configuration."""
        model = MLP(input_dim=25, hidden_layers=[64, 32], num_classes=2, dropout=0.3)
        model.eval()
        x = torch.randn(8, 25)
        out = model(x)
        assert out.shape == (8, 2)

    def test_custom_input_dim(self):
        """MLP with input_dim=10 accepts 10-feature input and rejects 25-feature input."""
        model = MLP(input_dim=10, hidden_layers=[64, 32], num_classes=2, dropout=0.3)
        model.eval()

        # Should work with matching input dimension
        x_valid = torch.randn(4, 10)
        out = model(x_valid)
        assert out.shape == (4, 2)

        # Should fail with mismatched input dimension
        x_invalid = torch.randn(4, 25)
        with pytest.raises(RuntimeError):
            model(x_invalid)

    def test_dropout_layers_present(self, default_model):
        """Model contains Dropout layers with the configured probability."""
        dropout_layers = [
            m for m in default_model.modules() if isinstance(m, torch.nn.Dropout)
        ]
        assert len(dropout_layers) > 0, "No Dropout layers found in model"
        for layer in dropout_layers:
            assert layer.p == pytest.approx(0.3), (
                f"Dropout probability is {layer.p}, expected 0.3"
            )

    def test_parameter_count(self, default_model):
        """Model has a reasonable number of trainable parameters."""
        param_count = sum(p.numel() for p in default_model.parameters() if p.requires_grad)
        # With hidden_layers=[128,64,32], input=25, output=2:
        # Layer 1: 25*128 + 128 = 3328
        # Layer 2: 128*64 + 64 = 8256
        # Layer 3: 64*32 + 32 = 2080
        # Output:  32*2 + 2 = 66
        # Total expected: 13730
        assert param_count > 1000, "Too few parameters"
        assert param_count < 100000, "Too many parameters"


class TestMLPForward:
    """Tests for MLP forward pass behavior."""

    def test_forward_shape(self, default_model, sample_batch):
        """Forward pass on batch of 16 produces shape (16, 2)."""
        default_model.eval()
        out = default_model(sample_batch)
        assert out.shape == (16, 2)

    def test_forward_single_sample(self, default_model):
        """Forward pass on single sample produces shape (1, 2)."""
        default_model.eval()
        x = torch.randn(1, 25)
        out = default_model(x)
        assert out.shape == (1, 2)

    def test_output_is_raw_logits(self, default_model):
        """Output contains raw logits (can be negative, not softmax-clamped)."""
        default_model.eval()
        torch.manual_seed(0)
        # Run multiple batches to increase chance of seeing negative values
        found_negative = False
        for _ in range(10):
            x = torch.randn(32, 25)
            out = default_model(x)
            if (out < 0).any():
                found_negative = True
                break
        assert found_negative, (
            "No negative output values found across 10 batches -- "
            "outputs may be softmax-clamped instead of raw logits"
        )
