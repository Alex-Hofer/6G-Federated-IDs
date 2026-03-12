"""Device auto-detection for compute acceleration.

Automatically selects the best available compute device for PyTorch
operations. Priority order: NVIDIA CUDA GPU > Apple MPS > CPU.

Example:
    >>> from federated_ids.device import get_device
    >>> device = get_device()
    >>> print(device)
    cpu
"""

import torch


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Checks for GPU acceleration in priority order:
    1. **CUDA** -- NVIDIA GPUs (best performance for training)
    2. **MPS** -- Apple Silicon GPUs (Metal Performance Shaders)
    3. **CPU** -- Fallback, always available

    Returns:
        A ``torch.device`` instance for the best available backend.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
