"""Global seed utility for reproducibility.

Sets all random number generator seeds (Python random, NumPy, PyTorch)
to ensure identical results across repeated experiment runs. This is
critical for thesis reproducibility: same seed must produce same model
weights, same data splits, and same evaluation metrics.

Example:
    >>> from federated_ids.seed import set_global_seed
    >>> set_global_seed(42)
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for all random number generators for reproducibility.

    Configures Python's ``random`` module, NumPy, PyTorch CPU, and
    PyTorch CUDA (if available) to use the specified seed. Also enables
    deterministic cuDNN behavior at the cost of some GPU performance.

    Args:
        seed: The random seed to set across all RNG sources.

    Note:
        Setting ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False`` may reduce GPU
        training speed by 10-20%, but ensures reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
