"""
Set all random seeds for reproducibility.

Call set_all_seeds() at the top of every script and notebook before any
stochastic operations (model loading, data splitting, etc.).

Usage:
    from src.utils.reproducibility import set_all_seeds
    set_all_seeds(42)
"""

import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and HuggingFace Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (small perf cost, but required for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
