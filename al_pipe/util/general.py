"""General functions to help with the project making code neater."""

import random
import numpy as np
import torch


def seed_all(seed):
    """
    Set random seeds for reproducibility across PyTorch, NumPy and Python's random.

    Args:
        seed (int): The random seed value to use for all random number generators.

    Returns:
        np.random.RandomState(): random state from np lib could be used in scikit-learn
    """  # noqa: E501
    # https://scikit-learn.org/stable/common_pitfalls.html
    # TODO: account for the random state instance

    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.RandomState(seed)


def avail_device(device):
    """
    Check device availability and return appropriate device.

    Args:
        device (str): Requested device ('cuda' or 'cpu')

    Returns:
        torch.device: Available device to use
    """
    torch.backends.cudnn.enabled = False
    use_cuda = torch.cuda.is_available()
    device = torch.device(device if use_cuda else "cpu")
    print("device: ", device)
