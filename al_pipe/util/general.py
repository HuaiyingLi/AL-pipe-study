"""General functions to help with the project making code neater."""

import random
import numpy as np
import torch
import torch.nn.functional as F


SEQUENCE_CODE = {"A": 0, "C": 1, "T": 2, "G": 3}


def seed_all(seed: int) -> np.random.RandomState:
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


def avail_device(device: str) -> str:
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
    return device


def validate_dna(seq: str) -> None:
    """
    Validates that the DNA sequence contains only the characters A, C, G, and T.

    Args:
        seq (str): The DNA sequence.

    Raises:
        ValueError: If the sequence contains invalid characters.
    """
    allowed = set("ACGT")
    invalid = set(seq.upper()) - allowed
    if invalid:
        raise ValueError(f"Invalid characters in DNA sequence: {invalid}")


def onehot_encode_dna(seq: str) -> torch.LongTensor:
    """Given DNA sequence input covert it to onehot encoded form."""
    # TODO: could extend this to other sequence type
    if validate_dna(seq):
        # Convert the string into a NumPy array
        arr = np.array(list(seq))

        # Use vectorized indexing to map each character to its corresponding index.
        indices = SEQUENCE_CODE[arr]

        # Create the one-hot encoding matrix by indexing into an identity matrix.
        return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=len(SEQUENCE_CODE)).float()


def pad_collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.
    `batch` is a list of tensors of shape (seq_len, num_classes).
    Returns a tensor of shape (batch_size, max_seq_len, num_classes).
    """
    # Use PyTorch's pad_sequence to pad the batch along dimension 0 (time dimension)
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch
