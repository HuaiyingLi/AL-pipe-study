"""
    1. Load data from disk
    2. Splits into L and U.

------
Write function to load data

Split data into labeled/unlabeled

Test the function on a small sample

"""  # noqa: D205

import os

import pandas as pd
import torch
import torch.nn.functional as F


def load_data(data_path: str) -> pd.DataFrame:
    # load data from path
    """
    Load data from the given file path.

    Parameters:
    - data_path: str, the path to the data file (e.g., CSV)

    Returns:
    - pd.DataFrame containing the loaded data.
    TODO: could support different file type
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    if data_path.lower().endswith(".csv"):
        # default separator here is \t
        data = pd.read_csv(data_path, sep="\t")
        if data.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns.")
        data.columns = ["sequences", "values"]
    else:
        raise ValueError("Unsupported file type. Only CSV are supported.")

    return data


# def dna_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Collate function for DNA sequences of various lengths.
#     """
#     inputs = [item[0] for item in batch]
#     labels = torch.stack([item[1] for item in batch])
#     return torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0), labels


def dna_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]], max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DNA sequences of various lengths.
    """
    inputs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    processed_inputs = []
    for tensor in inputs:
        if tensor.shape[0] < max_length:
            processed_inputs.append(F.pad(tensor, (0, 0, 0, max_length - tensor.shape[0]), "constant", 0))
            # padding = torch.zeros(max_length - tensor.shape[0], *tensor.shape[1:])
            # processed_inputs.append(torch.cat([tensor, padding], dim=0))
        else:
            processed_inputs.append(tensor[:max_length])
    return torch.stack(processed_inputs), labels
