"""Simulated `oracle` labeling."""

import os

import torch

from al_pipe.util.data import load_data


class InSilicoLabeler:
    """
    A class that simulates an oracle labeler for DNA sequences.

    This class provides functionality to generate labels for DNA sequences in silico,
    acting as a simulated oracle in an active learning pipeline. It can be used to
    evaluate active learning strategies without requiring real experimental validation.
    """

    def __init__(self, path: str, data_name: str) -> None:
        self.ground_truth_data = load_data(os.path.join(path, data_name))

    def return_label(self, sequences: list[str] | list[torch.Tensor]) -> torch.Dataset:
        pass
