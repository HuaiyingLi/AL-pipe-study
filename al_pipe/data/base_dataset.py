"""Abstract base class for dataset."""

import os

from abc import ABC, abstractmethod

import torch

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for dataset."""

    def __init__(self, data_path: str, data_name: str, **kwargs) -> None:
        """
        Initialize the dataset.
        """
        self.data_path = os.path.join(data_path, data_name)
        self.transform = kwargs.get("transform", None)

    @abstractmethod
    def _load_data(self) -> list[torch.Tensor]:
        """Load data from the data path."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        raise NotImplementedError()

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the item at the given index."""
        raise NotImplementedError()
