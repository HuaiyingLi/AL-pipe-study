"""Abstract base class for dataset."""

import os

from abc import ABC, abstractmethod

import torch

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for dataset."""

    def __init__(
        self,
        data_path: str,
        data_name: str,
        batch_size: int,
        train_val_test_pool_split: list[float],
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        """
        Initialize the dataset.
        """
        super().__init__()
        self.data_path = os.path.join(data_path, data_name)
        self.batch_size = batch_size
        self.train_val_test_pool_split = train_val_test_pool_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @abstractmethod
    def _load_data(self) -> list[torch.Tensor]:
        """Load data from the data path."""
        raise NotImplementedError()

    # @abstractmethod
    # def get_subset(self, indices: list[int]) -> "BaseDataset":
    #     """Return a subset of the dataset."""
    #     raise NotImplementedError()

    def delete(self, indices: list[int]) -> None:
        """Delete the items at the given indices."""
        for index in sorted(indices, reverse=True):
            del self.data[index]
