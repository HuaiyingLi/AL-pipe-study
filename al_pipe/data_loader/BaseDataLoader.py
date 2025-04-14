"""Base data loader class for loading and managing datasets."""

from abc import ABC, abstractmethod

import torch

from torch.utils.data import DataLoader

from al_pipe.data.base_dataset import BaseDataset


class BaseDataLoader(DataLoader, ABC):
    """Abstract base class for data loading and management.

    This class provides a template for data loaders that handle dataset loading,
    splitting, and batch creation for active learning pipelines.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
    ) -> None:
        """Initialize the data loader.

        Args:
            dataset: BaseDataset object containing the data
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            pin_memory: If True, pin memory for faster data transfer to GPU
            shuffle: If True, shuffle the data at every epoch
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle

        # Initialize data splits
        self._train_dataset: BaseDataset | None = None
        self._val_dataset: BaseDataset | None = None
        self._test_dataset: BaseDataset | None = None
        self._pool_dataset: BaseDataset | None = None

    @abstractmethod
    def load_data(self) -> None:
        """Load and prepare the datasets."""
        raise NotImplementedError()

    def update_train_dataset(self, new_indices: list[int]) -> None:
        """Update training dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to training set
        """
        self._train_dataset = torch.utils.data.Subset(self._dataset, new_indices)

    def update_val_dataset(self, new_indices: list[int]) -> None:
        """Update validation dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to validation set
        """
        self._val_dataset = torch.utils.data.Subset(self._dataset, new_indices)

    def update_test_dataset(self, new_indices: list[int]) -> None:
        """Update test dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to test set
        """
        self._test_dataset = torch.utils.data.Subset(self._dataset, new_indices)

    def update_pool_dataset(self, new_indices: list[int], mode: str = "subset") -> None:
        """Update pool dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to pool set
        """
        if mode == "subset":
            self._pool_dataset = torch.utils.data.Subset(self._dataset, new_indices)
        elif mode == "remove":
            self._pool_dataset = self._pool_dataset.delete(new_indices)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_train_loader(self) -> DataLoader:
        """Get DataLoader for training data.

        Returns:
            DataLoader for training dataset
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset has not been initialized")
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def get_val_loader(self) -> DataLoader:
        """Get DataLoader for validation data.

        Returns:
            DataLoader for validation dataset
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset has not been initialized")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def get_test_loader(self) -> DataLoader:
        """Get DataLoader for test data.

        Returns:
            DataLoader for test dataset
        """
        if self.test_dataset is None:
            raise ValueError("Test dataset has not been initialized")
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def get_pool_loader(self) -> DataLoader:
        """Get DataLoader for pool data.

        Returns:
            DataLoader for pool dataset
        """
        if self.pool_dataset is None:
            raise ValueError("Pool dataset has not been initialized")
        return DataLoader(
            self._pool_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
