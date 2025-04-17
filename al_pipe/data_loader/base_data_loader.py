"""Base data loader class for loading and managing datasets."""

from abc import ABC
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from al_pipe.data.base_dataset import BaseDataset

if TYPE_CHECKING:
    pass


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
        first_batch_strategy=None,
    ) -> None:
        """Initialize the data loader.

        Args:
            dataset: BaseDataset object containing the data
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            pin_memory: If True, pin memory for faster data transfer to GPU
            shuffle: If True, shuffle the data at every epoch
        """
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
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

        if first_batch_strategy is not None:
            self._first_batch_strategy = first_batch_strategy
            # TODO: all need getters to fix
            self._first_batch_strategy.select_first_batch(self, self._dataset.train_val_test_pool_split)

    def update_train_pool_dataset(self, new_indices: list[int]) -> None:
        """Move the selected indices from pool_dataset to train dataset.

        Args:
            new_indices: Indices of new samples to add to training set
        """
        data_to_move = self._pool_dataset.get_subset(new_indices)
        self._train_dataset.append(data_to_move)
        self._pool_dataset.delete(new_indices)
        self._train_dataset.update_embedded_data()
        self._pool_dataset.update_embedded_data()

    def update_train_dataset(self, new_indices: list[int], action_type: str) -> None:
        """Update training dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to training set
            action_type: Type of action to take on the training dataset
        """
        if action_type == "first-set":
            self._train_dataset = self._dataset.return_subset(new_indices)
            self._train_dataset.update_embedded_data()
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def update_val_dataset(self, new_indices: list[int], action_type: str) -> None:
        """Update validation dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to validation set
            action_type: Type of action to take on the validation dataset
        """
        if action_type == "first-set":
            self._val_dataset = self._dataset.return_subset(new_indices)
            self._val_dataset.update_embedded_data()
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def update_test_dataset(self, new_indices: list[int], action_type: str) -> None:
        """Update test dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to test set
            action_type: Type of action to take on the test dataset
        """
        if action_type == "first-set":
            self._test_dataset = self._dataset.return_subset(new_indices)
            self._test_dataset.update_embedded_data()
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def update_pool_dataset(self, new_indices: list[int], action_type: str) -> None:
        """Update pool dataset with new samples.

        Args:
            new_indices: Indices of new samples to add to pool set
            action_type: Type of action to take on the pool dataset
        """
        if action_type == "first-set":
            self._pool_dataset = self._dataset.return_subset(new_indices)
        elif action_type == "remove":
            self._pool_dataset = self._pool_dataset.delete(new_indices)
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        self._pool_dataset.update_embedded_data()

    def get_train_loader(self) -> DataLoader:
        """Get DataLoader for training data.

        Returns:
            DataLoader for training dataset
        """
        if self._train_dataset is None:
            raise ValueError("Training dataset has not been initialized")
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def get_val_loader(self) -> DataLoader:
        """Get DataLoader for validation data.

        Returns:
            DataLoader for validation dataset
        """
        if self._val_dataset is None:
            raise ValueError("Validation dataset has not been initialized")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def get_test_loader(self) -> DataLoader:
        """Get DataLoader for test data.

        Returns:
            DataLoader for test dataset
        """
        if self._test_dataset is None:
            raise ValueError("Test dataset has not been initialized")
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def get_pool_loader(self) -> DataLoader:
        """Get DataLoader for pool data.

        Returns:
            DataLoader for pool dataset
        """
        if self._pool_dataset is None:
            raise ValueError("Pool dataset has not been initialized")
        return DataLoader(
            self._pool_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def get_dataset(self) -> BaseDataset:
        """Get the dataset.

        Returns:
            BaseDataset: The dataset
        """
        return self._dataset
