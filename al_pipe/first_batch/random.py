"""Random selection strategy for first batch."""

import numpy as np

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.data_loader import BaseDataLoader
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy


class RandomFirstBatch(FirstBatchStrategy):
    """
    A strategy that randomly selects sequences for the first batch.

    This class implements random selection of sequences from the initial pool
    of unlabeled data to form the first training batch in an active learning
    pipeline.
    """

    def __init__(self, dataset: BaseDataset, batch_size: dict[str, int]) -> None:
        super().__init__(dataset, batch_size)

    def select_first_batch(self, data_loader: BaseDataLoader) -> BaseDataLoader:
        """
        Randomly select sequences for the first batch and update the data loader.

        This method randomly splits the initial dataset into training, validation, test
        and pool sets according to the sizes specified in self.batch_size. The splits
        are then used to update the provided data loader.

        Args:
            data_loader (BaseDataLoader): The data loader containing the full dataset

        Returns:
            BaseDataLoader: The updated data loader with train/val/test/pool splits
        """
        # TODO: There might be a faster way to split using train_test_split
        # Get total dataset size
        dataset = data_loader.dataset
        total_size = len(dataset)

        # Randomly select indices for all splits
        indices = np.random.permutation(total_size)
        split_sizes = list(self.batch_size.values())

        # Split indices into train/val/test/pool
        start_idx = 0
        split_indices = []
        for size in split_sizes:
            end_idx = start_idx + size
            split_indices.append(indices[start_idx:end_idx])
            start_idx = end_idx

        # Update data loader with new splits
        data_loader.update_train_dataset(split_indices[0])
        data_loader.update_val_dataset(split_indices[1])
        data_loader.update_test_dataset(split_indices[2])
        data_loader.update_pool_dataset(indices[start_idx:])

        return data_loader.get_train_loader()
