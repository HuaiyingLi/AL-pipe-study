"""Random selection strategy for first batch."""

import numpy as np

from al_pipe.data_loader.base_data_loader import BaseDataLoader
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy


class RandomFirstBatch(FirstBatchStrategy):
    """
    A strategy that randomly selects sequences for the first batch.

    This class implements random selection of sequences from the initial pool
    of unlabeled data to form the first training batch in an active learning
    pipeline.
    """

    @staticmethod
    def select_first_batch(data_loader: BaseDataLoader, data_size: dict[str, int]) -> BaseDataLoader:
        """
        Randomly select sequences for the first batch and update the data loader.

        This method randomly splits the initial data_set into training, validation, test
        and pool sets according to the sizes specified in self.batch_size. The splits
        are then used to update the provided data loader.

        Args:
            data_loader (BaseDataLoader): The data loader containing the full data_set

        Returns:
            BaseDataLoader: The updated data loader with train/val/test/pool splits
        """
        # TODO: There might be a faster way to split using train_test_split
        # Get total data_set size
        total_size = len(data_loader.get_dataset())

        # Randomly select indices for all splits
        indices = np.random.permutation(total_size)
        split_sizes_ratio = list(data_size.values())

        # Calculate absolute sizes from ratios
        split_sizes = [int(ratio * total_size) for ratio in split_sizes_ratio]
        # Adjust the last element to ensure the sum equals total_size
        split_sizes[-1] = total_size - sum(split_sizes[:-1])

        # Split indices into train/val/test/pool
        start_idx = 0
        split_indices = []
        for size in split_sizes:
            end_idx = start_idx + size
            split_indices.append(indices[start_idx:end_idx])
            start_idx = end_idx

        # Update data loader with new splits
        data_loader.update_train_dataset(split_indices[0], action_type="first-set")
        data_loader.update_val_dataset(split_indices[1], action_type="first-set")
        data_loader.update_test_dataset(split_indices[2], action_type="first-set")
        data_loader.update_pool_dataset(split_indices[3], action_type="first-set")

        return data_loader
