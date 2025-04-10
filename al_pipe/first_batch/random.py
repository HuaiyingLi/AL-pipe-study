"""Random selection strategy for first batch."""

import numpy as np

from torch.utils.data import DataLoader

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy


class RandomFirstBatch(FirstBatchStrategy):
    """
    A strategy that randomly selects sequences for the first batch.

    This class implements random selection of sequences from the initial pool
    of unlabeled data to form the first training batch in an active learning
    pipeline.
    """

    def __init__(self, dataset: BaseDataset, batch_size: int) -> None:
        super().__init__(dataset, batch_size)

    def select_first_batch(self, dataset: BaseDataset) -> tuple[DataLoader, DataLoader]:
        """
        Randomly select sequences for the first batch.

        This method selects a random subset of sequences from the provided dataset
        to form the initial training batch for the active learning process.

        Args:
            dataset (BaseDataset): The dataset from which to select sequences.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple of two DataLoaders, one for the training batch
            and one for the unlabeled data.
        """
        indices = np.random.choice(len(dataset), self.batch_size, replace=False)
        first_batch_loader = DataLoader(dataset[indices], batch_size=self.batch_size, shuffle=True)
        pool_loader = DataLoader(dataset[~indices], batch_size=self.batch_size, shuffle=False)

        return first_batch_loader, pool_loader
