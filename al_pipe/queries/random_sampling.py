"""Simple query strategy."""

import random

import torch

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.data_loader import BaseDataLoader
from al_pipe.queries.base_strategy import BaseQueryStrategy


class RandomQueryStrategy(BaseQueryStrategy):
    """
    Random query strategy for active learning.

    This strategy randomly selects samples from the unlabeled pool for labeling.

    Attributes:
        batch_size (int): Number of samples to select in each query.
    """

    def __init__(self, dataset: BaseDataset, batch_size: int) -> None:
        """
        Initialize the RandomQueryStrategy.

        Args:
            dataset (BaseDataset): The dataset to select samples from.
            batch_size (int): Number of samples to select in each query.
        """
        super().__init__()
        # TODO: what kind of datasets are we importing
        self.dataset = dataset
        self.batch_size = batch_size

    # TODO: First thing tommorow finish this bit of code
    def select_samples(self, model: torch.nn.Module, pool_loader: BaseDataLoader, batch_size: int) -> list[int]:
        """
        Select samples from the unlabeled pool for labeling.

        Args:
            model: The current model being trained (not used in this strategy).
            unlabeled_data: Pool of unlabeled samples to select from.
            batch_size: Number of samples to select.

        Returns:
            List of indices of selected samples from unlabeled pool.
        """
        if batch_size > len(pool_loader):
            raise ValueError("Batch size cannot be greater than the number of unlabeled samples.")
        else:
            selected_indices = random.sample(range(len(pool_loader)), batch_size)

        # update the pool loader
        pool_loader.update_pool_dataset(selected_indices, action_type="remove")

        return selected_indices

    def get_status(self) -> None:
        """
        Returns the status of the selection. Always returns None for this strategy.
        """
        return None
