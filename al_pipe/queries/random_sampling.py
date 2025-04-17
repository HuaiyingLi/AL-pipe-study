"""Simple query strategy."""

import random

from al_pipe.data_loader.base_data_loader import BaseDataLoader
from al_pipe.queries.base_strategy import BaseQueryStrategy


class RandomQueryStrategy(BaseQueryStrategy):
    """
    Random query strategy for active learning.

    This strategy randomly selects samples from the unlabeled pool for labeling.

    Attributes:
        selection_size (int): Number of samples to select in each query.
    """

    def __init__(self, selection_size: int) -> None:
        super().__init__(selection_size)

    # TODO: add status tracker later
    def select_samples(self, full_data_loader: BaseDataLoader) -> None:
        """
        Select samples from the unlabeled pool for labeling.

        Args:
            pool_loader: Pool of unlabeled samples to select from.

        Returns:
            List of indices of selected samples from unlabeled pool.
        """
        try:
            selected_indices = random.sample(range(len(full_data_loader.get_pool_loader())), self.selection_size)
            # update the pool loader
            full_data_loader.update_train_pool_dataset(selected_indices)
        except ValueError:
            return
