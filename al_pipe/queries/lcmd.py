"""Simple query strategy."""

import torch

from al_pipe.bmdal_reg.bmdal.algorithms import select_batch
from al_pipe.bmdal_reg.bmdal.feature_data import TensorFeatureData
from al_pipe.data_loader.base_data_loader import BaseDataLoader
from al_pipe.queries.base_strategy import BaseQueryStrategy
from al_pipe.util.general import flat_list_tensor


class LCMDQueryStrategy(BaseQueryStrategy):
    """
    Random query strategy for active learning.

    This strategy randomly selects samples from the unlabeled pool for labeling.

    Attributes:
        selection_size (int): Number of samples to select in each query.
    """

    def __init__(self, selection_size: int) -> None:
        super().__init__(selection_size)

    # TODO: add status tracker later
    def select_samples(self, regressor: torch.nn.Module, full_data_loader: BaseDataLoader) -> None:
        """
        Select samples from the unlabeled pool for labeling.

        Args:
            pool_loader: Pool of unlabeled samples to select from.

        Returns:
            List of indices of selected samples from unlabeled pool.
        """
        # TODO: check if the new indices is from the pool dataset
        # TODO: data is of varying length fix to same length by trimming (should we consistently fix to same length?)
        new_idxs, _ = select_batch(
            batch_size=self.selection_size,
            models=[regressor],
            data={
                "train": TensorFeatureData(flat_list_tensor(full_data_loader.get_train_loader().dataset.embedded_data)),
                "pool": TensorFeatureData(flat_list_tensor(full_data_loader.get_pool_loader().dataset.embedded_data)),
            },
            y_train=full_data_loader.get_train_loader().dataset.get_labels(),
            selection_method="lcmd",
            sel_with_train=True,
            base_kernel="grad",
            kernel_transforms=[("rp", [512])],
        )
        full_data_loader.update_train_pool_dataset(new_idxs)
