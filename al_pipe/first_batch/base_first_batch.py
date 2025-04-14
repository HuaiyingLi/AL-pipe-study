"""Abstract base class for first-batch strategies."""

from abc import ABC, abstractmethod

import torch

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.data_loader.base_data_loader import BaseDataLoader


class FirstBatchStrategy(ABC):
    """
    An abstract base class for different first batch strategies.
    """

    def __init__(self, dataset: BaseDataset, data_size: dict[str, int]) -> None:
        self.dataset = dataset
        self.data_size = data_size

    # TODO: move data_size to the constructor of the class (design decision pending)
    @abstractmethod
    def select_first_batch(self, data_loader: BaseDataLoader, data_size: dict[str, int]) -> list[torch.Tensor]:
        """
        Given a pd.Series of initial set of sequence input, return an subset
        of sequences as the first batch of training input.

        Returns:
            list[torch.Tensor]: A list of torch tensors selected as first batch.
        """
        raise NotImplementedError()
