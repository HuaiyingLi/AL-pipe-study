"""Abstract base class for first-batch strategies."""

from abc import ABC, abstractmethod

import torch


class FirstBatchStrategyFactory(ABC):
    """
    An abstract base class for different first batch strategies.
    """

    @abstractmethod
    def select_first_batch(self) -> list[torch.Tensor]:
        """
        Given a pd.Series of initial set of sequence input, return an subset
        of sequences as the first batch of training input.

        Returns:
            list[torch.Tensor]: A list of torch tensors selected as first batch.
        """
        raise NotImplementedError()
