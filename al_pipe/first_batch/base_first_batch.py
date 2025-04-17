"""Abstract base class for first-batch strategies."""

from abc import ABC, abstractmethod

from al_pipe.data_loader.base_data_loader import BaseDataLoader


class FirstBatchStrategy(ABC):
    """
    An abstract base class for different first batch strategies.
    """

    @abstractmethod
    def select_first_batch(self, data_loader: BaseDataLoader, data_size: dict[str, int]) -> BaseDataLoader:
        """
        Given a pd.Series of initial set of sequence input, return an subset
        of sequences as the first batch of training input.

        Returns:
            BaseDataLoader: A data loader with the first batch of training input.
        """
        raise NotImplementedError()
