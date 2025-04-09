"""Random selection strategy for first batch."""

import torch

from al_pipe.first_batch.base_first_batch import FirstBatchStrategy


class RandomSelect(FirstBatchStrategy):
    """
    A strategy that randomly selects sequences for the first batch.

    This class implements random selection of sequences from the initial pool
    of unlabeled data to form the first training batch in an active learning
    pipeline.
    """

    def __init__(self) -> None:
        super().__init__()

    def select_first_batch(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Randomly select sequences for the first batch.

        Args:
            sequences (list[torch.Tensor]): List of sequences to select from

        Returns:
            list[torch.Tensor]: Randomly selected subset of sequences for first batch
        """
        indices = torch.randperm(len(sequences))[: self.batch_size]
        return [sequences[i] for i in indices]
