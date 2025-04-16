"""DNA data loader module for loading and managing DNA sequence datasets."""

from collections.abc import Callable

import torch

from torch.nn import functional as F

from al_pipe.data.dna_dataset import DNADataset
from al_pipe.data_loader.base_data_loader import BaseDataLoader


class DnaDataLoader(BaseDataLoader):
    r"""Data loader class for DNA sequence data.

    This class handles loading and managing DNA sequence datasets, providing functionality
    for training, validation, testing and pool data loading.
    """

    def __init__(
        self,
        dataset: DNADataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize the DNA data loader.

        Args:
            dataset: The DNA dataset to load data from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data during loading
            num_workers: Number of subprocesses to use for data loading
            pin_memory: Whether to pin memory in GPU training
            collate_fn: The collate function to use for the data
        """
        super().__init__(dataset, batch_size, shuffle, num_workers, pin_memory)
        # TODO: move max_length to the constructor of the class
        self._max_length = dataset.max_length
        self.collate_fn = self.get_collate_fn()

    def dna_collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for DNA sequences of various lengths.
        """
        inputs = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])

        processed_inputs = []
        for tensor in inputs:
            if tensor.shape[0] < self._max_length:
                processed_inputs.append(F.pad(tensor, (0, 0, 0, self._max_length - tensor.shape[0]), "constant", 0))
                # padding = torch.zeros(max_length - tensor.shape[0], *tensor.shape[1:])
                # processed_inputs.append(torch.cat([tensor, padding], dim=0))
            else:
                processed_inputs.append(tensor[: self._max_length])
        return torch.stack(processed_inputs), labels

    def get_collate_fn(self) -> Callable:
        return self.dna_collate_fn
