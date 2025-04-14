"""DNA data loader module for loading and managing DNA sequence datasets."""

from al_pipe.data.dna_dataset import DNADataset
from al_pipe.data_loader.base_data_loader import BaseDataLoader


class DnaDataLoader(BaseDataLoader):
    """Data loader class for DNA sequence data.

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
        """Initialize the DNA data loader.

        Args:
            dataset: The DNA dataset to load data from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data during loading
            num_workers: Number of subprocesses to use for data loading
            pin_memory: Whether to pin memory in GPU training
        """
        super().__init__(dataset, batch_size, shuffle, num_workers, pin_memory)
