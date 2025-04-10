"""Abstract base class for DNA embedding."""

from abc import ABC, abstractmethod

import pandas as pd
import torch

from al_pipe.data.base_dataset import BaseDataset


class BaseStaticEmbedder(ABC):
    """
    An abstract base class for all DNA embedding models.
    """

    def __init__(self, dataset: BaseDataset, device="cuda") -> None:
        from al_pipe.util.general import avail_device

        self.sequence_data = dataset
        self.device = avail_device(device)

    @abstractmethod
    def embed_any_sequences(dataset: BaseDataset | pd.Series) -> list[torch.Tensor]:
        """
        Given a pd.Series of DNA sequences, return their embedding representations.

        Args:
            dataset (BaseDataset | pd.Series): Dataset | pd.Series containing DNA sequences to embed

        Returns:
            list[torch.Tensor]: List of torch tensors containing embedded representations
                of the input DNA sequences
        """
        raise NotImplementedError()

    @abstractmethod
    def embed_loaded_sequences(self) -> list[torch.Tensor]:
        """
        Generate embeddings for DNA sequences that were loaded during initialization.

        This method should be implemented by subclasses to embed the DNA sequences
        that were loaded when the class was instantiated.

        Returns:
            list[torch.Tensor]: List of torch tensors containing embedded representations
                of the loaded DNA sequences
        """
        raise NotImplementedError()
