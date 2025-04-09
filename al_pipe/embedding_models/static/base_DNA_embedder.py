"""Abstract base class for DNA embedding."""

from abc import ABC, abstractmethod

import pandas as pd
import torch

from al_pipe.util.general import avail_device


class BaseStaticEmbedder(ABC):
    """
    An abstract base class for all DNA embedding models.
    """

    def __init__(self, sequence_data: pd.Series, device="cuda") -> None:
        self.sequence_data = sequence_data
        self.device = avail_device(device)

    @abstractmethod
    def embed_any_sequences(sequences: pd.Series) -> list[torch.Tensor]:
        """
        Given a pd.Series of DNA sequences, return their embedding representations.

        Args:
            sequences (pd.Series): Series containing DNA sequences to embed

        Returns:
            list[torch.Tensor]: List of torch tensors containing embedded representations
                of the input DNA sequences
        """
        pass

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
        pass
