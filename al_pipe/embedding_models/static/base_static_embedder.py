"""Abstract base class for DNA embedding."""

from abc import ABC, abstractmethod

import pandas as pd
import torch


class BaseStaticEmbedder(ABC):
    """
    An abstract base class for all DNA embedding models.
    """

    def __init__(self, device="cuda") -> None:
        # TODO: fix circular import
        from al_pipe.util.general import avail_device

        self.device = avail_device(device)

    @staticmethod
    @abstractmethod
    def embed_any_sequences(sequences: pd.Series) -> list[torch.Tensor]:
        """
        Given a pd.Series of DNA sequences, return their embedding representations.

        Args:
            sequences (pd.Series): pd.Series containing DNA sequences to embed

        Returns:
            list[torch.Tensor]: List of torch tensors containing embedded representations
                of the input DNA sequences
        """
        raise NotImplementedError()
