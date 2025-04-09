"""Abstract base class for DNA embedding."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStaticEmbedder(ABC):
    """
    An abstract base class for all DNA embedding models.
    """

    @abstractmethod
    def embed_any_sequences(sequences: pd.Series):
        """
        Given a list or tensor of DNA sequences, return an embedding representation.

        Returns:
            pd.Series: A pandas Series containing embedded representations of the loaded DNA sequences.
        """
        pass

    @abstractmethod
    def embed_loaded_sequences(self):
        """
        Generate embeddings for DNA sequences loaded in the class.

        This method should be implemented by subclasses to embed the DNA sequences
        that were loaded during class initialization.

        Returns:
            pd.Series: A pandas Series containing embedded representations of the loaded DNA sequences.
        """
        pass
