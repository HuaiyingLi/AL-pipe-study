"""Onehot encoding class for data embedding."""

import pandas as pd
import torch

from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder


class OneHotEmbedder(BaseStaticEmbedder):
    """
    A class for one-hot encoding categorical data.

    This class provides functionality to convert categorical data into one-hot encoded
    numerical representations, which can be used as input features for machine learning models.

    Attributes:
        params: Parameters for the one-hot encoding process
        device (str): Device to run computations on ('cuda' or 'cpu')
        al_data (Data): Data object containing the dataset to be encoded
    """

    def __init__(self, device="cuda") -> None:
        super().__init__(device)

    @staticmethod
    def embed_any_sequences(sequences: pd.Series) -> list[torch.Tensor]:
        """
        Generate one-hot encoded embeddings for any input DNA sequences.

        Args:
            sequences (pd.Series): A pandas Series containing DNA sequences to be encoded.

        Returns:
            list[torch.Tensor]: A list of tensors containing one-hot encoded tensors for each DNA sequence.
                      Each tensor has shape (sequence_length, 4) where 4 represents the four
                      possible nucleotides (A,C,G,T).
        """
        # TODO: fix circular import
        from al_pipe.util.general import onehot_encode_dna

        # TODO: this is not efficient, should be vectorized (if possible)
        return [encoded for seq in sequences if (encoded := onehot_encode_dna(seq)) is not None]
