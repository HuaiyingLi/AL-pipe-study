"""Onehot encoding class for data embedding."""

import pandas as pd

from al_pipe.embedding_models.static.base_DNA_embedder import BaseStaticEmbedder
from al_pipe.util.general import onehot_encode_dna


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

    def __init__(self, sequence_data: pd.Series, device="cuda") -> None:
        super().__init__(sequence_data, device)

    def embed_loaded_sequences(self) -> pd.Series:
        """
        Generate one-hot encoded embeddings for DNA sequences loaded in the class.

        Converts each DNA sequence in self.sequence_data into a one-hot encoded tensor
        representation using the onehot_encode_dna function.

        Returns:
            pd.Series: A pandas Series containing one-hot encoded tensors for each DNA sequence.
                      Each tensor has shape (sequence_length, 4) where 4 represents the four
                      possible nucleotides (A,C,G,T).
        """
        # TODO: stacking can be done later since they are of different length

        # # Convert the encoded series into a list of tensors, then stack them.
        # encoded_tensor = torch.stack(encoded_series.tolist())

        return self.sequence_data.apply(onehot_encode_dna)

    def embed_any_sequences(sequences: pd.Series) -> pd.Series:
        """
        Generate one-hot encoded embeddings for any input DNA sequences.

        Args:
            sequences (pd.Series): A pandas Series containing DNA sequences to be encoded.

        Returns:
            pd.Series: A pandas Series containing one-hot encoded tensors for each DNA sequence.
                      Each tensor has shape (sequence_length, 4) where 4 represents the four
                      possible nucleotides (A,C,G,T).
        """
        return sequences.apply(onehot_encode_dna)
