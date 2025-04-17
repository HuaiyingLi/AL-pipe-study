"""Onehot encoding class for data embedding."""

import pandas as pd
import torch
import torch.nn.functional as F

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

        ## 1) Find the maximum sequence length
        #    (fall back to 0 if there are no sequences)
        # seq_lens = sequences.str.len().tolist()
        # TODO: set max_len outside of this function
        max_len = 100
        # max_len = max(seq_lens) if seq_lens else 0

        embeddings: list[torch.Tensor] = []
        for seq in sequences:
            encoded = onehot_encode_dna(seq)
            if encoded is None:
                continue

            L = encoded.size(0)
            if L < max_len:
                # pad along dim=0 (the sequence dimension) on the "bottom"
                # pad format is (pad_last_dim_left, pad_last_dim_right,
                #                pad_second_last_dim_left, pad_second_last_dim_right, ...)
                encoded = F.pad(encoded, (0, 0, 0, max_len - L), mode="constant", value=0)
                print(f"padded {encoded.shape}")
            else:
                # truncate if longer than max_len
                encoded = encoded[:max_len]

            embeddings.append(encoded)

        return embeddings
