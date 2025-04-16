"""Specialized dataset class(es) for DNA data."""

import pandas as pd
import torch

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.util.data import load_data


class DNADataset(BaseDataset):
    """Dataset for DNA data."""

    def __init__(
        self,
        data_path: str,
        data_name: str,
        batch_size: int,
        train_val_test_pool_split: list[float],
        max_length: int,
        embedding_model: BaseStaticEmbedder,
    ) -> None:
        super().__init__(data_path, data_name, batch_size, train_val_test_pool_split, embedding_model)
        self.data: pd.DataFrame = self._load_data()
        self.embedded_data: list[torch.Tensor] = self._embed_data()
        self.max_length: int = max_length

    def _embed_data(self) -> list[torch.Tensor]:
        """Embed the data."""
        return self.embedding_model.embed_any_sequences(self.data["sequences"])

    def _load_data(self) -> pd.DataFrame:
        """Load data from the data path.

        This method utilizes the load_data function to read the dataset from the specified
        data path. It returns the loaded data as a list of torch tensors.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded DNA sequences and their values.
        """
        return load_data(self.data_path)

    def get_sequences(self) -> pd.Series:
        """Get the sequences from the dataset.

        Returns:
            pd.Series: A pandas Series containing the sequences from the dataset.
        """
        return self.data["sequences"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embedded_data[index].float(), torch.tensor(self.data["values"].iloc[index]).float()
